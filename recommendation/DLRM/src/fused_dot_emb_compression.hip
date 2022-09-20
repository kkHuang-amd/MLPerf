#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPStream.h>
#include "common.h"

struct __align__(8) half4 {
  half2 vals[2];
};

template <uint THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void fused_index_select_dot_f16_nonaligned_kernel(
        const half *__restrict emb_output, // input
        const half *__restrict bottom_mlp_output, // input
        const int *__restrict indices,
        const int *__restrict vecs_per_gpu,
        const int *__restrict indices_offset,
        half *__restrict output,
        uint batch_size,
        uint num_rows,
        uint num_cols,
        uint input_size,
        uint output_size,
        uint interaction_output_size,
        const int num_gpus) {
  extern __shared__ half smem_f16_fwd[];
  half *smem_in = &smem_f16_fwd[0];

  // Input
  const half *gmem_bottom_mlp_output = &bottom_mlp_output[blockIdx.x * num_cols];

  uint output_batch_offset = blockIdx.x * output_size;
  half *gmem_out_bottom_mlp = &output[output_batch_offset];
  half *gmem_out_interaction = &output[output_batch_offset + num_cols];

  // Load bottom MLP into smem
  for (uint idx = threadIdx.x; idx < num_cols; idx += blockDim.x) {
    gmem_out_bottom_mlp[idx] = smem_in[idx] = gmem_bottom_mlp_output[idx];
  }

  // Load embedding output into smem
  int vecs_per_gpu_offset = 0;
  int fea_id = 1;
  for (uint gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
    const int *gpu_indices = &indices[vecs_per_gpu_offset * batch_size];
    const half *gpu_emb_output = &emb_output[indices_offset[gpu_id] * num_cols];
    const int num_vecs = vecs_per_gpu[gpu_id];
    for (uint vec_id = 0; vec_id < num_vecs; vec_id++) {
      uint idx = gpu_indices[vec_id * batch_size + blockIdx.x];
      for (uint col = threadIdx.x; col < num_cols; col += blockDim.x) {
        smem_in[fea_id * num_cols + col] = gpu_emb_output[idx * num_cols + col];
      }
      fea_id++;
    }
    vecs_per_gpu_offset += num_vecs;
  }
  __syncthreads();

  for (uint idx = threadIdx.x; idx < (interaction_output_size); idx += blockDim.x) {
    uint elems_per_row = 1;
    uint index = idx;
    while (index >= elems_per_row) {
      index -= elems_per_row;
      elems_per_row++;
    }
    uint target_row = elems_per_row;
    uint target_col = index;

    half sum = __float2half(0);
    for (uint i = 0; i < num_cols; i++) {
      half tmp1 = smem_in[target_row * num_cols + i];
      half tmp2 = smem_in[target_col * num_cols + i];
      sum = __hfma(tmp1, tmp2, sum);
    }

    gmem_out_interaction[idx] = sum;
  }

  gmem_out_interaction[interaction_output_size] = __float2half(0);
}

template <uint THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void fused_index_select_dot_f16_kernel(
        const half *__restrict emb_output, // input
        const half *__restrict bottom_mlp_output, // input
        const int *__restrict indices,
        const int *__restrict vecs_per_gpu,
        const int *__restrict indices_offset,
        half *__restrict output,
        uint batch_size,
        uint num_rows,
        uint num_cols,
        uint input_size,
        uint output_size,
        uint interaction_output_size,
        const int num_gpus) {
  extern __shared__ half smem_f16_fwd[];
  half *smem_in = &smem_f16_fwd[0];

  // Input
  const half *gmem_bottom_mlp_output = &bottom_mlp_output[blockIdx.x * num_cols];

  uint output_batch_offset = blockIdx.x * output_size;
  half *gmem_out_bottom_mlp = &output[output_batch_offset];
  half *gmem_out_interaction = &output[output_batch_offset + num_cols];

  // Load bottom MLP into smem
  for (uint idx = threadIdx.x; idx < num_cols; idx += blockDim.x) {
    gmem_out_bottom_mlp[idx] = smem_in[idx] = gmem_bottom_mlp_output[idx];
  }

  // Load embedding output into smem
  int vecs_per_gpu_offset = 0;
  int fea_id = 1;
  for (uint gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
    const int *gpu_indices = &indices[vecs_per_gpu_offset * batch_size];
    const half *gpu_emb_output = &emb_output[indices_offset[gpu_id] * num_cols];
    const int num_vecs = vecs_per_gpu[gpu_id];
    for (uint vec_id = 0; vec_id < num_vecs; vec_id++) {
      uint idx = gpu_indices[vec_id * batch_size + blockIdx.x];
      for (uint col = threadIdx.x; col < num_cols; col += blockDim.x) {
        smem_in[fea_id * num_cols + col] = gpu_emb_output[idx * num_cols + col];
      }
      fea_id++;
    }
    vecs_per_gpu_offset += num_vecs;
  }
  __syncthreads();

  for (uint idx = threadIdx.x; idx < (interaction_output_size); idx += blockDim.x) {
    uint elems_per_row = 1;
    uint index = idx;
    while (index >= elems_per_row) {
      index -= elems_per_row;
      elems_per_row++;
    }
    uint target_row = elems_per_row;
    uint target_col = index;

    half4 sum;
    sum.vals[0] = __float2half2_rn(0);
    sum.vals[1] = __float2half2_rn(0);
    uint num_cols_half4 = num_cols >> 2;
    for (uint i = 0; i < num_cols_half4; i++) {
      half4 tmp1 = ((half4 *)smem_in)[target_row * num_cols_half4 + i];
      half4 tmp2 = ((half4 *)smem_in)[target_col * num_cols_half4 + i];
      sum.vals[0] = __hfma2(tmp1.vals[0], tmp2.vals[0], sum.vals[0]);
      sum.vals[1] = __hfma2(tmp1.vals[1], tmp2.vals[1], sum.vals[1]);
    }

    half sum_val0 = __hadd(__low2half(sum.vals[0]), __high2half(sum.vals[0]));
    half sum_val1 = __hadd(__low2half(sum.vals[1]), __high2half(sum.vals[1]));
    gmem_out_interaction[idx] = __hadd(sum_val0, sum_val1);
  }

  gmem_out_interaction[interaction_output_size] = __float2half(0);
}

torch::Tensor fused_index_select_dot(torch::Tensor emb_output,
                                     torch::Tensor bottom_mlp_output,
                                     torch::Tensor indices,
                                     torch::Tensor vecs_per_gpu,
                                     torch::Tensor indices_offset,
                                     uint batch_size,
                                     uint num_features,
                                     uint emb_dim,
                                     uint pad) {
  uint interaction_output_size = (num_features * (num_features - 1)) >> 1;
  uint output_size = interaction_output_size + emb_dim + pad;
  int64_t output_shape[2] = {batch_size, output_size};
  auto output = torch::empty(c10::IntArrayRef(output_shape), emb_output.options());
  uint input_size = num_features * emb_dim;

  const uint kNumThreads = 128;
  uint num_blocks = batch_size;
  uint shared_mem_size_elems = num_features * emb_dim;

  const int num_gpus = vecs_per_gpu.numel();
  bool float4_predicate = !((emb_dim & 3) || (output_size & 3));

  if (float4_predicate) {
    if (emb_output.scalar_type() == torch::ScalarType::Half) {
      uint shared_mem_size_bytes = shared_mem_size_elems << 1;
      fused_index_select_dot_f16_kernel<kNumThreads>
          <<<num_blocks, kNumThreads, shared_mem_size_bytes>>>(
              (const half *) emb_output.contiguous().data_ptr<at::Half>(),
              (const half *) bottom_mlp_output.contiguous().data_ptr<at::Half>(),
              indices.contiguous().data_ptr<int>(),
              vecs_per_gpu.data_ptr<int>(),
              indices_offset.data_ptr<int>(),
              (half *) output.contiguous().data_ptr<at::Half>(),
              batch_size,
              num_features, // num_rows
              emb_dim, // num_cols
              input_size,
              output_size,
              interaction_output_size,
              num_gpus);
    }
    else {
      throw std::invalid_argument("fused_index_select_dot does not support FP32");
    }
  }
  else {
    if (emb_output.scalar_type() == torch::ScalarType::Half) {
      uint shared_mem_size_bytes = shared_mem_size_elems << 1;
      fused_index_select_dot_f16_nonaligned_kernel<kNumThreads>
          <<<num_blocks, kNumThreads, shared_mem_size_bytes>>>(
              (const half *) emb_output.contiguous().data_ptr<at::Half>(),
              (const half *) bottom_mlp_output.contiguous().data_ptr<at::Half>(),
              indices.contiguous().data_ptr<int>(),
              vecs_per_gpu.data_ptr<int>(),
              indices_offset.data_ptr<int>(),
              (half *) output.contiguous().data_ptr<at::Half>(),
              batch_size,
              num_features, // num_rows
              emb_dim, // num_cols
              input_size,
              output_size,
              interaction_output_size,
              num_gpus);
    }
    else {
      throw std::invalid_argument("fused_index_select_dot does not support FP32");
    }
  }
  return output;
}

template <uint THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void fused_dot_dedup_f16_kernel(const half *__restrict emb_output, // input
                                    const half *__restrict bottom_mlp_output, // input
                                    const int *__restrict indices,
                                    const int *__restrict vecs_per_gpu,
                                    const int *__restrict indices_offset,
                                    const half *__restrict upstream_grad,
                                    float *__restrict grad,
                                    uint batch_size,
                                    uint num_rows,
                                    uint num_cols,
                                    uint input_size,
                                    uint ugrad_size,
                                    uint interaction_ugrad_size,
                                    const int num_gpus,
                                    const int smem_offset) {
  extern __shared__ half smem_f16_bwd[];
  uint *smem_idx = reinterpret_cast<uint *>(smem_f16_bwd);
  half *smem_in = &smem_f16_bwd[smem_offset];
  half *smem_interaction_ugrad = &smem_f16_bwd[smem_offset + input_size];

  // Input
  const half *gmem_bottom_mlp_output = &bottom_mlp_output[blockIdx.x * num_cols];

  // Gradient
  float *gmem_mlp_grad = &grad[blockIdx.x * num_cols];
  float *gmem_grad = &grad[batch_size * num_cols];

  // Upstream Gradient
  uint upstream_grad_batch_offset = blockIdx.x * ugrad_size;
  const half *gmem_mlp_ugrad = &upstream_grad[upstream_grad_batch_offset];
  const half *gmem_interaction_ugrad = &upstream_grad[upstream_grad_batch_offset + num_cols];

  // Load bottom MLP into smem
  for (uint idx = threadIdx.x; idx < num_cols; idx += blockDim.x) {
    smem_in[idx] = gmem_bottom_mlp_output[idx];
  }

  // Load embedding output into smem
  int vecs_per_gpu_offset = 0;
  int fea_id = 1;
  const int num_emb_features = num_rows - 1;
  for (uint gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
    const int *gpu_indices = &indices[vecs_per_gpu_offset * batch_size];
    const int index_offset = indices_offset[gpu_id];
    const half *gpu_emb_output = &emb_output[index_offset * num_cols];
    const int num_vecs = vecs_per_gpu[gpu_id];
    for (uint vec_id = 0; vec_id < num_vecs; vec_id++) {
      uint idx = gpu_indices[vec_id * batch_size + blockIdx.x];
      if (threadIdx.x == 0) {
        // Store in smem for future use
        smem_idx[fea_id - 1] = (idx + index_offset) * num_cols;
      }
      for (uint col = threadIdx.x; col < num_cols; col += blockDim.x) {
        smem_in[fea_id * num_cols + col] = gpu_emb_output[idx * num_cols + col];
      }
      fea_id++;
    }
    vecs_per_gpu_offset += num_vecs;
  }

  // Interaction Upstream Grad -> Shared Memory
  for (uint idx = threadIdx.x; idx < interaction_ugrad_size; idx += blockDim.x) {
    smem_interaction_ugrad[idx] = gmem_interaction_ugrad[idx];
  }
  __syncthreads();

  // Special case for the bottom MLP gradients
  for (uint idx = threadIdx.x; idx < num_cols; idx += blockDim.x) {
    half sum = __float2half(0);
    for (int k = 1; k < num_rows; k++) {
      size_t upstream_grad_offset = (k * (k - 1)) >> 1;
      sum = __hfma(smem_in[k * num_cols + idx], smem_interaction_ugrad[upstream_grad_offset], sum);
    }
    // Add bottom grad and bottom MLP grad together
    gmem_mlp_grad[idx] = __half2float(__hadd(sum, gmem_mlp_ugrad[idx]));
  }

  // Compute embedding gradient
  for (uint idx = threadIdx.x; idx < num_cols; idx += blockDim.x) {
    for (uint row_idx = 1; row_idx < num_rows; row_idx++) {
      half sum = __float2half(0);
      size_t upstream_grad_offset = (row_idx * (row_idx - 1)) >> 1;
      for (int k = 0; k < row_idx; k++) {
        sum = __hfma(smem_in[k * num_cols + idx], smem_interaction_ugrad[upstream_grad_offset + k], sum);
      }
      for (int k = row_idx + 1; k < num_rows; k++) {
        upstream_grad_offset = (k * (k - 1)) >> 1;
        sum = __hfma(smem_in[k * num_cols + idx], smem_interaction_ugrad[upstream_grad_offset + row_idx], sum);
      }
      atomicAddNoRet(&gmem_grad[smem_idx[row_idx - 1] + idx], __half2float(sum));
    }
  }
}

torch::Tensor fused_dot_dedup(torch::Tensor emb_output,
                              torch::Tensor bottom_mlp_output,
                              torch::Tensor indices,
                              torch::Tensor vecs_per_gpu,
                              torch::Tensor indices_offset,
                              torch::Tensor upstream_grad,
                              uint batch_size,
                              uint num_features,
                              uint emb_dim,
                              uint pad) {
  // Type is float32 because atomicAdd does not support half
  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  int64_t total_vectors = emb_output.sizes()[0] + bottom_mlp_output.sizes()[0];
  auto emb_grad = torch::zeros({total_vectors, (int64_t) emb_dim}, opts);

  const uint kNumThreads = 128;
  uint num_blocks = batch_size;
  uint input_size = num_features * emb_dim;

  // 1D ugrad size
  uint interaction_ugrad_size = num_features * (num_features - 1) >> 1;
  uint interaction_ugrad_size_with_padding = interaction_ugrad_size + pad;
  uint ugrad_size = emb_dim + interaction_ugrad_size_with_padding;

  // input space + upstream grad space
  uint smem_size_elems = input_size + interaction_ugrad_size;
  const int num_gpus = vecs_per_gpu.numel();

  if (emb_output.scalar_type() == torch::ScalarType::Half) {
    uint smem_offset = (num_features - 1) << 1; // << 1 because of half elements
    uint smem_size_bytes = (smem_size_elems + smem_offset) << 1;
    fused_dot_dedup_f16_kernel<kNumThreads>
        <<<num_blocks, kNumThreads, smem_size_bytes>>>(
            (const half *) emb_output.contiguous().data_ptr<at::Half>(),
            (const half *) bottom_mlp_output.contiguous().data_ptr<at::Half>(),
            indices.contiguous().data_ptr<int>(),
            vecs_per_gpu.data_ptr<int>(),
            indices_offset.data_ptr<int>(),
            (const half *) upstream_grad.contiguous().data_ptr<at::Half>(),
            (float *) emb_grad.data_ptr<float>(),
            batch_size,
            num_features,
            emb_dim,
            input_size,
            ugrad_size,
            interaction_ugrad_size,
            num_gpus,
            smem_offset);
    emb_grad = emb_grad.to(torch::kFloat16);
  }
  else {
    throw std::invalid_argument("fused_index_select_dot does not support FP32");
  }
  return emb_grad;
}
