#include "dot_based_interact_common.cuh"

using namespace dlrm_dot;

template <uint THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
  void dotBasedInteractNoWmmaFwdKernelNonAligned(const half *__restrict input,
                                                 half *__restrict output,
                                                 uint batch_size,
                                                 uint num_rows,
                                                 uint num_cols,
                                                 uint input_size,
                                                 uint output_size,
                                                 uint interaction_output_size) {
  extern __shared__ half smem_f16_fwd_na[];
  half *smem_in = &smem_f16_fwd_na[0];

  uint input_batch_offset = blockIdx.x * input_size;
  const half *gmem_in = &input[input_batch_offset];

  uint output_batch_offset = blockIdx.x * output_size;
  half *gmem_out_bottom_mlp = &output[output_batch_offset];
  half *gmem_out_interaction = &output[output_batch_offset + num_cols];

  // Load the input - one sample per block
  for (uint idx = threadIdx.x; idx < input_size; idx += blockDim.x) {
    smem_in[idx] = gmem_in[idx];
  }
  __syncthreads();

  // Copy bottom MLP output to output
  for (uint idx = threadIdx.x; idx < num_cols; idx += blockDim.x) {
    gmem_out_bottom_mlp[idx] = smem_in[idx];
  }

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
  void dotBasedInteractNoWmmaFwdKernel(const half *__restrict input,
                                       half *__restrict output,
                                       uint batch_size,
                                       uint num_rows,
                                       uint num_cols,
                                       uint input_size,
                                       uint output_size,
                                       uint interaction_output_size) {
  extern __shared__ half smem_f16_fwd[];
  half *smem_in = &smem_f16_fwd[0];

  uint input_batch_offset = blockIdx.x * input_size;
  const half *gmem_in = &input[input_batch_offset];

  uint output_batch_offset = blockIdx.x * output_size;
  half *gmem_out_bottom_mlp = &output[output_batch_offset];
  half *gmem_out_interaction = &output[output_batch_offset + num_cols];

  // Load the input - one sample per block
  uint input_size_half4 = input_size >> 2;
  for (uint idx = threadIdx.x; idx < input_size_half4; idx += blockDim.x) {
    ((half4 *)smem_in)[idx] = ((half4 *)gmem_in)[idx];
  }
  __syncthreads();

  // Copy bottom MLP output to output
  uint btm_mlp_out_size_half4 = num_cols >> 2;
  for (uint idx = threadIdx.x; idx < btm_mlp_out_size_half4; idx += blockDim.x) {
    ((half4 *)gmem_out_bottom_mlp)[idx] = ((half4 *)smem_in)[idx];
  }

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

template <uint THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
  void dotBasedInteractNoWmmaFwdKernelNonAligned(const float *__restrict input,
                                                 float *__restrict output,
                                                 uint batch_size,
                                                 uint num_rows,
                                                 uint num_cols,
                                                 uint input_size,
                                                 uint output_size,
                                                 uint interaction_output_size) {
  extern __shared__ float smem_f32_fwd[];
  float *smem_in = &smem_f32_fwd[0];

  uint input_batch_offset = blockIdx.x * input_size;
  const float *gmem_in = &input[input_batch_offset];

  uint output_batch_offset = blockIdx.x * output_size;
  float *gmem_out_bottom_mlp = &output[output_batch_offset];
  float *gmem_out_interaction = &output[output_batch_offset + num_cols];

  // Load the input - one sample per block
  for (uint idx = threadIdx.x; idx < input_size; idx += blockDim.x) {
    smem_in[idx] = gmem_in[idx];
  }
  __syncthreads();

  // Copy bottom MLP output to output
  for (uint idx = threadIdx.x; idx < num_cols; idx += blockDim.x) {
    gmem_out_bottom_mlp[idx] = smem_in[idx];
  }

  for (uint idx = threadIdx.x; idx < (interaction_output_size); idx += blockDim.x) {
    uint elems_per_row = 1;
    uint index = idx;
    while (index >= elems_per_row) {
      index -= elems_per_row;
      elems_per_row++;
    }
    uint target_row = elems_per_row;
    uint target_col = index;

    float sum = 0;
    for (uint i = 0; i < num_cols; i++) {
      float tmp1 = smem_in[target_row * num_cols + i];
      float tmp2 = smem_in[target_col * num_cols + i];
      sum = fmaf(tmp1, tmp2, sum);
    }

    gmem_out_interaction[idx] = sum;
  }

  gmem_out_interaction[interaction_output_size] = 0;
}

template <uint THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
  void dotBasedInteractNoWmmaFwdKernel(const float *__restrict input,
                                       float *__restrict output,
                                       uint batch_size,
                                       uint num_rows,
                                       uint num_cols,
                                       uint input_size,
                                       uint output_size,
                                       uint interaction_output_size) {
  extern __shared__ float smem_f32_fwd[];
  float *smem_in = &smem_f32_fwd[0];

  uint input_batch_offset = blockIdx.x * input_size;
  const float *gmem_in = &input[input_batch_offset];

  uint output_batch_offset = blockIdx.x * output_size;
  float *gmem_out_bottom_mlp = &output[output_batch_offset];
  float *gmem_out_interaction = &output[output_batch_offset + num_cols];

  // Load the input - one sample per block
  uint input_size_float4 = input_size >> 2;
  for (uint idx = threadIdx.x; idx < input_size_float4; idx += blockDim.x) {
    ((float4 *)smem_in)[idx] = ((float4 *)gmem_in)[idx];
  }
  __syncthreads();

  // Copy bottom MLP output to output
  uint btm_mlp_out_size_float4 = num_cols >> 2;
  for (uint idx = threadIdx.x; idx < btm_mlp_out_size_float4; idx += blockDim.x) {
    ((float4 *)gmem_out_bottom_mlp)[idx] = ((float4 *)smem_in)[idx];
  }

  for (uint idx = threadIdx.x; idx < (interaction_output_size); idx += blockDim.x) {
    uint elems_per_row = 1;
    uint index = idx;
    while (index >= elems_per_row) {
      index -= elems_per_row;
      elems_per_row++;
    }
    uint target_row = elems_per_row;
    uint target_col = index;

    float4 sum;
    sum.x = 0;
    sum.y = 0;
    sum.z = 0;
    sum.w = 0;
    uint num_cols_float4 = num_cols >> 2;
    for (uint i = 0; i < num_cols_float4; i++) {
      float4 tmp1 = ((float4 *)smem_in)[target_row * num_cols_float4 + i];
      float4 tmp2 = ((float4 *)smem_in)[target_col * num_cols_float4 + i];
      sum.x = fmaf(tmp1.x, tmp2.x, sum.x);
      sum.y = fmaf(tmp1.y, tmp2.y, sum.y);
      sum.z = fmaf(tmp1.z, tmp2.z, sum.z);
      sum.w = fmaf(tmp1.w, tmp2.w, sum.w);
    }

    gmem_out_interaction[idx] = sum.x + sum.y + sum.z + sum.w;
  }

  gmem_out_interaction[interaction_output_size] = 0;
}

template <uint THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
  void dotBasedInteractNoWmmaBwdKernelNonAligned(const half *__restrict input,
                                                 const half *__restrict upstream_grad,
                                                 half *__restrict grad,
                                                 half *__restrict bottom_mlp_grad,
                                                 uint batch_size,
                                                 uint num_rows,
                                                 uint num_cols,
                                                 uint input_size,
                                                 uint ugrad_size,
                                                 uint interaction_ugrad_size) {
  extern __shared__ half smem_f16_bwd[];
  half *smem_in = &smem_f16_bwd[0];
  half *smem_interaction_ugrad = &smem_f16_bwd[input_size];

  // Input
  uint input_batch_offset = blockIdx.x * input_size;
  const half *gmem_in = &input[input_batch_offset];

  // Gradient
  const uint &grad_batch_offset = input_batch_offset;
  half *gmem_mlp_grad = &bottom_mlp_grad[blockIdx.x * num_cols];
  half *gmem_interaction_grad = &grad[grad_batch_offset];

  // Upstream Gradient
  uint upstream_grad_batch_offset = blockIdx.x * ugrad_size;
  const half *gmem_mlp_ugrad = &upstream_grad[upstream_grad_batch_offset];
  const half *gmem_interaction_ugrad = &upstream_grad[upstream_grad_batch_offset + num_cols];

  // input -> shared memory
  for (uint idx = threadIdx.x; idx < input_size; idx += blockDim.x) {
    smem_in[idx] = gmem_in[idx];
  }

  // Interaction Upstream Grad -> Shared Memory
  for (uint idx = threadIdx.x; idx < interaction_ugrad_size; idx += blockDim.x) {
    smem_interaction_ugrad[idx] = gmem_interaction_ugrad[idx];
  }
  __syncthreads();

  // Copy the upstream gradient w.r.t to mlp to it's corresponding memory location.
  for (uint idx = threadIdx.x; idx < num_cols; idx += blockDim.x) {
    gmem_mlp_grad[idx] = gmem_mlp_ugrad[idx];
  }

  for (uint idx = threadIdx.x; idx < num_cols; idx += blockDim.x) {
    size_t grad_idx = idx;
    for (uint row_idx = 0; row_idx < num_rows; row_idx++) {
      half sum = __float2half(0);
      size_t upstream_grad_offset = (row_idx * (row_idx - 1)) >> 1;
      for (int k = 0; k < row_idx; k++) {
        sum = __hfma(smem_in[k * num_cols + idx], smem_interaction_ugrad[upstream_grad_offset + k], sum);
      }
      for (int k = row_idx + 1; k < num_rows; k++) {
        upstream_grad_offset = (k * (k - 1)) >> 1;  // TODO: this can become a sum
        sum = __hfma(smem_in[k * num_cols + idx], smem_interaction_ugrad[upstream_grad_offset + row_idx], sum);
      }
      gmem_interaction_grad[grad_idx] = sum;
      grad_idx += num_cols;
    }
  }
}

template <uint THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
  void dotBasedInteractNoWmmaBwdKernel(const half *__restrict input,
                                       const half *__restrict upstream_grad,
                                       half *__restrict grad,
                                       half *__restrict bottom_mlp_grad,
                                       uint batch_size,
                                       uint num_rows,
                                       uint num_cols,
                                       uint input_size,
                                       uint ugrad_size,
                                       uint interaction_ugrad_size) {
  extern __shared__ half smem_f16_bwd[];
  half *smem_in = &smem_f16_bwd[0];
  half *smem_interaction_ugrad = &smem_f16_bwd[input_size];

  // Input
  uint input_batch_offset = blockIdx.x * input_size;
  const half *gmem_in = &input[input_batch_offset];

  // Gradient
  const uint &grad_batch_offset = input_batch_offset;
  half *gmem_mlp_grad = &bottom_mlp_grad[blockIdx.x * num_cols];
  half *gmem_interaction_grad = &grad[grad_batch_offset];

  // Upstream Gradient
  uint upstream_grad_batch_offset = blockIdx.x * ugrad_size;
  const half *gmem_mlp_ugrad = &upstream_grad[upstream_grad_batch_offset];
  const half *gmem_interaction_ugrad = &upstream_grad[upstream_grad_batch_offset + num_cols];

  // input -> shared memory
  uint input_size_half4 = input_size >> 2;
  for (uint idx = threadIdx.x; idx < input_size_half4; idx += blockDim.x) {
    ((half4 *)smem_in)[idx] = ((half4 *)gmem_in)[idx];
  }

  // Interaction Upstream Grad -> Shared Memory
  uint upstream_grad_size_half4 = interaction_ugrad_size >> 2;
  for (uint idx = threadIdx.x; idx < upstream_grad_size_half4; idx += blockDim.x) {
    ((half4 *)smem_interaction_ugrad)[idx] = ((half4 *)gmem_interaction_ugrad)[idx];
  }

  uint vectorized_load_offset = (upstream_grad_size_half4 << 2);
  for (uint idx = vectorized_load_offset + threadIdx.x; idx < interaction_ugrad_size; idx += blockDim.x) {
    smem_interaction_ugrad[idx] = gmem_interaction_ugrad[idx];
  }
  __syncthreads();

  // Copy the upstream gradient w.r.t to mlp to it's corresponding memory location.
  for (uint idx = threadIdx.x; idx < (num_cols >> 2); idx += blockDim.x) {
    ((half4 *)gmem_mlp_grad)[idx] = ((half4 *)gmem_mlp_ugrad)[idx];
  }

  for (uint idx = threadIdx.x; idx < num_cols; idx += blockDim.x) {
    size_t grad_idx = idx;
    for (uint row_idx = 0; row_idx < num_rows; row_idx++) {
      half sum = __float2half(0);
      size_t upstream_grad_offset = (row_idx * (row_idx - 1)) >> 1;
      for (int k = 0; k < row_idx; k++) {
        sum = __hfma(smem_in[k * num_cols + idx], smem_interaction_ugrad[upstream_grad_offset + k], sum);
      }
      for (int k = row_idx + 1; k < num_rows; k++) {
        upstream_grad_offset = (k * (k - 1)) >> 1;  // TODO: this can become a sum
        sum = __hfma(smem_in[k * num_cols + idx], smem_interaction_ugrad[upstream_grad_offset + row_idx], sum);
      }
      gmem_interaction_grad[grad_idx] = sum;
      grad_idx += num_cols;
    }
  }
}

template <uint THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
  void dotBasedInteractNoWmmaBwdKernelNonAligned(const float *__restrict input,
                                                 const float *__restrict upstream_grad,
                                                 float *__restrict grad,
                                                 float *__restrict bottom_mlp_grad,
                                                 uint batch_size,
                                                 uint num_rows,
                                                 uint num_cols,
                                                 uint input_size,
                                                 uint ugrad_size,
                                                 uint interaction_ugrad_size) {
  extern __shared__ float smem_f32_bwd_na[];
  float *smem_in = &smem_f32_bwd_na[0];
  float *smem_interaction_ugrad = &smem_f32_bwd_na[input_size];

  // Input
  uint input_batch_offset = blockIdx.x * input_size;
  const float *gmem_in = &input[input_batch_offset];

  // Gradient
  const uint &grad_batch_offset = input_batch_offset;
  float *gmem_mlp_grad = &bottom_mlp_grad[blockIdx.x * num_cols];
  float *gmem_interaction_grad = &grad[grad_batch_offset];

  // Upstream Gradient
  uint upstream_grad_batch_offset = blockIdx.x * ugrad_size;
  const float *gmem_mlp_ugrad = &upstream_grad[upstream_grad_batch_offset];
  const float *gmem_interaction_ugrad = &upstream_grad[upstream_grad_batch_offset + num_cols];

  // input -> shared memory
  for (uint idx = threadIdx.x; idx < input_size; idx += blockDim.x) {
    smem_in[idx] = gmem_in[idx];
  }

  // Interaction Upstream Grad -> Shared Memory
  for (uint idx = threadIdx.x; idx < interaction_ugrad_size; idx += blockDim.x) {
    smem_interaction_ugrad[idx] = gmem_interaction_ugrad[idx];
  }
  __syncthreads();

  // Copy the upstream gradient w.r.t to mlp to it's corresponding memory location.
  for (uint idx = threadIdx.x; idx < num_cols; idx += blockDim.x) {
    gmem_mlp_grad[idx] = gmem_mlp_ugrad[idx];
  }

  for (uint idx = threadIdx.x; idx < num_cols; idx += blockDim.x) {
    size_t grad_idx = idx;
    for (uint row_idx = 0; row_idx < num_rows; row_idx++) {
      float sum = 0;
      size_t upstream_grad_offset = (row_idx * (row_idx - 1)) >> 1;
      for (int k = 0; k < row_idx; k++) {
        sum = fmaf(smem_in[k * num_cols + idx], smem_interaction_ugrad[upstream_grad_offset + k], sum);
      }
      for (int k = row_idx + 1; k < num_rows; k++) {
        upstream_grad_offset = (k * (k - 1)) >> 1;  // TODO: this can become a sum
        sum = fmaf(smem_in[k * num_cols + idx], smem_interaction_ugrad[upstream_grad_offset + row_idx], sum);
      }
      gmem_interaction_grad[grad_idx] = sum;
      grad_idx += num_cols;
    }
  }
}

template <uint THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
  void dotBasedInteractNoWmmaBwdKernel(const float *__restrict input,
                                       const float *__restrict upstream_grad,
                                       float *__restrict grad,
                                       float *__restrict bottom_mlp_grad,
                                       uint batch_size,
                                       uint num_rows,
                                       uint num_cols,
                                       uint input_size,
                                       uint ugrad_size,
                                       uint interaction_ugrad_size) {
  extern __shared__ float smem_f32_bwd[];
  float *smem_in = &smem_f32_bwd[0];
  float *smem_interaction_ugrad = &smem_f32_bwd[input_size];

  // Input
  uint input_batch_offset = blockIdx.x * input_size;
  const float *gmem_in = &input[input_batch_offset];

  // Gradient
  const uint &grad_batch_offset = input_batch_offset;
  float *gmem_mlp_grad = &bottom_mlp_grad[blockIdx.x * num_cols];
  float *gmem_interaction_grad = &grad[grad_batch_offset];

  // Upstream Gradient
  uint upstream_grad_batch_offset = blockIdx.x * ugrad_size;
  const float *gmem_mlp_ugrad = &upstream_grad[upstream_grad_batch_offset];
  const float *gmem_interaction_ugrad = &upstream_grad[upstream_grad_batch_offset + num_cols];

  // input -> shared memory
  uint input_size_float4 = input_size >> 2;
  for (uint idx = threadIdx.x; idx < input_size_float4; idx += blockDim.x) {
    ((float4 *)smem_in)[idx] = ((float4 *)gmem_in)[idx];
  }

  // Interaction Upstream Grad -> Shared Memory
  uint upstream_grad_size_float4 = interaction_ugrad_size >> 2;
  for (uint idx = threadIdx.x; idx < upstream_grad_size_float4; idx += blockDim.x) {
    ((float4 *)smem_interaction_ugrad)[idx] = ((float4 *)gmem_interaction_ugrad)[idx];
  }

  uint vectorized_load_offset = (upstream_grad_size_float4 << 2);
  for (uint idx = vectorized_load_offset + threadIdx.x; idx < interaction_ugrad_size; idx += blockDim.x) {
    smem_interaction_ugrad[idx] = gmem_interaction_ugrad[idx];
  }
  __syncthreads();

  // Copy the upstream gradient w.r.t to mlp to it's corresponding memory location.
  for (uint idx = threadIdx.x; idx < (num_cols >> 2); idx += blockDim.x) {
    ((float4 *)gmem_mlp_grad)[idx] = ((float4 *)gmem_mlp_ugrad)[idx];
  }

  for (uint idx = threadIdx.x; idx < num_cols; idx += blockDim.x) {
    size_t grad_idx = idx;
    for (uint row_idx = 0; row_idx < num_rows; row_idx++) {
      float sum = 0;
      size_t upstream_grad_offset = (row_idx * (row_idx - 1)) >> 1;
      for (int k = 0; k < row_idx; k++) {
        sum = fmaf(smem_in[k * num_cols + idx], smem_interaction_ugrad[upstream_grad_offset + k], sum);
      }
      for (int k = row_idx + 1; k < num_rows; k++) {
        upstream_grad_offset = (k * (k - 1)) >> 1;  // TODO: this can become a sum
        sum = fmaf(smem_in[k * num_cols + idx], smem_interaction_ugrad[upstream_grad_offset + row_idx], sum);
      }
      gmem_interaction_grad[grad_idx] = sum;
      grad_idx += num_cols;
    }
  }
}

inline void dotBasedInteractNoWmmaFwd(const void *input,
                                      const void *bottom_mlp_output,
                                      const void *output,
                                      uint batch_size,
                                      uint num_rows,
                                      uint num_cols,
                                      uint pad,
                                      bool amp_train) {
  const uint kNumThreads = 128;
  uint num_blocks = batch_size;

  // Output
  uint interaction_output_size = (num_rows * (num_rows - 1)) >> 1;
  uint output_size = num_cols + interaction_output_size + pad;

  // Input
  uint input_size = num_rows * num_cols;

  uint shared_mem_size_elems = input_size;
  uint shared_mem_size_bytes = shared_mem_size_elems << 2;  // F32 Kernel

  bool float4_predicate = !((num_cols & 3) || (output_size & 3));

  if (float4_predicate) {
    if (amp_train) {
      dotBasedInteractNoWmmaFwdKernel<kNumThreads>
          <<<num_blocks, kNumThreads, shared_mem_size_bytes>>>((const half *)input,
                                                               (half *)output,
                                                               batch_size,
                                                               num_rows,
                                                               num_cols,
                                                               input_size,
                                                               output_size,
                                                               interaction_output_size);
    }
    else {
      dotBasedInteractNoWmmaFwdKernel<kNumThreads>
          <<<num_blocks, kNumThreads, shared_mem_size_bytes>>>((const float *)input,
                                                               (float *)output,
                                                               batch_size,
                                                               num_rows,
                                                               num_cols,
                                                               input_size,
                                                               output_size,
                                                               interaction_output_size);
    }
  } else {
    if (amp_train) {
      dotBasedInteractNoWmmaFwdKernelNonAligned<kNumThreads>
          <<<num_blocks, kNumThreads, shared_mem_size_bytes>>>((const half *)input,
                                                               (half *)output,
                                                               batch_size,
                                                               num_rows,
                                                               num_cols,
                                                               input_size,
                                                               output_size,
                                                               interaction_output_size);
    }
    else {
      dotBasedInteractNoWmmaFwdKernelNonAligned<kNumThreads>
          <<<num_blocks, kNumThreads, shared_mem_size_bytes>>>((const float *)input,
                                                               (float *)output,
                                                               batch_size,
                                                               num_rows,
                                                               num_cols,
                                                               input_size,
                                                               output_size,
                                                               interaction_output_size);
    }
  }
}

inline void dotBasedInteractNoWmmaBwd(const void *input,
                                      const void *upstream_grad,
                                      void *grad,
                                      void *bottom_mlp_grad,
                                      uint batch_size,
                                      uint num_rows,
                                      uint num_cols,
                                      uint pad,
                                      bool amp_train) {
  const uint kNumThreads = 128;

  uint num_blocks = batch_size;

  uint input_size = num_rows * num_cols;

  // 1D ugrad size
  uint interaction_ugrad_size = num_rows * (num_rows - 1) >> 1;
  uint interaction_ugrad_size_with_padding = interaction_ugrad_size + pad;
  uint ugrad_size = num_cols + interaction_ugrad_size_with_padding;

  // input space + upstream grad space
  uint smem_size_elems = input_size + interaction_ugrad_size;
  uint smem_size_bytes = smem_size_elems << 2;  // F32 Kernel

  bool float4_predicate = !((interaction_ugrad_size_with_padding & 3) || (num_cols & 3));
  if (float4_predicate) {
    if (amp_train) {
      dotBasedInteractNoWmmaBwdKernel<kNumThreads>
          <<<num_blocks, kNumThreads, smem_size_bytes>>>((const half *)input,
                                                         (const half *)upstream_grad,
                                                         (half *)grad,
                                                         (half *)bottom_mlp_grad,
                                                         batch_size,
                                                         num_rows,
                                                         num_cols,
                                                         input_size,
                                                         ugrad_size,
                                                         interaction_ugrad_size);
    }
    else {
      dotBasedInteractNoWmmaBwdKernel<kNumThreads>
          <<<num_blocks, kNumThreads, smem_size_bytes>>>((const float *)input,
                                                         (const float *)upstream_grad,
                                                         (float *)grad,
                                                         (float *)bottom_mlp_grad,
                                                         batch_size,
                                                         num_rows,
                                                         num_cols,
                                                         input_size,
                                                         ugrad_size,
                                                         interaction_ugrad_size);
    }
  } else {
    if (amp_train) {
      dotBasedInteractNoWmmaBwdKernelNonAligned<kNumThreads>
          <<<num_blocks, kNumThreads, smem_size_bytes>>>((const half *)input,
                                                         (const half *)upstream_grad,
                                                         (half *)grad,
                                                         (half *)bottom_mlp_grad,
                                                         batch_size,
                                                         num_rows,
                                                         num_cols,
                                                         input_size,
                                                         ugrad_size,
                                                         interaction_ugrad_size);
    }
    else {
      dotBasedInteractNoWmmaBwdKernelNonAligned<kNumThreads>
          <<<num_blocks, kNumThreads, smem_size_bytes>>>((const float *)input,
                                                         (const float *)upstream_grad,
                                                         (float *)grad,
                                                         (float *)bottom_mlp_grad,
                                                         batch_size,
                                                         num_rows,
                                                         num_cols,
                                                         input_size,
                                                         ugrad_size,
                                                         interaction_ugrad_size);
    }
  }
}
