#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPStream.h>
#include "common.h"

template <typename T>
__global__ void index_select_transpose_kernel(
    T *input,
    int *indices,
    T *output,
    int *vec_to_gid,
    int *input_offset,
    int *indices_offset,
    const int blocks_per_vec,
    const int batch_size_per_gpu,
    const int num_vecs,
    const int vector_dim,
    const int bottom_mlp_rank) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;

  int vec_id = bid / blocks_per_vec;
  const int chunk_id = bid % blocks_per_vec;
  const int gid = vec_to_gid[vec_id];
  const int out_stride = num_vecs * vector_dim;;

  const int start_idx = (batch_size_per_gpu * chunk_id) / blocks_per_vec;
  const int end_idx = (batch_size_per_gpu * (chunk_id + 1)) / blocks_per_vec;
  const int start_lane = (vector_dim * tid) / block_size;
  const int end_lane = (vector_dim * (tid + 1)) / block_size;

  input += input_offset[gid];
  output += vec_id * vector_dim;

  if (gid == bottom_mlp_rank) {
    // Simply transpose bottom MLP
    for (int i = start_idx; i < end_idx; i++) {
      for (int l = start_lane; l < end_lane; l++) {
        output[i * out_stride + l] = input[i * vector_dim + l];
      }
    }
  }
  else {
    if (gid > bottom_mlp_rank) {
      vec_id--;
    }
    indices += vec_id * batch_size_per_gpu;

    for (int i = start_idx; i < end_idx; i++) {
      auto idx = indices[i];
      for (int l = start_lane; l < end_lane; l++) {
        output[i * out_stride + l] = input[idx * vector_dim + l];
      }
    }
  }
}

torch::Tensor vec_to_int_tensor(std::vector<int> vec, at::TensorOptions opt) {
  return torch::from_blob(vec.data(), {(int64_t) vec.size()}, opt).to(torch::kCUDA);
}

torch::Tensor index_select_transpose(
    torch::Tensor input,
    torch::Tensor indices, // Indices for embedding output only
    std::vector<int> num_unique_indices,
    std::vector<int> vecs_per_gpu,
    const int batch_size_per_gpu,
    const int bottom_mlp_rank) {
  const int vector_dim = input.size(1);
  const int grid_size = (indices.size(0) + batch_size_per_gpu) / ENTRIES_PER_BLOCK;
  const int blocks_per_vec = batch_size_per_gpu / ENTRIES_PER_BLOCK;

  assert(ENTRIES_PER_BLOCK <= batch_size_per_gpu);
  assert(num_unique_indices.size() == vecs_per_gpu.size());

  std::vector<int> indices_offset;
  std::vector<int> vec_to_gid;
  int offset = 0, gid = 0, num_vecs = 0, sum = 0, tmp = 0, num_embs;
  for (int i = 0; i < num_unique_indices.size(); i++) {
    tmp = num_unique_indices[i];
    num_unique_indices[i] = sum * vector_dim;
    sum += tmp;

    if (i != bottom_mlp_rank) {
      num_embs = vecs_per_gpu[i];
      for (int j = 0; j < num_embs; j++) {
        indices_offset.push_back(offset + j);
        vec_to_gid.push_back(gid);
      }
      offset += num_embs * batch_size_per_gpu;
      num_vecs += num_embs;
    }
    else {
      indices_offset.push_back(-1);
      vec_to_gid.push_back(gid);
      num_vecs++;
    }
    gid++;
  }

  auto output = torch::empty({batch_size_per_gpu, num_vecs, vector_dim}, input.options());
  auto current_stream = at::hip::getCurrentHIPStream(at::hip::current_device()).stream();

  auto int_opts = torch::TensorOptions().dtype(torch::kInt32);
  auto vec_to_gid_tensor = vec_to_int_tensor(vec_to_gid, int_opts);
  auto indices_offset_tensor = vec_to_int_tensor(indices_offset, int_opts);
  auto input_offset_tensor = vec_to_int_tensor(num_unique_indices, int_opts);

  if (input.scalar_type() == torch::ScalarType::Float) {
    hipLaunchKernelGGL(index_select_transpose_kernel, grid_size, warpSize, 0, current_stream,
        input.contiguous().data_ptr<float>(),
        indices.contiguous().data_ptr<int>(),
        output.data_ptr<float>(),
        vec_to_gid_tensor.contiguous().data_ptr<int>(),
        input_offset_tensor.contiguous().data_ptr<int>(),
        indices_offset_tensor.contiguous().data_ptr<int>(),
        blocks_per_vec,
        batch_size_per_gpu,
        num_vecs,
        vector_dim,
        bottom_mlp_rank);
  }
  else if (input.scalar_type() == torch::ScalarType::Half) {
    hipLaunchKernelGGL(index_select_transpose_kernel, grid_size, warpSize, 0, current_stream,
        (half *) input.contiguous().data_ptr<at::Half>(),
        indices.contiguous().data_ptr<int>(),
        (half *) output.data_ptr<at::Half>(),
        vec_to_gid_tensor.contiguous().data_ptr<int>(),
        input_offset_tensor.contiguous().data_ptr<int>(),
        indices_offset_tensor.contiguous().data_ptr<int>(),
        blocks_per_vec,
        batch_size_per_gpu,
        num_vecs,
        vector_dim,
        bottom_mlp_rank);
  }
  else {
    throw std::invalid_argument("Invalid input type.");
  }

  return output;
}
