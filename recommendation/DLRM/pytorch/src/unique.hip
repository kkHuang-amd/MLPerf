#include <torch/extension.h>
#include <vector>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPStream.h>

#define NUM_THREADS 1024

template <typename T1, typename T2>
__device__ void bitonic_sort_pair(
    T1 *key, T2 *value, int n) {
  int tid = threadIdx.x;
  int block_size = blockDim.x;
  int start = (n * tid) / block_size;
  int end = (n * (tid + 1)) / block_size;
  for (int ss = 2; ss <= n; ss *= 2) {
    for (int sg = ss; sg >= 2; sg /= 2) {
      int d = sg / 2;
      for (int i = start; i < end; i++) {
        if (i % sg < d) {
          int sg_t = (i / ss) % 2;
          int j = i + d;
          T1 a = key[i];
          T1 b = key[j];
          if ((sg_t == 0 && a > b) ||
              (sg_t == 1 && a < b)) {
            // swap keys
            key[i] = b;
            key[j] = a;

            // swap values
            T2 val = value[i];
            value[i] = value[j];
            value[j] = val;
          }
        }
      }
      __syncthreads();
    }
  }
}

__device__ void exclusive_sum_scan(
  int *arr,
  const int num_entries,
  const int block_size,
  const int tid) {
  int offset = 1;
  int num_blocks = num_entries / block_size;
  for (int d = num_entries >> 1; d > 0; d >>= 1) {
    __syncthreads();
    for (int b = 0; b < num_blocks; b++) {
      int pid = tid + (block_size * b);
      if (pid < d) {
        int ai = offset * (2 * pid + 1) - 1;
        int bi = offset * (2 * pid + 2) - 1;
        arr[bi] += arr[ai];
      }
    }
    offset *= 2;
  }

  if (tid == 0) {
    arr[num_entries - 1] = 0;
  }

  for (int d = 1; d < num_entries; d *= 2) {
    offset >>= 1;
    __syncthreads();
    for (int b = 0; b < num_blocks; b++) {
      int pid = tid + (block_size * b);
      if (pid < d) {
        int ai = offset * (2 * pid + 1) - 1;
        int bi = offset * (2 * pid + 2) - 1;
        float tmp = arr[ai];
        arr[ai] = arr[bi];
        arr[bi] += tmp;
      }
    }
  }
}

__global__ void __launch_bounds__(NUM_THREADS) large_unique_transpose_kernel(
    int *input,
    int *sorted_indices,
    const int num_entries,
    int *output,
    int *inverse_indices,
    int *num_uniques) {
  HIP_DYNAMIC_SHARED(int, smem);

  const int row = blockIdx.x;
  const int col = blockIdx.y;
  const int num_rows = gridDim.x;
  const int num_cols = gridDim.y;
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;
  const int in_block_offset = (row * num_cols + col) * num_entries;
  const int out_bid = col * num_rows + row;
  const int out_block_offset = out_bid * num_entries;
  const int start_idx = (num_entries * tid) / block_size;
  const int end_idx = (num_entries * (tid + 1)) / block_size;

  // Shift input and output global memory buffers
  int *input_ = input + in_block_offset;
  int *sorted_indices_ = sorted_indices + in_block_offset;
  int *output_ = output + out_block_offset;
  int *inverse_indices_ = inverse_indices + out_block_offset;
  int *num_uniques_ = num_uniques + out_bid + 1;

  // Shift shared memory buffers
  int *smem_input = smem;
  int *idx_offsets = smem;

  // Create indices
  for (int i = start_idx; i < end_idx; i++) {
    smem_input[i] = input_[i];
    sorted_indices_[i] = i;
  }

  __syncthreads();

  bitonic_sort_pair(smem_input, sorted_indices_, num_entries);

  int idx_offset = 0;
  bool is_last_uniq = false;

  if (tid == 0) {
    input_[0] = smem_input[0];
    idx_offsets[0] = 1;
    idx_offset = 1;
  }

  int idx, pidx;

  if (tid == 0) {
    pidx = input_[start_idx + idx_offset - 1];
  }
  else {
    pidx = input_[start_idx + idx_offset - 1] = smem_input[start_idx + idx_offset - 1];
  }
  __syncthreads();

  for (int i = start_idx + idx_offset; i < end_idx; i++) {
    input_[i] = idx = smem_input[i];
    idx_offsets[i] = pidx != idx;
    pidx = idx;
  }

  if (tid == block_size - 1) {
    is_last_uniq = idx_offsets[end_idx - 1];
  }

  __syncthreads();

  exclusive_sum_scan(idx_offsets, num_entries, block_size, tid);

  __syncthreads();

  // Copy outputs from shared memory to global memory
  if (tid == block_size - 1) {
    idx_offset = 1;
  }
  else {
    idx_offset = 0;
  }

  int pos = idx_offsets[start_idx];
  int npos;
  for (int i = start_idx; i < end_idx - idx_offset; i++) {
    npos = idx_offsets[i + 1];
    // Use inclusive sum for inverese indices (shift exclusive sum to the left)
    inverse_indices_[sorted_indices_[i]] = npos - 1;
    if (npos - pos == 1) {
      output_[pos] = input_[i];
    }
    pos = npos;
  }

  if (tid == block_size - 1) {
    // Set last unique index
    int i = end_idx - 1;
    int pos = idx_offsets[i];
    if (is_last_uniq) {
      inverse_indices_[sorted_indices_[i]] = pos;
      output_[pos] = input_[i];
      *num_uniques_ = pos + 1;
    }
    else {
      inverse_indices_[sorted_indices_[i]] = pos - 1;
      *num_uniques_ = pos;
    }
    if (out_bid == 0) {
      // For cumsum in the next step
      num_uniques[0] = 0;
    }
  }
}

__global__ void __launch_bounds__(NUM_THREADS) small_unique_transpose_kernel(
    int *input,
    int *sorted_indices,
    const int num_entries,
    int *output,
    int *inverse_indices,
    int *num_uniques) {
  HIP_DYNAMIC_SHARED(int, smem);

  const int row = blockIdx.x;
  const int col = blockIdx.y;
  const int num_rows = gridDim.x;
  const int num_cols = gridDim.y;
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;
  const int in_block_offset = (row * num_cols + col) * num_entries;
  const int out_bid = col * num_rows + row;
  const int out_block_offset = out_bid * num_entries;
  const int start_idx = (num_entries * tid) / block_size;
  const int end_idx = (num_entries * (tid + 1)) / block_size;

  // Shift input and output global memory buffers
  int *input_ = input + in_block_offset;
  int *sorted_indices_ = sorted_indices + in_block_offset;
  int *output_ = output + out_block_offset;
  int *inverse_indices_ = inverse_indices + out_block_offset;
  int *num_uniques_ = num_uniques + out_bid + 1;

  // Shift shared memory buffers
  int *smem_input = smem;
  int *idx_offsets = smem + num_entries;
  int *smem_indices = smem + num_entries;

  // Load data to smem for sorting
  for (int i = start_idx; i < end_idx; i++) {
    smem_input[i] = input_[i];
    smem_indices[i] = i;
  }

  __syncthreads();

  bitonic_sort_pair(smem_input, smem_indices, num_entries);

  // Write sorted indices to HBM due to insufficient smem
  for (int i = start_idx; i < end_idx; i++) {
    sorted_indices_[i] = smem_indices[i];
  }

  int idx_offset = 0;
  bool is_last_uniq = false;

  if (tid == 0) {
    idx_offsets[0] = 1;
    idx_offset = 1;
  }

  int idx;
  int pidx = smem_input[start_idx + idx_offset - 1];
  for (int i = start_idx + idx_offset; i < end_idx; i++) {
    idx = smem_input[i];
    idx_offsets[i] = pidx != idx;
    pidx = idx;
  }

  if (tid == block_size - 1) {
    is_last_uniq = idx_offsets[end_idx - 1];
  }

  __syncthreads();

  exclusive_sum_scan(idx_offsets, num_entries, block_size, tid);

  __syncthreads();

  // Copy outputs from shared memory to global memory
  if (tid == block_size - 1) {
    idx_offset = 1;
  }
  else {
    idx_offset = 0;
  }

  int pos = idx_offsets[start_idx];
  int npos;
  for (int i = start_idx; i < end_idx - idx_offset; i++) {
    npos = idx_offsets[i + 1];
    // Use inclusive sum for inverese indices (shift exclusive sum to the left)
    inverse_indices_[sorted_indices_[i]] = npos - 1;
    if (npos - pos == 1) {
      output_[pos] = smem_input[i];
    }
    pos = npos;
  }

  if (tid == block_size - 1) {
    // Set last unique index
    int i = end_idx - 1;
    int pos = idx_offsets[i];
    if (is_last_uniq) {
      inverse_indices_[sorted_indices_[i]] = pos;
      output_[pos] = smem_input[i];
      *num_uniques_ = pos + 1;
    }
    else {
      inverse_indices_[sorted_indices_[i]] = pos - 1;
      *num_uniques_ = pos;
    }
    if (out_bid == 0) {
      // For cumsum in the next step
      num_uniques[0] = 0;
    }
  }
}

__global__ void __launch_bounds__(NUM_THREADS) compact_tensor(
    int *input,
    int *output,
    int *inverse_indices,
    int in_stride,
    int64_t *offsets,
    int64_t *num_uniques) {
  const int row = blockIdx.x;
  const int col = blockIdx.y;
  const int num_cols = gridDim.y;
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;
  const int bid = row * num_cols + col;
  const int64_t offset = offsets[bid];
  const int64_t num_entries = offsets[bid + 1] - offset;
  const int block_offset = bid * in_stride;
  int *input_ = input + block_offset;
  int *output_ = output + offset;
  int *inverse_indices_ = inverse_indices + block_offset;
  const int inverse_offset = offsets[bid] - offsets[row * num_cols];

  if (num_entries > block_size) {
    const int64_t start_idx = (num_entries * tid) / block_size;
    const int64_t end_idx = (num_entries * (tid + 1)) / block_size;

    for (int i = start_idx; i < end_idx; i++) {
      output_[i] = input_[i];
    }
  }
  else if (tid < num_entries) {
    output_[tid] = input_[tid];
  }

  // Shift inverse indices based on number of unique indices for each GPU
  if (inverse_offset != 0) {
    const int start_idx = (in_stride * tid) / block_size;
    const int end_idx = (in_stride * (tid + 1)) / block_size;
    for (int i = start_idx; i < end_idx; i++) {
      inverse_indices_[i] += inverse_offset;
    }
  }

  if (col == 0 && tid == 0) {
    num_uniques[row] = offsets[(row + 1) * num_cols] - offsets[row * num_cols];
  }
}

std::vector<torch::Tensor> unique_transpose(
    torch::Tensor input,
    const int num_embs,
    const int num_gpus) {
  dim3 grid_size(num_embs, num_gpus, 1);
  dim3 transposed_grid_size(num_gpus, num_embs, 1);
  const int num_blocks = num_embs * num_gpus;
  const int num_threads = NUM_THREADS;
  const int num_entries = input.numel() / num_blocks;
  int smem_size = num_entries * 2 * sizeof(int);
  bool is_small = true;

  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, at::hip::current_device());

  if ((size_t) smem_size > prop.maxSharedMemoryPerMultiProcessor) {
    is_small = false;
    smem_size = num_entries * sizeof(int);
    if (smem_size > prop.maxSharedMemoryPerMultiProcessor) {
      throw std::invalid_argument("Number of unique indices is too large");
    }
  }

  auto tmp_output = torch::empty_like(input);
  auto sorted_indices = torch::empty_like(input);
  auto inverse_indices = torch::empty_like(input);
  auto input_opts = input.options();
  auto num_uniques = torch::empty({num_blocks + 1}, input_opts);

  int *tmp_output_ptr = tmp_output.contiguous().data_ptr<int>();
  int *inverse_indices_ptr = inverse_indices.contiguous().data_ptr<int>();

  auto current_stream = at::hip::getCurrentHIPStream(at::hip::current_device()).stream();

  if (is_small) {
    hipLaunchKernelGGL(
        small_unique_transpose_kernel, grid_size, num_threads, smem_size, current_stream,
        input.contiguous().data_ptr<int>(),
        sorted_indices.contiguous().data_ptr<int>(),
        num_entries,
        tmp_output_ptr,
        inverse_indices_ptr,
        num_uniques.contiguous().data_ptr<int>());
  } else {
    hipLaunchKernelGGL(
        large_unique_transpose_kernel, grid_size, num_threads, smem_size, current_stream,
        input.contiguous().data_ptr<int>(),
        sorted_indices.contiguous().data_ptr<int>(),
        num_entries,
        tmp_output_ptr,
        inverse_indices_ptr,
        num_uniques.contiguous().data_ptr<int>());
  }

  auto num_unique_offsets = at::cumsum(num_uniques, 0);
  int total_uniques = num_unique_offsets[num_blocks].item<int>();
  auto output = torch::empty({total_uniques}, input_opts);
  torch::TensorOptions long_opts = torch::TensorOptions()
    .dtype(torch::kInt64)
    .layout(input_opts.layout())
    .device(input_opts.device());
  auto num_uniques_gpu = torch::empty({num_gpus}, long_opts);

  hipLaunchKernelGGL(
      compact_tensor, transposed_grid_size, num_threads, 0, current_stream,
      tmp_output_ptr,
      output.contiguous().data_ptr<int>(),
      inverse_indices_ptr,
      num_entries,
      num_unique_offsets.contiguous().data_ptr<int64_t>(),
      num_uniques_gpu.contiguous().data_ptr<int64_t>());

  return {output.view(total_uniques), inverse_indices, num_uniques_gpu};
}
