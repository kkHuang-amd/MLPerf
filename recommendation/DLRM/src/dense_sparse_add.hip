#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPStream.h>
#include <THH/THHAtomics.cuh>

__global__ void dense_sparse_add_kernel(
    float *src,
    float *dst,
    int64_t *indices,
    const int num_entries,
    const int vector_dim,
    const float alpha) {
  for (int i = blockIdx.x; i < num_entries; i += gridDim.x) {
    for (int l = threadIdx.x; l < vector_dim; l += blockDim.x) {
#if ROCM_VERSION > 402
      atomicAddNoRet(&dst[indices[i] * vector_dim + l], src[i * vector_dim + l] * alpha);
#else
      gpuAtomicAddNoReturn(&dst[indices[i] * vector_dim + l], src[i * vector_dim + l] * alpha);
#endif
    }
  }
}

__global__ void dense_sparse_add_fp16_kernel(
    half *src,
    float *dst,
    int64_t *indices,
    const int num_entries,
    const int vector_dim,
    const float alpha) {
  for (int i = blockIdx.x; i < num_entries; i += gridDim.x) {
    for (int l = threadIdx.x; l < vector_dim; l += blockDim.x) {
#if ROCM_VERSION > 402
      atomicAddNoRet(&dst[indices[i] * vector_dim + l], __half2float(src[i * vector_dim + l]) * alpha);
#else
      gpuAtomicAddNoReturn(&dst[indices[i] * vector_dim + l], __half2float(src[i * vector_dim + l]) * alpha);
#endif
    }
  }
}

void dense_sparse_add(
    torch::Tensor src, // sparse tensor
    torch::Tensor dst, // dense tensor
    torch::Tensor indices,
    const float alpha) {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, at::hip::current_device());

  const int num_entries = indices.size(0);
  const int vector_dim = src.size(1);
  const int grid_size = prop.multiProcessorCount;
  auto current_stream = at::hip::getCurrentHIPStream(at::hip::current_device()).stream();

  assert(dst.is_contiguous());

  if (src.scalar_type() == torch::ScalarType::Float) {
    hipLaunchKernelGGL(dense_sparse_add_kernel, grid_size, warpSize, 0, current_stream,
        src.contiguous().data_ptr<float>(),
        dst.data_ptr<float>(),
        indices.contiguous().data_ptr<int64_t>(),
        num_entries,
        vector_dim,
        alpha);
  }
  else if (src.scalar_type() == torch::ScalarType::Half) {
    hipLaunchKernelGGL(dense_sparse_add_fp16_kernel, grid_size, warpSize, 0, current_stream,
        (half *) src.contiguous().data_ptr<at::Half>(),
        dst.data_ptr<float>(),
        indices.contiguous().data_ptr<int64_t>(),
        num_entries,
        vector_dim,
        alpha);
  }
  else {
    throw std::invalid_argument("Invalid input type.");
  }
}
