#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#include "type_shim.h"
#include "multi_tensor_apply.cuh"

#define BLOCK_SIZE 512
#define ILP 4

template<typename T_grad, typename T_weight>
struct LARCFunctor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<2>& tl,
    float *grad_norms,
    float *param_norms,
    const float lr,
    const float trust_coefficient,
    const float epsilon,
    const float weight_decay,
    const bool is_skipped) {
    int tensor_loc = tl.block_to_tensor[blockIdx.x];

    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];
    n -= chunk_idx * chunk_size;
    n = min(n, chunk_size);

    T_grad* g = (T_grad*) tl.addresses[0][tensor_loc];
    g += chunk_idx * chunk_size;

    T_weight* p = (T_weight*) tl.addresses[1][tensor_loc];
    p += chunk_idx * chunk_size;

    float adaptive_lr;
    float trust_ratio = 1.0;

    if (is_skipped) {
      adaptive_lr = lr;
    }
    else {
      int tensor_offset = tl.start_tensor_this_launch + tensor_loc;
      float g_norm = grad_norms[tensor_offset];
      float p_norm = param_norms[tensor_offset];

      //float trust_ratio = 1.0;
      if (g_norm > 0.0f && p_norm > 0.0f) {
        trust_ratio = trust_coefficient * p_norm / (g_norm + p_norm * weight_decay + epsilon);
      }
      //adaptive_lr = lr * trust_ratio;
    }

    //if (weight_decay != 0.0f) {
    for (int i_start = 0; i_start < n; i_start += blockDim.x * ILP) {
#pragma unroll
      for (int i = i_start + threadIdx.x;
          i < i_start + threadIdx.x + ILP * blockDim.x && i < n;
          i += blockDim.x) {
	//FIXME
        g[i] = (g[i] + (weight_decay * p[i])) * trust_ratio;
      }
    }
    //}
#if 0
    else {
      // Avoid reading p when weight decay = 0.0f
      for (int i_start = 0; i_start < n; i_start += blockDim.x * ILP) {
#pragma unroll
        for (int i = i_start + threadIdx.x;
            i < i_start + threadIdx.x + ILP * blockDim.x && i < n;
            i += blockDim.x) {
          g[i] *= trust_ratio;
        }
      }
    }
#endif
  }
};

void multi_tensor_larc_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor grad_norms,
  at::Tensor param_norms,
  const float lr,
  const float trust_coefficient,
  const float epsilon,
  const float weight_decay,
  const bool is_skipped)
{
  using namespace at;

  auto grad_type = tensor_lists[0][0].scalar_type();
  auto weight_type = tensor_lists[1][0].scalar_type();

  if (grad_type == at::ScalarType::Float &&
      weight_type == at::ScalarType::Float) {
      multi_tensor_apply<2>(
          BLOCK_SIZE,
          chunk_size,
          noop_flag,
          tensor_lists,
          LARCFunctor<float, float>(),
          grad_norms.DATA_PTR<float>(),
          param_norms.DATA_PTR<float>(),
          lr,
          trust_coefficient,
          epsilon,
          weight_decay,
          is_skipped);
  }
  else if (grad_type == at::ScalarType::Half &&
           weight_type == at::ScalarType::Half) {
      multi_tensor_apply<2>(
          BLOCK_SIZE,
          chunk_size,
          noop_flag,
          tensor_lists,
          LARCFunctor<at::Half, at::Half>(),
          grad_norms.DATA_PTR<float>(),
          param_norms.DATA_PTR<float>(),
          lr,
          trust_coefficient,
          epsilon,
          weight_decay,
          is_skipped);
  }
  else if (grad_type == at::ScalarType::Half &&
           weight_type == at::ScalarType::Float) {
      multi_tensor_apply<2>(
          BLOCK_SIZE,
          chunk_size,
          noop_flag,
          tensor_lists,
          LARCFunctor<at::Half, float>(),
          grad_norms.DATA_PTR<float>(),
          param_norms.DATA_PTR<float>(),
          lr,
          trust_coefficient,
          epsilon,
          weight_decay,
          is_skipped);
  }
  else {
    AT_ERROR("multi_tensor_larc only supports some combinations of gradient & weight types. Given: ",
             "gradient: ", grad_type, ", weight: ", weight_type);
  }

  AT_CUDA_CHECK(cudaGetLastError());
}
