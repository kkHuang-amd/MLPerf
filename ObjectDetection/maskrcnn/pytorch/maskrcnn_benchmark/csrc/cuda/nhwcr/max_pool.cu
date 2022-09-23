#include "Descriptors.h"
#include <miopen/miopen.h>
#include <miopen/config.h>
#include <miopen/export.h>
#include <torch/extension.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/TensorUtils.h>
#include "ParamsHash.h"
#include "Exceptions.h"
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include <ATen/NamedTensorUtils.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/Pool.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/macros/Macros.h>
#include <ATen/native/cuda/LaunchUtils.h>

#include "ceil_div.h"

namespace at { namespace native { namespace nhwc {

namespace {

__device__ inline int min(int a, int b) {
  return a <= b ? a : b;
}

#define CUDA_MAX_THREADS 1024 // this is safe, in reality 256 is our limit

#define BLOCK_STRIDE 2 // increasing block_stride to lower # of blocks launched

static __device__ inline int p_start(int size, int pad, int kernel, int dilation, int stride) {
  return (size + pad < ((kernel - 1) * dilation + 1)) ? 0 : (size + pad - ((kernel - 1) * dilation + 1)) / stride + 1;
}

static __device__ inline int p_end(int size, int pad, int pooled_size, int stride) {
  return min((size + pad) / stride + 1, pooled_size);
}

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(CUDA_MAX_THREADS)
__global__ void max_pool_forward_nhwc(const scalar_t* bottom_data, const int nbatch,
                                   const int channels, const int height,
                                   const int width, const int pooled_height, const int pooled_width,
                                   const int kernel_h, const int kernel_w, const int stride_h,
                                   const int stride_w, const int pad_h, const int pad_w,
                                   const int dilation_h, const int dilation_w,
                                   const int in_stride_n, const int in_stride_c,
                                   const int in_stride_h, const int in_stride_w,
                                   const int kernel_stride_C, const int kernel_size_C,
                                   scalar_t* top_data, int64_t* top_mask) {
  extern __shared__ int smem[];
  int *out_mask_cached = smem;
  scalar_t *out_cached = reinterpret_cast<scalar_t*>(&out_mask_cached[kernel_size_C*blockDim.x*blockDim.y*blockDim.z]);

  // flattening cta for pre-computation & smem initialization;
  int thread_id = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  int block_size = blockDim.x * blockDim.y * blockDim.z;

  // use shared memory to store temporary output value. This is simply to
  // reduce register usage.
  for (int i = thread_id; i < kernel_size_C*blockDim.x*blockDim.y*blockDim.z; i+= block_size) {
    out_cached[i] = at::numeric_limits<scalar_t>::lower_bound();
    out_mask_cached[i] = 0;
  }

  __syncthreads();

  int batch_id = blockIdx.x % nbatch;
  int channel_id = blockIdx.x / nbatch;
  int channel_offset = threadIdx.x + channel_id * blockDim.x;

  top_data = top_data + batch_id * pooled_height * pooled_width * channels;
  top_mask = top_mask + batch_id * pooled_height * pooled_width * channels;
  bottom_data = bottom_data + batch_id * in_stride_n;

  out_cached = &out_cached[(threadIdx.z * blockDim.y + threadIdx.y) * kernel_size_C*blockDim.x];
  out_mask_cached = &out_mask_cached[(threadIdx.z * blockDim.y + threadIdx.y) * kernel_size_C*blockDim.x];

  int oH = (pooled_height + gridDim.z-1) / gridDim.z;
  int oW = (pooled_width + gridDim.y-1) / gridDim.y;
  int ostartH = threadIdx.z + blockIdx.z*oH;
  int oendH = ::min(ostartH+oH, pooled_height);
  int ostartW = threadIdx.y + blockIdx.y*oW;
  int oendW = ::min(ostartW+oW, pooled_width);

  for (int oh = ostartH; oh < oendH; oh+=blockDim.z) {
    int hstart = oh * stride_h - pad_h;
    int hend = min(hstart + (kernel_h - 1) * dilation_h + 1, height);
    for (int ow = ostartW; ow < oendW; ow+=blockDim.y) {
      int wstart = ow * stride_w - pad_w;
      int wend = min(wstart + (kernel_w - 1) * dilation_w + 1, width);
      while(hstart < 0)
        hstart += dilation_h;
      while(wstart < 0)
        wstart += dilation_w;
      for (int ih = hstart; ih < hend; ih += dilation_h) {
        for (int iw = wstart; iw < wend; iw += dilation_w) {
          int cached_index = threadIdx.x;
          const scalar_t *ptr_input = bottom_data + ih * in_stride_h + iw * in_stride_w;
          for(int c = channel_offset; c < channels; c+= blockDim.x*kernel_stride_C) {
            scalar_t val = ptr_input[c*in_stride_c];
            if ((static_cast<accscalar_t>(val) > out_cached[cached_index]) || at::_isnan(val)) {
              out_cached[cached_index] = static_cast<accscalar_t>(val);
              out_mask_cached[cached_index] = ih * width + iw;
            }
            cached_index += blockDim.x;
          }
        }
      }
      scalar_t *ptr_output_data = top_data + (oh * pooled_width + ow) * channels;
      int64_t *ptr_output_mask = top_mask + (oh * pooled_width + ow) * channels;

      int cached_index = threadIdx.x;
      for(int c = channel_offset; c < channels; c+= blockDim.x*kernel_stride_C) {
        ptr_output_data[c] = out_cached[cached_index];
        ptr_output_mask[c] = out_mask_cached[cached_index];
        out_cached[cached_index] = at::numeric_limits<scalar_t>::lower_bound();
        out_mask_cached[cached_index] = 0;
        cached_index += blockDim.x;
      }
    }
  }
}

static const int BLOCK_THREADS = 256;

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(CUDA_MAX_THREADS)
__global__ void max_pool_backward_nhwc(const scalar_t* top_diff,
                                    const int64_t* top_mask, const int nbatch, const int channels,
                                    const int height, const int width, const int pooled_height,
                                    const int pooled_width, const int kernel_h, const int kernel_w,
                                    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
                                    const int dilation_h, const int dilation_w,
                                    const int out_stride_c, const int out_stride_h, const int out_stride_w,
                                    const int kernel_stride_C, const int kernel_size_C,
                                    scalar_t* bottom_diff) {
  extern __shared__ int smem[];
  accscalar_t *out_cached = reinterpret_cast<accscalar_t*>(smem);

  int thread_id = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  int block_size = blockDim.x * blockDim.y * blockDim.z;

  int batch_id = blockIdx.x % nbatch;
  int channel_id = blockIdx.x / nbatch;
  int channel_offset = threadIdx.x + channel_id * blockDim.x;

  for (int i = thread_id; i < kernel_size_C*blockDim.x*blockDim.y*blockDim.z; i+= block_size) {
    out_cached[i] = accscalar_t(0.0);
  }

  __syncthreads();

  out_cached = &out_cached[(threadIdx.z * blockDim.y + threadIdx.y) * kernel_size_C*blockDim.x];

  bottom_diff = bottom_diff + batch_id * height * width * channels;
  top_mask = top_mask + batch_id * pooled_height * pooled_width * channels;
  top_diff = top_diff + batch_id * pooled_height * pooled_width * channels;

  int iH = (height + gridDim.z-1) / gridDim.z;
  int iW = (width + gridDim.y-1) / gridDim.y;
  int istartH = threadIdx.z + blockIdx.z*iH;
  int iendH = ::min(istartH+iH, height);
  int istartW = threadIdx.y + blockIdx.y*iW;
  int iendW = ::min(istartW+iW, width);

  for (int ih = istartH; ih < iendH; ih+=blockDim.z) {
    int phstart = p_start(ih, pad_h, kernel_h, dilation_h, stride_h);
    int phend = p_end(ih, pad_h, pooled_height, stride_h);
    for (int iw = istartW; iw < iendW; iw+=blockDim.y) {
      int pwstart = p_start(iw, pad_w, kernel_w, dilation_w, stride_w);
      int pwend = p_end(iw, pad_w, pooled_width, stride_w);
      int index_shift = ih * width + iw;
      if ((phstart + 1 != phend) || (pwstart + 1 != pwend)) {
        for(int oh = phstart; oh < phend; ++oh) {
          for(int ow = pwstart; ow < pwend; ++ow) {
            int cached_index = threadIdx.x;
            const int64_t* ptr_top_mask = top_mask + oh * out_stride_h + ow * out_stride_w;
            for (int c = channel_offset; c < channels; c += blockDim.x*kernel_stride_C) {
              if (ptr_top_mask[c*out_stride_c] == index_shift) {
                out_cached[cached_index] +=
                  static_cast<accscalar_t>(top_diff[oh * out_stride_h + ow * out_stride_w + c*out_stride_c]);
              }
              cached_index += blockDim.x;
            }
          }
        }
        scalar_t *ptr_bottom_diff = bottom_diff + index_shift * channels;
        int cached_index = threadIdx.x;
        for (int c = channel_offset; c < channels; c += blockDim.x*kernel_stride_C) {
          ptr_bottom_diff[c] = static_cast<scalar_t>(out_cached[cached_index]);
          out_cached[cached_index] = accscalar_t(0.0);
          cached_index += blockDim.x;
        }
      } else {
        const int64_t* ptr_top_mask = top_mask + phstart * out_stride_h + pwstart * out_stride_w;
        scalar_t *ptr_bottom_diff = bottom_diff + index_shift * channels;
        int cached_index = threadIdx.x;
        for (int c = channel_offset; c < channels; c += blockDim.x*kernel_stride_C) {
          if (ptr_top_mask[c*out_stride_c] == index_shift) {
            ptr_bottom_diff[c] =
              static_cast<scalar_t>(top_diff[phstart * out_stride_h + pwstart * out_stride_w + c*out_stride_c]);
          }
          cached_index += blockDim.x;
        }
      }
    }
  }
}

} // namespace

void max_pool2d_with_indices_out_cuda
(const Tensor& input_,
IntArrayRef kernel_size,
IntArrayRef stride,
IntArrayRef padding,
IntArrayRef dilation,
bool ceil_mode,
const Tensor& output,
const Tensor& indices) {
  NoNamesGuard guard;

  TensorArg output_arg{ output, "output", 1 };
  TensorArg indices_arg{ indices, "indices", 2 };
  TensorArg input_arg{ input_, "input_", 3 };

  checkAllSameGPU(__func__, {output_arg, indices_arg, input_arg});
  if (output.numel() == 0) {
    return;
  }

  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  const int64_t nbatch = input_.ndimension() == 4 ? input_.size(0) : 1;
  const int64_t nInputPlane = input_.size(3);
  const int64_t inputHeight = input_.size(1);
  const int64_t inputWidth = input_.size(2);

  const int64_t outputHeight = output.size(1);
  const int64_t outputWidth = output.size(2);

  Tensor input = input_;

  const int64_t in_stride_n = input_.ndimension() == 4 ? input.stride(0) : 0;
  const int64_t in_stride_c = input.stride(3);
  const int64_t in_stride_h = input.stride(1);
  const int64_t in_stride_w = input.stride(2);

  const int count = safe_downcast<int, int64_t>(output.numel());

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
    "max_pool2d_with_indices_out_cuda_frame",
    [&] {
      using accscalar_t = acc_type<scalar_t, true>;

      scalar_t *output_data = output.data_ptr<scalar_t>();
      scalar_t *input_data = input.data_ptr<scalar_t>();
      int64_t *indices_data = indices.data_ptr<int64_t>();
      
      const int max_threads = std::min<int>(
          at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, CUDA_MAX_THREADS);
      int* maxThreadsDim = at::cuda::getCurrentDeviceProperties()->maxThreadsDim;
      int block_x = std::min<int>(
          maxThreadsDim[0], std::min<int>(lastPow2(nInputPlane), at::cuda::warp_size()));
      int block_y = std::min<int>(
          maxThreadsDim[1], std::min<int>(lastPow2(outputWidth), max_threads / block_x));
      int block_z = std::min<int>(
          maxThreadsDim[2], std::min<int>(lastPow2(outputHeight), max_threads / block_x / block_y));
      block_x = std::min<int>(
          maxThreadsDim[0], std::min<int>(lastPow2(nInputPlane), max_threads / block_y / block_z));
      const dim3 block(block_x, block_y, block_z);

      int kernel_stride_C = ceil_div(
          safe_downcast<int, int64_t>(nInputPlane), block_x * 4);
      int kernel_size_C = ceil_div(
          safe_downcast<int, int64_t>(nInputPlane), block_x * kernel_stride_C);

      int grid_x = nbatch*kernel_stride_C;
      int grid_y = std::min<int>(
          at::cuda::getCurrentDeviceProperties()->maxGridSize[1],
          ceil_div(safe_downcast<int, int64_t>(outputWidth), block_y*BLOCK_STRIDE));
      int grid_z = std::min<int>(
          at::cuda::getCurrentDeviceProperties()->maxGridSize[2],
          ceil_div(safe_downcast<int, int64_t>(outputHeight), block_z*BLOCK_STRIDE));
      const dim3 grid(grid_x, grid_y, grid_z);

      size_t shmem_size = (kernel_size_C * block_x*block_y*block_z) * (sizeof(int) + sizeof(scalar_t));
      AT_ASSERT(shmem_size <= at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock);

      max_pool_forward_nhwc<scalar_t, scalar_t>
      <<<grid, block, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
          input_data, nbatch,
              nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
              kH, kW, dH, dW, padH, padW, dilationH, dilationW,
              in_stride_n, in_stride_c,
              in_stride_h, in_stride_w,
              kernel_stride_C, kernel_size_C,
              output_data, indices_data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
       
     
    }
  );
}

void max_pool2d_with_indices_backward_out_cuda
(const Tensor& gradOutput_,
const Tensor& input_,
IntArrayRef kernel_size,
IntArrayRef stride,
IntArrayRef padding,
IntArrayRef dilation,
bool ceil_mode,
const Tensor& indices,
const Tensor& gradInput) {
  NoNamesGuard guard;

  TensorArg gradInput_arg{ gradInput, "gradInput", 1 };
  TensorArg gradOutput_arg{ gradOutput_, "gradOutput_", 2 };
  TensorArg input_arg{ input_, "input_", 3 };
  TensorArg indices_arg{ indices, "indices", 4 };

  checkAllSameGPU(__func__,
                  {gradInput_arg, gradOutput_arg, input_arg, indices_arg});
  if (gradOutput_.numel() == 0) {
    return;
  }

  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  const Tensor input = input_;

  const int64_t nbatch = input.ndimension() == 4 ? input.size(0) : 1;
  const int64_t nInputPlane = input.size(3);
  const int64_t inputHeight = input.size(1);
  const int64_t inputWidth = input.size(2);

  const int64_t in_stride_n = input.ndimension() == 4 ? input.stride(0) : 0;
  const int64_t in_stride_c = input.stride(3);
  const int64_t in_stride_h = input.stride(1);
  const int64_t in_stride_w = input.stride(2);

  const Tensor gradOutput = gradOutput_;

  const int64_t outputHeight = gradOutput.size(1);
  const int64_t outputWidth = gradOutput.size(2);

  const int64_t out_stride_c = gradOutput.stride(3);
  const int64_t out_stride_h = gradOutput.stride(1);
  const int64_t out_stride_w = gradOutput.stride(2);

  gradInput.zero_();

  int64_t count = input.numel();

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
    "max_pool2d_with_indices_out_cuda_frame",
    [&] {
      using accscalar_t = acc_type<scalar_t, true>;

      scalar_t *gradOutput_data = gradOutput.data_ptr<scalar_t>();
      scalar_t *gradInput_data = gradInput.data_ptr<scalar_t>();
      int64_t *indices_data = indices.data_ptr<int64_t>();
      
      const int max_threads = std::min<int>(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, CUDA_MAX_THREADS);
      int* maxThreadsDim = at::cuda::getCurrentDeviceProperties()->maxThreadsDim;
      int block_x = std::min<int>(
          maxThreadsDim[0], std::min<int>(lastPow2(nInputPlane), at::cuda::warp_size()));
      int block_y = std::min<int>(
          maxThreadsDim[1], std::min<int>(lastPow2(inputWidth), max_threads / block_x));
      int block_z = std::min<int>(
          maxThreadsDim[2], std::min<int>(lastPow2(inputHeight), max_threads / block_x / block_y));
      block_x = std::min<int>(
          maxThreadsDim[0], std::min<int>(lastPow2(nInputPlane), max_threads / block_y / block_z));
      const dim3 block(block_x, block_y, block_z);

      int kernel_stride_C = ceil_div(
          safe_downcast<int, int64_t>(nInputPlane), block_x * 4);
      int kernel_size_C = ceil_div(
          safe_downcast<int, int64_t>(nInputPlane), block_x * kernel_stride_C);

      int grid_x = nbatch*kernel_stride_C;
      int grid_y = std::min<int>(
          at::cuda::getCurrentDeviceProperties()->maxGridSize[1],
          ceil_div(safe_downcast<int, int64_t>(inputWidth), block_y*BLOCK_STRIDE));
      int grid_z = std::min<int>(
          at::cuda::getCurrentDeviceProperties()->maxGridSize[2],
          ceil_div(safe_downcast<int, int64_t>(inputHeight), block_z*BLOCK_STRIDE));
      const dim3 grid(grid_x, grid_y, grid_z);

      size_t shmem_size = (kernel_size_C * block_x*block_y*block_z) * sizeof(accscalar_t);
      AT_ASSERT(shmem_size <= at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock);

      // The backward kernel is launched on input instead output.
      // If it is launched on output layer, atomic_add would not provide much benefit on FP16.
      // Please check comments at https://github.com/pytorch/pytorch/pull/34519.
      max_pool_backward_nhwc<scalar_t, accscalar_t>
      <<<grid, block, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
              gradOutput_data,
              indices_data,
              nbatch,
              nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
              kH, kW, dH, dW, padH, padW, dilationH, dilationW,
              out_stride_c, out_stride_h, out_stride_w,
              kernel_stride_C, kernel_size_C,
              gradInput_data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
        
      
    }
  );
}

//Forward
std::tuple<at::Tensor,at::Tensor> max_pool_nhwc_fwd(
                       const at::Tensor& x_t,
                       const int kernel,
                       const int stride,
                       const int padding,
                       const int dilation) {
  // we assume later x is contiguous
  at::Tensor x = x_t.contiguous();

  // dimensions
  const int N = x.size(0);
  const int C = x.size(3);
  const int H = x.size(1);
  const int W = x.size(2);

  //if (H != W) {
  //  throw std::invalid_argument("H and W must be the same.");
  //}
  
  int d_out_h = ((H + 2 * padding - dilation * (kernel-1) - 1)/stride) + 1;
  int d_out_w = ((W + 2 * padding - dilation * (kernel-1) - 1)/stride) + 1;
  at::Tensor y = at::empty({N, d_out_h, d_out_w, C}, x.options());
  at::Tensor indices = at::empty({N, d_out_h, d_out_w, C}, x.options().dtype(at::kLong));
  max_pool2d_with_indices_out_cuda(x,
          IntArrayRef({kernel, kernel}), IntArrayRef({stride, stride}), IntArrayRef({padding, padding}), IntArrayRef({dilation, dilation}), 
          false, y, indices);

  return std::tuple<at::Tensor,at::Tensor>{y, indices};

}
  
//Backward
at::Tensor max_pool_nhwc_bwd(const at::Tensor& x_t,
                             const at::Tensor& y_t,
                             const at::Tensor& grad_y_t,
                             const at::Tensor& indices,
                             const int kernel,
                             const int stride,
                             const int padding,
                             const int dilation) {
  // we assume later x, y and grad_y are contiguous
  at::Tensor x = x_t.contiguous();
  at::Tensor grad_y = grad_y_t.contiguous();

  at::Tensor dx = at::empty_like(x);
  max_pool2d_with_indices_backward_out_cuda(grad_y, x, 
          IntArrayRef({kernel, kernel}), IntArrayRef({stride, stride}), IntArrayRef({padding, padding}), IntArrayRef({dilation, dilation}), 
          false, indices, dx);
  return dx;

}

}}}

