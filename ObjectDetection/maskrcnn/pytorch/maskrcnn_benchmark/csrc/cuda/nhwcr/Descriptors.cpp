#include "Descriptors.h"
#include <miopen/miopen.h>
#include <miopen/config.h>
#include <miopen/export.h>
#include <torch/extension.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/hip/detail/DeviceThreadHandles.h>
#include <c10/hip/HIPStream.h>
#include <iostream>
#include <sstream>
#include <string>

namespace at { namespace native { namespace nhwc {

namespace {

inline miopenDataType_t getDataType(const at::Tensor& t) {
  auto scalar_type = t.scalar_type();
  if (scalar_type == at::kFloat) {
    return miopenFloat;
  } else if (scalar_type == at::kHalf) {
    return miopenHalf;
  } else if (scalar_type == at::kBFloat16) {
    return miopenBFloat16;
  } else {
  throw std::runtime_error("TensorDescriptor only supports float, half and bfloat16 tensors");
  }
}

void createMIOpenHandle(miopenHandle_t *handle) {
  MIOPEN_CHECK(miopenCreate(handle));
}

void destroyMIOpenHandle(miopenHandle_t handle) {

}

using MIOpenPoolType = at::cuda::DeviceThreadHandlePool<miopenHandle_t, createMIOpenHandle, destroyMIOpenHandle>;

} // anonymous namespace

void TensorDescriptor::set(const at::Tensor &t, size_t pad) {
  set(getDataType(t), t.sizes(), t.strides(), pad);
}

static int MIOPEN_DIM_MAX = 5;

void TensorDescriptor::set(miopenDataType_t datatype, IntArrayRef t_sizes, IntArrayRef t_strides, size_t pad) {
  size_t dim = t_sizes.size();
  if (dim > MIOPEN_DIM_MAX || pad > MIOPEN_DIM_MAX)
#define _STR(X) #X
#define STR(X) _STR(X)
    throw std::runtime_error("MIOpen supports only up to " STR(MIOPEN_DIM_MAX) " dimensions");
#undef _STR
#undef STR
  int size[MIOPEN_DIM_MAX];
  int stride[MIOPEN_DIM_MAX];
  for (size_t i = 0; i < dim; ++i) {
    size[i] = static_cast<int>(t_sizes[i]);
    stride[i] = static_cast<int>(t_strides[i]);
  }
  for (size_t i = dim; i < pad; ++i) {
    size[i] = 1;
    stride[i] = 1;
  }

  int stride_value_1;
  stride_value_1 = stride[3];
  stride[3] = stride[1];
  stride[1] = stride_value_1;

  int stride_value_2;
  stride_value_2 = stride[2];
  stride[2] = stride[3];
  stride[3] = stride_value_2;

  int size_value_1;
  size_value_1 = size[3];
  size[3] = size[1];
  size[1] = size_value_1;

  int size_value_2;
  size_value_2 = size[2];
  size[2] = size[3];
  size[3] = size_value_2;

  set(datatype, static_cast<int>(std::max(dim, pad)), size, stride);
}

std::string miopenTypeToString(miopenDataType_t dtype) {
  switch (dtype) {
    case miopenFloat:
      return "miopenFloat";
    case miopenHalf:
      return "miopenHalf";
    case miopenBFloat16:
      return "miopenBFloat16";
    default:
      std::ostringstream oss;
      oss << "(unknown data-type " << static_cast<int>(dtype) << ")";
      return oss.str();
  }
}

std::ostream& operator<<(std::ostream & out, const TensorDescriptor& d) {
  out << "TensorDescriptor " << static_cast<void*>(d.desc()) << "\n";
  int nbDims = 4;
  int dimA[MIOPEN_DIM_MAX];
  int strideA[MIOPEN_DIM_MAX];
  miopenDataType_t dtype;
  miopenGetTensorDescriptor(d.desc(), &dtype, dimA, strideA);
  out << "    type = " << miopenTypeToString(dtype) << "\n";
  out << "    nbDims = " << nbDims << "\n";
  // Read out only nbDims of the arrays!
  out << "    dimA = ";
  for (auto i : ArrayRef<int>{dimA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << "\n";
  out << "    strideA = ";
  for (auto i : ArrayRef<int>{strideA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << "\n";
  return out;
}

void TensorDescriptor::print() { std::cout << *this; }

void FilterDescriptor::set(const at::Tensor &t, int64_t pad) {
  auto dim = t.ndimension();
  if (dim > MIOPEN_DIM_MAX || pad > MIOPEN_DIM_MAX)
#define _STR(X) #X
#define STR(X) _STR(X)
    throw std::runtime_error("MIOpen supports only up to " STR(MIOPEN_DIM_MAX) " dimensions");
#undef _STR
#undef STR
  if (!t.is_contiguous()) {
      throw std::runtime_error("MIOpen filters (a.k.a. weights) must be contiguous");
  }

  int size[MIOPEN_DIM_MAX];
  int stride[MIOPEN_DIM_MAX];
  for (int i = 0; i < dim; ++i) {
    size[i] = (int) t.size(i);
  }
  for (int i = dim; i < pad; ++i) {
    size[i] = (int) 1;
  }

  for (int i = pad; i >= dim; --i ) {
      stride[i] = 1;
  }
  for (int i = dim-1 ; i >=0; --i ) {
      // Pass-through
      stride[i] = t.stride(i);
  }

  int stride_value_1;
  stride_value_1 = stride[3];
  stride[3] = stride[1];
  stride[1] = stride_value_1;

  int stride_value_2;
  stride_value_2 = stride[2];
  stride[2] = stride[3];
  stride[3] = stride_value_2;

  int size_value_1;
  size_value_1 = size[3];
  size[3] = size[1];
  size[1] = size_value_1;

  dim = std::max(dim, pad);
  set(getDataType(t), (int) dim, size, stride);
}

miopenHandle_t getMiopenHandle() {
  int device;
  HIP_CHECK(hipGetDevice(&device));

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.
  static auto pool = std::make_shared<MIOpenPoolType>();
  thread_local std::unique_ptr<MIOpenPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  auto handle = myPoolWindow->reserve(device);
  MIOPEN_CHECK(miopenSetStream(handle, at::hip::getCurrentHIPStream()));
  return handle;
}

}}}
