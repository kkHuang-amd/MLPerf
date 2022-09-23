#pragma once

#include "Exceptions.h"

#include <miopen/miopen.h>
#include <ATen/TensorUtils.h>
#include "ATen/cuda/ATenCUDAGeneral.h"

namespace at { namespace native { namespace nhwc {


inline int dataSize(miopenDataType_t dataType)
{
  switch (dataType) {
    case miopenHalf: return 2;
    case miopenFloat: return 4;
    case miopenBFloat16: return 2;
    default: return 8;
  }
}

template <typename T, miopenStatus_t (*dtor)(T*)>
struct DescriptorDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      MIOPEN_CHECK(dtor(x));
    }
  }
};

// A generic class for wrapping MIOpen descriptor types.  All you need
// is to give the underlying type the Descriptor_t points to (usually,
// if it's miopenTensorDescriptor_t it points to miopenTensorStruct),
// the constructor and the destructor.  Subclasses are responsible
// for defining a set() function to actually set the descriptor.
//
// Descriptors default construct to a nullptr, and have a descriptor
// initialized the first time you call set() or any other initializing
// function.
template <typename T, miopenStatus_t (*ctor)(T**), miopenStatus_t (*dtor)(T*)>
class Descriptor
{
public:
  // Use desc() to access the underlying descriptor pointer in
  // a read-only fashion.  Most client code should use this.
  // If the descriptor was never initialized, this will return
  // nullptr.
  T* desc() const { return desc_.get(); }
  T* desc() { return desc_.get(); }

  // Use mut_desc() to access the underlying descriptor pointer
  // if you intend to modify what it points to (e.g., using
  // miopenSetFooDescriptor).  This will ensure that the descriptor
  // is initialized.  Code in this file will use this function.
  T* mut_desc() { init(); return desc_.get(); }
protected:
  void init() {
    if (desc_ == nullptr) {
      T* raw_desc;
      MIOPEN_CHECK(ctor(&raw_desc));
      desc_.reset(raw_desc);
    }
  }
private:
  std::unique_ptr<T, DescriptorDeleter<T, dtor>> desc_;
};


class TensorDescriptor
  : public Descriptor<miopenTensorDescriptor,
                      &miopenCreateTensorDescriptor,
                      &miopenDestroyTensorDescriptor>
{
public:
  TensorDescriptor() {}
  explicit TensorDescriptor(const at::Tensor &t, size_t pad = 0) {
    set(t, pad);
  }

  void set(const at::Tensor &t, size_t pad = 0);
  void set(miopenDataType_t dataType, IntArrayRef sizes, IntArrayRef strides, size_t pad = 0);

  void print();

private:
  void set(miopenDataType_t dataType, int dim, int* size, int* stride) {
    MIOPEN_CHECK(miopenSetTensorDescriptor(mut_desc(), dataType, dim, size, stride));
  }
};

std::ostream& operator<<(std::ostream & out, const TensorDescriptor& d);

class FilterDescriptor
  : public Descriptor<miopenTensorDescriptor,
                      &miopenCreateTensorDescriptor,
                      &miopenDestroyTensorDescriptor>
{
 public:
  void set(const at::Tensor &t, int64_t pad = 0);

private:
  void set(miopenDataType_t dataType, int dim, int* size, int* stride) {
    MIOPEN_CHECK(miopenSetTensorDescriptor(mut_desc(), dataType, dim, size, stride));
  }
};

struct ConvolutionDescriptor
  : public Descriptor<miopenConvolutionDescriptor,
                      &miopenCreateConvolutionDescriptor,
                      &miopenDestroyConvolutionDescriptor>
{
  void set(miopenDataType_t dataType, miopenConvolutionMode_t c_mode,  int dim, int* pad, int* stride, int * upscale /* aka dilation */, int groups) {
    MIOPEN_CHECK(miopenInitConvolutionNdDescriptor(mut_desc(), dim, pad, stride, upscale, c_mode));
    MIOPEN_CHECK(miopenSetConvolutionGroupCount(mut_desc(), groups));
  }
};

union Constant
{
  float f;
  double d;
  Constant(miopenDataType_t dataType, double value) {
    if (dataType == miopenHalf || dataType == miopenFloat || dataType == miopenBFloat16) {
      f = static_cast<float>(value);
    } else {
      d = value;
    }
  }
};

miopenHandle_t getMiopenHandle();

}}}  // namespace
