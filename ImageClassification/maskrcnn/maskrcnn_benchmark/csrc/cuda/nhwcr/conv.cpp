#include <miopen/miopen.h>
#include <miopen/config.h>
#include <miopen/export.h>
#include <torch/extension.h>
#include <ATen/native/ConvUtils.h>
#include <functional>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <unordered_map>
#include "Descriptors.h"
#include "ParamsHash.h"
#include "Exceptions.h"
#include <c10/hip/HIPCachingAllocator.h>
#include <iostream>

namespace at { namespace native { namespace nhwc {

// ---------------------------------------------------------------------
//
// Math
//
// ---------------------------------------------------------------------

miopenDataType_t getDataType(const Tensor& tensor) {
  if (tensor.scalar_type() == at::kFloat) {
    return miopenFloat;
  } else if (tensor.scalar_type() == at::kBFloat16) {
    return miopenBFloat16;
  } else if (tensor.scalar_type() == at::kHalf) {
    return miopenHalf;
  }
  std::string msg("getDataType() not supported for ");
  msg += toString(tensor.scalar_type());
  throw std::runtime_error(msg);
}


constexpr int input_batch_size_dim = 0;  // also grad_input
constexpr int input_channels_dim = 3;
constexpr int output_batch_size_dim = 0;  // also grad_output
constexpr int output_channels_dim = 3;
constexpr int weight_output_channels_dim = 0;
constexpr int weight_input_channels_dim = 3;

// Often written as 2 + max_dim (extra dims for batch size and channels)
constexpr int max_dim = 3;


Tensor narrowGroup(const Tensor& t, int dim, int group_idx, int64_t groups) {
  auto group_size = t.size(dim) / groups;
  return t.narrow(dim, group_idx * group_size, group_size);
}


std::vector<int64_t> conv_output_size(
    IntArrayRef input_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[1] = weight_size[weight_output_channels_dim];
  for (size_t d = 2; d < dim; ++d) {
    auto kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2])
                        - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

// Handle [N, H, W, C] format -- adjust offsets into pad, stride, dilation, etc.
std::vector<int64_t> conv_output_size_nhwc(
    IntArrayRef input_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[3] = weight_size[weight_output_channels_dim];
  for (size_t d = 1; d < dim-1; ++d) {
    auto kernel = dilation[d - 1] * (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 1])
                        - kernel) / stride[d - 1] + 1;
  }
  return output_size;
}

std::vector<int64_t> conv_input_size(
    IntArrayRef output_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  
  auto dim = output_size.size();
  std::vector<int64_t> input_size(dim);
  input_size[0] = output_size[output_batch_size_dim];
  input_size[3] = weight_size[weight_input_channels_dim] * groups;
  for (size_t d = 1; d < dim-1; ++d) {
    int kernel = dilation[d - 1] * (weight_size[d] - 1) + 1;
    input_size[d] = (output_size[d] - 1) * stride[d - 1] - (2 * padding[d - 1]) +
                     kernel + output_padding[d - 1];
  }
  return input_size;
}

std::vector<int64_t> conv_weight_size(
    IntArrayRef input_size, IntArrayRef output_size,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  auto dim = input_size.size();
  std::vector<int64_t> weight_size(dim);
  weight_size[0] = output_size[1];
  weight_size[1] = input_size[1] / groups;
  for (size_t d = 2; d < dim; ++d) {
    int kernel = input_size[d] - (output_size[d] - 1) * stride[d - 2]
               + 2 * padding[d - 2] - output_padding[d - 2];
    weight_size[d] = (kernel - 1) / dilation[d - 2] + 1;
  }
  return weight_size;
}
// ---------------------------------------------------------------------
//
// Checking
//
// ---------------------------------------------------------------------

// Used on pad, stride and dilation
static void check_args(CheckedFrom c, IntArrayRef args, size_t expected_size, const char* arg_name)
{
  if (args.size() > expected_size){
    std::stringstream ss;
    ss << "Too many " << arg_name << " values (" << args.size() << ") supplied, expecting " << expected_size << " (while checking arguments for " << c << ")";
    throw std::runtime_error(ss.str());
  }
  else if (args.size() < expected_size){
    std::stringstream ss;
    ss << "Not enough " << arg_name << " values (" << args.size() << ") supplied, expecting " << expected_size << " (while checking arguments for " << c << ")";
    throw std::runtime_error(ss.str());
  }

  auto num_negative_values = std::count_if(args.begin(), args.end(), [](int x){return x < 0;});
  if (num_negative_values > 0){
    std::stringstream ss;
    ss << arg_name << " should be greater than zero but got (";
    std::copy(args.begin(), args.end() - 1, std::ostream_iterator<int>(ss,", "));
    ss << args.back() <<  ")" << " (while checking arguments for " << c << ")";
    throw std::runtime_error(ss.str());
  }
}

// see NOTE [ Convolution checks] in src/Aten/native/cudnn/Conv.cpp

static void convolution_shape_check(
    CheckedFrom c,
    const TensorGeometryArg& input, const TensorGeometryArg& weight, const TensorGeometryArg& output,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups)
{

  check_args(c, padding, input->dim() - 2, "padding");
  check_args(c, stride, padding.size(), "stride");
  check_args(c, dilation, padding.size(), "dilation");

  // Input
  checkDimRange(c, input, 3, 6 /* exclusive */);
  checkSize(c, input, input_channels_dim, weight->size(1) * groups);

  // Weight
  checkSameDim(c, input, weight);

  checkSameDim(c, input, output);

}

static void convolution_shape_check_nhwc(
    CheckedFrom c,
    const TensorGeometryArg& input, const TensorGeometryArg& weight, const TensorGeometryArg& output,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups)
{
  check_args(c, padding, input->dim() - 2, "padding");
  check_args(c, stride, padding.size(), "stride");
  check_args(c, dilation, padding.size(), "dilation");

  // Input
  checkDimRange(c, input, 3, 6 /* exclusive */);
  checkSize(c, input, input_channels_dim, weight->size(1) * groups);

  // Weight
  checkSameDim(c, input, weight);

  // TODO: check that output->size() matches output_sizes
  // TODO: check that weight matches output->sizes()
  checkSameDim(c, input, output);
}


// This POD struct is used to let us easily compute hashes of the
// parameters
struct ConvolutionParams
{
  miopenDataType_t dataType;
  int input_size[2 + max_dim];
  int input_stride[2 + max_dim];
  int weight_size[2 + max_dim];
  int padding[max_dim];
  int stride[max_dim];
  int dilation[max_dim];
  int64_t groups;
  bool deterministic;
  //This is needed to distinguish between miopen handles of multiple gpus.

  // NB: transposed purposely omitted: transposed just swaps
  // forward and backward, so you can reuse the benchmark entry,
};
// ConvolutionParams must be a POD because we read out its memory
// contenst as char* when hashing

void setConvolutionParams(
    ConvolutionParams* params,
    const at::Tensor& input, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool deterministic) {

  miopenDataType_t dataType = getDataType(input);

  memset(params, 0, sizeof(ConvolutionParams));
  params->dataType = dataType;
  
  for (int i = 0; i != input.dim(); ++i) {
    params->input_size[i] = (int) input.size(i);
    params->input_stride[i] = (int) input.stride(i);
    params->weight_size[i] = (int) weight.size(i);
  }

  for (size_t i = 0; i != padding.size(); ++i) {
    params->padding[i] = padding[i];
    params->stride[i] = stride[i];
    params->dilation[i] = dilation[i];
  }
  params->groups = groups;
  params->deterministic = deterministic;
}

// Convenience struct for passing around descriptors and data
// pointers
struct ConvolutionArgs {
  miopenHandle_t handle;
  ConvolutionParams params;
  TensorDescriptor idesc, odesc;
  FilterDescriptor wdesc;
  const Tensor& input, output, weight;
  ConvolutionDescriptor cdesc;

  ConvolutionArgs(const Tensor& input, const Tensor& output, const Tensor& weight) : input(input), output(output), weight(weight) {
  }
};

// ---------------------------------------------------------------------
//
// Benchmarking
//
// ---------------------------------------------------------------------

template <typename T>
struct BenchmarkCache {
  std::mutex mutex;
  std::unordered_map<ConvolutionParams, T, ParamsHash<ConvolutionParams>, ParamsEqual<ConvolutionParams>> map;

  bool find(const ConvolutionParams& params, T* results) {
    std::lock_guard<std::mutex> guard(mutex);
    auto it = map.find(params);
    if (it == map.end()) {
      return false;
    }
    *results = it->second;
    return true;
  }

  void insert(const ConvolutionParams& params, const T& results) {
    std::lock_guard<std::mutex> guard(mutex);
    map[params] = results;
  }

};

BenchmarkCache<miopenConvFwdAlgorithm_t> fwd_algos;
BenchmarkCache<miopenConvBwdDataAlgorithm_t> bwd_data_algos;
BenchmarkCache<miopenConvBwdWeightsAlgorithm_t> bwd_filter_algos;

BenchmarkCache<size_t> fwd_wssizes;
BenchmarkCache<size_t> bwd_data_wssizes;
BenchmarkCache<size_t> bwd_filter_wssizes;

struct Workspace {
  Workspace(size_t size) : size(size), data(NULL) {

    data = c10::hip::HIPCachingAllocator::raw_alloc(size);

  }
  Workspace(const Workspace&) = delete;
  Workspace(Workspace&&) = default;
  Workspace& operator=(Workspace&&) = default;
  ~Workspace() {

    if (data) {
      c10::hip::HIPCachingAllocator::raw_delete(data);
    }

  }

  size_t size;
  void* data;
};

template<typename algo_t>
struct algorithm_search {
};

size_t getWorkspaceSize(
    const ConvolutionArgs& args, const miopenConvFwdAlgorithm_t)
{
    size_t sz = 0;
    miopenConvolutionForwardGetWorkSpaceSize(
        args.handle,
        args.wdesc.desc(),
        args.idesc.desc(),
        args.cdesc.desc(),
        args.odesc.desc(),
        &sz);
    return sz;
}
size_t getWorkspaceSize(
    const ConvolutionArgs& args, const miopenConvBwdDataAlgorithm_t)
{
    size_t sz = 0;
    miopenConvolutionBackwardDataGetWorkSpaceSize(
        args.handle,
        args.odesc.desc(),
        args.wdesc.desc(),
        args.cdesc.desc(),
        args.idesc.desc(),
        &sz);
    return sz;
}
size_t getWorkspaceSize(
    const ConvolutionArgs& args, const miopenConvBwdWeightsAlgorithm_t)
{
    size_t sz = 0;
    miopenConvolutionBackwardWeightsGetWorkSpaceSize(
        args.handle,
        args.odesc.desc(),
        args.idesc.desc(),
        args.cdesc.desc(),
        args.wdesc.desc(),
        &sz);
    return sz;
}

template<typename perf_t>
perf_t getBestAlgorithm(perf_t *perfResults, bool deterministic, int n_algo) {
  return perfResults[0];
}

template<>
struct algorithm_search<miopenConvFwdAlgorithm_t> {
  using perf_t = miopenConvAlgoPerf_t;
  using algo_t = miopenConvFwdAlgorithm_t;

  static constexpr auto DEFAULT_ALGO = miopenConvolutionFwdAlgoGEMM;
  static BenchmarkCache<algo_t>& cache() { return fwd_algos; }
  static BenchmarkCache<size_t>& wsscache() { return fwd_wssizes; }

  static perf_t findAlgorithm(const ConvolutionArgs& args) {
    int perf_count;
    perf_t perf_results;
    size_t max_ws_size = getWorkspaceSize(args, DEFAULT_ALGO);
    Workspace ws(max_ws_size);
    MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(
        args.handle,
        args.idesc.desc(), args.input.data_ptr(),
        args.wdesc.desc(), args.weight.data_ptr(),
        args.cdesc.desc(),
        args.odesc.desc(), args.output.data_ptr(),
        1,        // just return the fastest
        &perf_count,
        &perf_results,
        ws.data,
        ws.size,
        false));
    return perf_results;
  }
};

template<>
struct algorithm_search<miopenConvBwdDataAlgorithm_t> {
  using perf_t = miopenConvAlgoPerf_t;
  using algo_t = miopenConvBwdDataAlgorithm_t;

  static constexpr auto DEFAULT_ALGO = miopenConvolutionBwdDataAlgoGEMM;
  static BenchmarkCache<algo_t>& cache() { return bwd_data_algos; }
  static BenchmarkCache<size_t>& wsscache() { return bwd_data_wssizes; }

  static perf_t findAlgorithm(const ConvolutionArgs& args) {
    int perf_count;
    perf_t perf_results;
    size_t max_ws_size = getWorkspaceSize(args, DEFAULT_ALGO);
    Workspace ws(max_ws_size);
    MIOPEN_CHECK(miopenFindConvolutionBackwardDataAlgorithm(
        args.handle,
        args.odesc.desc(), args.output.data_ptr(),
        args.wdesc.desc(), args.weight.data_ptr(),
        args.cdesc.desc(),
        args.idesc.desc(), args.input.data_ptr(),
        1,      // just return the fastest
        &perf_count,
        &perf_results,
        ws.data,
        ws.size,
        false));
    return perf_results;
  }
};

template<>
struct algorithm_search<miopenConvBwdWeightsAlgorithm_t> {
  using perf_t = miopenConvAlgoPerf_t;
  using algo_t = miopenConvBwdWeightsAlgorithm_t;

  static constexpr auto DEFAULT_ALGO = miopenConvolutionBwdWeightsAlgoGEMM;
  static BenchmarkCache<algo_t>& cache() { return bwd_filter_algos; }
  static BenchmarkCache<size_t>& wsscache() { return bwd_filter_wssizes; }

  static perf_t findAlgorithm(const ConvolutionArgs& args) {
    int perf_count;
    perf_t perf_results;
    size_t max_ws_size = getWorkspaceSize(args, DEFAULT_ALGO);
    Workspace ws(max_ws_size);
    MIOPEN_CHECK(miopenFindConvolutionBackwardWeightsAlgorithm(
        args.handle,
        args.odesc.desc(), args.output.data_ptr(),
        args.idesc.desc(), args.input.data_ptr(),
        args.cdesc.desc(),
        args.wdesc.desc(), args.weight.data_ptr(),
        1,      // just return the fastest
        &perf_count,
        &perf_results,
        ws.data,
        ws.size,
        false));
    return perf_results;
  }
};

template<typename algo_t>
void findAlgorithm(const ConvolutionArgs& args, bool benchmark, algo_t* algo) {
  using search = algorithm_search<algo_t>;
  auto& cache = search::cache();
  auto& wsscache = search::wsscache();

  if (cache.find(args.params, algo)) {
    return;
  }

  if (args.params.deterministic && !benchmark) {
    *algo = search::DEFAULT_ALGO;
    return;
  }

  if (cache.find(args.params, algo)) {
    // re-check cache since another thread may have benchmarked the algorithm
    return;
  }

  auto perfResults = search::findAlgorithm(args);
  *algo = reinterpret_cast<algo_t&>(perfResults);

  cache.insert(args.params, *algo);
  wsscache.insert(args.params, perfResults.memory);

  c10::hip::HIPCachingAllocator::emptyCache();

}

template<typename algo_t>
Workspace chooseAlgorithm(
    const ConvolutionArgs& args,
    bool benchmark,
    algo_t* algo)
{
  findAlgorithm(args, benchmark, algo);

  using search = algorithm_search<algo_t>;
  size_t workspace_size;
  search::wsscache().find(args.params, &workspace_size);
  try {
    return Workspace(workspace_size);
  } catch (const std::exception& e) {
    hipGetLastError(); // clear OOM error

    // switch to default algorithm and record it in the cache to prevent
    // further OOM errors
    *algo = search::DEFAULT_ALGO;
    workspace_size = getWorkspaceSize(args, *algo);
    search::cache().insert(args.params, *algo);
    search::wsscache().insert(args.params, workspace_size);
    return Workspace(workspace_size);
  }
}

// ---------------------------------------------------------------------
//
// Bias addition
//
// ---------------------------------------------------------------------

// In-place!

void miopen_convolution_add_bias_nhwc_(CheckedFrom c, const TensorArg& output, const TensorArg& bias)
{
  checkAllSameGPU(c, {output, bias});
  checkSize(c, bias, { output->size(output_channels_dim) });

  TensorDescriptor bdesc, odesc;
  std::vector<int64_t> shape( output->dim(), 1);
  shape[output_channels_dim] = -1;
  at::Tensor bias_contig =  bias->reshape(shape).contiguous();

  output->add_( bias_contig );

}

// see NOTE [ Convolution design ] in src/Aten/native/cudnn/Conv.cpp


// ---------------------------------------------------------------------
//
// Convolution forward / Transposed convolution backward
//
// ---------------------------------------------------------------------

// The raw API directly invokes MIOpen.
//
// There are a few reasons this should never be directly exposed
// via ATen:
//
//    - It takes output as a parameter (this should be computed!)
//    - It doesn't do input checking
//    - It doesn't resize output (it is assumed to be correctly sized)
//

void raw_miopen_convolution_forward_out_nhwc(
    const Tensor& output, const Tensor& input, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {

  auto dataType = getDataType(input);
  miopenConvolutionMode_t c_mode = miopenConvolution;

  ConvolutionArgs args{ input, output, weight };

  args.handle = getMiopenHandle();
  setConvolutionParams(&args.params, input, weight, padding, stride, dilation, groups, deterministic);
  args.idesc.set(input);
  args.wdesc.set(weight);
  args.odesc.set(output);
  args.cdesc.set(dataType, c_mode, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

  miopenConvFwdAlgorithm_t fwdAlg;
  Workspace workspace = chooseAlgorithm(args, benchmark, &fwdAlg);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  MIOPEN_CHECK(miopenConvolutionForward(
    args.handle,
    &one, args.idesc.desc(), input.data_ptr(),
    args.wdesc.desc(), weight.data_ptr(),
    args.cdesc.desc(), fwdAlg, &zero,
    args.odesc.desc(), output.data_ptr(), workspace.data, workspace.size));

}


Tensor miopen_convolution_forward_nhwc(
    CheckedFrom c,
    const TensorArg& input, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{

  //checkAllSameType(c, {input, weight});
  checkAllSameGPU(c, {input, weight});

  auto output_t = at::empty(conv_output_size_nhwc(input->sizes(), weight->sizes(),
                                                  padding, stride, dilation, groups),
                            input->options());

  // Avoid ambiguity of "output" when this is being used as backwards
  TensorArg output{ output_t, "result", 0 };

  //convolution_shape_check(c, input, weight, output, padding, stride, dilation, groups);

  // See #4500
  Tensor weight_contig = weight->contiguous();

  raw_miopen_convolution_forward_out_nhwc(
      *output, *input, weight_contig,
      padding, stride, dilation, groups, benchmark, deterministic);

  return *output;
}

Tensor miopen_convolution_nhwc(
    const Tensor& input_t, const Tensor& weight_t,
    std::vector<long> padding, std::vector<long> stride, std::vector<long> dilation, int64_t groups, bool benchmark, bool deterministic)
{

  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 };

  CheckedFrom c = "miopen_convolution_nhwc";

  auto output_t = miopen_convolution_forward_nhwc(
    c, input, weight, padding, stride, dilation, groups, benchmark, deterministic);

  return output_t;
}


Tensor miopen_convolution_with_bias_nhwc(
   const at::Tensor& input_t, const at::Tensor& weight_t, const at::Tensor& bias_t,
    std::vector<long> padding, std::vector<long> stride, std::vector<long> dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 },
            bias   { bias_t,   "bias",   3 };
  CheckedFrom c = "miopen_convolution_nhwc";
  auto output_t = miopen_convolution_forward_nhwc(
    c, input, weight, padding, stride, dilation, groups, benchmark, deterministic);

  miopen_convolution_add_bias_nhwc_(c, { output_t, "result", 0 }, bias);

  return output_t;
}

// NB: output_padding not needed here, as there is no ambiguity to
// resolve
Tensor miopen_convolution_transpose_backward_input_nhwc(
    const Tensor& grad_output_t, const Tensor& weight_t,
    std::vector<long> padding, std::vector<long> stride, std::vector<long> dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  TensorArg grad_output { grad_output_t,  "grad_output", 1 },
            weight      { weight_t, "weight", 2 };
  return miopen_convolution_forward_nhwc(
    "miopen_convolution_transpose_backward_input",
    grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

// ---------------------------------------------------------------------
//
// Convolution backward / Transposed convolution forward
//
// ---------------------------------------------------------------------

void raw_miopen_convolution_backward_input_out_nhwc(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {

  auto dataType = getDataType(grad_output);
  miopenConvolutionMode_t c_mode = miopenConvolution;

  ConvolutionArgs args{ grad_input, grad_output, weight };
  args.handle = getMiopenHandle();
  setConvolutionParams(&args.params, grad_input, weight, padding, stride, dilation, groups, deterministic);
  args.idesc.set(grad_input);
  args.wdesc.set(weight);
  args.odesc.set(grad_output);
  args.cdesc.set(dataType, c_mode, grad_output.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

  miopenConvBwdDataAlgorithm_t bwdDataAlg;
  Workspace workspace = chooseAlgorithm(args, benchmark, &bwdDataAlg);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  MIOPEN_CHECK(miopenConvolutionBackwardData(
      args.handle,
      &one, args.odesc.desc(), grad_output.data_ptr(),
      args.wdesc.desc(), weight.data_ptr(),
      args.cdesc.desc(), bwdDataAlg, &zero,
      args.idesc.desc(), grad_input.data_ptr(), workspace.data, workspace.size));

}

// see NOTE [ Backward vs transpose convolutions ] in src/Aten/native/cudnn/Conv.cpp


Tensor miopen_convolution_backward_input_nhwc(
    CheckedFrom c,
    IntArrayRef input_size, const TensorArg& grad_output, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  //checkAllSameType(c, {grad_output, weight});
  checkAllSameGPU(c, {grad_output, weight});

  auto grad_input_t = at::empty(input_size, grad_output->options());

  // Avoid "grad_input" when this is being used as transposed convolution
  TensorArg grad_input{ grad_input_t, "result", 0 };
  //convolution_shape_check(c, grad_input, weight, grad_output, padding, stride, dilation, groups);

  // See #4500
  Tensor weight_contig = weight->contiguous();

  raw_miopen_convolution_backward_input_out_nhwc(
      *grad_input, *grad_output, weight_contig,
      padding, stride, dilation, groups, benchmark, deterministic);

  return *grad_input;
}

Tensor miopen_convolution_transpose_forward_nhwc(
    CheckedFrom c,
    const TensorArg& grad_output, const TensorArg& weight,
    std::vector<long> padding, std::vector<long> output_padding, std::vector<long> stride, std::vector<long> dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  auto input_size = conv_input_size(grad_output->sizes(), weight->sizes(),
                                    padding, output_padding, stride, dilation, groups);
  return miopen_convolution_backward_input_nhwc(c, input_size, grad_output, weight,
                                    padding, stride, dilation, groups, benchmark, deterministic);
}


Tensor miopen_convolution_backward_input_nhwc(
    IntArrayRef input_size, const Tensor& grad_output_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  TensorArg grad_output{ grad_output_t, "grad_output", 1 },
            weight{ weight_t, "weight", 2 };
  return miopen_convolution_backward_input_nhwc(
      "miopen_convolution_backward_input_nhwc",
      input_size, grad_output, weight,
      padding, stride, dilation, groups, benchmark, deterministic);
}

// ---------------------------------------------------------------------
//
// Convolution backward (weight)
//
// ---------------------------------------------------------------------

void raw_miopen_convolution_backward_weight_out_nhwc(
    const Tensor& grad_weight, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {

  auto dataType = getDataType(input);
  miopenConvolutionMode_t c_mode = miopenConvolution;

  ConvolutionArgs args{ input, grad_output, grad_weight };
  args.handle = getMiopenHandle();
  setConvolutionParams(&args.params, input, grad_weight, padding, stride, dilation, groups, deterministic);
  args.idesc.set(input);
  args.wdesc.set(grad_weight);
  args.odesc.set(grad_output);
  args.cdesc.set(dataType, c_mode, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

  miopenConvBwdWeightsAlgorithm_t bwdFilterAlg;
  Workspace workspace = chooseAlgorithm(args, benchmark, &bwdFilterAlg);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  MIOPEN_CHECK(miopenConvolutionBackwardWeights(
      args.handle,
      &one, args.odesc.desc(), grad_output.data_ptr(),
      args.idesc.desc(), input.data_ptr(),
      args.cdesc.desc(), bwdFilterAlg, &zero,
      args.wdesc.desc(), grad_weight.data_ptr(), workspace.data, workspace.size));

}


Tensor miopen_convolution_backward_weight_nhwc(
    CheckedFrom c,
    IntArrayRef weight_size, const TensorArg& grad_output, const TensorArg& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{

  //checkAllSameType(c, {grad_output, input});
  checkAllSameGPU(c, {grad_output, input});

  auto grad_weight_t = at::empty(weight_size, grad_output->options());

  // For uniformity with everything else, although it seems grad_weight
  // would be unambiguous too.
  TensorArg grad_weight{ grad_weight_t, "result", 0 };

  raw_miopen_convolution_backward_weight_out_nhwc(
      *grad_weight, *grad_output, *input,
      padding, stride, dilation, groups, benchmark, deterministic);

  return grad_weight_t;
}


Tensor miopen_convolution_backward_weight_nhwc(
    IntArrayRef weight_size,
    const Tensor& grad_output_t,
    const Tensor& input_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  TensorArg grad_output{ grad_output_t, "grad_output", 1 },
            input{ input_t, "input", 2 };
  return miopen_convolution_backward_weight_nhwc(
      "miopen_convolution_backward_weight_nhwc",
      weight_size, grad_output, input,
      padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor miopen_convolution_transpose_backward_weight_nhwc(
    IntArrayRef weight_size,
    const Tensor& grad_output_t,
    const Tensor& input_t,
    std::vector<long> padding, std::vector<long> stride, std::vector<long> dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  TensorArg grad_output{ grad_output_t, "grad_output", 1 },
            input{ input_t, "input", 2 };
  return miopen_convolution_backward_weight_nhwc(
      "miopen_convolution_backward_weight_nhwc",
      weight_size, input, grad_output,
      padding, stride, dilation, groups, benchmark, deterministic);
}

// This is the main bprop entry-point
std::tuple<at::Tensor,at::Tensor> miopen_convolution_backward_nhwc(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight, std::vector<long> padding, std::vector<long> stride, std::vector<long> dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,2> output_mask) {

  Tensor grad_output = grad_output_t.contiguous();

  Tensor grad_input = at::empty({}, grad_output_t.options()), grad_weight;

  if (output_mask[0]) {
    grad_input = miopen_convolution_backward_input_nhwc(input.sizes(), grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }

  if (output_mask[1]) {
    grad_weight = miopen_convolution_backward_weight_nhwc(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
  }
  return std::tuple<Tensor,Tensor>{grad_input, grad_weight};
}
// ---------------------------------------------------------------------
//
// Convolution backward (bias)
//
// ---------------------------------------------------------------------

Tensor miopen_convolution_backward_bias_nhwc(
    const Tensor& grad_output_t)
{
  TensorArg grad_output{ grad_output_t, "grad_output", 1 };

  std::vector<int64_t> discard_dims;
  for( int i = 0; i < grad_output_t.dim(); i++ ) {
      if(i != output_channels_dim ) {
          discard_dims.push_back(i);
      }
  }
  Tensor outputBias = at::squeeze( at::sum(grad_output_t, discard_dims, true) );
  if( outputBias.dim() == 0 ) {
      // always return a tensor of shape [_]
      return outputBias.unsqueeze(0);
  }
  else {
      return outputBias;
  }

}



// This is the main bprop entry-point
std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_convolution_backward_with_bias_nhwc(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    std::vector<long> padding, std::vector<long> stride, std::vector<long> dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask) {

  Tensor grad_output = grad_output_t.contiguous();

  Tensor grad_input = at::empty({}, grad_output_t.options()), grad_weight, grad_bias;

  if (output_mask[0]) {
    grad_input = miopen_convolution_backward_input_nhwc(input.sizes(), grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }

  if (output_mask[1]) {
    grad_weight = miopen_convolution_backward_weight_nhwc(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
  }

  grad_bias = miopen_convolution_backward_bias_nhwc(grad_output);

  return std::tuple<Tensor,Tensor,Tensor>{grad_input, grad_weight, grad_bias};
}

Tensor miopen_convolution_transpose_nhwc(
    const Tensor& input_t, const Tensor& weight_t,
    std::vector<long> padding, std::vector<long> output_padding, std::vector<long> stride, std::vector<long> dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 };
  CheckedFrom c = "miopen_convolution_transpose";
  auto output_t = miopen_convolution_transpose_forward_nhwc(
    c, input, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  return output_t;
}

Tensor miopen_convolution_transpose_with_bias_nhwc(
    const Tensor& input_t, const Tensor& weight_t, const Tensor& bias_t,
    std::vector<long> padding, std::vector<long> output_padding, std::vector<long> stride, std::vector<long> dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 },
            bias   { bias_t,   "bias",   3 };
  CheckedFrom c = "miopen_convolution_transpose";
  auto output_t = miopen_convolution_transpose_forward_nhwc(
    c, input, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  if (bias->defined()) {
    miopen_convolution_add_bias_nhwc_(c, { output_t, "result", 0 }, bias);
  }
  return output_t;
}

std::tuple<at::Tensor,at::Tensor> miopen_convolution_transpose_backward_nhwc(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    std::vector<long> padding, std::vector<long> output_padding, std::vector<long> stride, std::vector<long> dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,2> output_mask) {

  Tensor grad_output = grad_output_t.contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = miopen_convolution_transpose_backward_input_nhwc(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  if (output_mask[1]) {
    grad_weight = miopen_convolution_transpose_backward_weight_nhwc(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
  }

  return std::tuple<Tensor,Tensor>{grad_input, grad_weight};
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_convolution_transpose_backward_with_bias_nhwc(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    std::vector<long> padding, std::vector<long> output_padding, std::vector<long> stride, std::vector<long> dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask) {

  Tensor grad_output = grad_output_t.contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = miopen_convolution_transpose_backward_input_nhwc(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  if (output_mask[1]) {
    grad_weight = miopen_convolution_transpose_backward_weight_nhwc(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
  }
  if (output_mask[2]) {
    grad_bias = miopen_convolution_backward_bias_nhwc(grad_output);
  }

  return std::tuple<Tensor,Tensor,Tensor>{grad_input, grad_weight, grad_bias};
}

}}} // namespace

