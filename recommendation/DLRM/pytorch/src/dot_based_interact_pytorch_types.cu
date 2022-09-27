#include <torch/extension.h>
#include <torch/types.h>
#include <stdexcept>
#include "dot_based_interact.cu"
#include "dot_based_interact_no_wmma.cu"

#ifdef USE_WMMA
torch::Tensor dotBasedInteractFwdTorch(torch::Tensor input, torch::Tensor bottom_mlp_output, uint output_padding_width) {
  auto size = input.sizes();
  auto batch_size = size[0];
  auto num_rows = size[1];
  auto num_cols = size[2];
  uint output_size = ((num_rows * (num_rows - 1)) >> 1) + num_cols + output_padding_width;

  int64_t outputShape[2] = {batch_size, output_size};
  auto output = torch::empty(c10::IntArrayRef(outputShape), input.options());

  if (input.scalar_type() == torch::ScalarType::Half &&
      bottom_mlp_output.scalar_type() == torch::ScalarType::Half) {
    dotBasedInteractFwd(input.contiguous().data_ptr<at::Half>(),
                        bottom_mlp_output.contiguous().data_ptr<at::Half>(),
                        output.contiguous().data_ptr<at::Half>(),
                        batch_size,
                        num_rows,
                        num_cols,
                        output_padding_width,
                        true);
  }
  else if (input.scalar_type() == torch::ScalarType::Float &&
           bottom_mlp_output.scalar_type() == torch::ScalarType::Float) {
#ifdef __HIP_PLATFORM_HCC__
    dotBasedInteractFwd(input.contiguous().data_ptr<float>(),
                        bottom_mlp_output.contiguous().data_ptr<float>(),
                        output.contiguous().data_ptr<float>(),
                        batch_size,
                        num_rows,
                        num_cols,
                        output_padding_width,
                        /*amp_train=*/false);
#else
    // CUDA does not support FP32 WMMA
    dotBasedInteractNoWmmaFwd(input.contiguous().data_ptr<float>(),
                              bottom_mlp_output.contiguous().data_ptr<float>(),
                              output.contiguous().data_ptr<float>(),
                              batch_size,
                              num_rows,
                              num_cols,
                              output_padding_width,
                              /*amp_train=*/false);
#endif
  }
  else {
    throw std::invalid_argument("Invalid input type.");
  }
  return output;
}

std::vector<torch::Tensor> dotBasedInteractBwdTorch(torch::Tensor input, torch::Tensor upstreamGrad, uint output_padding_width) {
  auto size = input.sizes();
  auto batch_size = size[0];
  auto num_rows = size[1];
  auto num_cols = size[2];

  auto outputGrad = torch::empty_like(input);
  int64_t outputShape[2] = {batch_size, num_cols};
  auto mlp_grad = torch::empty(c10::IntArrayRef(outputShape), input.options());

  if (input.scalar_type() == torch::ScalarType::Half &&
      upstreamGrad.scalar_type() == torch::ScalarType::Half) {
    dotBasedInteractBwd(input.contiguous().data_ptr<at::Half>(),
                        upstreamGrad.contiguous().data_ptr<at::Half>(),
                        outputGrad.contiguous().data_ptr<at::Half>(),
                        mlp_grad.contiguous().data_ptr<at::Half>(),
                        batch_size,
                        num_rows,
                        num_cols,
                        output_padding_width,
                        true);
  }
  else if (input.scalar_type() == torch::ScalarType::Float &&
           upstreamGrad.scalar_type() == torch::ScalarType::Float) {
#ifdef __HIP_PLATFORM_HCC__
    dotBasedInteractBwd(input.contiguous().data_ptr<float>(),
                        upstreamGrad.contiguous().data_ptr<float>(),
                        outputGrad.contiguous().data_ptr<float>(),
                        mlp_grad.contiguous().data_ptr<float>(),
                        batch_size,
                        num_rows,
                        num_cols,
                        output_padding_width,
                        false);
#else
    dotBasedInteractNoWmmaBwd(input.contiguous().data_ptr<float>(),
                              upstreamGrad.contiguous().data_ptr<float>(),
                              outputGrad.contiguous().data_ptr<float>(),
                              mlp_grad.contiguous().data_ptr<float>(),
                              batch_size,
                              num_rows,
                              num_cols,
                              output_padding_width,
                              /*amp_train=*/false);
#endif
  }
  else {
    throw std::invalid_argument("Invalid input type.");
  }
  return {outputGrad, mlp_grad};
}
#endif // USE_WMMA

torch::Tensor dotBasedInteractNoWmmaFwdTorch(torch::Tensor input, torch::Tensor bottom_mlp_output, uint output_padding_width) {
  auto size = input.sizes();
  auto batch_size = size[0];
  auto num_rows = size[1];
  auto num_cols = size[2];
  uint output_size = ((num_rows * (num_rows - 1)) >> 1) + num_cols + output_padding_width;

  int64_t outputShape[2] = {batch_size, output_size};
  auto output = torch::empty(c10::IntArrayRef(outputShape), input.options());

  if (input.scalar_type() == torch::ScalarType::Half &&
      bottom_mlp_output.scalar_type() == torch::ScalarType::Half) {
    dotBasedInteractNoWmmaFwd(input.contiguous().data_ptr<at::Half>(),
                              bottom_mlp_output.contiguous().data_ptr<at::Half>(),
                              output.contiguous().data_ptr<at::Half>(),
                              batch_size,
                              num_rows,
                              num_cols,
                              output_padding_width,
                              /*amp_train=*/true);
  }
  else if (input.scalar_type() == torch::ScalarType::Float &&
           bottom_mlp_output.scalar_type() == torch::ScalarType::Float) {
    dotBasedInteractNoWmmaFwd(input.contiguous().data_ptr<float>(),
                              bottom_mlp_output.contiguous().data_ptr<float>(),
                              output.contiguous().data_ptr<float>(),
                              batch_size,
                              num_rows,
                              num_cols,
                              output_padding_width,
                              /*amp_train=*/false);
  }
  else {
    throw std::invalid_argument("Invalid input type.");
  }
  return output;
}

std::vector<torch::Tensor> dotBasedInteractNoWmmaBwdTorch(torch::Tensor input, torch::Tensor upstreamGrad, uint output_padding_width) {
  auto size = input.sizes();
  auto batch_size = size[0];
  auto num_rows = size[1];
  auto num_cols = size[2];

  auto outputGrad = torch::empty_like(input);
  int64_t outputShape[2] = {batch_size, num_cols};
  auto mlp_grad = torch::empty(c10::IntArrayRef(outputShape), input.options());

  if (input.scalar_type() == torch::ScalarType::Half &&
      upstreamGrad.scalar_type() == torch::ScalarType::Half) {
    dotBasedInteractNoWmmaBwd(input.contiguous().data_ptr<at::Half>(),
                              upstreamGrad.contiguous().data_ptr<at::Half>(),
                              outputGrad.contiguous().data_ptr<at::Half>(),
                              mlp_grad.contiguous().data_ptr<at::Half>(),
                              batch_size,
                              num_rows,
                              num_cols,
                              output_padding_width,
                              /*amp_train=*/true);
  }
  else if (input.scalar_type() == torch::ScalarType::Float &&
           upstreamGrad.scalar_type() == torch::ScalarType::Float) {
    dotBasedInteractNoWmmaBwd(input.contiguous().data_ptr<float>(),
                              upstreamGrad.contiguous().data_ptr<float>(),
                              outputGrad.contiguous().data_ptr<float>(),
                              mlp_grad.contiguous().data_ptr<float>(),
                              batch_size,
                              num_rows,
                              num_cols,
                              output_padding_width,
                              /*amp_train=*/false);
  }
  else {
    throw std::invalid_argument("Invalid input type.");
  }
  return {outputGrad, mlp_grad};
}
