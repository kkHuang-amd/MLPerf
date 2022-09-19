#include <torch/extension.h>

void multi_tensor_scale_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  float scale);

void multi_tensor_axpby_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  float a,
  float b,
  int arg_to_check);

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::optional<bool> per_tensor_python);

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_mp_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::optional<bool> per_tensor_python);

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_scale_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  float scale,
  at::optional<bool> per_tensor_python);

void multi_tensor_lars_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor grad_norms,
  at::Tensor param_norms,
  float lr,
  float trust_coefficient,
  float epsilon,
  float weight_decay,
  float momentum,
  float dampening,
  bool nesterov,
  bool first_run,
  bool wd_after_momentum,
  float scale,
  const bool is_skipped);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multi_tensor_scale", &multi_tensor_scale_cuda,
        "Fused overflow check + scale for a list of contiguous tensors");
  m.def("multi_tensor_axpby", &multi_tensor_axpby_cuda,
        "out = a*x + b*y for a list of contiguous tensors");
  m.def("multi_tensor_l2norm", &multi_tensor_l2norm_cuda,
        "Computes L2 norm for a list of contiguous tensors");
  m.def("multi_tensor_l2norm_mp", &multi_tensor_l2norm_mp_cuda,
        "Computes L2 norm for a list of contiguous tensors");
  m.def("multi_tensor_l2norm_scale", &multi_tensor_l2norm_scale_cuda,
        "Computes L2 norm for a list of contiguous tensors and does scaling");
  m.def("multi_tensor_lars", &multi_tensor_lars_cuda,
        "Computes new gradients using LARS");
}
