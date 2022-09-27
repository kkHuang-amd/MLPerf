#include <torch/extension.h>

#ifdef USE_WMMA
torch::Tensor dotBasedInteractFwdTorch(
    torch::Tensor input,
    torch::Tensor bottom_mlp_output,
    uint pad);
std::vector<torch::Tensor> dotBasedInteractBwdTorch(
    torch::Tensor input,
    torch::Tensor upstreamGrad,
    uint pad);
#endif
torch::Tensor dotBasedInteractNoWmmaFwdTorch(
    torch::Tensor input,
    torch::Tensor bottom_mlp_output,
    uint output_padding_width);
std::vector<torch::Tensor> dotBasedInteractNoWmmaBwdTorch(
    torch::Tensor input,
    torch::Tensor upstreamGrad,
    uint output_padding_width);
torch::Tensor gather_gpu_fwd(const torch::Tensor weight, const torch::Tensor indices, const bool to_fp16);
torch::Tensor gather_gpu_bwd(const torch::Tensor grad, const torch::Tensor indices, const int num_features);
void dense_sparse_add(
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor indices,
    const float alpha);
std::vector<torch::Tensor> unique_transpose(
    torch::Tensor input,
    const int num_embs,
    const int num_gpus);
torch::Tensor deduplicate_tensor_atm(
    torch::Tensor input,
    torch::Tensor indices,
    std::vector<int> num_unique_indices,
    std::vector<int> vecs_per_gpu,
    const int batch_size_per_gpu,
    const int bottom_mlp_rank);
torch::Tensor index_select_transpose(
    torch::Tensor input,
    torch::Tensor indices,
    std::vector<int> num_unique_indices,
    std::vector<int> embs_per_gpu,
    const int batch_size_per_gpu,
    const int bottom_mlp_rank);
torch::Tensor fused_index_select_dot(
    torch::Tensor emb_output,
    torch::Tensor bottom_mlp_output,
    torch::Tensor indices,
    torch::Tensor vecs_per_gpu,
    torch::Tensor indices_offset,
    uint batch_size,
    uint num_features,
    uint emb_dim,
    uint pad);
torch::Tensor fused_dot_dedup(
    torch::Tensor emb_output,
    torch::Tensor bottom_mlp_output,
    torch::Tensor indices,
    torch::Tensor vecs_per_gpu,
    torch::Tensor indices_offset,
    torch::Tensor upstream_grad,
    uint batch_size,
    uint num_features,
    uint emb_dim,
    uint pad);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef USE_WMMA
  m.def("dotBasedInteractFwd", &dotBasedInteractFwdTorch, "", py::arg("input"), py::arg("bottom_mlp_output"), py::arg("pad"));
  m.def("dotBasedInteractBwd", &dotBasedInteractBwdTorch, "", py::arg("input"), py::arg("upstreamGrad"), py::arg("pad"));
#endif
  m.def("dotBasedInteractNoWmmaFwd", &dotBasedInteractNoWmmaFwdTorch, "", py::arg("input"), py::arg("bottom_mlp_output"), py::arg("pad"));
  m.def("dotBasedInteractNoWmmaBwd", &dotBasedInteractNoWmmaBwdTorch, "", py::arg("input"), py::arg("upstreamGrad"), py::arg("pad"));
  m.def("gather_gpu_fwd", &gather_gpu_fwd, "Embedding gather", py::arg("weight"), py::arg("indices"), py::arg("to_fp16"));
  m.def("gather_gpu_bwd", &gather_gpu_bwd, "Embedding gather backward",
        py::arg("grad"), py::arg("indices"), py::arg("num_features"));
  m.def("dense_sparse_add", &dense_sparse_add, "Dense-sparse add for the embedding optimizer");
  m.def("unique_transpose", &unique_transpose, "Unique and transpose");
  m.def("deduplicate_tensor_atm", &deduplicate_tensor_atm, "Deduplicate tensor using Atomic");
  m.def("index_select_transpose", &index_select_transpose, "Index select and transpose");
  m.def("fused_index_select_dot", &fused_index_select_dot, "Fused index select and dot interaction");
  m.def("fused_dot_dedup", &fused_dot_dedup, "Fused dot interaction and dedup");
}
