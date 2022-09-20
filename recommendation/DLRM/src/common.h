#include <torch/extension.h>
#include <vector>

#define ENTRIES_PER_BLOCK 128

torch::Tensor vec_to_int_tensor(std::vector<int> vec, at::TensorOptions opt);
