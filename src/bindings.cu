#include <cuda_runtime.h>
#include <torch/extension.h>

namespace py = pybind11;

void launch_reshape_and_cache(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping
);


void launch_paged_attention_v1(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    int max_context_len
);


void launch_paged_attention_v2(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    int max_context_len
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TinyServe Low-Level Bindings";
    m.def("paged_attention_v1", &launch_paged_attention_v1);
    m.def("paged_attention_v2", &launch_paged_attention_v2);
    m.def("reshape_and_cache", &launch_reshape_and_cache);
}