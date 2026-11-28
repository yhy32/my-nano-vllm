#include <torch/extension.h>

at::Tensor dequant_fp8(const at::Tensor& weight, const at::Tensor& scale_inv);

// 把 C++ 的 Tensor 接口暴露为 Python 扩展的 dequant_fp8()。
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def("dequant_fp8", &dequant_fp8, "Blockwise FP8 to BF16 dequantization",
               pybind11::arg("weight"), pybind11::arg("scale_inv"));
}
