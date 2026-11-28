#include "my_op.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace {
void check_fp8_weight(const at::Tensor& weight, const at::Tensor& scale_inv)
{
    // 包装层先检查输入，kernel 可以直接按连续布局访问裸指针。
    TORCH_CHECK(weight.is_cuda() && scale_inv.device() == weight.device(),
                "weight and scale_inv must be on the same CUDA device");
    TORCH_CHECK(weight.is_contiguous() && scale_inv.is_contiguous(),
                "weight and scale_inv must be contiguous");
    constexpr int64_t BlockSize = 128;
    TORCH_CHECK(weight.scalar_type() == at::kFloat8_e4m3fn && weight.dim() == 2,
                "weight must be a 2D FP8 E4M3FN tensor");
    const auto rows = weight.size(0);
    const auto cols = weight.size(1);
    TORCH_CHECK(rows > 0 && cols > 0 && rows % BlockSize == 0 && cols % BlockSize == 0,
                "weight dimensions must be positive multiples of 128");
    TORCH_CHECK(scale_inv.scalar_type() == at::kBFloat16 && scale_inv.dim() == 2 &&
                scale_inv.size(0) == rows / BlockSize && scale_inv.size(1) == cols / BlockSize,
                "scale_inv must be BF16 with one entry per 128x128 weight block");

}
}

at::Tensor dequant_fp8(const at::Tensor& weight, const at::Tensor& scale_inv)
{
    check_fp8_weight(weight, scale_inv);
    const auto rows = weight.size(0);
    const auto cols = weight.size(1);
    // PyTorch 分配输出，并提供当前设备和 stream，与后面的 F.linear 保持执行顺序。
    const c10::cuda::CUDAGuard guard(weight.device());
    auto output = at::empty(weight.sizes(), weight.options().dtype(at::kBFloat16));
    const auto stream = c10::cuda::getCurrentCUDAStream(weight.get_device());
    // 这里只把设备指针和形状传给 CUDA 启动函数，权重始终留在 GPU 上。
    const auto status = launch_dequant_fp8_weight(
        weight.data_ptr(), scale_inv.data_ptr(), output.data_ptr(), rows, cols, stream.stream());
    TORCH_CHECK(status == cudaSuccess, "dequant_fp8 launch failed: ", cudaGetErrorString(status));
    return output;
}
