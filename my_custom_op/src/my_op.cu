#include "my_op.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace {
constexpr int BlockSize = 128;

__global__ void dequant_fp8_weight(const __nv_fp8_e4m3* weight,
                                  const __nv_bfloat16* scale_inv,
                                  __nv_bfloat16* output, int64_t elements, int64_t cols)
{
    const int64_t id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= elements)
        return;

    const int64_t row = id / cols;
    const int64_t col = id % cols;
    const int64_t scale_id = (row / BlockSize) * (cols / BlockSize) + col / BlockSize;
    const float value = static_cast<float>(weight[id]);
    const float scale = __bfloat162float(scale_inv[scale_id]);
    output[id] = __float2bfloat16_rn(value * scale);
}

}
cudaError_t launch_dequant_fp8_weight(const void* weight, const void* scale_inv,
                                      void* output, int64_t rows, int64_t cols, cudaStream_t stream)
{
    constexpr int Threads = 256;
    const int64_t elements = rows * cols;
    dequant_fp8_weight<<<(elements + Threads - 1) / Threads, Threads, 0, stream>>>(
        static_cast<const __nv_fp8_e4m3*>(weight),
        static_cast<const __nv_bfloat16*>(scale_inv),
        static_cast<__nv_bfloat16*>(output), elements, cols);
    // 检查 launch 是否成功，后续算子在同一 stream 上按顺序消费输出。
    return cudaGetLastError();
}
