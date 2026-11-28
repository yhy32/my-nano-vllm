#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

// 接收 GPU 指针，在调用方的 CUDA stream 上启动 kernel，结果写入 output。
cudaError_t launch_dequant_fp8_weight(const void* weight, const void* scale_inv,
                                      void* output, int64_t rows, int64_t cols, cudaStream_t stream);
