"""Compare the CUDA extension with a PyTorch blockwise reference."""
import torch

from setup import build_extension


def main():
    extension = build_extension()
    torch.manual_seed(0)
    weight = torch.randn(4096, 1024, device="cuda").to(torch.float8_e4m3fn)
    scale_inv = torch.rand(32, 8, device="cuda", dtype=torch.bfloat16)

    # 在非默认 stream 上测试，验证包装层使用的是调用方的当前 stream。
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = extension.dequant_fp8(weight, scale_inv)
        # 展开每个块的 scale，独立计算 PyTorch 参考结果。
        expanded_scale = scale_inv.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
        expected = (weight.float() * expanded_scale).bfloat16()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    print(f"PASS: {actual.numel():,} elements match the PyTorch reference")


if __name__ == "__main__":
    main()

