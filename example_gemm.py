"""Run the dequant_fp8.cu kernel directly on PyTorch CUDA tensors."""
import ctypes
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import torch


def main():
    source = Path(__file__).resolve().parent / "gemm/dequant_fp8.cu"
    with TemporaryDirectory() as build_dir:
        library_path = Path(build_dir) / "dequant_fp8.so"
        major, minor = torch.cuda.get_device_capability()
        subprocess.run(
            [
                os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc"),
                "-O3", f"-arch=sm_{major}{minor}", "-shared", "-Xcompiler", "-fPIC",
                "-DDEQUANT_SHARED", str(source), "-o", str(library_path),
            ],
            check=True,
        )
        library = ctypes.CDLL(str(library_path))
        launch = library.launch_dequant_fp8
        launch.argtypes = [ctypes.c_void_p] * 4
        launch.restype = ctypes.c_int

        weight = torch.randn(4096, 1024, device="cuda").to(torch.float8_e4m3fn)
        scale = torch.rand(32, 8, device="cuda", dtype=torch.bfloat16)
        output = torch.empty_like(weight, dtype=torch.bfloat16)

        stream = torch.cuda.current_stream()
        status = launch(weight.data_ptr(), scale.data_ptr(), output.data_ptr(), stream.cuda_stream)
        if status != 0:
            raise RuntimeError(f"CUDA launch failed with error code {status}")

        expanded = scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
        reference = (weight.float() * expanded).bfloat16()
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        print(f"PASS: {source.name}, {output.numel():,} elements match PyTorch")


if __name__ == "__main__":
    main()

