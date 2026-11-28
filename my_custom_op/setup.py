"""Build and load the local PyTorch CUDA extension."""
from pathlib import Path

from torch.utils.cpp_extension import load


def build_extension(*, verbose=False):
    root = Path(__file__).resolve().parent
    sources = ["my_op.cc", "my_op_bind.cc", "my_op.cu"]
    # 编译 Tensor 包装、Python 绑定和 CUDA kernel，并复用 PyTorch 的扩展缓存。
    return load(
        name="my_custom_op_ext",
        sources=[str(root / "src" / source) for source in sources],
        extra_include_paths=[str(root / "include")],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O3"],
        verbose=verbose,
    )


if __name__ == "__main__":
    extension = build_extension(verbose=True)
    print(extension.__file__)

