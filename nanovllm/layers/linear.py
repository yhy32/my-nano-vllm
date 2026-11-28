import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        *, local: bool = False,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = 0 if local else dist.get_rank()
        self.tp_size = 1 if local else dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        self.register_buffer("scale_inv", None, persistent=False)
        # ModelRunner 加载权重后可绑定自定义 CUDA 函数，默认走 PyTorch 路径。
        self.dequant_kernel = None
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
    def weight_for_compute(self, dtype: torch.dtype) -> torch.Tensor:
        if self.weight.dtype != torch.float8_e4m3fn:
            return self.weight

        # 自定义算子返回临时 BF16 权重供 F.linear 使用，原始参数继续以 FP8 保存。
        if self.dequant_kernel is not None:
            return self.dequant_kernel(self.weight, self.scale_inv).to(dtype)

        # PyTorch 反量化路径按块广播 scale，供未绑定扩展的 FP8 层使用。
        out_block, in_block = self.weight_block_size
        out_blocks, in_blocks = self.scale_inv.shape
        blocks = self.weight.float().view(out_blocks, out_block, in_blocks, in_block)
        scales = self.scale_inv.float()[:, None, :, None]
        blocks.mul_(scales)
        return blocks.reshape_as(self.weight).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_for_compute(x.dtype), self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        *, local: bool = False,
    ):
        tp_size = 1 if local else dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0, local=local)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_for_compute(x.dtype), self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        *, local: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias, local=local)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int, scale_inv=None, weight_block_size=None):
        if loaded_weight.dtype == torch.float8_e4m3fn:
            if scale_inv is None or weight_block_size is None:
                raise ValueError("FP8 权重缺少 scale 或 block size")

            if param.dtype != loaded_weight.dtype:
                param.requires_grad_(False)
                param.data = torch.empty_like(param, dtype=loaded_weight.dtype)
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
        if scale_inv is not None:
            self.weight_block_size = weight_block_size
            out_block, in_block = weight_block_size
            # 只有第一次加载scale的时候走分配，之后self.scale_inv已经是tensor，跳过
            if self.scale_inv is None:
                local_out_dim = self.weight.shape[0]
                in_dim = self.weight.shape[1]
                out_blk = local_out_dim // out_block
                in_blk = in_dim // in_block
                self.scale_inv = torch.empty(out_blk, in_blk, dtype=torch.bfloat16, device=self.weight.device)

            scale_shard = scale_inv.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
            block_size = out_block
            blk_offset = shard_offset // block_size
            blk_size = shard_size // block_size
            scale_dst = self.scale_inv.narrow(0, blk_offset, blk_size)
            scale_dst.copy_(scale_shard)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str, scale_inv=None, weight_block_size=None):
        if loaded_weight.dtype == torch.float8_e4m3fn:
            if scale_inv is None or weight_block_size is None:
                raise ValueError("FP8 权重缺少 scale 或 block size")

            if param.dtype != loaded_weight.dtype:
                param.requires_grad_(False)
                param.data = torch.empty_like(param, dtype=loaded_weight.dtype)
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

        if scale_inv is not None:
            self.weight_block_size = weight_block_size
            out_block, in_block = weight_block_size
            if self.scale_inv is None:
                local_out_dim = self.weight.shape[0]
                in_dim = self.weight.shape[1]
                out_blk = local_out_dim // out_block
                in_blk = in_dim // in_block
                self.scale_inv = torch.empty(out_blk, in_blk, dtype=torch.bfloat16, device=self.weight.device)

            scale_shard = scale_inv.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
            block_size = out_block
            blk_offset = shard_offset // block_size
            blk_size = shard_size // block_size
            scale_dst = self.scale_inv.narrow(0, blk_offset, blk_size)
            scale_dst.copy_(scale_shard)

class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        *, local: bool = False,
    ):
        tp_size = 1 if local else dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1, local=local)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, scale_inv=None, weight_block_size=None):
        if loaded_weight.dtype == torch.float8_e4m3fn:
            if scale_inv is None or weight_block_size is None:
                raise ValueError("FP8 权重缺少 scale 或 block size")

            if param.dtype != loaded_weight.dtype:
                param.requires_grad_(False)
                param.data = torch.empty_like(param, dtype=loaded_weight.dtype)
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

        if scale_inv is not None:
            self.weight_block_size = weight_block_size
            out_block, in_block = weight_block_size
            if self.scale_inv is None:
                local_out_dim = self.weight.shape[0]
                in_dim = self.weight.shape[1]
                out_blk = local_out_dim // out_block
                in_blk = in_dim // in_block
                self.scale_inv = torch.empty(out_blk, in_blk, dtype=torch.bfloat16, device=self.weight.device)

            scale_shard = scale_inv.chunk(self.tp_size, 1)[self.tp_rank]
            blk_offset = 0
            blk_size = scale_shard.shape[1]
            scale_dst = self.scale_inv.narrow(1, blk_offset, blk_size)
            scale_dst.copy_(scale_shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight_for_compute(x.dtype), self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
