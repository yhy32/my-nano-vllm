"""Run with torchrun --standalone --nproc-per-node=2 test_moe_ep.py [--cuda]."""
import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig

from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM
from nanovllm.utils.loader import load_model


@torch.inference_mode()
def check(device, fp8):
    # Small block-aligned expert matrices also exercise separate FP8 scale files.
    h = 128
    cfg = Qwen3MoeConfig(hidden_size=h, moe_intermediate_size=h, intermediate_size=h,
                         num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=32, num_experts=4, num_experts_per_tok=2, norm_topk_prob=True, vocab_size=32,
                         decoder_sparse_step=1, mlp_only_layers=[])
    model = Qwen3MoeForCausalLM(cfg, expert_parallel=True).to(device)
    mlp = model.model.layers[0].mlp
    count = 4 // dist.get_world_size()
    start = dist.get_rank() * count
    assert list(mlp.experts) == [str(i) for i in range(start, start + count)]
    prefix = "model.layers.0.mlp."
    gate = torch.eye(4, h)
    weights = {prefix + "gate.weight": gate}
    scales = {}
    reference = []
    gen = torch.Generator().manual_seed(31)
    for expert in range(4):
        matrices = []
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"{prefix}experts.{expert}.{proj}.weight"
            w = torch.randn(h, h, generator=gen) * 0.1
            if fp8:
                scale = torch.tensor([[0.25 * (1 + expert)]])
                w = (w / scale).to(torch.float8_e4m3fn)
                scales[key + "_scale_inv"] = scale
                matrices.append((w.float() * scale).to(device))
            else:
                matrices.append(w.to(device))
            weights[key] = w
        reference.append(matrices)

    with TemporaryDirectory() as path:
        save_file(weights, str(Path(path) / "weights.safetensors"))
        if scales:
            save_file(scales, str(Path(path) / "scales.safetensors"))
        load_model(model, path, fp8Config=SimpleNamespace(is_FP8=fp8, weight_block_size=(128, 128)))

    # Force cross-rank routes, then a decode that sends everything to rank 1.
    for routes in ([(0, 2), (0, 1), (2, 3), (2, 3)], [(2, 3)], [(0, 1)] * 5):
        x = torch.randn(len(routes), h, generator=gen).to(device)
        x[:, :4] = 0
        for row, (first, second) in enumerate(routes):
            x[row, first], x[row, second] = 3, 2
        y = mlp(x)
        scores = torch.tensor([3., 2.], device=device).softmax(0)
        expected = torch.zeros_like(x)
        for row, pair in enumerate(routes):
            for slot, expert in enumerate(pair):
                g, u, d = reference[expert]
                inter = F.silu(F.linear(x[row], g)) * F.linear(x[row], u)
                expected[row] += scores[slot] * F.linear(inter, d)
        torch.testing.assert_close(y, expected, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    cuda = parser.parse_args().cuda
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if cuda else "gloo")
    try:
        for fp8 in (False, True):
            check(device, fp8)
        if dist.get_rank() == 0:
            print("PASS: expert placement, checkpoint loading, dispatch and combine (float32 / FP8)")
    finally:
        dist.destroy_process_group()
