# my-nano-vllm

## Qwen3-MoE support

The model runner selects Qwen3 or Qwen3-MoE from the checkpoint's `model_type`.
The MoE layer routes tokens to their top-k experts and combines the expert outputs
using the router scores. The loader maps the MoE router weight to the local
parameter name and combines each expert's gate and up projections when loading
per-expert checkpoints.

Run the single-GPU eager smoke test from the repository root:

```bash
python example_qwen3_moe.py
```

The example creates a temporary random checkpoint with two layers, four experts
and two experts per token. It checks the router weight after loading, runs prefill
and five cached decode steps, and compares logits at every step against
Transformers recomputing the full token prefix in FP32.

A single-GPU run with BF16 weights passed, with a maximum absolute decode logit
difference of 0.004136. This test covers one request within one KV cache block.
The full Qwen3-30B-A3B checkpoint, multi-GPU execution and CUDA Graph decode
remain unverified.
