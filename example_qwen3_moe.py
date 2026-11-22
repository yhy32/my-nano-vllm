"""Compare eager GPU prefill and cached decode against Transformers."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context


def save_checkpoint(reference: Qwen3MoeForCausalLM, directory: Path) -> None:
    # Match the per-expert weight layout in the official Qwen3-30B-A3B checkpoint.
    weights = {}
    for name, tensor in reference.state_dict().items():
        if name.endswith(".mlp.experts.gate_up_proj"):
            prefix = name.removesuffix(".gate_up_proj")
            for expert_id, expert_weights in enumerate(tensor):
                gate, up = expert_weights.chunk(2, dim=0)
                weights[f"{prefix}.{expert_id}.gate_proj.weight"] = gate.clone()
                weights[f"{prefix}.{expert_id}.up_proj.weight"] = up.clone()
        elif name.endswith(".mlp.experts.down_proj"):
            prefix = name.removesuffix(".down_proj")
            for expert_id, expert_weights in enumerate(tensor):
                weights[f"{prefix}.{expert_id}.down_proj.weight"] = expert_weights.clone()
        else:
            weights[name] = tensor.clone()
    reference.config.save_pretrained(directory)
    save_file(weights, directory / "model.safetensors")


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(42)
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=256,
        attention_bias=False,
        tie_word_embeddings=False,
    )
    reference = Qwen3MoeForCausalLM(config).eval().to(torch.bfloat16)
    config.dtype = torch.bfloat16
    tokens = [1, 17, 42, 9]

    with TemporaryDirectory(prefix="nano-qwen3-moe-") as directory:
        save_checkpoint(reference, Path(directory))
        expected = reference.float()(input_ids=torch.tensor([tokens]), use_cache=False).logits[0, -1]
        runner = ModelRunner(
            Config(directory, enforce_eager=True, max_model_len=16, max_num_batched_tokens=16,
                   max_num_seqs=1, gpu_memory_utilization=0.4),
            rank=0,
            event=[],
        )
        try:
            router = runner.model.model.layers[0].mlp.topkRouter.router.weight
            expected_router = reference.state_dict()["model.layers.0.mlp.gate.weight"]
            torch.testing.assert_close(router.cpu().float(), expected_router, atol=0, rtol=0)

            seq = Sequence(tokens)
            seq.block_table = [0]
            seq.num_scheduled_tokens = len(tokens)
            input_ids, positions = runner.prepare_prefill([seq])
            logits = runner.run_model(input_ids, positions, is_prefill=True)
            assert logits.is_cuda and logits.shape == (1, config.vocab_size)
            actual = logits[0].float().cpu()
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
            print(f"Router weights match exactly. Eager GPU logits shape: {tuple(logits.shape)}")
            print(f"Max absolute logit difference vs Transformers: {(actual - expected).abs().max().item():.6f}")

            for step in range(1, 6):
                # Both implementations see the same token chosen by the GPU runner.
                token = actual.argmax().item()
                seq.append_token(token)
                input_ids, positions = runner.prepare_decode([seq])
                logits = runner.run_model(input_ids, positions, is_prefill=False)
                assert logits.is_cuda and logits.shape == (1, config.vocab_size)
                actual = logits[0].float().cpu()

                # Recompute the full prefix as a reference for the cached result.
                expected = reference(input_ids=torch.tensor([seq.token_ids]), use_cache=False).logits[0, -1]
                torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
                max_error = (actual - expected).abs().max().item()
                print(f"Decode step {step}: token={token}, max absolute logit difference={max_error:.6f}")
        finally:
            reset_context()
            runner.exit()


if __name__ == "__main__":
    main()
