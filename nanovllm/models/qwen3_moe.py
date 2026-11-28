import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3MoeConfig

from nanovllm.layers.moe import expert_parallel_forward
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3MoeAttetion(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output

class Qwen3MoeMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3MoeExperts(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, *, local: bool = False):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False, local=local,
        )
        self.act_fn = SiluAndMul()
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False, local=local,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        activated = self.act_fn(gate_up)
        return self.down_proj(activated)

class Qwen3MoeTopKRouter(nn.Module):

    def __init__(
        self,
        num_experts_per_tok: int = 0,
        num_experts: int = 0,
        norm_topk_prob: bool = True,
        hidden_size: int = 0
    ) -> None:
        assert num_experts_per_tok > 0 and num_experts > 0
        assert num_experts_per_tok <= num_experts
        super().__init__()
        self.top_k = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.hidden_dim = hidden_size
        self.router = ReplicatedLinear(hidden_size, num_experts)

    def forward(
        self,
        hidden_states # [B*L, H]
    ):
        router_logits = self.router(hidden_states) # [B*L, num_experts]
        router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value # [B*L, num_experts_per_tok]
        return router_logits, router_scores, router_indices


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 128,
        num_experts_per_tok: int = 8,
        norm_topk_prob: bool = True,
        expert_parallel: bool = False,
    ):
        super().__init__()
        self.topkRouter = Qwen3MoeTopKRouter(num_experts_per_tok=num_experts_per_tok, num_experts=num_experts, norm_topk_prob=norm_topk_prob, hidden_size=hidden_size)
        self.expert_parallel = expert_parallel
        if expert_parallel:
            world = dist.get_world_size()
            if num_experts % world:
                raise ValueError("Unequal expert partitions are unsupported")
            self.local_count = num_experts // world
            self.first_expert = dist.get_rank() * self.local_count
            self.experts = nn.ModuleDict({
                str(i): Qwen3MoeExperts(hidden_size, intermediate_size, local=True)
                for i in range(self.first_expert, self.first_expert + self.local_count)
            })
            return
        self.experts = nn.ModuleList([
            Qwen3MoeExperts(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [T, H]，T = B * L
        if self.expert_parallel:
            return expert_parallel_forward(hidden_states, self.topkRouter, self.experts,
                                           self.first_expert, self.local_count)
        _, router_scores, router_indices = self.topkRouter(hidden_states)

        output = torch.zeros_like(hidden_states)

        for expert_id, expert in enumerate(self.experts):
            token_idx, topk_slot = torch.where(router_indices == expert_id)
            if token_idx.numel() == 0:
                continue

            expert_input = hidden_states.index_select(0, token_idx)  # [N, H]
            expert_output = expert(expert_input)                    # [N, H]

            weights = router_scores[token_idx, topk_slot]           # [N]
            weighted_output = expert_output * weights[:, None]     # [N, H]
            output.index_add_(0, token_idx, weighted_output.to(output.dtype))

        return output

class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_idx: int, expert_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3MoeAttetion(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(
                            hidden_size=config.hidden_size,
                            intermediate_size=config.moe_intermediate_size,
                            num_experts=config.num_experts,
                            num_experts_per_tok=config.num_experts_per_tok,
                            expert_parallel=expert_parallel,
                            norm_topk_prob=config.norm_topk_prob
                        )
        else:
            self.mlp = Qwen3MoeMLP(
                            hidden_size=config.hidden_size,
                            intermediate_size=config.intermediate_size,
                            hidden_act=config.hidden_act
                        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_size = config.hidden_size

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

class Qwen3MoeModel(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig, expert_parallel: bool = False,
    ):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3MoeDecoderLayer(config, layer_idx, expert_parallel) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions=positions, hidden_states=hidden_states, residual=residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

class Qwen3MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    weight_suffix_mapping = {
        ".mlp.gate.weight": ".mlp.topkRouter.router.weight",
    }

    def __init__(
        self,
        config: Qwen3MoeConfig, expert_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.model = Qwen3MoeModel(config, expert_parallel)
        self.local_expert_ids = None
        if expert_parallel:
            count = config.num_experts // dist.get_world_size()
            start = dist.get_rank() * count
            self.local_expert_ids = range(start, start + count)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
