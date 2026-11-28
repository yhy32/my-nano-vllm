import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    enable_expert_parallel: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.enable_expert_parallel:
            if self.hf_config.model_type != "qwen3_moe":
                raise ValueError("Expert parallelism requires qwen3_moe")
            if self.hf_config.num_experts % self.tensor_parallel_size:
                raise ValueError("num_experts must be divisible by tensor_parallel_size")
            self.enforce_eager = True  # Route counts vary on each step.
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
