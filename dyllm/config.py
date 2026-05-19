import os
from dataclasses import dataclass
from transformers import AutoConfig

from dyllm.configs.llada_moe import LLaDAMoEConfig  # noqa: F401 (registers AutoConfig side effect)


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 65536
    max_num_seqs: int = 16
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    mask_id: int = -1
    num_full_steps: int = 8
    threshold: float = 0.99
    dist_port: int | None = None
    
    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        if getattr(self.hf_config, "model_type", None) == "llada_moe":
            # LLaDA-MoE ships mask as a special token (<|mask|>) rather than via
            # `mask_token_id`; resolve via the tokenizer.
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
            self.mask_id = tokenizer.convert_tokens_to_ids("<|mask|>")
        else:
            self.mask_id = self.hf_config.mask_token_id
        if hasattr(self.hf_config, "max_position_embeddings"):
            max_context_length = self.hf_config.max_position_embeddings
        elif hasattr(self.hf_config, "max_sequence_length"):
            max_context_length = self.hf_config.max_sequence_length
        self.max_model_len = min(self.max_model_len, max_context_length)
        assert self.max_num_batched_tokens >= self.max_model_len
