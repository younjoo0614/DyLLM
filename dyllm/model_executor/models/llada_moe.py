import torch
from torch import nn

from dyllm.configs.llada_moe import LLaDAMoEConfig
from dyllm.utils.context import get_context
from dyllm.utils.metadata import get_metadata
from dyllm.model_executor.layers.layernorm import RMSNorm
from dyllm.model_executor.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from dyllm.model_executor.layers.moe import LLaDAMoESparseMoeBlock
from dyllm.model_executor.models.llada import (
    LLaDAAttention,
    LLaDADecoderLayer,
    LLaDAMLP,
    LLaDAModel,
    LLaDAForDLM,
)


class LLaDAMoEAttention(LLaDAAttention):
    def __init__(self, config: LLaDAMoEConfig, threshold: float) -> None:
        super().__init__(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            threshold=threshold,
        )
        self.qk_layernorm = bool(getattr(config, "qk_layernorm", False))
        if self.qk_layernorm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        metadata = get_metadata()

        q = self.q_proj(hidden_states)
        if ctx.is_full:
            kv = self.kv_proj(hidden_states)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        else:
            kv = self.kv_proj(hidden_states[ctx.idx_salient_row])
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)

        # qk_layernorm is applied immediately after projection and before RoPE.
        if self.qk_layernorm:
            q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(q.shape)
            k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(k.shape)

        if not ctx.is_full:
            if ctx.idx_salient_row_k is not None:
                k_temp = torch.zeros(
                    ctx.total_seqlen, self.num_kv_heads * self.head_dim, dtype=k.dtype, device=k.device
                )
                k_temp[ctx.idx_salient_row] = k
            else:
                k_temp = torch.zeros(
                    ctx.total_seqlen_k, self.num_kv_heads * self.head_dim, dtype=k.dtype, device=k.device
                )
                k_temp[ctx.idx_salient_row] = k
            k = k_temp

        def split_last(x, h, d):
            *prefix, _ = x.shape
            return x.view(*prefix, h, d)

        q = split_last(q, self.num_heads, self.head_dim)
        k = split_last(k, self.num_heads, self.head_dim)
        v = split_last(v, self.num_heads, self.head_dim)

        q, k = self.rotary_emb(positions, q, k)

        if ctx.is_full:
            self.k_cache.reset_full(k.flatten(-2, -1), metadata.running_seqs_tensor, seq_ids_list=metadata.running_seqs)
            o = self.attn(q, k, v)
        else:
            if ctx.idx_salient_row_k is not None:
                self.k_cache.scatter_update(
                    metadata.running_seqs_tensor, ctx.idx_salient_row_k, k[ctx.idx_salient_row].flatten(-2, -1)
                )
            else:
                self.k_cache.scatter_update(
                    metadata.running_seqs_tensor, ctx.idx_salient_row, k[ctx.idx_salient_row].flatten(-2, -1)
                )
            o = self.attn(
                q, self.k_cache.get_seqs(metadata.running_seqs_tensor).view(-1, self.num_kv_heads, self.head_dim), v
            )
        output = self.o_proj(o.flatten(-2, -1))
        self.k_cache.finish(metadata.finished_seqs)
        return output


class LLaDAMoEDecoderLayer(LLaDADecoderLayer):
    def __init__(self, config: LLaDAMoEConfig, layer_idx: int, threshold: float) -> None:
        nn.Module.__init__(self)
        self.self_attn = LLaDAMoEAttention(config, threshold)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.moe_layer_freq[layer_idx] == 0:
            self.mlp = LLaDAMLP(hidden_size=config.hidden_size, intermediate_size=config.dense_intermediate_size)
        else:
            self.mlp = LLaDAMoESparseMoeBlock(config)


class LLaDAMoEModel(LLaDAModel):
    def __init__(self, config: LLaDAMoEConfig, threshold: float):
        # Avoid LLaDAModel.__init__ since it instantiates LLaDADecoderLayer
        # that expects legacy config fields (n_kv_heads/max_sequence_length/mlp_hidden_size).
        nn.Module.__init__(self)
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LLaDAMoEDecoderLayer(config, layer_idx=i, threshold=threshold) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class LLaDAMoEForDLM(LLaDAForDLM):
    packed_modules_mapping = {
        "kv_proj": ["k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, config: LLaDAMoEConfig, threshold: float):
        nn.Module.__init__(self)
        self.config = config
        self.model = LLaDAMoEModel(config, threshold)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def normalize_weight_name(self, name: str) -> str:
        return name
