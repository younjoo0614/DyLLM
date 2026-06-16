"""LLaDA-MoE model configuration."""

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class LLaDAMoEConfig(PretrainedConfig):
    model_type = "llada_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=157184,
        hidden_size=2048,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_act="silu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=False,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        attention_bias=False,
        clip_qkv=None,
        dense_intermediate_size=8192,
        expert_intermediate_size=1024,
        num_experts=64,
        num_experts_per_tok=8,
        shared_expert_intermediate_size=None,
        moe_layer_freq=None,
        qk_layernorm=True,
        moe_router_score_function="softmax",
        moe_router_enable_expert_bias=False,
        output_router_logits=False,
        router_aux_loss_coef=0.01,
        router_num_group=None,
        router_topk_group=None,
        routed_scaling_factor=1,
        norm_topk_prob=None,
        rope_theta=50000.0,
        rope_scaling=None,
        partial_rotary_factor=1.0,
        eos_token_id=156892,
        pad_token_id=156892,
        mask_token_id=156892,
        architectures=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.clip_qkv = clip_qkv

        self.dense_intermediate_size = dense_intermediate_size
        self.expert_intermediate_size = expert_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.moe_layer_freq = moe_layer_freq or [1] * num_hidden_layers

        self.qk_layernorm = qk_layernorm
        self.moe_router_score_function = moe_router_score_function
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_num_group = router_num_group
        self.router_topk_group = router_topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob

        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        if architectures is None:
            architectures = ["LLaDAMoEModel"]

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            architectures=architectures,
            **kwargs,
        )
        self.mask_token_id = mask_token_id


AutoConfig.register("llada_moe", LLaDAMoEConfig)
