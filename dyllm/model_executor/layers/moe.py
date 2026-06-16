"""LLaDA-MoE sparse expert block (packed `[num_tokens, hidden_size]` layout)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from dyllm.configs.llada_moe import LLaDAMoEConfig
from dyllm.model_executor.layers.fused_moe_triton import fused_moe_triton
from dyllm.model_executor.layers.mlp_cache_manage import MLPcache

__all__ = [
    "LLaDAExpertMLP",
    "LLaDAMoESparseMoeBlock",
]


class LLaDAExpertMLP(nn.Module):
    """One expert: `down( act(gate(x)) * up(x) )`, matching HF `LLaDAMoEMLP(..., mlp_type='expert')`."""

    def __init__(self, config: LLaDAMoEConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.expert_intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LLaDAMoESparseMoeBlock(nn.Module):
    """
    MoE FFN for DyLLM packed tensors `[num_tokens, hidden_size]`.

    - Router: softmax over experts (fixed; no `score_func` branching).
    - `expert_bias` and `shared_expert` are not implemented.

    After the mixture output, applies `MLPcache` like `LLaDAMLP`. Routing uses
    full vs sparse expert lists; compute uses stacked `gate_up`/`down` weights
    fused via Triton (`fused_moe_triton`).
    """

    # When True, route the gate matmul in fp32 (numerically stable top-k selection).
    # Toggle as a class attribute before model construction.
    use_fp32_gate: bool = False

    def __init__(self, config: LLaDAMoEConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", False))
        self.act_fn = ACT2FN[config.hidden_act]

        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([LLaDAExpertMLP(config) for _ in range(self.num_experts)])
        self.cache_update = MLPcache(config.hidden_size)

        self.gate_up_weight: torch.Tensor | None = None
        self.down_proj_weight: torch.Tensor | None = None

    def _stack_weights(self) -> None:
        """Stack per-expert `Linear` weights once for batched `bmm` (lazy). `gate_up`: `[E, 2I, H]`, `down`: `[E, H, I]`."""
        if self.gate_up_weight is not None:
            return
        gate_stacked = torch.stack([e.gate_proj.weight for e in self.experts], dim=0)
        up_stacked = torch.stack([e.up_proj.weight for e in self.experts], dim=0)
        self.gate_up_weight = torch.cat([gate_stacked, up_stacked], dim=1)
        self.down_proj_weight = torch.stack([e.down_proj.weight for e in self.experts], dim=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() != 2:
            raise ValueError(
                f"Expected packed hidden_states [num_tokens, hidden_size], got shape {tuple(hidden_states.shape)}"
            )

        out = self._route_and_mix(hidden_states)
        return self.cache_update(out)

    def _route_and_mix(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MoE forward without MLPcache; input/output `[num_tokens, H]`."""
        self._stack_weights()
        assert self.gate_up_weight is not None
        assert self.down_proj_weight is not None

        num_tokens, hidden_dim = hidden_states.shape
        if num_tokens == 0:
            return hidden_states

        device = hidden_states.device

        if self.use_fp32_gate:
            if self.gate.weight.dtype != torch.float32:
                self.gate.weight.data = self.gate.weight.data.float()
            router_logits = F.linear(hidden_states.float(), self.gate.weight)
        else:
            router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        token_idx = (
            torch.arange(num_tokens, device=device)
            .unsqueeze(1)
            .expand_as(selected_experts)
            .reshape(-1)
        )
        expert_flat = selected_experts.reshape(-1)
        sort_idx = expert_flat.argsort(stable=True)

        token_idx_sorted = token_idx[sort_idx]
        expert_ids_sorted = expert_flat[sort_idx]
        x_sorted = hidden_states[token_idx_sorted]

        e_ids, counts = torch.unique_consecutive(expert_ids_sorted, return_counts=True)
        if e_ids.numel() == 0:
            return hidden_states.new_zeros((num_tokens, hidden_dim))

        expert_start = torch.zeros(e_ids.shape[0] + 1, device=device, dtype=torch.long)
        expert_start[1:] = counts.cumsum(0)
        expert_ids = e_ids.to(torch.int32)

        out_sorted = fused_moe_triton(
            x_sorted, expert_start, expert_ids,
            self.gate_up_weight, self.down_proj_weight,
        )

        unsort_idx = torch.empty_like(sort_idx)
        unsort_idx[sort_idx] = torch.arange(len(sort_idx), device=device)

        out_token = out_sorted[unsort_idx].view(num_tokens, self.top_k, hidden_dim)
        return (out_token * routing_weights.unsqueeze(-1)).sum(dim=1).to(hidden_states.dtype)
