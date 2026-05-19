"""
Triton fused MoE - expert-parallel launch (packed x_sorted, CSR expert_start).

Weight gather eliminated: kernels receive full [E_total, ...] weight tensors
and an expert_ids mapping; each CTA loads actual_eid = expert_ids[pid_e] to
index into the correct expert slice, avoiding the costly
`gate_up_weight[e_ids]` / `down_weight[e_ids]` advanced-indexing copies.

Fixed BLOCK sizes: BLOCK_M=64, BLOCK_N=128, BLOCK_K=64. No autotune -
avoids per-key re-benchmarking that causes multi-minute stalls in real
inference.
"""

from __future__ import annotations
import torch
import triton
import triton.language as tl

BLOCK_M: int = 64
BLOCK_N: int = 128
BLOCK_K: int = 64


@triton.jit
def _gate_up_kernel(
    X_ptr,
    W_ptr,
    Inter_ptr,
    Expert_start_ptr,
    Expert_ids_ptr,
    H: tl.constexpr,
    I: tl.constexpr,
    stride_xm,
    stride_xh,
    stride_we,
    stride_wn,
    stride_wh,
    stride_om,
    stride_oi,
    batch_m: tl.int32,
    max_t: tl.int32,
    num_active: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    actual_eid = tl.load(Expert_ids_ptr + pid_e)

    e_start = tl.load(Expert_start_ptr + pid_e)
    e_end = tl.load(Expert_start_ptr + pid_e + 1)
    num_tokens_e = e_end - e_start

    m_start = e_start + pid_m * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    m_mask = (m_offs - e_start) < num_tokens_e
    m_mask = m_mask & (m_offs < batch_m)
    m_mask = m_mask & (pid_e < num_active)
    m_mask = m_mask & (max_t > 0)

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < I

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_base = actual_eid * stride_we

    for k in range(0, tl.cdiv(H, BLOCK_K)):
        k_offs = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offs < H

        x_mask = m_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(
            X_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xh,
            mask=x_mask,
            other=0.0,
        )

        wg_mask = n_mask[:, None] & k_mask[None, :]
        wg_tile = tl.load(
            W_ptr + w_base + n_offs[:, None] * stride_wn + k_offs[None, :] * stride_wh,
            mask=wg_mask,
            other=0.0,
        )
        wu_tile = tl.load(
            W_ptr + w_base + (I + n_offs[:, None]) * stride_wn + k_offs[None, :] * stride_wh,
            mask=wg_mask,
            other=0.0,
        )
        gate_acc = tl.dot(x_tile, tl.trans(wg_tile), acc=gate_acc)
        up_acc = tl.dot(x_tile, tl.trans(wu_tile), acc=up_acc)

    gate_f = gate_acc.to(tl.float32)
    inter = (gate_f * tl.sigmoid(gate_f)) * up_acc

    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(
        Inter_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_oi,
        inter.to(Inter_ptr.dtype.element_ty),
        mask=out_mask,
    )


@triton.jit
def _down_kernel(
    Inter_ptr,
    W_ptr,
    Out_ptr,
    Expert_start_ptr,
    Expert_ids_ptr,
    H: tl.constexpr,
    I: tl.constexpr,
    stride_im,
    stride_ii,
    stride_we,
    stride_wh,
    stride_wi,
    stride_om,
    stride_oh,
    batch_m: tl.int32,
    max_t: tl.int32,
    num_active: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    actual_eid = tl.load(Expert_ids_ptr + pid_e)

    e_start = tl.load(Expert_start_ptr + pid_e)
    e_end = tl.load(Expert_start_ptr + pid_e + 1)
    num_tokens_e = e_end - e_start

    m_start = e_start + pid_m * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    m_mask = (m_offs - e_start) < num_tokens_e
    m_mask = m_mask & (m_offs < batch_m)
    m_mask = m_mask & (pid_e < num_active)
    m_mask = m_mask & (max_t > 0)

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_base = actual_eid * stride_we

    for k in range(0, tl.cdiv(I, BLOCK_K)):
        k_offs = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offs < I

        inter_mask = m_mask[:, None] & k_mask[None, :]
        inter_tile = tl.load(
            Inter_ptr + m_offs[:, None] * stride_im + k_offs[None, :] * stride_ii,
            mask=inter_mask,
            other=0.0,
        )

        w_mask = k_mask[:, None] & n_mask[None, :]
        w_tile = tl.load(
            W_ptr + w_base + k_offs[:, None] * stride_wi + n_offs[None, :] * stride_wh,
            mask=w_mask,
            other=0.0,
        )
        acc = tl.dot(inter_tile, w_tile, acc=acc)

    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(
        Out_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_oh,
        acc.to(Out_ptr.dtype.element_ty),
        mask=out_mask,
    )


def fused_moe_triton(
    x_sorted: torch.Tensor,
    expert_start: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    M, H = x_sorted.shape
    num_active = expert_start.shape[0] - 1
    _, two_I, _ = gate_up_weight.shape
    I = two_I // 2

    max_t = int((expert_start[1:] - expert_start[:-1]).max().item())
    batch_m = int(M)
    num_active_i = int(num_active)

    inter = torch.empty((M, I), device=x_sorted.device, dtype=x_sorted.dtype)
    out_sorted = torch.empty((M, H), device=x_sorted.device, dtype=x_sorted.dtype)

    grid_mu = (num_active, triton.cdiv(max_t, BLOCK_M), triton.cdiv(I, BLOCK_N))
    grid_md = (num_active, triton.cdiv(max_t, BLOCK_M), triton.cdiv(H, BLOCK_N))

    _gate_up_kernel[grid_mu](
        x_sorted,
        gate_up_weight,
        inter,
        expert_start,
        expert_ids,
        H,
        I,
        x_sorted.stride(0),
        x_sorted.stride(1),
        gate_up_weight.stride(0),
        gate_up_weight.stride(1),
        gate_up_weight.stride(2),
        inter.stride(0),
        inter.stride(1),
        batch_m,
        max_t,
        num_active_i,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_stages=3,
        num_warps=8,
    )

    _down_kernel[grid_md](
        inter,
        down_weight,
        out_sorted,
        expert_start,
        expert_ids,
        H,
        I,
        inter.stride(0),
        inter.stride(1),
        down_weight.stride(0),
        down_weight.stride(1),
        down_weight.stride(2),
        out_sorted.stride(0),
        out_sorted.stride(1),
        batch_m,
        max_t,
        num_active_i,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_stages=3,
        num_warps=8,
    )

    return out_sorted
