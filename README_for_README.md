# README reference for this PR (LLaDA-MoE + Triton fused MoE)

> Temporary file for the upstream maintainer to use as reference when
> updating `README.md` for this PR. Safe to delete once `README.md` is
> updated; the PR does not depend on this file.

## 1. What this PR adds

### New files

| Path | Lines | Role |
| --- | --- | --- |
| `dyllm/configs/llada_moe.py` | 104 | `LLaDAMoEConfig` (`PretrainedConfig`) + `AutoConfig.register("llada_moe", ...)` |
| `dyllm/model_executor/models/llada_moe.py` | 136 | `LLaDAMoEAttention` (QK-LayerNorm), `LLaDAMoEDecoderLayer`, `LLaDAMoEModel`, `LLaDAMoEForDLM` — all inherit from the existing LLaDA classes |
| `dyllm/model_executor/layers/moe.py` | 140 | `LLaDAMoESparseMoeBlock` — top-k routing + Triton fused matmul + MLPcache integration |
| `dyllm/model_executor/layers/fused_moe_triton.py` | 252 | Two Triton kernels (`_gate_up_kernel`, `_down_kernel`) that fuse expert dispatch + matmul. Fixed `BLOCK_M=64, BLOCK_N=128, BLOCK_K=64`, no autotune |
| `scripts/run_llada_moe.sh` | 30 | Example invocation (`CUDA_VISIBLE_DEVICES=0`, gsm8k, `num_full_steps=4`, `refresh_interval=64`) |

### Modified files

| Path | Lines | Change |
| --- | --- | --- |
| `dyllm/config.py` | +12 / -1 | Side-effect import of `LLaDAMoEConfig` (registers with `AutoConfig`); when `hf_config.model_type == "llada_moe"`, resolve mask id via tokenizer (`<|mask|>` special token) instead of `hf_config.mask_token_id` |
| `dyllm/configs/__init__.py` | +1 | Re-export `LLaDAMoEConfig` |
| `dyllm/model_executor/models/__init__.py` | +1 | Re-export `LLaDAMoEForDLM` |
| `dyllm/engine/model_runner.py` | +5 | Dispatch `LLaDAMoEForDLM(...)` when `model_type == "llada_moe"`; same `LLaDASampler("confidence")` as the dense path |
| `dyllm/sampling_params.py` | +1 | New field `refresh_interval: int = 0` (defaults to friend-master behavior) |
| `dyllm/engine/sequence.py` | +2 | Plumb `sampling_params.refresh_interval` onto the `Sequence` |
| `dyllm/engine/scheduler.py` | +8 / -1 | On the sparse-path scheduling branch, return `is_full=True` whenever any sequence has `refresh_interval > 0` and `(processed_steps − num_full_steps) % refresh_interval == 0`. With `refresh_interval=0` (the default) the path is identical to master |
| `dyllm/eval/adapter.py` | +35 | Add `refresh_interval` param to the adapter; markdown extraction for instruct-model code tasks; **add `<\|role_end\|>` to `all_stops`** (see §5) |
| `dyllm/eval/eval.py` | +7 | New CLI flag `--refresh-interval`, piped through to the adapter |

Total: **9 modifications + 5 new files**, +662 lines of new code, +77 / -8 in modifications.

## 2. Default config behavior

All new parameters are additive and default to friend-master behavior:

| Parameter | Default | Effect when default |
| --- | --- | --- |
| `SamplingParams.refresh_interval` | `0` | Sparse path runs identically to current master (no forced refresh) |
| `DyLLMAdapter(refresh_interval=)` | `0` | Same |
| `eval.py --refresh-interval` | `0` | Same |
| `dyllm/eval/adapter.py` stop list | adds `<\|role_end\|>` | New tokenizer stop — see §5; benign for LLaDA-8B / Dream (token not in their vocab effective output) |

LLaDA / Dream paths are byte-identical to master apart from the `<|role_end|>` stop addition (which is a strict superset of the existing list).

## 3. CLI usage

```bash
# LLaDA-MoE sparse_r64 setup that produced the headline numbers
bash scripts/run_llada_moe.sh

# Or directly:
CUDA_VISIBLE_DEVICES=0 python -m dyllm.eval.eval \
  --model-path /path/to/LLaDA-MoE-7B-A1B-Instruct \
  --tasks gsm8k --num-shot 5 \
  --batch-size 16 --max-new-tokens 256 \
  --num-steps 256 \
  --num-full-steps 4 \
  --refresh-interval 64 \
  --block-size 32 \
  --threshold 0.99
```

## 4. Measurement results (gsm8k 5-shot, `limit=200`, `batch_size=16`)

| # | Model | DyLLM config | gsm8k flex | wall (s) | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | LLaDA-MoE-7B-A1B-Instruct | dense (`num_full_steps=256`) | 75.0% | 2436 | Reference baseline |
| 2 | LLaDA-MoE-7B-A1B-Instruct | sparse_r0 (`num_full_steps=4, refresh=0`) | 68.5% | 446 | No refresh — fastest, lowest acc |
| 3 | LLaDA-MoE-7B-A1B-Instruct | sparse_r64 (`num_full_steps=4, refresh=64`) | **76.0%** | **482** | **Headline config** (after `<\|role_end\|>` fix) |
| 4 | LLaDA-8B-Instruct (dense arch) | sparse_r0 | 77.0% | 939 | Cross-model reference |

**Speedup (cross-model)**: LLaDA-MoE sparse_r64 (482 s) vs LLaDA-8B sparse_r0 (939 s) = **1.95×**.

Hardware: 1× A100 80GB PCIe (GPU 1 of a 2× node).

## 5. Adding `<|role_end|>` to the eval stop list

LLaDA-MoE introduces a per-turn terminator (`<|role_end|>`) in its
chat template that earlier supported models (LLaDA-8B, Dream) do not
use (see `tokenizer_config.json:chat_template`). Extending the
adapter's `all_stops` to include this token is needed so that
post-response tokens don't leak into the answer extracted by
`lm_eval`'s gsm8k flex matcher.

Measured impact on the headline config:

| sparse_r64 (LLaDA-MoE) | gsm8k flex | Δ |
| --- | --- | --- |
| Without `<\|role_end\|>` in stops | 72.5% | — |
| With `<\|role_end\|>` added to `all_stops` | **76.0%** | **+3.5pp** |

The change is at `dyllm/eval/adapter.py:210` and is a strict superset
of the existing list — LLaDA-8B / Dream paths are unaffected (their
generations don't emit `<|role_end|>`; cross-model LLaDA-8B run
(#4 above) reproduces the expected ~77% number).

## 6. Triton fused MoE (kernel details)

- File: `dyllm/model_executor/layers/fused_moe_triton.py`
- Two kernels: `_gate_up_kernel` (fuses gate + up projections in a
  single CTA, then SiLU + multiply), `_down_kernel` (down projection
  back to hidden size).
- Expert-parallel launch: each program block handles one `(expert, M-tile, N-tile)`. Expert weights are passed as full `[E_total, ...]` tensors
  plus an `expert_ids` mapping, so no `gate_up_weight[e_ids]` advanced
  indexing copies are needed.
- Fixed `BLOCK_M=64, BLOCK_N=128, BLOCK_K=64`. No `@triton.autotune` —
  Triton's autotune triggers minute-scale re-benchmarking on each new
  shape, which torpedoes wall-clock in real inference. The fixed sizes
  are tuned for the 64-expert / top-8 / hidden=2048 / intermediate=1024
  LLaDA-MoE regime at A100 PCIe; runs reasonably across the batch-size
  range we tested.
- nsys profile of the headline config (sparse_r64, `limit=16`):
  - `_gate_up_kernel`: 30.2% of GPU kernel time (avg 2.35 ms / launch, 4096 launches)
  - `_down_kernel`: 12.5% (avg 0.97 ms / launch)
  - MoE total: ~43% of kernel time, attention/QKV: ~25%, sparse attention: ~7%
  - Device-to-Host memcpy total: 2.0 MB (scheduler token-count reads only — no problematic CPU↔GPU sync inside the fused path)
  - GPU utilization: ~67% of wall time (consistent with diffusion-style multi-step inference)

## 7. Environment

| Component | Version |
| --- | --- |
| GPU | NVIDIA A100 80GB PCIe |
| CUDA toolkit | 13.0 (driver 13.1) |
| Python | 3.10.20 |
| PyTorch | 2.9.1+cu130 |
| Triton | 3.5.1 |
| transformers | 4.57.6 |

CUDA extensions (`attention_ops`, `cache`, `custom_ops`) build from the
existing `dyllm/csrc/` via `pip install -e .` — no changes to the build.

## 8. Backward compatibility verification

- LLaDA-8B-Instruct end-to-end gsm8k run (entry #4 above): 77.0% flex vs
  historical 78.5% → within ±2pp, on a single 200-example sample
  (per-sample stderr ≈3pp).
- Dream / LLaDA branches in `model_runner.py` and the existing engine /
  scheduler paths are untouched apart from the additive
  `refresh_interval` check (default 0 = no-op).
- No changes to `csrc/`, no new build steps.

---

*This file is for the maintainer's convenience. Delete after updating
`README.md`.*
