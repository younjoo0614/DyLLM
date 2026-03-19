import argparse
import json
import torch
import os
from lm_eval.evaluator import simple_evaluate
from dyllm.eval.adapter import DyLLMAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--tasks", type=str, default="gsm8k")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--num-shot", type=int, default=5)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--ignore-eos", action="store_true", default=False)
    ap.add_argument("--num-steps", type=int, default=256)
    ap.add_argument("--num-full-steps", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument("--output-file", type=str, default=None)
    ap.add_argument("--log-samples", action="store_true", default=False)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model_args_dict = {
        "model_path": args.model_path,
        "max_new_toks": args.max_new_tokens,
        "tensor_parallel_size": args.tp_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "ignore_eos": args.ignore_eos,
        "trust_remote_code": args.trust_remote_code,
        "num_steps": args.num_steps,
        "num_full_steps": args.num_full_steps,
        "block_size": args.block_size,
        "threshold": args.threshold,
    }
    model_args = ",".join([f"{k}={v}" for k, v in model_args_dict.items()])

    with torch.inference_mode():
        results = simple_evaluate(
            model="dyllm",
            model_args=model_args,
            tasks=args.tasks.split(","),
            num_fewshot=args.num_shot,
            batch_size=args.batch_size,
            device="cuda",
            limit=args.limit,
            log_samples=args.log_samples,
            confirm_run_unsafe_code=True,
            verbosity="INFO",
        )

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output_file}")
    else:
        print(json.dumps(results.get("results", results), indent=2, default=str))


if __name__ == "__main__":
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    main()
