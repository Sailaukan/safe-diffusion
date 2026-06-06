import argparse
import gc
import json
import os
import sys
from pathlib import Path
from time import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import torch
from tdc import Evaluator, Oracle

from genmol.sampler import (
    DEFAULT_DE_NOVO_MIN_ADD_LEN,
    DEFAULT_DE_NOVO_RANDOMNESS,
    Sampler,
)


DEFAULT_EVAL_SOFTMAX_TEMP = 0.5


def resolve_project_path(path):
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def log_to_wandb(run_path, metrics):
    import wandb

    api = wandb.Api()
    run = api.run(run_path)
    for key, value in metrics.items():
        run.summary[key] = value
    run.update()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-path")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--softmax-temp", type=float, default=DEFAULT_EVAL_SOFTMAX_TEMP)
    parser.add_argument("--randomness", type=float, default=DEFAULT_DE_NOVO_RANDOMNESS)
    parser.add_argument("--min-add-len", type=int, default=DEFAULT_DE_NOVO_MIN_ADD_LEN)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = Sampler(str(resolve_project_path(args.model_path)))
    all_samples = []
    t_start = time()

    for start in range(0, args.num_samples, args.chunk_size):
        current = min(args.chunk_size, args.num_samples - start)
        samples = sampler.de_novo_generation(
            current,
            softmax_temp=args.softmax_temp,
            randomness=args.randomness,
            min_add_len=args.min_add_len,
        )
        all_samples.extend(samples)
        print(
            f"chunk {start // args.chunk_size + 1}: requested={current} "
            f"valid={len(samples)} total_valid={len(all_samples)}",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    elapsed = time() - t_start
    oracle_qed = Oracle("qed")
    oracle_sa = Oracle("sa")
    evaluator = Evaluator("diversity")

    df = pd.DataFrame(
        {
            "smiles": all_samples,
            "qed": oracle_qed(all_samples) if all_samples else [],
            "sa": oracle_sa(all_samples) if all_samples else [],
        }
    )
    valid_count = len(df["smiles"])
    unique_df = df.drop_duplicates("smiles") if valid_count else df
    unique_count = len(unique_df["smiles"]) if valid_count else 0
    quality_df = unique_df[(unique_df["qed"] >= 0.6) & (unique_df["sa"] <= 4)] if valid_count else unique_df

    metrics = {
        "denovo/time_sec": elapsed,
        "denovo/num_samples": args.num_samples,
        "denovo/chunk_size": args.chunk_size,
        "denovo/valid_count": valid_count,
        "denovo/unique_count": unique_count,
        "denovo/quality_count": len(quality_df),
        "denovo/validity": valid_count / args.num_samples,
        "denovo/uniqueness": unique_count / valid_count if valid_count else 0.0,
        "denovo/diversity": evaluator(unique_df["smiles"]) if unique_count > 1 else 0.0,
        "denovo/quality": len(quality_df) / args.num_samples,
        "denovo/softmax_temp": args.softmax_temp,
        "denovo/randomness": args.randomness,
        "denovo/min_add_len": args.min_add_len,
    }

    samples_path = output_dir / "samples.csv"
    metrics_path = output_dir / "metrics.json"
    df.to_csv(samples_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"wrote {samples_path}", flush=True)
    print(f"wrote {metrics_path}", flush=True)

    if args.run_path:
        log_to_wandb(args.run_path, metrics)
        print(f"logged metrics to {args.run_path}", flush=True)


if __name__ == "__main__":
    main()
