#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch


def _insert_import_paths(model_dir: Path, metrics_dir: Path) -> None:
    sys.path.insert(0, str(metrics_dir))
    sys.path.insert(0, str(model_dir / "src"))


def _model_class(model_name: str):
    if model_name == "safe-ar":
        from genmol.model import SafeAR

        return SafeAR
    if model_name == "safe-mdlm":
        from genmol.model import SafeMDLM

        return SafeMDLM
    if model_name == "safe-udlm":
        from genmol.model import SafeUDLM

        return SafeUDLM
    raise ValueError(f"Unsupported genmol model: {model_name}")


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_model(model_name: str, checkpoint: Path, device: torch.device):
    model_cls = _model_class(model_name)
    model = model_cls.load_from_checkpoint(
        str(checkpoint),
        map_location=device,
        strict=False,
    )
    model.to(device)
    model.eval()
    if hasattr(model, "backbone"):
        model.backbone.eval()
    if getattr(model, "ema", None):
        model.ema.store(model.backbone.parameters())
        model.ema.copy_to(model.backbone.parameters())
    if hasattr(model, "diffusion"):
        model.diffusion.to_device(device)
    return model


def _sample_legacy_repaired_smiles(checkpoint: Path, args) -> list[str]:
    from genmol.sampler import Sampler

    sampler = Sampler(str(checkpoint))
    if hasattr(sampler.model, "diffusion"):
        sampler.model.diffusion.sampling_steps = int(args.steps)
    smiles = []
    remaining = int(args.num_samples)
    while remaining > 0:
        batch_size = min(int(args.batch_size), remaining)
        smiles.extend(
            sampler.de_novo_generation(
                num_samples=batch_size,
                softmax_temp=args.temperature,
                randomness=args.randomness,
                min_add_len=args.min_add_len,
                max_length=args.max_length,
                top_k=args.top_k,
                fix=True,
            )
        )
        remaining -= batch_size
    return smiles[: int(args.num_samples)]


def _pad_dropped_samples(records: list, target_count: int, molecule_utils) -> list:
    missing = target_count - len(records)
    if missing <= 0:
        return records[:target_count]
    records.extend(
        molecule_utils.MoleculeEvaluation(
            source=None,
            smiles=None,
            canonical_smiles=None,
            valid=False,
            qed=None,
            sa=None,
            quality=False,
        )
        for _ in range(missing)
    )
    return records


def _max_length(model) -> int:
    return int(
        model.config.model.get(
            "max_position_embeddings",
            model.config.model.get("length", 256),
        )
    )


def _load_length_distribution(model_dir: Path):
    path = model_dir / "data" / "len.pk"
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _sample_add_lengths(
    *,
    model_dir: Path,
    batch_size: int,
    min_add_len: int,
    max_length: int,
    length_mode: str,
) -> list[int]:
    max_add_len = max(max_length - 2, 1)
    if length_mode == "fixed":
        return [min(max(int(min_add_len), 1), max_add_len)] * batch_size

    distribution = _load_length_distribution(model_dir)
    if distribution is None:
        return [min(max(int(min_add_len), 1), max_add_len)] * batch_size

    add_lengths = []
    for _ in range(batch_size):
        target_length = int(random.choice(distribution))
        add_len = max(target_length - 2, int(min_add_len), 1)
        add_lengths.append(min(add_len, max_add_len))
    return add_lengths


def _build_diffusion_template(
    model,
    model_dir: Path,
    batch_size: int,
    min_add_len: int,
    length_mode: str,
) -> torch.Tensor:
    add_lengths = _sample_add_lengths(
        model_dir=model_dir,
        batch_size=batch_size,
        min_add_len=min_add_len,
        max_length=_max_length(model),
        length_mode=length_mode,
    )
    width = max(add_lengths) + 2
    template = torch.full(
        (batch_size, width),
        int(model.pad_index),
        dtype=torch.long,
    )
    for row_idx, add_len in enumerate(add_lengths):
        template[row_idx, 0] = int(model.bos_index)
        template[row_idx, 1 : 1 + add_len] = int(model.mask_index)
        template[row_idx, 1 + add_len] = int(model.eos_index)
    return template.to(model.device)


@torch.no_grad()
def _sample_ar_batch(model, batch_size: int, args) -> torch.Tensor:
    return model.sample_ids(
        num_samples=batch_size,
        max_length=args.max_length or _max_length(model),
        temperature=args.temperature,
        randomness=args.randomness,
        min_new_tokens=args.min_add_len,
        top_k=args.top_k,
        stop_at_eos=True,
        ban_special_tokens=True,
    )


@torch.no_grad()
def _sample_diffusion_batch(model, model_dir: Path, batch_size: int, args) -> torch.Tensor:
    template = _build_diffusion_template(
        model=model,
        model_dir=model_dir,
        batch_size=batch_size,
        min_add_len=args.min_add_len,
        length_mode=args.length_mode,
    )
    attention_mask = template.ne(model.pad_index)
    model_attention_mask = attention_mask.long()
    editable_mask = template.eq(model.mask_index) & attention_mask

    x = model.diffusion.initialize_sample(template, editable_mask)
    num_steps = max(int(args.steps), 2)
    timestep_grid = model.diffusion.get_sampling_timesteps(
        model.device,
        num_steps=num_steps,
    )

    for step_idx in range(num_steps):
        t = timestep_grid[step_idx].expand(x.shape[0])
        sigma_t = model.diffusion.time_conditioning(t)
        logits = model(
            x,
            attention_mask=model_attention_mask,
            timesteps=sigma_t,
        )
        x = model.diffusion.step_confidence(
            logits,
            x,
            step_idx,
            num_steps,
            args.temperature,
            args.randomness,
            editable_mask=editable_mask,
            timestep_grid=timestep_grid,
        )

    if bool(getattr(model.diffusion, "final_denoise", True)):
        sigma_0 = torch.zeros(x.shape[0], device=x.device)
        logits = model(
            x,
            attention_mask=model_attention_mask,
            timesteps=sigma_0,
        )
        x = model.diffusion.final_denoise_step(
            logits,
            x,
            editable_mask,
            softmax_temp=args.temperature,
            randomness=0.0,
        )
    return x


def _write_records_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "safe",
        "source",
        "smiles",
        "canonical_smiles",
        "valid",
        "qed",
        "sa",
        "quality",
        "raw_token_ids",
        "raw_length",
        "first_token_id",
        "first_token_is_bos",
        "decoded_length",
        "eos_position",
        "pad_position",
    ]
    extras = sorted({key for row in rows for key in row if key not in preferred})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=preferred + extras)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Sample and score de novo molecules from genmol SAFE checkpoints."
    )
    parser.add_argument("--model", required=True, choices=["safe-ar", "safe-mdlm", "safe-udlm"])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--records-output")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--randomness", type=float, default=0.5)
    parser.add_argument("--min-add-len", type=int, default=40)
    parser.add_argument("--length-mode", choices=["distribution", "fixed"], default="distribution")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--qed-threshold", type=float, default=0.6)
    parser.add_argument("--sa-threshold", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_dir = Path(args.model_dir).resolve()
    metrics_dir = Path(args.metrics_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()

    _insert_import_paths(model_dir, metrics_dir)
    import molecule_utils

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    smiles = _sample_legacy_repaired_smiles(checkpoint, args)
    records = molecule_utils.evaluate_molecules(
        smiles=smiles,
        qed_threshold=args.qed_threshold,
        sa_threshold=args.sa_threshold,
    )
    records = _pad_dropped_samples(records, int(args.num_samples), molecule_utils)
    summary = {
        "task": "denovo",
        "model": args.model,
        "checkpoint": str(checkpoint),
        "metrics": molecule_utils.generation_metrics(records),
        "token_diagnostics": {},
        "sampling": {
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "temperature": args.temperature,
            "randomness": args.randomness,
            "min_add_len": args.min_add_len,
            "length_mode": args.length_mode,
            "seed": args.seed,
            "sampler": "genmol.sampler.Sampler.de_novo_generation",
            "repair_fragments": True,
            "num_repaired_smiles": len(smiles),
            "num_dropped_by_sampler": int(args.num_samples) - len(smiles),
        },
    }
    molecule_utils.write_json(summary, args.output)

    if args.records_output:
        rows = []
        for record in records:
            row = record.to_dict()
            row["safe"] = ""
            row["raw_token_ids"] = ""
            rows.append(row)
        _write_records_csv(rows, Path(args.records_output))


if __name__ == "__main__":
    main()
