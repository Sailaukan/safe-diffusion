#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    summary = {"models": {}, "missing": []}
    for model in args.models:
        path = run_root / model / "denovo_metrics.json"
        if not path.exists():
            summary["missing"].append(model)
            continue
        with path.open() as f:
            data = json.load(f)
        summary["models"][model] = {
            "checkpoint": data.get("checkpoint"),
            "metrics": data.get("metrics", {}),
            "token_diagnostics": data.get("token_diagnostics", {}),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

