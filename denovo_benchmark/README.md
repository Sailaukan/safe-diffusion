# De Novo SAFE-GPT Benchmark

This folder launches the fair comparison run across:

- `safe-ar`
- `safe-mdlm`
- `safe-udlm`
- `safe-duo` with `algo=duo_base`

Defaults:

- SAFE-GPT revision: `b83175cd7394`
- training steps: `10000`
- global batch size: `256`
- generated de novo samples per model: `10000`
- sampling steps: `128`

Run all training jobs:

```bash
bash denovo_benchmark/train_all.sh
```

Run all evaluations after checkpoints exist:

```bash
bash denovo_benchmark/eval_all.sh
```

Useful overrides:

```bash
RUN_ROOT=/path/to/run MAX_STEPS=10000 SAMPLE_COUNT=10000 bash denovo_benchmark/train_all.sh
RUN_ROOT=/path/to/run bash denovo_benchmark/eval_all.sh
RESUME=1 bash denovo_benchmark/train_one.sh safe-mdlm
```

The scripts refuse to overwrite an existing checkpoint directory unless
`RESUME=1` is set. Evaluation writes per-model `denovo_metrics.json`,
`denovo_records.csv`, and a combined `summary.json` under `RUN_ROOT`.

