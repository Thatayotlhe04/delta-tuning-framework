# Delta Tuning Framework

This repository contains a synthetic experiment for testing whether a nonlinear
adapter residual can be approximated by a mergeable low-rank affine update on a
task-like hidden-state distribution.

The current artifact is Experiment 1 only: a controlled simulation with
synthetic hidden states and synthetic adapter weights. It does not claim that
real pretrained language model adapters behave the same way.

## Reproduce

Install the small scientific stack:

```bash
python3 -m pip install -r requirements.txt
```

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover tests
```

Run the full synthetic experiment:

```bash
PYTHONPATH=src python3 scripts/run_synthetic_adapter_compression.py \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --n-train 20000 \
  --n-test 5000 \
  --hidden-dim 256 \
  --adapter-rank 16 \
  --intrinsic-dims 4 8 16 32 64 128 256 \
  --noise-sigmas 0 0.01 0.05 \
  --activations identity relu gelu tanh \
  --replacement-ranks 1 2 4 8 16 32 64 \
  --cluster-counts 2 4 8 \
  --out-dir results/synthetic_adapter_compression
```

## Outputs

- `results/synthetic_adapter_compression/runs.csv`: one row per seed,
  configuration, method, rank, and cluster count.
- `results/synthetic_adapter_compression/summary.csv`: grouped means and
  standard deviations.
- `results/synthetic_adapter_compression/config.json`: exact grid and package
  versions.
- `results/synthetic_adapter_compression/*.svg`: the three required plots.
- `reports/synthetic_adapter_compression.md`: concise interpretation of the
  simulation.
