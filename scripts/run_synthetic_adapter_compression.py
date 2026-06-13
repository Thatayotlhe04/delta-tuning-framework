#!/usr/bin/env python3
"""Run the synthetic adapter compression experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from delta_tuning_framework.synthetic_adapter import ExperimentConfig, run_experiment, write_all_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--n-train", type=int, required=True)
    parser.add_argument("--n-test", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--adapter-rank", type=int, required=True)
    parser.add_argument("--intrinsic-dims", nargs="+", type=int, required=True)
    parser.add_argument("--noise-sigmas", nargs="+", type=float, required=True)
    parser.add_argument("--activations", nargs="+", required=True)
    parser.add_argument("--replacement-ranks", nargs="+", type=int, required=True)
    parser.add_argument("--cluster-counts", nargs="+", type=int, required=True)
    parser.add_argument("--output-classes", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        seeds=args.seeds,
        n_train=args.n_train,
        n_test=args.n_test,
        hidden_dim=args.hidden_dim,
        adapter_rank=args.adapter_rank,
        intrinsic_dims=args.intrinsic_dims,
        noise_sigmas=args.noise_sigmas,
        activations=args.activations,
        replacement_ranks=args.replacement_ranks,
        cluster_counts=args.cluster_counts,
        output_classes=args.output_classes,
    )
    rows = run_experiment(config, progress=not args.quiet)
    write_all_outputs(config, rows, args.out_dir)
    print(f"Wrote {len(rows)} rows to {args.out_dir}")


if __name__ == "__main__":
    main()
