"""Synthetic adapter residual compression experiment.

This module tests whether nonlinear adapter residuals can be approximated by a
mergeable low-rank affine update on controlled hidden-state distributions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import statistics
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "delta-tuning-matplotlib"))
warnings.simplefilter("ignore", RuntimeWarning)
import numpy as np
from scipy import __version__ as scipy_version
from scipy.cluster.vq import kmeans2
from scipy.special import erf

np.seterr(all="ignore")


DEFAULT_COLUMNS = [
    "seed",
    "d",
    "adapter_rank",
    "intrinsic_dim",
    "noise_sigma",
    "activation",
    "replacement_method",
    "replacement_rank",
    "num_clusters",
    "epsilon_train",
    "epsilon_test",
    "r2_train",
    "r2_test",
    "teacher_student_fidelity_train",
    "teacher_student_fidelity_test",
    "activation_entropy_mean",
    "activation_entropy_std",
    "activation_active_fraction_mean",
    "activation_active_fraction_std",
]

SUMMARY_KEYS = [
    "d",
    "adapter_rank",
    "intrinsic_dim",
    "noise_sigma",
    "activation",
    "replacement_method",
    "replacement_rank",
    "num_clusters",
]

METRIC_COLUMNS = [
    "epsilon_train",
    "epsilon_test",
    "r2_train",
    "r2_test",
    "teacher_student_fidelity_train",
    "teacher_student_fidelity_test",
    "activation_entropy_mean",
    "activation_entropy_std",
    "activation_active_fraction_mean",
    "activation_active_fraction_std",
]


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: Sequence[int]
    n_train: int
    n_test: int
    hidden_dim: int
    adapter_rank: int
    intrinsic_dims: Sequence[int]
    noise_sigmas: Sequence[float]
    activations: Sequence[str]
    replacement_ranks: Sequence[int]
    cluster_counts: Sequence[int]
    output_classes: int = 5


@dataclass
class LowRankAffineModel:
    """Model for R_hat = (H - input_mean) @ left @ right + bias."""

    input_mean: np.ndarray
    left: np.ndarray
    right: np.ndarray
    bias: np.ndarray


@dataclass
class RankPath:
    """Cached QR/SVD path for rank-constrained affine prediction."""

    input_mean: np.ndarray
    bias: np.ndarray
    eigvecs: np.ndarray
    inv_sqrt_eigs: np.ndarray
    svd_u: np.ndarray
    svd_s: np.ndarray
    svd_vt: np.ndarray

    @property
    def max_rank(self) -> int:
        return int(min(self.svd_s.size, self.eigvecs.shape[1]))

    def model(self, rank: Optional[int]) -> LowRankAffineModel:
        if rank is None:
            rank_eff = self.max_rank
        else:
            rank_eff = int(max(0, min(rank, self.max_rank)))

        d = self.input_mean.size
        out_d = self.bias.size
        if rank_eff == 0:
            return LowRankAffineModel(
                input_mean=self.input_mean,
                left=np.zeros((d, 0), dtype=np.float64),
                right=np.zeros((0, out_d), dtype=np.float64),
                bias=self.bias,
            )

        scaled = (
            self.svd_u[:, :rank_eff]
            * self.svd_s[:rank_eff][None, :]
            * self.inv_sqrt_eigs[:, None]
        )
        left = self.eigvecs @ scaled
        right = self.svd_vt[:rank_eff, :]
        return LowRankAffineModel(self.input_mean, left, right, self.bias)


def stable_rng(*parts: object) -> np.random.Generator:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**32)
    return np.random.default_rng(seed)


def orthonormal_rows(rng: np.random.Generator, k: int, d: int) -> np.ndarray:
    raw = rng.normal(size=(d, k))
    q, _ = np.linalg.qr(raw, mode="reduced")
    return q.T


def generate_hidden_states(
    rng: np.random.Generator,
    n: int,
    k: int,
    d: int,
    sigma: float,
    p: np.ndarray,
) -> np.ndarray:
    z = rng.normal(size=(n, k))
    h = z @ p
    if sigma:
        h = h + sigma * rng.normal(size=(n, d))
    if not np.isfinite(h).all():
        raise FloatingPointError("Generated hidden states contain non-finite values")
    return np.asarray(h, dtype=np.float64)


def generate_adapter(
    rng: np.random.Generator, d: int, adapter_rank: int
) -> Tuple[np.ndarray, np.ndarray]:
    w_down = rng.normal(0.0, 1.0 / math.sqrt(d), size=(d, adapter_rank))
    w_up = rng.normal(0.0, 1.0 / math.sqrt(adapter_rank), size=(adapter_rank, d))
    if not np.isfinite(w_down).all() or not np.isfinite(w_up).all():
        raise FloatingPointError("Generated adapter weights contain non-finite values")
    return np.asarray(w_down, dtype=np.float64), np.asarray(w_up, dtype=np.float64)


def apply_activation(x: np.ndarray, activation: str) -> np.ndarray:
    if activation == "identity":
        return x
    if activation == "relu":
        return np.maximum(x, 0.0)
    if activation == "gelu":
        return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))
    if activation == "tanh":
        return np.tanh(x)
    raise ValueError(f"Unsupported activation: {activation}")


def activation_stats(preactivation: np.ndarray, activation: str) -> Dict[str, float]:
    if activation not in {"relu", "gelu"}:
        return {
            "activation_entropy_mean": float("nan"),
            "activation_entropy_std": float("nan"),
            "activation_active_fraction_mean": float("nan"),
            "activation_active_fraction_std": float("nan"),
        }

    mask = preactivation > 0.0
    p = mask.mean(axis=0)
    eps = 1e-12
    entropy = -(p * np.log(p + eps) + (1.0 - p) * np.log(1.0 - p + eps))
    return {
        "activation_entropy_mean": float(entropy.mean()),
        "activation_entropy_std": float(entropy.std(ddof=0)),
        "activation_active_fraction_mean": float(p.mean()),
        "activation_active_fraction_std": float(p.std(ddof=0)),
    }


def fit_rank_path(
    h: np.ndarray,
    g: np.ndarray,
    w_up: np.ndarray,
    *,
    center: bool,
    eig_tol: float = 1e-10,
) -> RankPath:
    d = h.shape[1]
    out_d = w_up.shape[1]
    if center:
        input_mean = h.mean(axis=0)
        x = h - input_mean
        bias = g.mean(axis=0) @ w_up
    else:
        input_mean = np.zeros(d, dtype=np.float64)
        x = h
        bias = np.zeros(out_d, dtype=np.float64)

    xtx = x.T @ x
    eigvals, eigvecs = np.linalg.eigh(xtx)
    if eigvals.size == 0:
        keep = np.zeros(0, dtype=bool)
    else:
        keep = eigvals > (float(eigvals.max()) * eig_tol)

    if not np.any(keep):
        return RankPath(
            input_mean=input_mean,
            bias=bias,
            eigvecs=np.zeros((d, 0), dtype=np.float64),
            inv_sqrt_eigs=np.zeros(0, dtype=np.float64),
            svd_u=np.zeros((0, 0), dtype=np.float64),
            svd_s=np.zeros(0, dtype=np.float64),
            svd_vt=np.zeros((0, out_d), dtype=np.float64),
        )

    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]
    inv_sqrt = 1.0 / np.sqrt(eigvals)

    xtg = x.T @ g
    htr = xtg @ w_up
    y = (eigvecs.T @ htr) * inv_sqrt[:, None]
    svd_u, svd_s, svd_vt = np.linalg.svd(y, full_matrices=False)
    return RankPath(input_mean, bias, eigvecs, inv_sqrt, svd_u, svd_s, svd_vt)


def assign_nearest(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(distances, axis=1)


def fit_piecewise_paths(
    h: np.ndarray,
    g: np.ndarray,
    w_up: np.ndarray,
    preactivation: np.ndarray,
    clusters: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[Optional[RankPath]]]:
    centroids, labels = kmeans2(
        preactivation,
        clusters,
        iter=30,
        minit="points",
        missing="warn",
        seed=rng,
    )
    labels = np.asarray(labels, dtype=np.int64)
    paths: List[Optional[RankPath]] = []
    for cluster_id in range(clusters):
        idx = np.flatnonzero(labels == cluster_id)
        if idx.size < 2:
            paths.append(None)
            continue
        paths.append(fit_rank_path(h[idx], g[idx], w_up, center=True))
    return centroids, labels, paths


def residual_norm2(g: np.ndarray, w_up: np.ndarray) -> float:
    gram = w_up @ w_up.T
    return float(np.einsum("ij,ij->", g @ gram, g))


def centered_residual_norm2(g: np.ndarray, w_up: np.ndarray) -> float:
    norm2 = residual_norm2(g, w_up)
    mean = g.mean(axis=0) @ w_up
    return float(max(norm2 - g.shape[0] * float(mean @ mean), 0.0))


def evaluate_model_stats(
    h: np.ndarray,
    g: np.ndarray,
    w_up: np.ndarray,
    c_head: np.ndarray,
    model: LowRankAffineModel,
) -> Dict[str, float]:
    n = h.shape[0]
    x = h - model.input_mean
    f = x @ model.left if model.left.shape[1] else np.zeros((n, 0), dtype=np.float64)

    r_norm2 = residual_norm2(g, w_up)
    r_centered_norm2 = centered_residual_norm2(g, w_up)
    r_sum = g.sum(axis=0) @ w_up

    if f.shape[1]:
        bb_t = model.right @ model.right.T
        pred_no_bias_norm2 = float(np.einsum("ij,ij->", f @ bb_t, f))
        pred_sum = f.sum(axis=0) @ model.right
        cross_no_bias = float(np.trace((g.T @ f) @ (model.right @ w_up.T)))
    else:
        pred_no_bias_norm2 = 0.0
        pred_sum = np.zeros_like(model.bias)
        cross_no_bias = 0.0

    pred_norm2 = (
        pred_no_bias_norm2
        + 2.0 * float(pred_sum @ model.bias)
        + n * float(model.bias @ model.bias)
    )
    cross = cross_no_bias + float(r_sum @ model.bias)
    err2 = max(r_norm2 + pred_norm2 - 2.0 * cross, 0.0)

    epsilon = math.sqrt(err2) / math.sqrt(max(r_norm2, 1e-30))
    r2 = float("nan")
    if r_centered_norm2 > 1e-30:
        r2 = 1.0 - err2 / r_centered_norm2

    teacher_logits = h @ c_head + g @ (w_up @ c_head)
    if f.shape[1]:
        pred_delta_logits = f @ (model.right @ c_head)
    else:
        pred_delta_logits = np.zeros((n, c_head.shape[1]), dtype=np.float64)
    pred_delta_logits = pred_delta_logits + model.bias @ c_head
    student_logits = h @ c_head + pred_delta_logits
    fidelity = float(np.mean(np.argmax(teacher_logits, axis=1) == np.argmax(student_logits, axis=1)))

    return {
        "err2": float(err2),
        "target_norm2": float(r_norm2),
        "target_centered_norm2": float(r_centered_norm2),
        "epsilon": float(epsilon),
        "r2": float(r2),
        "fidelity": fidelity,
        "matches": float(fidelity * n),
        "n": float(n),
    }


def evaluate_piecewise_stats(
    h: np.ndarray,
    g: np.ndarray,
    w_up: np.ndarray,
    c_head: np.ndarray,
    preactivation: np.ndarray,
    centroids: np.ndarray,
    paths: Sequence[Optional[RankPath]],
    rank: int,
) -> Dict[str, float]:
    labels = assign_nearest(preactivation, centroids)
    total_err2 = 0.0
    total_r_norm2 = 0.0
    total_centered = centered_residual_norm2(g, w_up)
    total_matches = 0.0
    total_n = 0.0

    for cluster_id, path in enumerate(paths):
        idx = np.flatnonzero(labels == cluster_id)
        if idx.size == 0:
            continue
        if path is None:
            bias = g[idx].mean(axis=0) @ w_up
            model = LowRankAffineModel(
                input_mean=np.zeros(h.shape[1], dtype=np.float64),
                left=np.zeros((h.shape[1], 0), dtype=np.float64),
                right=np.zeros((0, w_up.shape[1]), dtype=np.float64),
                bias=bias,
            )
        else:
            model = path.model(rank)
        stats = evaluate_model_stats(h[idx], g[idx], w_up, c_head, model)
        total_err2 += stats["err2"]
        total_r_norm2 += stats["target_norm2"]
        total_matches += stats["matches"]
        total_n += stats["n"]

    epsilon = math.sqrt(max(total_err2, 0.0)) / math.sqrt(max(total_r_norm2, 1e-30))
    r2 = float("nan")
    if total_centered > 1e-30:
        r2 = 1.0 - total_err2 / total_centered
    return {
        "epsilon": float(epsilon),
        "r2": float(r2),
        "fidelity": float(total_matches / max(total_n, 1.0)),
    }


def row_from_metrics(
    *,
    seed: int,
    d: int,
    adapter_rank: int,
    intrinsic_dim: int,
    noise_sigma: float,
    activation: str,
    replacement_method: str,
    replacement_rank: int,
    num_clusters: int,
    train_stats: Mapping[str, float],
    test_stats: Mapping[str, float],
    activation_metrics: Mapping[str, float],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "seed": seed,
        "d": d,
        "adapter_rank": adapter_rank,
        "intrinsic_dim": intrinsic_dim,
        "noise_sigma": noise_sigma,
        "activation": activation,
        "replacement_method": replacement_method,
        "replacement_rank": replacement_rank,
        "num_clusters": num_clusters,
        "epsilon_train": train_stats["epsilon"],
        "epsilon_test": test_stats["epsilon"],
        "r2_train": train_stats["r2"],
        "r2_test": test_stats["r2"],
        "teacher_student_fidelity_train": train_stats["fidelity"],
        "teacher_student_fidelity_test": test_stats["fidelity"],
    }
    row.update(activation_metrics)
    return row


def run_one_configuration(
    *,
    seed: int,
    d: int,
    adapter_rank: int,
    intrinsic_dim: int,
    noise_sigma: float,
    activation: str,
    n_train: int,
    n_test: int,
    replacement_ranks: Sequence[int],
    cluster_counts: Sequence[int],
    output_classes: int,
) -> List[Dict[str, object]]:
    p = orthonormal_rows(stable_rng("basis", seed, intrinsic_dim, d), intrinsic_dim, d)
    h_train = generate_hidden_states(
        stable_rng("hidden_train", seed, intrinsic_dim, noise_sigma),
        n_train,
        intrinsic_dim,
        d,
        noise_sigma,
        p,
    )
    h_test = generate_hidden_states(
        stable_rng("hidden_test", seed, intrinsic_dim, noise_sigma),
        n_test,
        intrinsic_dim,
        d,
        noise_sigma,
        p,
    )
    w_down, w_up = generate_adapter(stable_rng("adapter", seed, d, adapter_rank), d, adapter_rank)
    c_head = stable_rng("head", seed, d, output_classes).normal(
        0.0, 1.0 / math.sqrt(d), size=(d, output_classes)
    )

    pre_train = h_train @ w_down
    pre_test = h_test @ w_down
    g_train = apply_activation(pre_train, activation)
    g_test = apply_activation(pre_test, activation)
    if not np.isfinite(pre_train).all() or not np.isfinite(pre_test).all():
        raise FloatingPointError("Adapter preactivations contain non-finite values")
    if not np.isfinite(g_train).all() or not np.isfinite(g_test).all():
        raise FloatingPointError("Adapter residual features contain non-finite values")
    activation_metrics = activation_stats(pre_train, activation)

    rows: List[Dict[str, object]] = []
    zero = LowRankAffineModel(
        input_mean=np.zeros(d, dtype=np.float64),
        left=np.zeros((d, 0), dtype=np.float64),
        right=np.zeros((0, d), dtype=np.float64),
        bias=np.zeros(d, dtype=np.float64),
    )
    train_stats = evaluate_model_stats(h_train, g_train, w_up, c_head, zero)
    test_stats = evaluate_model_stats(h_test, g_test, w_up, c_head, zero)
    rows.append(
        row_from_metrics(
            seed=seed,
            d=d,
            adapter_rank=adapter_rank,
            intrinsic_dim=intrinsic_dim,
            noise_sigma=noise_sigma,
            activation=activation,
            replacement_method="no_delta",
            replacement_rank=0,
            num_clusters=0,
            train_stats=train_stats,
            test_stats=test_stats,
            activation_metrics=activation_metrics,
        )
    )

    bias = g_train.mean(axis=0) @ w_up
    bias_only = LowRankAffineModel(
        input_mean=np.zeros(d, dtype=np.float64),
        left=np.zeros((d, 0), dtype=np.float64),
        right=np.zeros((0, d), dtype=np.float64),
        bias=bias,
    )
    train_stats = evaluate_model_stats(h_train, g_train, w_up, c_head, bias_only)
    test_stats = evaluate_model_stats(h_test, g_test, w_up, c_head, bias_only)
    rows.append(
        row_from_metrics(
            seed=seed,
            d=d,
            adapter_rank=adapter_rank,
            intrinsic_dim=intrinsic_dim,
            noise_sigma=noise_sigma,
            activation=activation,
            replacement_method="bias_only",
            replacement_rank=0,
            num_clusters=0,
            train_stats=train_stats,
            test_stats=test_stats,
            activation_metrics=activation_metrics,
        )
    )

    no_bias_path = fit_rank_path(h_train, g_train, w_up, center=False)
    lora_bias_path = fit_rank_path(h_train, g_train, w_up, center=True)

    for rank in replacement_ranks:
        model = no_bias_path.model(rank)
        train_stats = evaluate_model_stats(h_train, g_train, w_up, c_head, model)
        test_stats = evaluate_model_stats(h_test, g_test, w_up, c_head, model)
        rows.append(
            row_from_metrics(
                seed=seed,
                d=d,
                adapter_rank=adapter_rank,
                intrinsic_dim=intrinsic_dim,
                noise_sigma=noise_sigma,
                activation=activation,
                replacement_method="low_rank_no_bias",
                replacement_rank=rank,
                num_clusters=0,
                train_stats=train_stats,
                test_stats=test_stats,
                activation_metrics=activation_metrics,
            )
        )

        model = lora_bias_path.model(rank)
        train_stats = evaluate_model_stats(h_train, g_train, w_up, c_head, model)
        test_stats = evaluate_model_stats(h_test, g_test, w_up, c_head, model)
        rows.append(
            row_from_metrics(
                seed=seed,
                d=d,
                adapter_rank=adapter_rank,
                intrinsic_dim=intrinsic_dim,
                noise_sigma=noise_sigma,
                activation=activation,
                replacement_method="lora_bias",
                replacement_rank=rank,
                num_clusters=0,
                train_stats=train_stats,
                test_stats=test_stats,
                activation_metrics=activation_metrics,
            )
        )

    model = lora_bias_path.model(None)
    train_stats = evaluate_model_stats(h_train, g_train, w_up, c_head, model)
    test_stats = evaluate_model_stats(h_test, g_test, w_up, c_head, model)
    rows.append(
        row_from_metrics(
            seed=seed,
            d=d,
            adapter_rank=adapter_rank,
            intrinsic_dim=intrinsic_dim,
            noise_sigma=noise_sigma,
            activation=activation,
            replacement_method="full_affine",
            replacement_rank=d,
            num_clusters=0,
            train_stats=train_stats,
            test_stats=test_stats,
            activation_metrics=activation_metrics,
        )
    )

    for clusters in cluster_counts:
        centroids, _, paths = fit_piecewise_paths(
            h_train,
            g_train,
            w_up,
            pre_train,
            clusters,
            stable_rng("cluster", seed, intrinsic_dim, noise_sigma, activation, clusters),
        )
        for rank in replacement_ranks:
            train_stats = evaluate_piecewise_stats(
                h_train, g_train, w_up, c_head, pre_train, centroids, paths, rank
            )
            test_stats = evaluate_piecewise_stats(
                h_test, g_test, w_up, c_head, pre_test, centroids, paths, rank
            )
            rows.append(
                row_from_metrics(
                    seed=seed,
                    d=d,
                    adapter_rank=adapter_rank,
                    intrinsic_dim=intrinsic_dim,
                    noise_sigma=noise_sigma,
                    activation=activation,
                    replacement_method="piecewise_lora_bias",
                    replacement_rank=rank,
                    num_clusters=clusters,
                    train_stats=train_stats,
                    test_stats=test_stats,
                    activation_metrics=activation_metrics,
                )
            )

    return rows


def run_experiment(config: ExperimentConfig, progress: bool = True) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    total = (
        len(config.seeds)
        * len(config.intrinsic_dims)
        * len(config.noise_sigmas)
        * len(config.activations)
    )
    done = 0
    for seed in config.seeds:
        for intrinsic_dim in config.intrinsic_dims:
            for noise_sigma in config.noise_sigmas:
                for activation in config.activations:
                    done += 1
                    if progress:
                        print(
                            f"[{done:04d}/{total:04d}] "
                            f"seed={seed} k={intrinsic_dim} sigma={noise_sigma} act={activation}",
                            flush=True,
                        )
                    rows.extend(
                        run_one_configuration(
                            seed=seed,
                            d=config.hidden_dim,
                            adapter_rank=config.adapter_rank,
                            intrinsic_dim=intrinsic_dim,
                            noise_sigma=noise_sigma,
                            activation=activation,
                            n_train=config.n_train,
                            n_test=config.n_test,
                            replacement_ranks=config.replacement_ranks,
                            cluster_counts=config.cluster_counts,
                            output_classes=config.output_classes,
                        )
                    )
    return rows


def write_runs_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in DEFAULT_COLUMNS})


def read_runs_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in [
            "seed",
            "d",
            "adapter_rank",
            "intrinsic_dim",
            "replacement_rank",
            "num_clusters",
        ]:
            row[key] = int(row[key])
        for key in ["noise_sigma"] + METRIC_COLUMNS:
            value = row[key]
            row[key] = float(value) if value != "" else float("nan")
    return rows


def write_summary_csv(rows: Sequence[Mapping[str, object]], path: Path) -> List[Dict[str, object]]:
    grouped: MutableMapping[Tuple[object, ...], List[Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(row[column] for column in SUMMARY_KEYS)
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for key, group in sorted(grouped.items(), key=lambda item: item[0]):
        out: Dict[str, object] = dict(zip(SUMMARY_KEYS, key))
        out["n_runs"] = len(group)
        for metric in METRIC_COLUMNS:
            values = [float(row[metric]) for row in group]
            clean = [value for value in values if math.isfinite(value)]
            if clean:
                out[f"{metric}_mean"] = statistics.fmean(clean)
                out[f"{metric}_std"] = statistics.pstdev(clean) if len(clean) > 1 else 0.0
            else:
                out[f"{metric}_mean"] = float("nan")
                out[f"{metric}_std"] = float("nan")
        summary_rows.append(out)

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        SUMMARY_KEYS
        + ["n_runs"]
        + [f"{metric}_{suffix}" for metric in METRIC_COLUMNS for suffix in ("mean", "std")]
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def write_config(config: ExperimentConfig, out_dir: Path) -> None:
    import matplotlib

    payload = {
        "seeds": list(config.seeds),
        "n_train": config.n_train,
        "n_test": config.n_test,
        "hidden_dim": config.hidden_dim,
        "adapter_rank": config.adapter_rank,
        "intrinsic_dims": list(config.intrinsic_dims),
        "noise_sigmas": list(config.noise_sigmas),
        "activations": list(config.activations),
        "replacement_ranks": list(config.replacement_ranks),
        "cluster_counts": list(config.cluster_counts),
        "output_classes": config.output_classes,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy_version,
        "matplotlib": matplotlib.__version__,
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def mean_by(rows: Iterable[Mapping[str, object]], x_key: str) -> Dict[object, float]:
    grouped: Dict[object, List[float]] = {}
    for row in rows:
        value = float(row["epsilon_test"])
        if math.isfinite(value):
            grouped.setdefault(row[x_key], []).append(value)
    return {key: statistics.fmean(values) for key, values in grouped.items()}


def write_plots(rows: Sequence[Mapping[str, object]], out_dir: Path, adapter_rank: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    activations = ["identity", "relu", "gelu", "tanh"]
    colors = {
        "identity": "#2764c5",
        "relu": "#d04a3a",
        "gelu": "#3a8f58",
        "tanh": "#8a4fb3",
    }

    plt.figure(figsize=(7.2, 4.6))
    for activation in activations:
        selected = [
            row
            for row in rows
            if row["replacement_method"] == "lora_bias"
            and row["num_clusters"] == 0
            and row["activation"] == activation
        ]
        means = mean_by(selected, "replacement_rank")
        xs = sorted(means)
        ys = [means[x] for x in xs]
        plt.plot(xs, ys, marker="o", label=activation, color=colors[activation])
    plt.xlabel("Replacement rank q")
    plt.ylabel("Mean test normalized residual error")
    plt.title("Static LoRA+bias error vs replacement rank")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "epsilon_vs_rank.svg")
    plt.close()

    plt.figure(figsize=(7.2, 4.6))
    for activation in activations:
        selected = [
            row
            for row in rows
            if row["replacement_method"] == "lora_bias"
            and row["num_clusters"] == 0
            and row["replacement_rank"] == adapter_rank
            and row["activation"] == activation
        ]
        means = mean_by(selected, "intrinsic_dim")
        xs = sorted(means)
        ys = [means[x] for x in xs]
        plt.plot(xs, ys, marker="o", label=activation, color=colors[activation])
    plt.xlabel("Intrinsic dimension k")
    plt.ylabel("Mean test normalized residual error")
    plt.title(f"Static LoRA+bias error vs intrinsic dimension at q={adapter_rank}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "epsilon_vs_intrinsic_dim.svg")
    plt.close()

    plt.figure(figsize=(7.2, 4.6))
    for activation in ["relu", "gelu"]:
        selected = [
            row
            for row in rows
            if row["replacement_method"] == "lora_bias"
            and row["num_clusters"] == 0
            and row["replacement_rank"] == adapter_rank
            and row["activation"] == activation
        ]
        xs = [float(row["activation_entropy_mean"]) for row in selected]
        ys = [float(row["epsilon_test"]) for row in selected]
        plt.scatter(xs, ys, s=18, alpha=0.6, label=activation, color=colors[activation])
    plt.xlabel("Mean activation entropy")
    plt.ylabel("Test normalized residual error")
    plt.title(f"Activation entropy vs static LoRA+bias error at q={adapter_rank}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "activation_entropy_vs_epsilon.svg")
    plt.close()


def safe_float(row: Mapping[str, object], key: str) -> float:
    value = float(row[key])
    return value if math.isfinite(value) else float("nan")


def summarize_findings(rows: Sequence[Mapping[str, object]], adapter_rank: int) -> Dict[str, object]:
    lora_qr = [
        row
        for row in rows
        if row["replacement_method"] == "lora_bias"
        and row["replacement_rank"] == adapter_rank
        and row["num_clusters"] == 0
    ]
    by_activation: Dict[str, Dict[str, float]] = {}
    for activation in sorted({str(row["activation"]) for row in lora_qr}):
        selected = [row for row in lora_qr if row["activation"] == activation]
        by_activation[activation] = {
            "epsilon_test_mean": statistics.fmean(safe_float(row, "epsilon_test") for row in selected),
            "fidelity_test_mean": statistics.fmean(
                safe_float(row, "teacher_student_fidelity_test") for row in selected
            ),
        }

    full_affine = [
        row for row in rows if row["replacement_method"] == "full_affine" and row["num_clusters"] == 0
    ]
    full_affine_mean = statistics.fmean(safe_float(row, "epsilon_test") for row in full_affine)

    piecewise_qr = [
        row
        for row in rows
        if row["replacement_method"] == "piecewise_lora_bias"
        and row["replacement_rank"] == adapter_rank
    ]
    piecewise_mean_by_k = {
        clusters: statistics.fmean(
            safe_float(row, "epsilon_test") for row in piecewise_qr if row["num_clusters"] == clusters
        )
        for clusters in sorted({int(row["num_clusters"]) for row in piecewise_qr})
    }

    identity_exact = [
        row
        for row in rows
        if row["replacement_method"] == "lora_bias"
        and row["replacement_rank"] >= adapter_rank
        and row["activation"] == "identity"
    ]
    identity_max_error = (
        max(safe_float(row, "epsilon_test") for row in identity_exact)
        if identity_exact
        else float("nan")
    )

    return {
        "by_activation": by_activation,
        "full_affine_epsilon_test_mean": full_affine_mean,
        "piecewise_epsilon_test_mean_by_clusters": piecewise_mean_by_k,
        "identity_max_epsilon_test_rank_ge_adapter": identity_max_error,
        "num_rows": len(rows),
    }


def write_report(rows: Sequence[Mapping[str, object]], out_dir: Path, report_path: Path, adapter_rank: int) -> None:
    findings = summarize_findings(rows, adapter_rank)
    by_activation = findings["by_activation"]
    piecewise = findings["piecewise_epsilon_test_mean_by_clusters"]

    activation_lines = "\n".join(
        f"| {activation} | {stats['epsilon_test_mean']:.6f} | {stats['fidelity_test_mean']:.4f} |"
        for activation, stats in sorted(by_activation.items())
    )
    piecewise_lines = "\n".join(
        f"| {clusters} | {epsilon:.6f} |" for clusters, epsilon in sorted(piecewise.items())
    )

    body = f"""# Synthetic Adapter Compression Experiment

This report summarizes a controlled simulation of one claim: whether a nonlinear
adapter residual can be approximated by a mergeable low-rank affine update on a
task-like hidden-state distribution.

The experiment is synthetic only. It validates or falsifies the mechanism under
controlled hidden-state geometry; it does not claim that real pretrained language
model adapters behave the same way.

## Artifacts

- `results/synthetic_adapter_compression/runs.csv`: one row per run.
- `results/synthetic_adapter_compression/summary.csv`: grouped means and standard deviations.
- `results/synthetic_adapter_compression/config.json`: exact grid and package versions.
- `results/synthetic_adapter_compression/epsilon_vs_rank.svg`
- `results/synthetic_adapter_compression/epsilon_vs_intrinsic_dim.svg`
- `results/synthetic_adapter_compression/activation_entropy_vs_epsilon.svg`

## Main Results

The run produced {findings['num_rows']} rows. The identity sanity check passed:
the maximum test normalized residual error for identity with `q >= {adapter_rank}`
was `{findings['identity_max_epsilon_test_rank_ge_adapter']:.6e}`.

Mean static LoRA+bias performance at `q={adapter_rank}`:

| Activation | Mean epsilon_test | Mean test fidelity |
| --- | ---: | ---: |
{activation_lines}

Mean full-affine upper-bound `epsilon_test` across all synthetic configurations:
`{findings['full_affine_epsilon_test_mean']:.6f}`.

Mean piecewise LoRA+bias `epsilon_test` at `q={adapter_rank}`:

| Clusters | Mean epsilon_test |
| ---: | ---: |
{piecewise_lines}

## Interpretation

The identity activation behaves as expected, which checks the linear algebra and
rank-constrained fitting path. Nonlinear activations should be interpreted by
comparing the static LoRA+bias rows against the full-affine and piecewise rows:

- If static LoRA+bias is close to the full-affine bound, the main limitation is
  not the low-rank mergeable form.
- If piecewise LoRA+bias sharply improves over static LoRA+bias, the residual is
  behaving like a conditional low-rank map rather than a single mergeable affine
  update.
- If full affine is also poor, the sampled adapter residual is not well described
  by simple affine structure on that distribution.

The next step, if these synthetic results are promising enough for the project,
is Experiment 2: repeat the residual replacement on real hidden states from a
small transformer with a trained adapter.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(body, encoding="utf-8")


def write_all_outputs(config: ExperimentConfig, rows: Sequence[Mapping[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_runs_csv(rows, out_dir / "runs.csv")
    write_summary_csv(rows, out_dir / "summary.csv")
    write_config(config, out_dir)
    write_plots(rows, out_dir, config.adapter_rank)
    write_report(rows, out_dir, Path("reports/synthetic_adapter_compression.md"), config.adapter_rank)
