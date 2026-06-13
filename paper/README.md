# Adapter-to-LoRA Empirical Note

This folder contains the working manuscript and summary tables for the synthetic adapter-compression observation.

Working title: **When Is an Adapter LoRA-Like? Static and Piecewise Low-Rank Compression of Synthetic Nonlinear Adapter Residuals**

The manuscript is deliberately scoped as a synthetic mechanism note, not a real-PLM fine-tuning paper. It studies when a bottleneck adapter residual of the form `R = f(H W_d) W_u` can be approximated by a static low-rank affine replacement `R_hat = H U V + b`.

## Files

- `adapter_lora_empirical_note_v0_2.md` — current manuscript draft.
- `data/static_q16_summary.csv` — static rank-16 LoRA+bias summary by activation.
- `data/static_q16_by_intrinsic_dim.csv` — static rank-16 summary by activation and intrinsic dimension.
- `data/piecewise_q16_summary.csv` — piecewise rank-16 LoRA+bias summary by activation and cluster count.

## Core empirical observation

A bottleneck adapter with identity activation is exactly LoRA-like because its residual is a static low-rank affine map. Nonlinear adapters are different: their residual matrix remains low-rank, but the residual need not be affine in the hidden states. In this synthetic experiment, static LoRA+bias recovers identity adapters exactly, approximately recovers tanh under narrow hidden-state geometry, and does not faithfully recover GELU/ReLU with one static affine map. Piecewise low-rank affine models improve GELU/ReLU, which is consistent with nonlinear adapters acting as conditional low-rank maps.

## Caveats

- The adapters are random, not task-trained.
- The hidden states are synthetic Gaussian manifold samples, not real transformer hidden states.
- The current run does not establish an entropy-compressibility relationship because ReLU/GELU activation entropy barely varies.
- The piecewise result supports a conditional-map interpretation but weakens the clean merge-once deployment story.
