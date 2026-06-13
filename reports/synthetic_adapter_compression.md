# Synthetic Adapter Compression Experiment

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

The run produced 31920 rows. The identity sanity check passed:
the maximum test normalized residual error for identity with `q >= 16`
was `5.295526e-08`.

Mean static LoRA+bias performance at `q=16`:

| Activation | Mean epsilon_test | Mean test fidelity |
| --- | ---: | ---: |
| gelu | 0.326527 | 0.9048 |
| identity | 0.000000 | 1.0000 |
| relu | 0.427650 | 0.8472 |
| tanh | 0.108743 | 0.9637 |

Mean full-affine upper-bound `epsilon_test` across all synthetic configurations:
`0.215730`.

Mean piecewise LoRA+bias `epsilon_test` at `q=16`:

| Clusters | Mean epsilon_test |
| ---: | ---: |
| 2 | 0.194704 |
| 4 | 0.177914 |
| 8 | 0.164989 |

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
