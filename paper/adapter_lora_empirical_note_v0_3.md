# When Is an Adapter LoRA-Like?

## Static and Piecewise Low-Rank Approximation of Synthetic Nonlinear Adapter Residuals

**Author:** Thatayotlhe Tsenang  
**Affiliation:** B.Sc. Mathematics Undergraduate, University of Botswana  
**Email:** 202406227@ub.ac.bw  
**Draft:** Preprint v0.3, June 2026

## Abstract

Bottleneck adapters and LoRA-style updates are both parameter-efficient ways to modify a pretrained model, but they need not implement the same hidden-state function. This note studies a narrow mechanism question: when can an adapter residual be represented by a static low-rank affine map of the hidden states? For a batch of hidden states \(H\), a bottleneck adapter produces

\[
R=f(HW_d)W_u,
\]

whereas the static replacement studied here has the form

\[
\hat R_q = HUV+\mathbf{1}b^\top,\qquad \operatorname{rank}(UV)\le q.
\]

The residual matrix \(R\) always has rank at most the adapter bottleneck rank \(r\), but this output-rank bound does not imply that \(R\) is affine in \(H\). We test this distinction in a controlled synthetic study with hidden dimension \(d=256\), adapter rank \(r=16\), seven intrinsic dimensions, three noise levels, four activations, and ten random seeds. A rank-16 static affine replacement recovers identity adapters to numerical precision, approximately recovers tanh adapters on low-dimensional hidden-state manifolds, and leaves substantial error for GELU and ReLU. Increasing the replacement rank beyond \(r\) does not reduce the remaining error, indicating that the gap is not caused by insufficient residual output rank. Piecewise low-rank affine replacements reduce error for GELU and ReLU, suggesting that nonlinear adapters in this setting behave more like conditional low-rank maps than a single mergeable update. The result is deliberately scoped: the study uses synthetic hidden states and random adapters, not task-trained adapters inside real language models.

## 1. Introduction

Parameter-efficient fine-tuning methods reduce the cost of adapting large pretrained models by changing only a small subset of parameters, adding small trainable modules, or learning a low-dimensional reparameterization of a weight update. Adapters and LoRA are often placed under the same broad parameter-efficient umbrella, but this can hide a functional difference. A method's parameter count does not determine the class of hidden-state transformations it can express.

A bottleneck adapter adds a residual branch. For a hidden state \(h\in\mathbb{R}^{1\times d}\), the adapter update is

\[
h' = h + f(hW_d)W_u,
\]

where \(W_d\in\mathbb{R}^{d\times r}\), \(W_u\in\mathbb{R}^{r\times d}\), \(r\ll d\), and \(f\) is usually nonlinear. A LoRA-style update, by contrast, contributes a static low-rank linear map such as

\[
hUV.
\]

If \(f\) is the identity, the adapter residual becomes \(hW_dW_u\), which is exactly a low-rank map. The nonlinear case is more subtle. Even when the residual matrix produced by a nonlinear adapter has rank at most \(r\), the mapping from hidden states to residuals may not be affine. This note studies that distinction.

The contribution is modest but precise. We do not claim that adapters can generally be compressed into LoRA, and we do not claim real transformer adapters behave like the synthetic adapters studied here. Instead, we isolate a mechanism:

\[
\text{low residual rank} \not\Rightarrow \text{static affine compressibility in } H.
\]

We then test when a static low-rank affine replacement succeeds or fails under controlled synthetic hidden-state geometry.

## 2. Definitions and Scope

Let \(H\in\mathbb{R}^{n\times d}\) be a batch of hidden states at an adapter insertion point. A bottleneck adapter produces residuals

\[
R=f(HW_d)W_u.
\]

Define

\[
G=f(HW_d).
\]

Then

\[
R=GW_u.
\]

Since \(G\in\mathbb{R}^{n\times r}\), the residual matrix satisfies

\[
\operatorname{rank}(R)\le r.
\]

This is true regardless of whether \(f\) is identity, tanh, GELU, or ReLU. The bound is a statement about the rank of the realized residual matrix. It is not a statement that the residual is a static affine function of the input hidden states.

The replacement studied here is a low-rank affine regression model,

\[
\hat R_q=(H-\mathbf{1}\mu_H^\top)M_q+\mathbf{1}\mu_R^\top,
\]

where \(\operatorname{rank}(M_q)\le q\), \(\mu_H\) is the feature mean, and \(\mu_R\) is the residual mean. Equivalently,

\[
\hat R_q=HUV+\mathbf{1}b^\top,
\]

with \(UV=M_q\) and \(b=\mu_R-\mu_HM_q\). We call this a static LoRA+bias replacement because it has the same functional form as a low-rank linear update plus a bias shift. The name is descriptive only: the experiment does not train a LoRA module inside a pretrained language model.

The main metric is normalized residual reconstruction error,

\[
\epsilon=\frac{\lVert R-\hat R_q\rVert_F}{\lVert R\rVert_F}.
\]

We also report a synthetic teacher-student fidelity score. A random linear head maps \(H+R\) and \(H+\hat R_q\) to logits, and fidelity is the fraction of examples for which the predicted class is unchanged. This is not a downstream task metric; it is a coarse functional sensitivity check.

## 3. Linear Equivalence and Nonlinear Obstruction

### Proposition 1: identity adapters are static low-rank maps

If \(f(x)=x\), then

\[
R=HW_dW_u.
\]

Let

\[
M=W_dW_u.
\]

Because \(M\) factors through the adapter bottleneck dimension \(r\),

\[
\operatorname{rank}(M)\le r.
\]

Therefore,

\[
R=HM
\]

is exactly representable by a static rank-\(r\) affine replacement with zero bias. In this algebraic sense, a linear bottleneck adapter is LoRA-like.

### Proposition 2: nonlinear low-rank residuals need not be static affine maps

For nonlinear \(f\), the matrix \(R=f(HW_d)W_u\) still has rank at most \(r\), but the function \(H\mapsto R\) may be nonlinear. ReLU makes the obstruction explicit. For a single hidden state \(h\),

\[
\operatorname{ReLU}(hW_d)=hW_dD(h),
\]

where \(D(h)\) is a diagonal activation mask depending on \(h\). The residual is

\[
\Delta(h)=hW_dD(h)W_u.
\]

For each fixed \(h\), the effective map factors through dimension \(r\). But the matrix

\[
W_dD(h)W_u
\]

changes with the hidden state. A single static map \(hUV+b\) cannot generally reproduce this input-dependent behavior.

This leads to two testable predictions. First, once the replacement rank reaches the adapter rank \(r\), further increasing \(q\) should not remove error caused by non-affineness. Second, if the nonlinear adapter is behaving as a conditional low-rank map, piecewise affine replacements should reduce error for nonlinear activations.

## 4. Synthetic Study

Hidden states are sampled from a controlled low-dimensional Gaussian manifold:

\[
H=ZP+\sigma E.
\]

Here \(Z\in\mathbb{R}^{n\times k}\) is standard Gaussian, \(P\in\mathbb{R}^{k\times d}\) has orthonormal rows, \(E\) is standard Gaussian noise, \(k\) controls intrinsic dimension, and \(\sigma\) controls off-manifold noise.

The grid is:

- hidden dimension \(d=256\);
- adapter bottleneck rank \(r=16\);
- \(n_{\text{train}}=20{,}000\);
- \(n_{\text{test}}=5{,}000\);
- \(k\in\{4,8,16,32,64,128,256\}\);
- \(\sigma\in\{0,0.01,0.05\}\);
- activations identity, tanh, GELU, and ReLU;
- replacement ranks \(q\in\{1,2,4,8,16,32,64\}\);
- ten random seeds.

Adapter weights are random rather than trained. This makes the study a controlled mechanism test rather than a downstream fine-tuning evaluation.

The evaluated replacements are no-delta, bias-only, low-rank without bias, static LoRA+bias, full affine, and piecewise LoRA+bias. The piecewise model clusters adapter preactivations \(HW_d\) and fits a separate rank-\(q\) affine map per cluster.

## 5. Results

### 5.1 Static rank-16 replacement

At \(q=16\), static LoRA+bias gives the following test-set averages:

| activation | n | mean \(\epsilon_{\text{test}}\) | mean \(R^2_{\text{test}}\) | mean fidelity |
|---|---:|---:|---:|---:|
| identity | 210 | \(1.22\times 10^{-8}\) | 1.0000 | 1.0000 |
| tanh | 210 | 0.1087 | 0.9810 | 0.9637 |
| GELU | 210 | 0.3265 | 0.8643 | 0.9048 |
| ReLU | 210 | 0.4277 | 0.7318 | 0.8472 |

The identity result confirms the algebraic sanity check: a linear bottleneck adapter is recovered to numerical precision by a rank-16 static affine replacement.

Tanh is partly recovered. Its mean reconstruction error is much lower than GELU and ReLU, and its teacher-student fidelity remains high. However, the recovery is not exact and becomes worse as intrinsic dimension increases.

GELU and ReLU are not faithfully recovered by one static affine replacement in this synthetic setting. This does not mean they are high-rank as residual matrices. It means their residuals are not well approximated as a single affine function of \(H\).

### 5.2 Rank sweep and plateau

Mean test error for static LoRA+bias decreases as \(q\) approaches the adapter rank \(r=16\), then plateaus:

| rank \(q\) | identity | tanh | GELU | ReLU |
|---:|---:|---:|---:|---:|
| 1 | 0.8836 | 0.8862 | 0.8562 | 0.7570 |
| 2 | 0.7858 | 0.7904 | 0.7720 | 0.7023 |
| 4 | 0.5949 | 0.6035 | 0.6188 | 0.6137 |
| 8 | 0.3715 | 0.3886 | 0.4642 | 0.5181 |
| 16 | \(1.22\times10^{-8}\) | 0.1087 | 0.3265 | 0.4277 |
| 32 | \(1.18\times10^{-8}\) | 0.1087 | 0.3265 | 0.4277 |
| 64 | \(1.19\times10^{-8}\) | 0.1087 | 0.3265 | 0.4277 |

The plateau is important. Once \(q\ge r\), the static replacement has enough rank to match the residual output subspace available to the adapter. The remaining error for tanh, GELU, and ReLU is therefore not fixed by adding more static rank. It is approximation error from fitting a nonlinear hidden-state function with one affine map.

### 5.3 Intrinsic dimension

At \(q=16\), the effect of intrinsic dimension differs by activation:

| activation | \(k=4\) | \(k=8\) | \(k=16\) | \(k=32\) | \(k=64\) | \(k=128\) | \(k=256\) |
|---|---:|---:|---:|---:|---:|---:|---:|
| identity | \(1.37e{-8}\) | \(1.43e{-8}\) | \(1.27e{-8}\) | \(1.33e{-8}\) | \(1.14e{-8}\) | \(1.11e{-8}\) | \(8.91e{-9}\) |
| tanh | 0.0182 | 0.0314 | 0.0471 | 0.0817 | 0.1264 | 0.1918 | 0.2646 |
| GELU | 0.1584 | 0.2104 | 0.2591 | 0.3341 | 0.3965 | 0.4503 | 0.4769 |
| ReLU | 0.4251 | 0.4286 | 0.4267 | 0.4283 | 0.4280 | 0.4282 | 0.4287 |

Tanh and GELU become less affine-compressible as the intrinsic dimension of the hidden-state distribution increases. ReLU is already poorly approximated at low intrinsic dimension and remains almost flat around 0.428 across the grid.

The tanh result is the most favorable case for the static story: at \(k=4\), the mean reconstruction error is 0.0182 and fidelity is 0.9940. By \(k=256\), error rises to 0.2646 and fidelity falls to 0.9163. This supports a local-geometry interpretation: smooth nonlinear adapters may look nearly affine on narrow hidden-state manifolds but less so as the sampled region broadens.

### 5.4 Piecewise affine maps

At \(q=16\), piecewise LoRA+bias gives:

| activation | static eps | \(K=2\) eps | \(K=4\) eps | \(K=8\) eps | \(K=8\) fidelity |
|---|---:|---:|---:|---:|---:|
| identity | \(1.22e{-8}\) | \(1.45e{-8}\) | \(1.76e{-8}\) | \(2.13e{-8}\) | 1.0000 |
| tanh | 0.1087 | 0.1089 | 0.1079 | 0.1069 | 0.9637 |
| GELU | 0.3265 | 0.2924 | 0.2642 | 0.2427 | 0.9278 |
| ReLU | 0.4277 | 0.3775 | 0.3396 | 0.3103 | 0.8883 |

Piecewise modeling gives little benefit for identity or tanh but reduces error for GELU and ReLU. This supports the conditional-map interpretation: nonlinear adapters can behave as different low-rank affine maps over different regions of preactivation space.

The improvement is not a free deployment win. A single static affine update can be merged once into a compatible linear computation. A piecewise update needs routing or cluster assignment, so it weakens the clean merge-once story.

## 6. Discussion

The central lesson is that residual rank and static mergeability are different properties. The adapter residual \(R=f(HW_d)W_u\) has rank at most \(r\) as a realized matrix, but static LoRA+bias must also approximate the mapping from hidden states to residuals by one affine function. Identity activation satisfies both conditions. Tanh partially satisfies the affine condition when hidden-state geometry is narrow. GELU and ReLU show substantial non-affine error in this synthetic setting.

The rank plateau strengthens this interpretation. If the problem were merely insufficient output rank, increasing \(q\) beyond 16 would reduce the error. It does not. The error that remains at \(q=16\) is the error of replacing \(H\mapsto f(HW_d)W_u\) with a single affine map.

The piecewise results clarify the failure mode for GELU and ReLU. Separate affine maps over preactivation regions reduce reconstruction error, which is consistent with nonlinear adapters acting as conditional low-rank maps. For ReLU, this matches the decomposition \(hW_dD(h)W_u\), where the effective low-rank matrix changes with the activation mask. For GELU, the transition is smooth rather than binary, but the same broad interpretation applies: the local linear response varies across hidden-state regions.

The practical implication is diagnostic rather than prescriptive. Before claiming that an adapter can be replaced by LoRA-like weights, one should test affine approximation quality on the relevant hidden-state distribution. Low trainable parameter count and low residual matrix rank are not sufficient evidence.

## 7. Threats to Validity

This study is synthetic. Hidden states are sampled from Gaussian low-dimensional manifolds rather than collected from transformer layers. Real transformer hidden states may have structure not captured by this generator.

The adapters are random, not task-trained. Training could make adapter residuals more affine, less affine, lower magnitude, more structured, or more aligned with downstream decision boundaries. The present results should therefore be read as a mechanism observation, not a claim about trained adapters in production models.

The teacher-student fidelity metric uses a random linear head. It measures whether a random downstream projection changes class predictions, not whether a real task loss or accuracy is preserved.

The experiment does not establish that activation entropy causes compression failure. In the current run, ReLU and GELU activation entropies are nearly constant, so entropy is not a meaningful explanatory variable. A follow-up experiment should vary preactivation shifts or scales to directly test whether stable activation regimes improve static affine compressibility.

Finally, "mergeable" is used in a functional sense. A low-rank affine residual can be merged only when it is aligned with an existing compatible linear-plus-bias computation. If the adapter is inserted at a location that does not correspond to such a computation, deployment requires additional architectural care.

## 8. Conclusion

A bottleneck adapter with identity activation is exactly LoRA-like: its residual is a static low-rank affine function of the hidden state. Nonlinear adapters are different. Their realized residual matrices remain rank-bounded by the adapter bottleneck, but the residual function need not be affine in the hidden states.

In this synthetic study, static LoRA+bias recovers identity adapters exactly, recovers tanh well only under narrow hidden-state geometry, and does not faithfully recover GELU/ReLU adapters with one static affine map. Increasing static rank beyond the adapter rank does not help, while piecewise affine models improve GELU and ReLU. The evidence therefore supports a narrow but useful claim: nonlinear adapter residuals should be evaluated as functions over the hidden-state distribution, not judged by output rank or parameter count alone.

## References

[1] Ning Ding et al. "Parameter-efficient fine-tuning of large-scale pre-trained language models." *Nature Machine Intelligence*, 2023.

[2] Neil Houlsby et al. "Parameter-Efficient Transfer Learning for NLP." ICML, 2019.

[3] Edward J. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR, 2022.

## Reproducibility Notes

The experiment reports summary tables in `paper/data/`. The current manuscript uses:

- `static_q16_summary.csv` for aggregate rank-16 static results;
- `static_q16_by_intrinsic_dim.csv` for intrinsic-dimension trends;
- `piecewise_q16_summary.csv` for piecewise rank-16 results;
- `static_by_rank_summary.csv` for the static rank sweep.
