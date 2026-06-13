# When Is an Adapter LoRA-Like?

## Static and Piecewise Low-Rank Compression of Synthetic Nonlinear Adapter Residuals

**Author:** Thatayotlhe Tsenang  
**Affiliation:** B.Sc. Mathematics Undergraduate, University of Botswana  
**Email:** 202406227@ub.ac.bw  
**Draft:** Preprint v0.2, June 2026

## Abstract

Parameter-efficient fine-tuning adapts large pretrained language models by modifying only a small set of parameters or a low-dimensional reparameterization of them. Adapters and LoRA are often discussed in the same parameter-efficient family, but they implement different forward-pass functions: a bottleneck adapter adds a residual module of the form

\[
R = f(HW_d)W_u,
\]

whereas a LoRA-style update adds a static low-rank affine contribution. This note studies a narrow synthetic question: when can an adapter residual be replaced by a static, mergeable, low-rank affine map? We separate two properties that are easy to conflate. For any activation `f`, the residual matrix `R = f(H W_d) W_u` has output rank at most the adapter bottleneck rank `r`; however, low output rank does not imply that `R` is affine in the hidden states `H`. In a controlled experiment with `d=256`, `r=16`, seven intrinsic dimensions, three noise levels, four activations, and ten seeds, a static rank-16 affine replacement recovers identity adapters to numerical precision, approximately recovers tanh adapters on average, but leaves substantial error for GELU and ReLU. Increasing the static replacement rank beyond 16 does not improve the results. Piecewise low-rank affine maps reduce error for GELU and ReLU, which is consistent with nonlinear adapters behaving as conditional low-rank maps rather than a single mergeable update. The claim is deliberately limited: these results are synthetic and use random, not task-trained, adapters.

## 1. Introduction

Large pretrained language models are expensive to adapt by full fine-tuning because full fine-tuning modifies the entire backbone and typically stores a separate adapted model per task. Parameter-efficient fine-tuning methods reduce this burden by updating a small part of the model, adding small trainable modules, or learning a low-dimensional parameterization of the update. Delta-tuning provides a useful umbrella view: a pretrained parameter set is adapted into a new parameter set, and the changed portion is the delta.

This structural view matters because parameter count alone does not determine what a method can represent. Addition-based methods, such as adapters, add new modules to the network. Specification-based methods, such as BitFit, update selected existing parameters. Reparameterization-based methods, such as LoRA, express weight changes through lower-dimensional factors. Two methods can therefore have comparable parameter counts while implementing different hidden-state transformations.

This paper focuses on one precise relationship: the connection between bottleneck adapters and LoRA-like static low-rank updates. A common adapter residual has the form

\[
h' = h + f(hW_d)W_u,
\]

where `h` is a hidden state, `W_d` is the down-projection, `W_u` is the up-projection, `r << d`, and `f` is an elementwise activation. A LoRA-style linear update contributes

\[
hUV.
\]

If the adapter activation is the identity, then

\[
h' = h + hW_dW_u,
\]

which is exactly a low-rank residual map with rank at most `r`. This observation is simple but important: a linear bottleneck adapter is LoRA-like in the algebraic sense.

The nonlinear case is the actual question. Although `f(HW_d)W_u` is always low-rank as a residual matrix, it may not be an affine function of the hidden states. A static LoRA-style update can represent only one affine map. A nonlinear adapter can behave like an input-dependent low-rank map. The goal of this note is not to propose a universal compression method. It is to test this distinction under controlled conditions and to identify which claims the synthetic evidence supports.

## 2. Problem Setup

Let `H` denote hidden states at an adapter insertion point. A bottleneck adapter produces the residual

\[
R = f(HW_d)W_u.
\]

Define

\[
G = f(HW_d).
\]

Then

\[
R = GW_u.
\]

Since `G` has at most `r` columns,

\[
rank(R) \le r.
\]

This bound holds for identity, tanh, GELU, ReLU, or any other elementwise activation.

However, `rank(R) <= r` does not imply that `R` is a static affine function of `H`. The replacement studied here is a centered low-rank affine model,

\[
\hat R_q = (H - \bar H)M_q + 1\bar R,
\]

with `rank(M_q) <= q`, equivalently written as

\[
\hat R_q = HUV + 1b^T.
\]

We call this a static LoRA+bias replacement. The term is descriptive: the experiment fits a closed-form low-rank affine approximation rather than training a LoRA module inside a language model.

The primary error metric is normalized residual reconstruction error,

\[
\epsilon = \frac{||R - \hat R||_F}{||R||_F}.
\]

We also report teacher-student fidelity using a random linear head.

## 3. Why the Rank Bound Is Insufficient

If `f` is the identity, then

\[
R = HW_dW_u.
\]

Let `M = W_dW_u`. Since `M` factors through the adapter bottleneck,

\[
rank(M) \le r.
\]

Thus a rank-`r` static affine map can recover the residual exactly up to numerical precision.

For nonlinear `f`, the residual matrix still satisfies `R = G W_u`, but `G = f(HW_d)` may not be affine in `H`. ReLU gives a clear example. For a single hidden state `h`,

\[
ReLU(hW_d) = hW_dD(h),
\]

where `D(h)` is a diagonal mask depending on `h`. The adapter residual becomes

\[
\Delta(h) = hW_dD(h)W_u.
\]

For each fixed `h`, this factors through rank at most `r`, but the effective matrix `W_dD(h)W_u` changes with the input. A single static update `hUV+b` cannot generally reproduce this conditional behavior.

This motivates two empirical tests. First, if the failure is not caused by insufficient output rank, then increasing `q` beyond `r` should not help. Second, if the failure is caused by multiple hidden-state regimes, then piecewise affine maps should reduce error for nonlinear activations.

## 4. Synthetic Experiment

Hidden states are generated as

\[
H = ZP + \sigma E,
\]

where `Z` is standard Gaussian, `P` has orthonormal rows, and `E` is standard Gaussian noise. The parameter `k` controls intrinsic dimension, while `sigma` adds off-manifold noise.

The experiment uses:

- hidden dimension `d = 256`;
- adapter bottleneck rank `r = 16`;
- `n_train = 20,000`;
- `n_test = 5,000`;
- intrinsic dimensions `k in {4,8,16,32,64,128,256}`;
- noise levels `sigma in {0,0.01,0.05}`;
- activations identity, tanh, GELU, and ReLU;
- ten random seeds.

Adapter weights are random, not trained. This makes the experiment a mechanism test rather than a downstream-task evaluation.

The evaluated replacement methods are: no delta, bias-only, low-rank without bias, static LoRA+bias, full affine, and piecewise LoRA+bias. Piecewise LoRA+bias clusters preactivations `HW_d` and fits one affine map per cluster.

## 5. Results

### 5.1 Static rank-16 replacement

At replacement rank `q=16`, static LoRA+bias gives:

| activation | n | mean epsilon_test | mean R2_test | mean fidelity_test |
|---|---:|---:|---:|---:|
| identity | 210 | 1.22e-8 | 1.0000 | 1.0000 |
| tanh | 210 | 0.1087 | 0.9810 | 0.9637 |
| GELU | 210 | 0.3265 | 0.8643 | 0.9048 |
| ReLU | 210 | 0.4277 | 0.7318 | 0.8472 |

Identity is recovered to numerical precision. This validates the algebraic sanity check: a linear bottleneck adapter is equivalent to a static low-rank affine residual.

Tanh is partially recovered. Its mean reconstruction error is much lower than GELU and ReLU, and its fidelity remains high, but the result is not exact and worsens as intrinsic dimension grows.

GELU and ReLU are not faithfully recovered by one static affine replacement in this synthetic setting.

### 5.2 Intrinsic dimension

For tanh and GELU, static replacement error increases as intrinsic dimension increases. For ReLU, error stays almost flat around 0.428 across the intrinsic-dimension grid.

| activation | k=4 | k=8 | k=16 | k=32 | k=64 | k=128 | k=256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| identity | 1.37e-8 | 1.43e-8 | 1.27e-8 | 1.33e-8 | 1.14e-8 | 1.11e-8 | 8.91e-9 |
| tanh | 0.0182 | 0.0314 | 0.0471 | 0.0817 | 0.1264 | 0.1918 | 0.2646 |
| GELU | 0.1584 | 0.2104 | 0.2591 | 0.3341 | 0.3965 | 0.4503 | 0.4769 |
| ReLU | 0.4251 | 0.4286 | 0.4267 | 0.4283 | 0.4280 | 0.4282 | 0.4287 |

The low-intrinsic-dimension story is supported for tanh and GELU, but not for ReLU in this setup.

### 5.3 Piecewise affine maps

Piecewise LoRA+bias improves GELU and ReLU. At rank `q=16`:

| activation | static eps | K=2 eps | K=4 eps | K=8 eps | K=8 fidelity |
|---|---:|---:|---:|---:|---:|
| identity | 1.22e-8 | 1.45e-8 | 1.76e-8 | 2.13e-8 | 1.0000 |
| tanh | 0.1087 | 0.1089 | 0.1079 | 0.1069 | 0.9637 |
| GELU | 0.3265 | 0.2924 | 0.2642 | 0.2427 | 0.9278 |
| ReLU | 0.4277 | 0.3775 | 0.3396 | 0.3103 | 0.8883 |

Piecewise modeling gives little benefit for identity or tanh but reduces error for GELU and ReLU. This supports the interpretation that nonlinear adapters can behave as conditional low-rank maps.

## 6. Discussion

The main lesson is that low output rank and static mergeability are different. The adapter residual `R=f(HW_d)W_u` always has rank at most `r` as a matrix, but a static LoRA+bias replacement must also be affine in `H`. The identity activation satisfies both properties. Tanh and GELU satisfy the rank property but become less affine-compressible as the hidden-state distribution broadens. ReLU satisfies the rank property but is poorly approximated by one static affine map throughout this experiment.

This distinction explains the observed rank plateau. Once `q` reaches `r=16`, the static affine path has enough rank to represent the best affine solution produced by the fitting procedure. More rank cannot correct the nonlinearity in `H -> f(HW_d)`. The remaining error is approximation error, not a rank-capacity error in the output residual matrix.

The piecewise results clarify the failure mode. When separate affine maps are fitted to regions of preactivation space, GELU and ReLU errors fall. This is consistent with the ReLU decomposition above: the adapter applies different effective low-rank transformations in different activation regimes. The cost is that the deployment story changes. A single static update can be merged once into a weight matrix; a piecewise update requires routing.

## 7. Limitations

This is a synthetic mechanism study, not a real language-model fine-tuning result. The hidden states are generated from controlled Gaussian low-dimensional manifolds rather than collected from a transformer. The adapters are random rather than trained on downstream tasks. The teacher-student fidelity metric uses a random linear head rather than a task loss. Therefore, the results do not establish that real PLM adapters behave the same way.

The experiment also does not justify a claim about activation entropy causing compression failure. In the run table, ReLU/GELU active fractions are close to one half and activation entropy is nearly constant. Since entropy barely varies, this run cannot test whether lower activation-regime switching improves static compressibility. A follow-up should add preactivation shifts or scale controls to vary activation patterns directly.

Finally, the static LoRA+bias replacement is fitted as a closed-form low-rank affine approximation to the adapter residual. It is LoRA-like in functional form, but it is not a full PEFT training procedure inside a pretrained language model.

## 8. Conclusion

A bottleneck adapter with identity activation is exactly LoRA-like: its residual is a static low-rank affine function of the hidden state. Nonlinear adapters are different. They produce residual matrices with rank at most the adapter bottleneck rank, but those residuals may not be affine functions of the hidden states. In this synthetic experiment, static LoRA+bias recovers identity adapters exactly, approximately recovers tanh under narrow hidden-state geometry, and does not faithfully recover GELU/ReLU adapters with one static affine map. Piecewise low-rank affine models improve GELU/ReLU, which is consistent with nonlinear adapters acting as conditional low-rank maps. Adapter compression should therefore be tested as an affine approximation problem on the hidden-state distribution, not inferred from trainable parameter count or output rank alone.

## Claim Audit

Supported by the synthetic run table:

- The rank-16 static LoRA+bias replacement and maximum-rank affine baseline agree up to numerical noise in this run.
- Identity adapters are recovered to numerical precision at `q=16`.
- Tanh and GELU static replacement error increases with intrinsic dimension.
- ReLU static replacement error remains near `0.428` across intrinsic dimensions.
- Piecewise low-rank affine fitting reduces error for GELU and ReLU.

Not claimed:

- Real pretrained language model adapters will show the same numerical behavior.
- Activation entropy predicts compressibility in the current run.
- Nonlinear adapters can generally be merged into LoRA without loss.
- The proposed diagnostic replaces downstream task evaluation.
