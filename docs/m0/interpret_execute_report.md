# M0-Core — Contextual Relation Interpretation + Dynamic Semantic Execution

## A. Git provenance

- Starting commit was `30c0a3ce446c093c739c6097d3c13f4f56ac6904` on `exp/m0_stage1_relation`; the worktree was clean.
- The requested `exp/m0_interpret_execute` branch was absent locally and on `origin`. It was created from the clean, specified Stage-I HEAD. Stage-I files and artifacts remain unchanged.
- The complete experiment package is committed on this branch; its tip hash is recorded in the delivery handoff.
- Remote: `origin` (`CrisRipper777/new1`); push target: `exp/m0_interpret_execute`.

## B. Existing-code audit

The existing factory loads `src.models.<name>.Model`; the existing NC loop uses full-graph training, best validation accuracy, and the configured patience/split/metric protocol. M0-Core supplies the same `forward -> (z, None, None, aux_loss, extra)` contract and uses the existing data loader, optimizer, checkpoint, and inference paths. The shared `canonicalize_physical_edges`, `incoming_degree`, `incoming_mean`, and exact recipient leave-one-out utility were reused without changing M0-S1.

The runner reuses `scripts/run_m0_single.py`, whose NC label-list override uses the dataset-declared 20 classes and does not inspect test labels to form the Macro-F1 label set. All pilot invocations set `task.evaluate_test=false`; checkpoint metadata has no test metric keys. The previous model `src/models/interaction_m0.py`, its tests, and `results/m0/stage1_relation/` were not modified.

## C. Files added/modified

Added model and config files:

- `src/models/interaction_core.py`
- `src/models/interaction_core_components.py`
- `configs/model/interaction_core.yaml`
- `tests/test_interaction_core.py`

Added experiment tools and documentation:

- `scripts/run_m0_core.py`
- `scripts/analyze_m0_core.py`
- `scripts/summarize_m0_core.py`
- `docs/m0/interpret_execute_design.md`
- `docs/m0/interpret_execute_report.md`

Results are under `results/m0/interpret_execute/`. Validation-selected checkpoints and run logs are under the git-ignored `outputs/m0/interpret_execute/` tree. The prior model and results were left intact.

## D. Exact Stage-I formulation

Text and visual inputs are split in the loader's `[text | visual]` order and projected independently:

```text
H0_text = Projector_text(X_text)
H0_visual = Projector_visual(X_visual)
U_text = W_rel_text H0_text
U_visual = W_rel_visual H0_visual
```

The canonical graph removes self-loops, makes physical relations undirected, and coalesces duplicates. Its edge convention is `src=j, dst=i`, meaning `j -> i`. For each directed edge:

```text
sT = cosine(U_i_text, U_j_text)
sV = cosine(U_i_visual, U_j_visual)
d  = |sT - sV|
E_text = PairEncoder_text([U_i_text, U_j_text, |U_i_text-U_j_text|,
                           U_i_text*U_j_text, sT, sV, d])
E_visual = PairEncoder_visual([U_i_visual, U_j_visual, |U_i_visual-U_j_visual|,
                               U_i_visual*U_j_visual, sT, sV, d])
```

Each pair encoder maps `4*64+3` to 64 dimensions. Recipient context is the exact one-hop leave-one-out mean over the other incoming neighbors' `U` projections; degree-one recipients use learned modality-specific `NO_CONTEXT` vectors. Pair-only execution uses learned `NULL_CONTEXT` vectors instead of real recipient context.

For contextual variants, text relation evidence queries `[E_visual, C_text, C_visual]`; visual evidence queries `[E_text, C_text, C_visual]`. Each side uses one two-head cross-attention block with residual, dropout, and LayerNorm. The outputs `R_text,R_visual` form `[E,2,64]` relation memory. This memory is computed once from `H0` per model forward and held fixed across the two interaction steps. `global_dynamic` broadcasts learned text and visual relation slots and does not run the pair/context encoder in normal forward computation.

## E. Exact Stage-II formulation

For dynamic variants and modality `m`, the current edge state gives:

```text
q_ji,k^m = LN(W_t^m H_i,k^m + W_s^m H_j,k^m + e_m)
xi_ji,k^m = CrossAttention(q_ji,k^m, [R_text, R_visual])
```

`context_static` substitutes a learned modality-specific static query. The source base message and conditional low-rank deviation are:

```text
z_j,k^m     = W_msg^m H_j,k^m
v_j,k^m     = W_down^m z_j,k^m
a_ji,k^m    = tanh(W_a^m xi_ji,k^m)
Delta m     = W_up^m(a_ji,k^m * v_j,k^m)
m_ji,k^m    = z_j,k^m + Delta m
```

`operator_rank=32`. `W_up.weight` and `W_up.bias` start at zero. Incoming messages use mean aggregation; degree-zero nodes receive a zero aggregate. Each independent modality stream updates with `LayerNorm(H + Dropout(W_u GELU(M)))`. The two steps share all Stage-II weights. Only terminal text and visual states are fused through `Linear(512,256) -> LayerNorm -> GELU -> Dropout`.

No relation vector is added as message content. No cross-modality graph message is sent.

## F. Variant definitions

| Variant | Relation memory | Execution query | Intended contrast |
|---|---|---|---|
| `global_dynamic` | Two learned global slots | Dynamic | Edge-specific relation specificity control |
| `pair_dynamic` | Pair evidence with NULL context | Dynamic | Pair specificity without recipient context |
| `context_static` | Pair evidence with real LOO context | Static | Context without current-state query |
| `context_dynamic` | Pair evidence with real LOO context | Dynamic | Full M0-Core |

The planned comparisons were used: `pair_dynamic vs global_dynamic`, `context_dynamic vs pair_dynamic`, and `context_dynamic vs context_static`.

## G. Zero-initialized operator verification

Unit tests verified zero `W_up` weight/bias and exact `base + delta == base` at initialization. The conditional branch had finite, nonzero gradients. All four smoke checkpoints had nonzero text and visual `W_up` after training. Smoke analysis showed trained deviation ratios from 0.0067 to 0.0122, so the branch moved away from its initial identity behavior. In the selected pilot checkpoints, the context-dynamic mean deviation ratios ranged from 0.223 to 0.285 across modality/step averages.

## H. Unit/regression tests

- Before implementation, `tests/test_interaction_m0.py`: 20 passed.
- Before implementation, `tests/test_protocol.py tests/test_inference_equivalence.py`: 23 passed.
- After implementation, the combined new M0-Core tests and those regressions: **69 passed**.
- `py_compile` and `git diff --check` passed.
- Tests ran through `/home/m3/miniconda3/envs/yhf_env/bin/python` (Python 3.12), which has the project dependencies. The shell's default `pytest` entrypoint points to an unrelated Python 3.8 install and was not used for the final test run.

## I. Smoke results

Movies, seed 42, four variants, five epochs each. All losses/metrics were finite, gradients were finite, each trained operator became nonzero, no NaNs were observed, and the relation and operation diagnostics were finite. These short smoke metrics are engineering checks only.

| Variant | Val Acc | Val Macro-F1 | Active params | Total params | Peak GPU delta |
|---|---:|---:|---:|---:|---:|
| global_dynamic | 39.142% | 6.479% | 932,436 | 1,040,980 | 6,827 MB |
| pair_dynamic | 39.142% | 6.479% | 1,040,596 | 1,040,980 | 9,219 MB |
| context_static | 39.172% | 6.488% | 974,804 | 1,040,980 | 8,457 MB |
| context_dynamic | 39.142% | 6.479% | 1,040,596 | 1,040,980 | 9,391 MB |

Smoke step0-to-step1 modulation changed in both modalities for every dynamic variant. The operator, modulation, and dynamicity smoke files are separate from pilot results.

## J. 36-run pilot metrics

All runs used `unified_full_graph_nc_v1`, seeds 42/43/44, and validation-accuracy checkpoint selection. The table reports mean ± population SD across seeds; the Macro-F1 value is measured at the selected validation-accuracy epoch.

**Validation accuracy (%)**

| Dataset | global_dynamic | pair_dynamic | context_static | context_dynamic |
|---|---:|---:|---:|---:|
| Movies | 56.149 ± 0.191 | 56.189 ± 0.375 | 56.149 ± 0.149 | 56.019 ± 0.403 |
| Grocery | 82.987 ± 0.402 | 82.811 ± 0.172 | 83.065 ± 0.276 | 82.811 ± 0.316 |
| Reddit-S | 96.299 ± 0.180 | 96.466 ± 0.306 | 96.571 ± 0.395 | 96.487 ± 0.306 |

**Validation Macro-F1 (%)**

| Dataset | global_dynamic | pair_dynamic | context_static | context_dynamic |
|---|---:|---:|---:|---:|
| Movies | 46.540 ± 1.053 | 46.916 ± 2.186 | 46.911 ± 1.807 | 48.179 ± 0.856 |
| Grocery | 74.596 ± 1.634 | 74.544 ± 1.394 | 75.461 ± 1.573 | 73.781 ± 1.236 |
| Reddit-S | 91.811 ± 0.478 | 92.579 ± 0.222 | 92.760 ± 0.478 | 92.344 ± 0.466 |

There were 36 complete checkpoints; best epochs ranged from 39 to 116. Active/total model-plus-classifier parameter counts were: global_dynamic `932,436 / 1,040,980`; pair_dynamic `1,040,596 / 1,040,980`; context_static `974,804 / 1,040,980`; context_dynamic `1,040,596 / 1,040,980`.

Runtime and baseline-subtracted peak GPU memory averaged across the 12 runs per dataset:

| Dataset | Mean seconds/epoch | Peak GPU memory delta |
|---|---:|---:|
| Movies | 1.19 s | 9,391 MB |
| Grocery | 1.07 s | 8,483 MB |
| Reddit-S | 2.03 s | 14,663 MB |

The monitor sampled the assigned GPU with `nvidia-smi` once per second. Full per-run timing and memory records are in `runtime_memory.csv`.

Paired validation-accuracy differences in percentage points (`left minus right`):

| Dataset | pair_dynamic − global_dynamic | context_dynamic − pair_dynamic | context_dynamic − context_static |
|---|---:|---:|---:|
| Movies | +0.040 (2/3 seeds positive) | −0.170 (1/3) | −0.130 (1/3) |
| Grocery | −0.176 (1/3) | +0.000 (2/3) | −0.254 (0/3) |
| Reddit-S | +0.168 (2/3) | +0.021 (2/3) | −0.084 (1/3) |

Across all nine matched dataset/seed pairs, the mean accuracy differences were +0.011 pp, −0.050 pp, and −0.156 pp respectively. These small, mixed task effects are interpreted together with the operation diagnostics and interventions below.

## K. Operator deviation diagnostics

The table averages per-run edge quantiles over datasets, seeds, modalities, and both steps. Full `mean/median/p10/p90` values remain in `operator_diagnostics.csv`.

| Variant | Mean | Median | P10 | P90 |
|---|---:|---:|---:|---:|
| global_dynamic | 0.171 | 0.158 | 0.087 | 0.280 |
| pair_dynamic | 0.272 | 0.249 | 0.121 | 0.456 |
| context_static | 0.343 | 0.330 | 0.156 | 0.542 |
| context_dynamic | 0.251 | 0.231 | 0.111 | 0.416 |

For context_dynamic, mean ratios were text step0/step1 `0.223/0.253` and visual step0/step1 `0.242/0.285`. The deviation is nonzero and materially varies by variant, modality, step, and edge.

## L. Operation diversity

Averaged across the nine runs, two modalities, and two steps:

| Variant | Feature variance | Edge variance | Modulation norm mean ± SD | `|a| > 0.95` |
|---|---:|---:|---:|---:|
| global_dynamic | 0.343 | 0.397 | 3.600 ± 0.246 | 7.98% |
| pair_dynamic | 0.398 | 0.442 | 3.788 ± 0.297 | 11.67% |
| context_static | 0.492 | 0.541 | 4.199 ± 0.359 | 20.79% |
| context_dynamic | 0.397 | 0.436 | 3.764 ± 0.298 | 11.11% |

Thus the learned modulation is not a constant across edges/features. Saturation is present but does not dominate the rank-32 coordinates.

## M. Dynamicity

For dynamic variants, these are averages across datasets, seeds, and edges. `cos` change is `1-cos(a_step0,a_step1)`; L2 is `||a_step1-a_step0||`.

| Variant | Modality | 1−cos mean / median / p90 | L2 mean / median / p90 |
|---|---|---:|---:|
| global_dynamic | text | 0.083 / 0.070 / 0.152 | 1.272 / 1.240 / 1.729 |
| global_dynamic | visual | 0.055 / 0.043 / 0.108 | 1.147 / 1.092 / 1.673 |
| pair_dynamic | text | 0.054 / 0.043 / 0.104 | 1.012 / 0.969 / 1.440 |
| pair_dynamic | visual | 0.039 / 0.026 / 0.084 | 0.944 / 0.877 / 1.462 |
| context_dynamic | text | 0.053 / 0.041 / 0.103 | 1.031 / 0.979 / 1.483 |
| context_dynamic | visual | 0.035 / 0.024 / 0.075 | 0.902 / 0.830 / 1.398 |

This establishes that the dynamic variants produce different same-edge operations at their two states. `context_static` is correctly omitted.

## N. Context intervention

On each trained context_dynamic checkpoint, replacing real recipient contexts by `NULL_CONTEXT` changed mean modulation by L2 `0.389/0.393` for text steps 0/1 and `0.690/0.676` for visual steps 0/1. On the nine validation-only comparisons, no-context minus full-context changed accuracy by `−0.151 pp`, Macro-F1 by `−0.139 pp`, mean val-logit L2 by `0.170`, and prediction flip rate by `0.99%`. Accuracy and Macro-F1 decreased in all nine matched runs, though the average metric changes are small.

Compared with `pair_dynamic`, full `context_dynamic` had −0.050 pp average accuracy and +0.088 pp Macro-F1 across matched runs, with mixed per-dataset results. This does not show a stable cross-dataset task gain from recipient context, while the operation intervention shows that context changes execution.

## O. Relation-specificity intervention

For the same nine context_dynamic checkpoints, replacing each modality's edge relation state by its current-graph per-modality mean changed modulation by mean L2 `1.986/2.031` for text steps 0/1 and `1.907/1.881` for visual steps 0/1. Mean-relation minus edge-specific validation changes were `−0.626 pp` accuracy and `−0.906 pp` Macro-F1; mean val-logit L2 was `0.682`, p90 `1.371`, and prediction flip rate `3.12%`. The edge-specific checkpoint performed better on all nine validation comparisons for both reported metrics. This is a same-checkpoint diagnostic intervention, not a new trained variant.

## P. Operator-off intervention

Setting `Delta m=0` in the same context_dynamic checkpoints reduced validation accuracy by `0.434 pp` and Macro-F1 by `1.155 pp` on average. Mean val-logit L2 was `1.182`, p90 `1.975`, and prediction flip rate `5.59%`. Full-operator validation metrics were higher in 8/9 matched runs. The trained executor therefore uses the conditional deviation, though the intervention does not isolate the causal effect of any one learned parameter.

## Q. P0 operation hard-case analysis

Frozen P0 utility edges were mapped post hoc by directed `(neighbor,target)` keys after checking each P0 target/neighbor sequence against its split cache. No P0 row entered training, loss, checkpoint selection, or hyperparameter choice. The table gives normalized beneficial/harmful centroid separation for context_dynamic operations; all entries have nine dataset/seed groups. Values are shown for the actual `a` vectors, separately by modality and step.

| Modality | Similarity regime | Step 0 | Step 1 |
|---|---|---:|---:|
| Text | Q1 | 0.179 | 0.189 |
| Text | Q5 | 0.372 | 0.392 |
| Visual | Q1 | 0.290 | 0.294 |
| Visual | Q5 | 0.525 | 0.517 |

For every variant, the P0 CSV also records beneficial/harmful centroid norms, within-group radii, and group-specific operator-deviation ratios. As a descriptive reference only, the prior Stage-I `pair_compat_context` relation-state separations were Text Q1/Q5 `0.116/0.420` and Visual Q1/Q5 `0.183/0.852`. The operation is more separated for both Q1 cases, slightly less for Text Q5, and less for Visual Q5. Therefore actual operations show utility-group structure, but are not uniformly more aligned across all hard cases. This descriptive distance comparison is not a causal result or a quality score.

## R. Q1 — Relation specificity

Task differences for pair_dynamic versus global_dynamic are small and mixed by dataset: +0.040 pp Movies, −0.176 pp Grocery, and +0.168 pp Reddit-S accuracy. Across nine seeds the average is +0.011 pp. Functionally, the pair variant has nonconstant edge modulation and larger mean operator deviation than the global control (`0.272` versus `0.171`). In the same-checkpoint relation-mean intervention, edge specificity changes operations substantially and mean-relation replacement lowers validation metrics on all nine runs.

**Conclusion:** edge-specific interpretation is functionally active and affects predictions, but this pilot does not show a stable accuracy gain over global relation slots.

## S. Q2 — Recipient context

The context_dynamic-versus-pair_dynamic accuracy difference averages −0.050 pp and changes direction by dataset. The no-context intervention changes modulation in both streams and slightly lowers validation metrics in all nine context_dynamic checkpoints.

**Conclusion:** recipient context changes the interpreted operation and has small same-checkpoint validation sensitivity, but adding it did not produce a stable task improvement over pair-only interpretation.

## T. Q3 — State-conditioned execution

Context_dynamic is below context_static in mean accuracy on all three datasets (−0.130/−0.254/−0.084 pp); the nine-run average is −0.156 pp and only 2/9 paired accuracy differences favor dynamic execution. However, context_dynamic has nonzero same-edge step-to-step modulation change (`1-cos` mean 0.053 text and 0.035 visual).

**Conclusion:** current states do alter the executed operation, but this configuration did not improve task metrics over the static query control in this pilot.

## U. What the experiment supports

- The low-rank conditional branch starts as the identity and becomes nonzero during training.
- Relation specificity, recipient context, and current-state queries each change the operation in the expected controlled comparisons.
- The executor has measurable same-edge dynamicity and the model responds to operator-off and relation/context interventions.
- Functional changes and validation task changes are distinct: operations change more consistently than the task metrics improve.

## V. What the experiment does not support

- It does not establish a stable accuracy advantage for edge-specific interpretation, recipient context, or dynamic execution over their controls.
- It does not establish causal claims beyond the stated same-checkpoint intervention sensitivities.
- It does not establish test-set performance or generalization beyond these three datasets, seeds, and the fixed NC protocol.
- It does not imply that the prior relation interpretation stage is useless; the operation and intervention diagnostics show it can alter execution even when task gains are mixed.
- P0 hard-case separations are descriptive and are not training or selection evidence.

## W. Recommended next step

Stop M0-Core implementation here and review the committed artifacts and report manually. Decide whether the operation-level evidence justifies a separately specified follow-up; no additional model family or stage is started in this run.
