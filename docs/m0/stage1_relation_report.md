# M0-S1 — Contextual Relation Interpretation: Stage Report

## A. Git provenance

- Working branch: `exp/m0_stage1_relation`.
- Frozen parent: P0.2 commit `8748867c6265aacedacb62d1a2af4a2aa1935ae7`.
- The remote branch pointed to the same frozen parent before implementation.
- Formal outputs are under `outputs/m0/stage1_relation/` and
  `results/m0/stage1_relation/`; P0 output directories were not modified.
- The implementation, tooling, and result artifacts are committed separately on this
  branch; their exact commit hashes are available in `git log`.

## B. Existing-code audit

`src.models.factory.build_model` dynamically imports `src.models.<name>` and calls
`Model(cfg, data_info)`. The NC loop performs one full-graph training forward per
epoch, evaluates full-graph embeddings, selects by validation accuracy, and calls
test evaluation only when `task.evaluate_test` is true. The model uses the
existing `forward -> (z, None, None, aux_loss, extra)` and `inference` contracts.

The existing NC metric helper built its Macro-F1 label set by indexing labels from
train, validation, and test. To keep the test sealed, the M0-only CLI wrapper uses
the dataset-declared `num_classes` instead. Train plus validation contain all 20
declared classes for each of the nine dataset/seed combinations, so this yields the
same label set without reading test labels. `src/tasks/nc.py`, `src/tasks/lp.py`,
P0 analysis, split code, and existing baseline models remain unchanged.

## C. Files added/modified

- Model: `src/models/interaction_components.py`,
  `src/models/interaction_m0.py`, `configs/model/interaction_m0.yaml`, and the
  Hydra package marker `configs/__init__.py`.
- Tests: `tests/test_interaction_m0.py`.
- Experiment tools: `scripts/run_m0_single.py`, `scripts/run_m0_stage1.py`,
  `scripts/summarize_m0_stage1.py`, and `scripts/analyze_m0_stage1.py`.
- Design and outputs: `docs/m0/stage1_relation_design.md`, this report, and the
  files listed in `results/m0/stage1_relation/README.md`.
- Validation-selected checkpoints and per-run logs are in
  `outputs/m0/stage1_relation/`.

## D. Exact M0.0 / M0.1-P / PC / PCC formulas

Text and visual features are sliced separately and encoded into
`H_0^T,H_0^V ∈ R^(N×256)`. Each modality is projected to `U^m ∈ R^(N×64)`. For a
directed physical relation `j → i` (`src=j`, `dst=i`):

```text
E^m_ji = phi_pair^m([U_i^m, U_j^m, |U_i^m-U_j^m|, U_i^m ⊙ U_j^m])
s^T_ji = cosine(U_i^T, U_j^T), s^V_ji = cosine(U_i^V, U_j^V)
E^C_ji = phi_C([s^T_ji, s^V_ji, |s^T_ji-s^V_ji|])
R_ji = Mixer([E^T_ji, E^V_ji, E^C_ji, C^T_{i\setminus j}, C^V_{i\setminus j}])[0:2]
m^m_ji,k = W_s^m H^m_j,k + W_r^m R^m_ji
M^m_i,k = mean_{j in N(i)} m^m_ji,k
H^m_i,k+1 = LayerNorm(H^m_i,k + Dropout(W_u^m GELU(M^m_i,k)))
Z_i = Fuse([H^T_i,2 || H^V_i,2])
```

The generic baseline sets both relation states to zero and uses the same two-step
executor. All message/update parameters are shared between the two steps within a
modality; text and visual executor parameters are independent. Fusion happens only
after the terminal states.

## E. Evidence masking implementation

`pair`, `pair_compat`, and `pair_compat_context` instantiate the same pair
encoders, compatibility encoder, learned null tokens, type embeddings, and
one-layer two-head mixer. The fixed token order is TextPair, VisualPair,
Compatibility, TextContext, VisualContext. `pair` replaces positions 2–4 with
learned nulls; `pair_compat` replaces positions 3–4; PCC uses all five real
evidence tokens. All model variants have the same state-dict structure and
940,948 total trainable parameters including the NC classifier.

## F. LOO recipient-context implementation

The model removes self-loops, symmetrizes, and coalesces the physical graph. For
each modality it sums `U_src` by `dst` and counts incoming degree. For `j → i`:

```text
degree(i) > 1: C_{i\setminus j} = (sum_{u in N(i)} U_u - U_j) / (degree(i)-1)
degree(i) = 1: C_i\j = learned NO_CONTEXT_m
degree(i) = 0: no relation edge; incoming aggregate remains zero
```

This is exact one-hop recipient-side mean LOO context. Tests check that the
subtracted vector is `U_j`, not `U_i`. No line graph or multi-hop pooling is used.

## G. Unit/regression test results

- New M0 tests: **20 passed**.
- M0 + NC protocol + inference compatibility tests: **43 passed**.
- `git diff --check`: passed.
- Before changes, the default system Python pytest failed because that Python 3.8
  environment lacks `exceptiongroup`. In `yhf_env`, the full suite both before and
  after implementation stops during collection because the unchanged
  `tests/test_multi_order_analyzer.py` imports missing
  `scripts.summarize_multi_order_bank_nc`.
- Excluding that collection-blocking file yields 103 passed and one unrelated
  existing P0.2 test failure: `conditional_node_bootstrap` expects
  `delta_mean_margin`, which `_batch_metric_values` does not return. Neither that
  test nor its analysis code is in this change.

## H. Movies seed42 smoke results

All four variants completed five full epochs with finite training loss, gradients,
embeddings, and diagnostics. Checkpoint selection was by validation accuracy at
epoch 5 in each engineering run; smoke accuracy is not used for research
conclusions.

| Variant | Smoke val accuracy | Epoch time estimate | Peak device delta |
|---|---:|---:|---:|
| generic | 39.44% | 1.23 s | 581 MB |
| pair | 39.23% | 0.50 s | 5,695 MB |
| pair_compat | 39.23% | 0.50 s | 7,475 MB |
| pair_compat_context | 39.29% | 0.25 s | 5,273 MB |

P/PC/PCC relation states had nonzero variance, relation/source message ratios
around 1.34–1.48, and PCC within-model context correction norms around 0.42–0.44.
Mean-head attention weights were returned only during analysis; normal forward uses
`need_weights=false`. Smoke outputs are under the `smoke_*` files and are separate
from the pilot conclusions.

## I. Full 36-run pilot results

Each cell is validation accuracy mean ± population SD over seeds 42/43/44. No test
metrics appear in run manifests or checkpoints; all 36 runs set
`task.evaluate_test=false`, and every checkpoint records
`selection=best_val_accuracy`. Best epochs ranged from 42 to 130.

| Dataset | generic | pair | pair_compat | pair_compat_context |
|---|---:|---:|---:|---:|
| Movies | 56.03 ± 0.33% | 55.46 ± 0.30% | 55.48 ± 0.29% | 55.54 ± 0.54% |
| Grocery | 83.45 ± 0.44% | 82.98 ± 0.52% | 82.85 ± 0.47% | 82.97 ± 0.44% |
| Reddit-S | 96.46 ± 0.28% | 96.24 ± 0.19% | 96.21 ± 0.13% | 96.21 ± 0.15% |

Paired mean accuracy differences in percentage points:

| Dataset | pair − generic | PC − pair | PCC − PC |
|---|---:|---:|---:|
| Movies | −0.57 | +0.02 | +0.06 |
| Grocery | −0.47 | −0.13 | +0.12 |
| Reddit-S | −0.22 | −0.02 | +0.00 |

Across all nine matched dataset/seed pairs, pair beat generic once; PC beat pair
twice; PCC beat PC four times. M0.0 had the highest mean validation accuracy on
all three datasets. Macro-F1 is also available in `pilot_metrics.csv` and
`pilot_summary.csv`; it does not change this ordering consistently.

## J. Runtime / GPU memory

Approximate epoch times (seconds), averaged over seeds from log timestamps, were
about 0.08 for generic; 0.50/0.52/0.84 for pair on Grocery/Movies/Reddit-S;
0.41/0.45/0.67 for PC; and 0.53/0.55/0.81 for PCC. Timestamp precision is one
second, so these are rough averages. The greatest baseline-subtracted assigned
device sample was 21,004 MB on a 24 GB RTX 3090; no run OOMed.

The full pilot used the first two-worker dispatcher, whose dynamic scheduling
occasionally let two processes share one physical GPU. Thus this pilot's
`peak_gpu_memory_mb` is an assigned-device sample during that run and can include
the sibling process. It is a conservative device-load measurement, not exact
per-process allocation. The reusable runner was corrected afterward to maintain
one sequential queue per GPU, and the smoke was rerun with that corrected queue.
No model result was selected using runtime or memory.

## K. Relation-state diagnostics

For P/PC/PCC, mean relation-vector norms were approximately 7.95–8.01 and mean
per-feature variance was nonzero across all dataset/modality groups (roughly
0.14–0.93). Thus the relation states did not collapse to a constant. Mean
cross-modal relation cosine was small and positive, varying by dataset/run rather
than indicating redundant modality states. Generic relation states are zero by
design and are not a learned representation baseline.

## L. Relation-message contribution diagnostics

The mean ratio `||W_r R_ji|| / (||W_s H_j,0|| + eps)` was approximately 1.25–1.47
for P/PC/PCC, with full mean/median/P10/P90 values in the diagnostics CSV. This
shows that the executor used the relation channel; its contribution was not
numerically negligible. PCC ratios were close to PC ratios, so context altered
relation states without consistently changing the aggregate contribution scale.

## M. P0 hard-case post-hoc analysis

The analyzer verified the frozen P0.2 target/neighbor sequence against the split
cache before mapping any relation. It computed beneficial/harmful centroids and
within-group radii for P0 Q1/Q5, separately by modality. Mean normalized
beneficial-vs-harmful separations across the three datasets and three seeds were:

| Modality / regime | pair | pair_compat | pair_compat_context |
|---|---:|---:|---:|
| Text Q1 | 0.1193 | 0.1187 | 0.1161 |
| Text Q5 | 0.4066 | 0.4080 | 0.4196 |
| Visual Q1 | 0.1827 | 0.1841 | 0.1832 |
| Visual Q5 | 0.8547 | 0.8693 | 0.8522 |

Context modestly raised Text-Q5 separation on average, but slightly lowered
Text-Q1 separation, barely changed Visual-Q1, and lowered Visual-Q5 relative to
PC. Dataset-specific effects are mixed, with large between-dataset variability.
These are descriptive representation distances, not a learned classifier or
ground-truth relation labels.

## N. Attention diagnostics

For PCC, text-query attention to VisualPair was roughly 0.22–0.26 across datasets;
attention to text/visual context tokens was about 0.20–0.23. Visual-query
attention to TextPair was about 0.17–0.24 and to context tokens about 0.19–0.22.
The five-token uniform reference is 0.20. Q1/Q5 utility-sign breakdowns show
some local differences, but no stable broad shift toward context. Overall context-key
means ranged from about 0.19 to 0.23 against a 0.20 uniform reference. Attention is a
descriptive internal statistic and does not establish causal use.

## O. Q1 Pair conclusion

**Functional value: yes.** Pair relation states are non-collapsed and the relation
message channel has substantial magnitude. **Task value: not supported here.**
Pair is below generic in mean validation accuracy on all three datasets (−0.22 to
−0.57 percentage points), with only one positive seed-level comparison out of
nine. Thus endpoint pair evidence changed the model behavior but did not improve
this downstream task under the current additive executor.

## P. Q2 Compatibility conclusion

**No consistent incremental task value.** PC minus P averages −0.04 pp on Movies,
−0.13 pp on Grocery, and −0.02 pp on Reddit-S; only 2/9 matched seed comparisons
are positive. Hard-case separation shifts are small and mixed. This does not
refute P0's association between compatibility and utility: this experiment tests a
learned compatibility token in one mixer/executor, not the P0 association itself.

## Q. Q3 Context conclusion

**Functional and behavioral value: present but modest.** PCC's same-checkpoint
context correction averages about 0.54–0.74 in relation-state norm. P0 separation
improves in Text Q5 on average but does not improve consistently across modalities
and regimes; context attention stays near the uniform reference. **Task value: no
stable gain.** PCC vs PC is +0.00 to +0.12 pp in dataset means and +0.06 pp overall,
with four positive comparisons among nine seed pairs. PCC remains below generic.

## R. What the experiment supports

- Explicit pair evidence produces a distinct, actively used relation channel.
- Compatibility/context tokens can alter relation states under a shared mixer.
- Recipient LOO context changes PCC relation states and can improve separation in
  some fixed P0 regimes.
- These functional changes did not yield a stable NC validation improvement with
  the current generic additive message interface.

## S. What the experiment does NOT support

This pilot does not show that contextual relation interpretation improves NC
accuracy, that compatibility or context is universally useful, or that attention
weights explain relation behavior. It does not establish causality or prove that an
operator bank, MoE, expert routing, state-conditioned execution, cross-modal graph
propagation, history readout, or any Stage-II/III feature is necessary. It covers
three datasets and three seeds with fixed hyperparameters and one executor.

## T. Recommendation for the next stage

Stop at M0-S1 and wait for human research review, as requested. Do not automatically
implement Stage II. The present result does **not** justify discarding all relation
evidence as functionally inert, but it also does not justify a larger recipient
context encoder on task value alone. If the research advisor elects to continue,
the next decision should explicitly address whether the additive executor is
limiting the behavioral gains before authorizing a separate Stage-II experiment.
