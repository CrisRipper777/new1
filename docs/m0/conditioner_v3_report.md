# M0-Core v3 — Relation–State Conditional Execution Audit

## A. Git provenance

- Branch: `exp/m0_conditioner_v3`; frozen parent: `exp/m0_relation_grounded_v2` at `3e134eccec88605a2332bdaf15ea4c58e584298f`.
- Experiments were launched from the parent checkout after verifying a clean working tree; all v3 artifacts use dedicated `conditioner_v3` paths.

## B. Protocol/code audit

- Training protocol: `unified_full_graph_nc_v1`, NC full graph; all run records and resolved Hydra configs set `task.evaluate_test=false`.
- Checkpoints were selected by validation accuracy. The analyzer reproduced saved validation accuracy before any intervention.
- `src/tasks/nc.py`, `src/tasks/lp.py`, data splits, stopping, checkpoint selection, and metrics were not modified.
- P0 was post-hoc only. No LP, test evaluation, HPO, Stage III, routing, operator bank, or extra propagation was run.

## C. Files added/modified

New v3 implementation/config/tests: `src/models/interaction_core_v3.py`, `src/models/interaction_core_v3_components.py`, `configs/model/interaction_core_v3.yaml`, `tests/test_interaction_core_v3.py`.
New runners/docs: `scripts/run_m0_core_v3.py`, `scripts/analyze_m0_core_v3.py`, `scripts/summarize_m0_core_v3.py`, `docs/m0/conditioner_v3_design.md`, `docs/m0/conditioner_v3_report.md`.
Results: `results/m0/conditioner_v3/`; checkpoints and run logs: `outputs/m0/conditioner_v3/` (local, ignored by Git).

## D. Frozen Stage-I verification

The v3 path retains the v2 modality projectors, directed pair formula, exact incoming LOO context with degree-one `NO_CONTEXT`, symmetric cross-attention residual blocks, and `[R_T,R_V]` memory. The relation/context formulas are covered by regression tests.
Across the 36 selected checkpoints, mean relation feature variance was 0.685; rows distinguish modality, directionality, cross-modal discrepancy, and paired context-shuffle sensitivity.

## E. Frozen low-rank operator verification

The v2 source-only operator remains `z=W_msg H_j`, `a=tanh(W_a xi)`, `Delta=W_up(a*W_down z)`, `m=z+Delta`; rank 32 and zero `W_up` initialization are tested. Mean operator deviation ratio across reported modality/step rows was 0.338.
## F. Base relation code formulation

A learned modality-specific static query retrieves `r` from `[R_T,R_V]` using the bias-free, non-affine-normalized pure relation retriever. Zero relation memory gives exact zero `r`; no target-state residual or additive bypass exists.

## G. Four conditioner formulations

- `context_static`: `xi=r` at both steps.
- `context_attn_dynamic`: v2 `LN(W_q H_i,k + e_m)` retrieves from the same relation memory, with no query residual.
- `context_bilinear_absolute`: `xi=r+W_c(tanh(W_r r) ⊙ tanh(W_h H_i,k))`.
- `context_bilinear_delta`: `xi=r+W_c(tanh(W_r r) ⊙ tanh(W_h(H_i,k-H_i,0)))`; the analyzer verified exact step-0 equality at all nine checkpoints.

## H. Unit/regression tests

`tests/test_interaction_core_v3.py`: 26 passed. Coverage includes the 33 requested invariant groups, including initialization, grounding, target/source separation, gradients, exact step-0 behavior, output contract, and the variance identity.

## I. Smoke

All four Movies seed-42 five-epoch smoke runs completed with finite losses and validation metrics. `W_up` became nonzero in every variant; `W_c` became nonzero in both bilinear variants. Smoke values:

| Variant | Val Acc | Val Macro-F1 | epochs | mean epoch seconds |
|---|---:|---:|---:|---:|
| context_attn_dynamic | 0.398 | 0.071 | 5 | 2.45 |
| context_bilinear_absolute | 0.397 | 0.070 | 5 | 2.04 |
| context_bilinear_delta | 0.397 | 0.070 | 5 | 2.04 |
| context_static | 0.397 | 0.070 | 5 | 2.04 |

## J. 36-run validation results

All 36 dataset × seed × variant records are present. Values below are mean ± population SD across three seeds (percent).

| Dataset | Variant | Val Acc | Val Macro-F1 |
|---|---|---:|---:|
| Grocery | context_attn_dynamic | 82.83 ± 0.49 | 74.47 ± 2.29 |
| Grocery | context_bilinear_absolute | 83.16 ± 0.36 | 75.49 ± 1.83 |
| Grocery | context_bilinear_delta | 83.10 ± 0.37 | 74.87 ± 2.55 |
| Grocery | context_static | 83.02 ± 0.41 | 74.66 ± 2.32 |
| Movies | context_attn_dynamic | 56.40 ± 0.69 | 46.41 ± 1.33 |
| Movies | context_bilinear_absolute | 56.73 ± 0.46 | 47.48 ± 0.64 |
| Movies | context_bilinear_delta | 56.67 ± 0.37 | 48.35 ± 0.36 |
| Movies | context_static | 56.64 ± 0.37 | 47.61 ± 0.50 |
| Reddit-S | context_attn_dynamic | 96.24 ± 0.17 | 92.13 ± 0.14 |
| Reddit-S | context_bilinear_absolute | 96.53 ± 0.16 | 93.00 ± 0.28 |
| Reddit-S | context_bilinear_delta | 96.52 ± 0.19 | 92.93 ± 0.31 |
| Reddit-S | context_static | 96.47 ± 0.17 | 92.42 ± 0.13 |

Paired validation-accuracy contrasts are mean percentage-point differences, with positive seed-pair counts shown as `n/3`.

| Dataset | Contrast (left − right) | Acc difference (pp) | positive pairs | Macro-F1 difference (pp) |
|---|---|---:|---:|---:|
| Movies | A_attn_dynamic_vs_static | -0.240 | 1/3 | -1.205 |
| Movies | B_bilinear_absolute_vs_attn | 0.330 | 3/3 | 1.075 |
| Movies | C_bilinear_delta_vs_absolute | -0.060 | 1/3 | 0.866 |
| Movies | D_bilinear_delta_vs_static | 0.030 | 1/3 | 0.736 |
| Grocery | A_attn_dynamic_vs_static | -0.185 | 0/3 | -0.193 |
| Grocery | B_bilinear_absolute_vs_attn | 0.332 | 3/3 | 1.020 |
| Grocery | C_bilinear_delta_vs_absolute | -0.059 | 1/3 | -0.621 |
| Grocery | D_bilinear_delta_vs_static | 0.088 | 3/3 | 0.206 |
| Reddit-S | A_attn_dynamic_vs_static | -0.231 | 0/3 | -0.288 |
| Reddit-S | B_bilinear_absolute_vs_attn | 0.294 | 3/3 | 0.876 |
| Reddit-S | C_bilinear_delta_vs_absolute | -0.010 | 1/3 | -0.075 |
| Reddit-S | D_bilinear_delta_vs_static | 0.052 | 3/3 | 0.513 |

## K. Stage-I health

Relation feature variance mean: 0.685; mean `1-cos(R_T,R_V)`: 0.928; mean reverse-edge `1-cos`: 0.183. Removing context changes the relation correction by mean L2 1.556; degree-matched context shuffle changes it by mean L2 0.595. The Stage-I code is context-sensitive; task alignment is assessed separately in W.

## L. Base relation diversity

The base relation code `r` has mean feature variance 0.819 and mean code norm 8.000; per-variant and per-modality records are in `base_relation_diagnostics.csv`.

## M. Conditioner gradient/growth health

Gradient RMS is pre-clipping. The bilinear conditioner activates progressively: `W_up` has nonzero gradient from epoch 1, `W_c` from epoch 2, and the `W_r`/`W_h` branches from epoch 3 in all bilinear runs. The selected-epoch trace confirms the learned output path remains active.
- W_c output: first nonzero gradient epoch 2; median run-level trajectory RMS 9.65e-06; selected-epoch RMS 1.77e-05.
- W_r relation branch: first nonzero gradient epoch 3; median run-level trajectory RMS 7.04e-06; selected-epoch RMS 1.50e-05.
- W_h state branch: first nonzero gradient epoch 3; median run-level trajectory RMS 4.59e-06; selected-epoch RMS 1.22e-05.
- `context_bilinear_absolute`: 18/18 selected modality/checkpoint rows nonzero; mean Frobenius norm 1.260.
- `context_bilinear_delta`: 18/18 selected modality/checkpoint rows nonzero; mean Frobenius norm 0.872.

## N. Dynamic correction magnitude

Each row below averages the per-run/modality edge summaries; delta step 0 is exactly zero by construction and verified at all nine checkpoints. Absolute conditioning is active at both steps; delta conditioning is active at step 1. Correction sizes are measurable but moderate relative to the base relation code.

| Variant | Step | rho mean | median | p10 | p90 | p95 | mean ||delta_xi|| | mean ||W_c||_F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| context_bilinear_absolute | 0 | 0.106 | 0.097 | 0.045 | 0.183 | 0.208 | 0.848 | 1.260 |
| context_bilinear_absolute | 1 | 0.138 | 0.128 | 0.060 | 0.235 | 0.263 | 1.103 | 1.260 |
| context_bilinear_delta | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.872 |
| context_bilinear_delta | 1 | 0.128 | 0.127 | 0.073 | 0.186 | 0.197 | 1.026 | 0.872 |

## O. Total / within-target / between-target variance decomposition

Every `r`, `xi`, `a`, and operator `Delta_m` vector is decomposed across directed edges. The primary identity check is `V_total − V_within_target − V_between_target`; the node-balanced summaries are separately labeled descriptive.

## P. eta_relation / eta_target across r, xi, a, Delta_m

| Variant | Object | eta_relation | eta_target | identity error |
|---|---|---:|---:|---:|
| context_static | base_relation_r | 0.187 | 0.813 | -0.0000000 |
| context_static | execution_xi_step0 | 0.187 | 0.813 | -0.0000000 |
| context_static | execution_xi_step1 | 0.187 | 0.813 | 0.0000000 |
| context_static | operator_delta_message_step0 | 0.317 | 0.683 | -0.0000000 |
| context_static | operator_delta_message_step1 | 0.271 | 0.729 | -0.0000000 |
| context_static | operator_modulation_a_step0 | 0.186 | 0.814 | 0.0000000 |
| context_static | operator_modulation_a_step1 | 0.186 | 0.814 | 0.0000000 |
| context_attn_dynamic | base_relation_r | 0.207 | 0.793 | -0.0000000 |
| context_attn_dynamic | execution_xi_step0 | 0.179 | 0.821 | 0.0000000 |
| context_attn_dynamic | execution_xi_step1 | 0.177 | 0.823 | 0.0000000 |
| context_attn_dynamic | operator_delta_message_step0 | 0.290 | 0.710 | 0.0000000 |
| context_attn_dynamic | operator_delta_message_step1 | 0.257 | 0.743 | 0.0000000 |
| context_attn_dynamic | operator_modulation_a_step0 | 0.175 | 0.825 | -0.0000000 |
| context_attn_dynamic | operator_modulation_a_step1 | 0.171 | 0.829 | -0.0000000 |
| context_bilinear_absolute | base_relation_r | 0.188 | 0.812 | -0.0000000 |
| context_bilinear_absolute | execution_xi_step0 | 0.182 | 0.818 | -0.0000000 |
| context_bilinear_absolute | execution_xi_step1 | 0.179 | 0.821 | 0.0000000 |
| context_bilinear_absolute | operator_delta_message_step0 | 0.324 | 0.676 | -0.0000000 |
| context_bilinear_absolute | operator_delta_message_step1 | 0.278 | 0.722 | 0.0000000 |
| context_bilinear_absolute | operator_modulation_a_step0 | 0.182 | 0.818 | -0.0000000 |
| context_bilinear_absolute | operator_modulation_a_step1 | 0.180 | 0.820 | -0.0000000 |
| context_bilinear_delta | base_relation_r | 0.186 | 0.814 | -0.0000000 |
| context_bilinear_delta | execution_xi_step0 | 0.186 | 0.814 | 0.0000000 |
| context_bilinear_delta | execution_xi_step1 | 0.182 | 0.818 | -0.0000000 |
| context_bilinear_delta | operator_delta_message_step0 | 0.328 | 0.672 | 0.0000000 |
| context_bilinear_delta | operator_delta_message_step1 | 0.277 | 0.723 | 0.0000000 |
| context_bilinear_delta | operator_modulation_a_step0 | 0.184 | 0.816 | 0.0000000 |
| context_bilinear_delta | operator_modulation_a_step1 | 0.186 | 0.814 | 0.0000000 |

## Q. Dynamicity diagnostics

Means are across dataset/seed/modality rows; state changes are substantial, while conditioner changes differ by formulation:

| Variant | mean D_H | mean D_A | mean D_XI | Spearman(D_H,D_A) | Spearman(D_H,D_XI) |
|---|---:|---:|---:|---:|---:|
| context_static | 8.853 | 0.000 | 0.000 | NA | NA |
| context_attn_dynamic | 8.880 | 0.488 | 4.445 | 0.029 | 0.054 |
| context_bilinear_absolute | 8.924 | 0.206 | 1.098 | 0.157 | 0.083 |
| context_bilinear_delta | 8.899 | 0.363 | 1.010 | 0.064 | 0.141 |

## R. State change vs conditioner change association

The bilinear associations are weak: absolute has mean Spearman 0.158 for `D_H`–`D_A` and 0.083 for `D_H`–`D_XI`; delta has 0.064 and 0.141. A larger recipient state shift therefore does not consistently imply a larger conditioner or correction shift.

## S. Conditioner-off intervention

| Variant | Dataset | Intervention | Δ Val Acc (pp) | Δ Macro-F1 (pp) | flip rate | logit L2 | xi L2 | a L2 | Δ message L2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| context_bilinear_absolute | Grocery | conditioner_off | -0.185 | -0.173 | 0.010 | 0.262 | 1.233 | 0.665 | 0.550 |
| context_bilinear_absolute | Movies | conditioner_off | 0.040 | -0.060 | 0.005 | 0.066 | 0.530 | 0.241 | 0.099 |
| context_bilinear_absolute | Reddit-S | conditioner_off | -0.052 | -0.050 | 0.002 | 0.113 | 1.164 | 0.507 | 0.584 |
| context_bilinear_delta | Grocery | conditioner_off | 0.107 | 0.154 | 0.003 | 0.079 | 0.503 | 0.208 | 0.198 |
| context_bilinear_delta | Movies | conditioner_off | 0.020 | -0.039 | 0.002 | 0.023 | 0.321 | 0.118 | 0.042 |
| context_bilinear_delta | Reddit-S | conditioner_off | -0.021 | -0.052 | 0.001 | 0.054 | 0.716 | 0.222 | 0.302 |
The intervention produces visible code/logit changes, but validation accuracy effects are small and mixed (about −0.185 to +0.107 pp across dataset/variant cells); this is evidence of execution-path use, not strong task necessity.

## T. Step1-dynamic-off intervention

| Variant | Dataset | Intervention | Δ Val Acc (pp) | Δ Macro-F1 (pp) | flip rate | logit L2 | xi L2 | a L2 | Δ message L2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| context_bilinear_delta | Grocery | step1_recipient_state_change_off | 0.107 | 0.154 | 0.003 | 0.079 | 0.503 | 0.208 | 0.198 |
| context_bilinear_delta | Movies | step1_recipient_state_change_off | 0.020 | -0.039 | 0.002 | 0.023 | 0.321 | 0.118 | 0.042 |
| context_bilinear_delta | Reddit-S | step1_recipient_state_change_off | -0.021 | -0.052 | 0.001 | 0.054 | 0.716 | 0.222 | 0.302 |
For the delta model this intervention is numerically equivalent to conditioner-off: its only nonzero correction is the step-1 state-change term. Validation effects remain small and mixed.

## U. V2 frozen-query reference

| Variant | Dataset | Intervention | Δ Val Acc (pp) | Δ Macro-F1 (pp) | flip rate | logit L2 | xi L2 | a L2 | Δ message L2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| context_attn_dynamic | Grocery | step1_frozen_query_H0 | -0.029 | -0.226 | 0.006 | 0.148 | 0.536 | 0.273 | 0.288 |
| context_attn_dynamic | Movies | step1_frozen_query_H0 | -0.110 | 0.087 | 0.010 | 0.110 | 0.494 | 0.260 | 0.128 |
| context_attn_dynamic | Reddit-S | step1_frozen_query_H0 | 0.010 | -0.020 | 0.001 | 0.053 | 0.399 | 0.192 | 0.193 |

## V. Relation-alignment shuffle

| Variant | Dataset | Intervention | Δ Val Acc (pp) | Δ Macro-F1 (pp) | flip rate | logit L2 | xi L2 | a L2 | Δ message L2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| context_bilinear_delta | Grocery | within_target_relation_cyclic | -0.127 | -0.042 | 0.013 | 0.318 | 4.573 | 2.202 | 1.427 |
| context_bilinear_delta | Movies | within_target_relation_cyclic | -0.100 | -0.070 | 0.012 | 0.117 | 4.333 | 1.835 | 0.513 |
| context_bilinear_delta | Reddit-S | within_target_relation_cyclic | -0.031 | -0.068 | 0.000 | 0.039 | 2.043 | 0.965 | 0.937 |
Shuffling relation codes within each target changes execution code by mean L2 about 2.04–4.57 and changes modulation by 0.97–2.20, while validation accuracy drops only 0.031–0.127 pp. The execution is relation-alignment-sensitive, with modest task impact.

## W. Context-alignment shuffle

| Variant | Dataset | Intervention | Δ Val Acc (pp) | Δ Macro-F1 (pp) | flip rate | logit L2 | xi L2 | a L2 | Δ message L2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| context_bilinear_delta | Grocery | degree_matched_paired_context_shuffle | 0.166 | 0.341 | 0.010 | 0.196 | 0.997 | 0.558 | 0.379 |
| context_bilinear_delta | Movies | degree_matched_paired_context_shuffle | 0.090 | 0.136 | 0.007 | 0.076 | 0.999 | 0.440 | 0.123 |
| context_bilinear_delta | Reddit-S | degree_matched_paired_context_shuffle | -0.021 | -0.073 | 0.001 | 0.031 | 0.066 | 0.032 | 0.035 |
Degree-matched context shuffle changes the execution code by about 1.00 L2 on Movies/Grocery but only 0.066 on Reddit-S. Validation improves on Movies (+0.090 pp) and Grocery (+0.166 pp), and changes −0.021 pp on Reddit-S. Thus context alignment changes the operation, but these runs do not show that the original alignment improves task performance.

## X. Operator-off intervention

| Variant | Dataset | Intervention | Δ Val Acc (pp) | Δ Macro-F1 (pp) | flip rate | logit L2 | xi L2 | a L2 | Δ message L2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| context_bilinear_delta | Grocery | operator_off | -0.654 | -0.729 | 0.054 | 1.692 | 0.055 | 0.026 | 3.481 |
| context_bilinear_delta | Movies | operator_off | -0.700 | -1.099 | 0.068 | 0.806 | 0.008 | 0.003 | 1.284 |
| context_bilinear_delta | Reddit-S | operator_off | -0.566 | -1.397 | 0.020 | 1.704 | 0.061 | 0.025 | 6.539 |
Removing the operator lowers validation accuracy by 0.566–0.700 pp across datasets and also lowers macro-F1. The Stage-II operator has clearer task contribution than the dynamic conditioner.

## Y. Operation geometry

Across all variants/modalities/steps, mean `||Delta||/||z||` is 0.338, mean `cos(z,Delta)` is 0.117, mean message scale is 1.124, and mean `1-cos(z,z+Delta)` is 0.055. Reinforce/suppress/redirect thresholds are descriptive only.

## Z. P0 stage-wise hard-case analysis

P0 produced 792 post-hoc rows for Q1/Q5 beneficial/harmful groups. Mean normalized beneficial-vs-harmful centroid separation for the delta variant is:

| Similarity group | relation r | execution xi | modulation a |
|---|---:|---:|---:|
| Q1 | 0.202 | 0.206 | 0.212 |
| Q5 | 0.463 | 0.460 | 0.429 |

Q5 separation is larger than Q1 at all three stages; the analysis gives no evidence for an assumed Q1-over-Q5 ordering. Operator deviation averages 0.334 in both groups; geometry remains descriptive, and P0 is post-hoc rather than causal evidence.

## AA. Q1: attention-vs-bilinear conclusion

Absolute bilinear exceeds attention by 0.318 pp mean validation accuracy (all three matched seeds positive on each dataset); delta exceeds attention by 0.275 pp on average. The direction is consistent but the gains are small, so this is suggestive evidence for bilinear expressivity over the 2-slot attention baseline, not evidence of a large bottleneck or a significance claim.

## AB. Q2: absolute-vs-delta conclusion

Mean paired accuracy difference, delta minus absolute: -0.043 pp; delta is slightly lower on average. Its mean advantage over static is only 0.057 pp. Together with weak state-change associations and small step-1-off effects, there is no clear evidence that recipient-state-change conditioning adds value over absolute conditioning.

## AC. Q3: relation-specificity conclusion

For the bilinear variants, within-target relation variation remains around 18–19% in `r`/`xi`/`a`, while between-target variation accounts for roughly 81–82%. The operator difference `Delta_m` has a higher relation component (~28–33%), but between-target variation still dominates (~67–72%). Relation-specificity is retained rather than erased, yet target identity explains most edge-level variation.

## AD. Q4: dynamic-conditioner necessity conclusion

Conditioner-off and step-1-off produce nonzero execution/message changes and some prediction flips, but validation effects stay small and are mixed by dataset. The dynamic branch is used by the learned function; the intervention does not establish broad task necessity. Relation shuffle changes the operation more strongly than it changes accuracy.

## AE. What is supported

The implementation invariants, nonzero learned bilinear corrections, exact delta step-0 static behavior, and exact variance decomposition are directly supported. Absolute bilinear improves over attention by about 0.3 pp on each dataset across the three paired seeds. The Stage-II operator itself contributes to validation performance; dynamic conditioning changes execution but has only small and mixed same-checkpoint task effects.

## AF. What is not supported

Three validation seeds do not support a universal ranking or significance claim. The experiments do not show that correct paired-context alignment improves task performance, that delta conditioning is better than absolute conditioning, or that conditioner changes are broadly necessary for validation accuracy. P0 groups are post-hoc and do not establish causal edge utility.

## AG. Recommendation for Stage-I+II freeze

For the conservative M0 Stage-I+II freeze, use `context_static` relation-conditioned execution as the default. Keep absolute bilinear as an optional research comparator: it modestly beats attention, but same-checkpoint dynamic-off effects are small and target-level variation dominates. Do not claim an added benefit for delta conditioning. Stop here for human review; do not begin Stage III.
