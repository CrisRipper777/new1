# M0-S3 — Relation- and Operation-aware Interaction History Consolidation

## A. Git provenance

- Branch: `exp/m0_stage3_history`; frozen parent: `exp/m0_conditioner_v3` at `108368f72c85441f580a1dd13b9720058915d57b`.
- New outputs use dedicated `stage3_history` paths; v3 code, checkpoints, and results were not overwritten.

## B. Protocol/code audit

- `unified_full_graph_nc_v1`; all run manifests set `task.evaluate_test=false`; checkpoint selection is best validation accuracy.
- The NC/LP task code, data splits, early stopping, checkpoint selection, and metrics were not changed.
- P0 is post-hoc. No LP, test evaluation, HPO, History Transformer, GPR, operator bank, MoE, extra propagation, or auxiliary loss was used.

## C. Files added/modified

- Model/components/config/tests: `src/models/interaction_full_s3.py`, `src/models/interaction_history_components.py`, `configs/model/interaction_full_s3.yaml`, `tests/test_interaction_full_s3.py`.
- Run/analyze/summarize: `scripts/run_m0_stage3.py`, `scripts/analyze_m0_stage3.py`, `scripts/summarize_m0_stage3.py`.
- Design/report/results: `docs/m0/stage3_history_design.md`, `docs/m0/stage3_history_report.md`, `results/m0/stage3_history/`; checkpoints/logs are in ignored local `outputs/m0/stage3_history/`.

## D. Frozen Stage-I/II verification

Stage I and Stage II are inherited from `interaction_core_v3` with `variant=context_bilinear_absolute`; no alternate relation encoder or semantic operator is introduced. Shared checkpoint weights are loaded into the terminal S3 model and compared on identical features/edges.

## E. Exact operation-profile formulation

For each modality and interaction step, the profile is `concat(incoming_mean(a), incoming_population_std(a))` with width 64. Degree-zero nodes receive zero mean and zero standard deviation. It summarizes the modulation actually used by the Stage-II low-rank semantic operator.

## F. Exact interaction-history-token formulation

Tokens are ordered `T0,T1,T2,V0,V1,V2`. T0/V0 encode intrinsic H0 plus learned NO_OPERATION; later tokens encode H1/H2, transitions `D1=H1−H0` and `D2=H2−H1`, and the corresponding 64-d mean/std operation profile. State-only mode substitutes learned NULL_OPERATION profiles.

## G. Exact readout-query formulation

The query is LayerNorm of a learned global vector plus projected H0 text/visual semantics and projected 128-d relation environments. State-only and operation-only modes use learned NULL_REL_ENV vectors. The readout is one four-head query-to-six-token cross-attention followed by LayerNorm; there is no query residual or attention stack.

## H. Four Stage-III variants

- `terminal`: returns the v3 fused terminal representation directly.
- `state_history`: uses H0/H1/H2 and transitions, with NULL_OPERATION and NULL_REL_ENV.
- `operation_history`: adds measured operation profiles, retaining NULL_REL_ENV.
- `relation_operation_history`: adds both real operation profiles and real relation environments.
All variants share a zero-initialized `W_hist`; initialization therefore equals terminal output exactly.

## I. Unit/regression tests

`tests/test_interaction_full_s3.py`: 27 passed. Coverage includes the 36 requested invariant groups, exact shared-weight v3 mapping, profile/environment statistics, intervention isolation, optimizer opening of W_hist, downstream gradients, attention normalization, and the NC output contract.

## J. Smoke

Movies seed 42, four variants, five epochs. Smoke confirmed finite losses/gradients, W_hist growth only in active history variants, delayed upstream gradients, normalized finite attention, and successful checkpoint/validation reload. Values are diagnostic only:

| Variant | Val Acc | Macro-F1 | epochs | mean epoch seconds | peak GPU MB |
|---|---:|---:|---:|---:|---:|
| terminal | 39.20 | 6.49 | 5 | 2.85 | 15299 |
| state_history | 38.90 | 6.42 | 5 | 2.24 | 11737 |
| operation_history | 39.14 | 6.50 | 5 | 3.26 | 11717 |
| relation_operation_history | 40.73 | 8.62 | 5 | 2.25 | 11717 |

## K. 36-run validation pilot

All 36 dataset × seed × variant records are present. Values are mean ± population SD across three seeds (%).

| Dataset | Variant | Val Acc mean ± SD (%) | Macro-F1 mean ± SD (%) |
|---|---|---:|---:|
| Movies | terminal | 56.26 ± 0.31 | 46.64 ± 0.67 |
| Movies | state_history | 54.54 ± 0.18 | 43.67 ± 2.54 |
| Movies | operation_history | 54.68 ± 0.19 | 44.81 ± 1.56 |
| Movies | relation_operation_history | 54.76 ± 0.07 | 44.51 ± 1.35 |
| Grocery | terminal | 83.11 ± 0.07 | 75.26 ± 0.58 |
| Grocery | state_history | 82.08 ± 0.52 | 74.44 ± 0.76 |
| Grocery | operation_history | 82.41 ± 0.43 | 74.99 ± 0.42 |
| Grocery | relation_operation_history | 82.54 ± 0.27 | 75.02 ± 0.55 |
| Reddit-S | terminal | 96.41 ± 0.16 | 92.40 ± 0.78 |
| Reddit-S | state_history | 96.15 ± 0.19 | 92.05 ± 0.37 |
| Reddit-S | operation_history | 96.26 ± 0.12 | 92.10 ± 0.15 |
| Reddit-S | relation_operation_history | 96.26 ± 0.08 | 92.07 ± 0.16 |

Paired contrasts are percentage-point differences across matched seeds:

| Dataset | Contrast | Δ Acc (pp) | positive pairs | Δ Macro-F1 (pp) |
|---|---|---:|---:|---:|
| Movies | A_state_history_vs_terminal | -1.720 | 0/3 | -2.967 |
| Movies | B_operation_history_vs_state_history | 0.140 | 2/3 | 1.144 |
| Movies | C_relation_operation_history_vs_operation_history | 0.080 | 2/3 | -0.299 |
| Grocery | A_state_history_vs_terminal | -1.035 | 0/3 | -0.814 |
| Grocery | B_operation_history_vs_state_history | 0.332 | 3/3 | 0.545 |
| Grocery | C_relation_operation_history_vs_operation_history | 0.127 | 2/3 | 0.036 |
| Reddit-S | A_state_history_vs_terminal | -0.262 | 1/3 | -0.343 |
| Reddit-S | B_operation_history_vs_state_history | 0.105 | 2/3 | 0.046 |
| Reddit-S | C_relation_operation_history_vs_operation_history | 0.000 | 1/3 | -0.031 |

## L. Stage-I/II regression health

The regression artifact contains 855 comparisons; every comparison passed `atol=1e-5, rtol=1e-5` for H0/H1/H2, relation memory and base codes, per-step modulation/execution code, and terminal fusion. The summary below describes the trained pilot checkpoints.

The exact-formula/weight mapping gate passed on all nine frozen v3 absolute checkpoints and all 36 trained S3 checkpoints. The table compares trained S3 Stage-I/II health with the prior v3 absolute pilot; some movement is expected from the changed end-to-end training objective and should be interpreted alongside exact code-path regression.

| Measure | v3 absolute mean | S3 pilot mean |
|---|---:|---:|
| relation feature variance | 0.700 | 0.668 |
| 1−cos(R_T,R_V) | 0.857 | 0.898 |
| reverse-edge 1−cos | 0.191 | 0.196 |
| operator deviation ratio | 0.357 | 0.123 |

Q6 answer: the exact Stage-I/II implementation and shared-weight mapping are preserved (all 855 regression rows passed), but trained health is not numerically unchanged: relation variance is 0.668 vs v3 0.700; modality discrepancy is 0.898 vs 0.857; reverse-edge discrepancy is 0.196 vs 0.191; and operator deviation is 0.123 vs 0.357. This is a training-induced diagnostic shift, with a marked reduction in operator deviation; it is not a code-path regression or a demonstrated Stage-II collapse.

## M. History-branch magnitude/growth

`rho_hist=||c_hist||/(||z_term||+eps)` distributions and selected checkpoint output norms:

| Variant | rho mean | median | p10 | p90 | p95 | mean ||c_hist|| | mean ||W_hist||_F |
|---|---:|---:|---:|---:|---:|---:|---:|
| terminal | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| state_history | 1.276 | 1.278 | 1.114 | 1.434 | 1.474 | 14.876 | 3.132 |
| operation_history | 1.351 | 1.350 | 1.186 | 1.517 | 1.561 | 15.760 | 3.321 |
| relation_operation_history | 1.294 | 1.294 | 1.134 | 1.455 | 1.501 | 15.060 | 3.158 |

Every active history variant started with exactly zero W_hist. The output-growth traces record epoch 0 plus each post-update epoch.

## N. Stage-III gradient health

Gradient RMS is pre-clipping. W_hist receives task gradient first; state/transition/operation token, query, and attention groups should activate after the branch opens.

| Parameter group | median first nonzero epoch | median trajectory RMS | median best-epoch RMS |
|---|---:|---:|---:|
| stage3_history_output | 1.0 | 5.13e-04 | 4.54e-04 |
| stage3_history_token_operation | 2.0 | 1.69e-04 | 1.91e-04 |
| stage3_history_token_state | 2.0 | 5.06e-04 | 5.21e-04 |
| stage3_history_token_transition | 2.0 | 9.46e-05 | 9.61e-05 |
| stage3_query_intrinsic | 2.0 | 6.00e-05 | 7.19e-05 |
| stage3_query_relation_env | 2.0 | 2.13e-05 | 2.45e-05 |
| stage3_readout_attention | 2.0 | 2.29e-04 | 2.30e-04 |

## O. Readout attention

Mean attention and per-node across-node standard deviation, averaged over trained dataset/seed runs:

| Token | mean attention | across-node std |
|---|---:|---:|
| T0 | 0.045 | 0.046 |
| T1 | 0.037 | 0.034 |
| T2 | 0.127 | 0.112 |
| V0 | 0.187 | 0.113 |
| V1 | 0.115 | 0.063 |
| V2 | 0.488 | 0.174 |

Dataset-level mean attention for the full candidate, averaged across its three seeds:

| Dataset | Token | mean attention | across-node std |
|---|---|---:|---:|
| Movies | T0 | 0.049 | 0.049 |
| Movies | T1 | 0.015 | 0.017 |
| Movies | T2 | 0.034 | 0.035 |
| Movies | V0 | 0.318 | 0.152 |
| Movies | V1 | 0.062 | 0.036 |
| Movies | V2 | 0.522 | 0.166 |
| Grocery | T0 | 0.061 | 0.064 |
| Grocery | T1 | 0.050 | 0.046 |
| Grocery | T2 | 0.204 | 0.129 |
| Grocery | V0 | 0.114 | 0.115 |
| Grocery | V1 | 0.069 | 0.058 |
| Grocery | V2 | 0.502 | 0.202 |
| Reddit-S | T0 | 0.039 | 0.048 |
| Reddit-S | T1 | 0.041 | 0.044 |
| Reddit-S | T2 | 0.130 | 0.159 |
| Reddit-S | V0 | 0.124 | 0.086 |
| Reddit-S | V1 | 0.181 | 0.076 |
| Reddit-S | V2 | 0.486 | 0.195 |

## P. Node/modality/stage preference heterogeneity

Attention entropy is computed per node. `preference_variance` is mean token variance across nodes; degree bins are descriptive, not a degree-aware model mechanism.

| Degree group | stage0 | stage1 | stage2 | text | visual | entropy | preference variance |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_positive_degree | 0.233 | 0.152 | 0.615 | 0.210 | 0.790 | 1.175 | 0.011 |
| 1 | 0.294 | 0.162 | 0.543 | 0.237 | 0.763 | 1.259 | 0.012 |
| 2 | 0.246 | 0.153 | 0.601 | 0.215 | 0.785 | 1.199 | 0.011 |
| 3-4 | 0.222 | 0.149 | 0.629 | 0.208 | 0.792 | 1.163 | 0.011 |
| 5-8 | 0.207 | 0.147 | 0.646 | 0.200 | 0.800 | 1.137 | 0.010 |
| 9-16 | 0.199 | 0.147 | 0.654 | 0.190 | 0.810 | 1.119 | 0.010 |
| 17+ | 0.194 | 0.144 | 0.662 | 0.186 | 0.814 | 1.104 | 0.009 |

## Q. History-off intervention

| Dataset | Intervention | Δ Val Acc (pp) | Δ Macro-F1 (pp) | logit L2 | flip rate | tokens L2 | history L2 | attention L1/L2 | query L2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Grocery | history_branch_off | -1.806 | -2.091 | 14.634 | 0.105 | 0.000 | 0.000 | 0.000/0.000 | 0.000 |
| Movies | history_branch_off | -3.389 | -1.348 | 8.712 | 0.300 | 0.000 | 0.000 | 0.000/0.000 | 0.000 |
| Reddit-S | history_branch_off | -0.902 | -1.427 | 14.661 | 0.027 | 0.000 | 0.000 | 0.000/0.000 | 0.000 |

## R. Operation-profile-off intervention

| Dataset | Intervention | Δ Val Acc (pp) | Δ Macro-F1 (pp) | logit L2 | flip rate | tokens L2 | history L2 | attention L1/L2 | query L2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Grocery | operation_profile_off | -0.312 | -0.455 | 2.684 | 0.035 | 9.537 | 3.552 | 0.103/0.057 | 0.000 |
| Movies | operation_profile_off | -0.600 | -0.910 | 0.845 | 0.044 | 8.490 | 2.315 | 0.226/0.142 | 0.000 |
| Reddit-S | operation_profile_off | -0.262 | -0.549 | 2.068 | 0.006 | 8.786 | 2.458 | 0.123/0.065 | 0.000 |

## S. Operation-history alignment intervention

| Dataset | Intervention | Δ Val Acc (pp) | Δ Macro-F1 (pp) | logit L2 | flip rate | tokens L2 | history L2 | attention L1/L2 | query L2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Grocery | operation_history_alignment_swap | 0.010 | 0.044 | 0.080 | 0.001 | 0.328 | 0.104 | 0.005/0.003 | 0.000 |
| Movies | operation_history_alignment_swap | -0.010 | -0.002 | 0.015 | 0.000 | 0.123 | 0.040 | 0.005/0.003 | 0.000 |
| Reddit-S | operation_history_alignment_swap | 0.000 | 0.000 | 0.016 | 0.000 | 0.149 | 0.021 | 0.003/0.002 | 0.000 |

## T. Relation-environment-off intervention

| Dataset | Intervention | Δ Val Acc (pp) | Δ Macro-F1 (pp) | logit L2 | flip rate | tokens L2 | history L2 | attention L1/L2 | query L2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Grocery | relation_environment_off | -0.878 | -1.214 | 2.690 | 0.046 | 0.000 | 3.834 | 0.475/0.249 | 12.475 |
| Movies | relation_environment_off | -0.570 | -1.687 | 0.746 | 0.042 | 0.000 | 1.831 | 0.230/0.129 | 10.994 |
| Reddit-S | relation_environment_off | -0.252 | -0.465 | 1.156 | 0.006 | 0.000 | 1.621 | 0.278/0.150 | 11.108 |

## U. Node-conditioning/global-query interventions

| Dataset | Intervention | Δ Val Acc (pp) | Δ Macro-F1 (pp) | logit L2 | flip rate | tokens L2 | history L2 | attention L1/L2 | query L2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Grocery | global_query_only | -1.103 | -1.098 | 3.775 | 0.056 | 0.000 | 5.406 | 0.542/0.283 | 21.423 |
| Grocery | node_intrinsic_query_off | -0.361 | -0.564 | 2.075 | 0.032 | 0.000 | 2.922 | 0.251/0.134 | 12.248 |
| Movies | global_query_only | -1.300 | -5.272 | 1.853 | 0.093 | 0.000 | 4.609 | 0.570/0.296 | 20.735 |
| Movies | node_intrinsic_query_off | -0.160 | -0.221 | 0.585 | 0.035 | 0.000 | 1.538 | 0.168/0.092 | 10.504 |
| Reddit-S | global_query_only | -0.577 | -0.907 | 3.587 | 0.015 | 0.000 | 4.777 | 0.623/0.324 | 21.159 |
| Reddit-S | node_intrinsic_query_off | -0.252 | -0.390 | 1.060 | 0.006 | 0.000 | 1.526 | 0.198/0.104 | 10.974 |

## V. Token-drop intervention

| Dataset | Intervention | Δ Val Acc (pp) | Δ Macro-F1 (pp) | logit L2 | flip rate | tokens L2 | history L2 | attention L1/L2 | query L2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Grocery | drop_T0 | -0.264 | -0.498 | 0.573 | 0.009 | 0.000 | 0.830 | 0.122/0.071 | 0.000 |
| Grocery | drop_T1 | -0.137 | -0.124 | 0.423 | 0.006 | 0.000 | 0.608 | 0.100/0.059 | 0.000 |
| Grocery | drop_T2 | -0.830 | -1.213 | 2.188 | 0.035 | 0.000 | 2.966 | 0.408/0.238 | 0.000 |
| Grocery | drop_V0 | -0.312 | -0.588 | 1.131 | 0.021 | 0.000 | 1.569 | 0.228/0.135 | 0.000 |
| Grocery | drop_V1 | -0.000 | 0.028 | 0.477 | 0.006 | 0.000 | 0.677 | 0.138/0.083 | 0.000 |
| Grocery | drop_V2 | -2.938 | -2.783 | 4.916 | 0.089 | 0.000 | 6.580 | 1.004/0.593 | 0.000 |
| Movies | drop_T0 | -0.210 | -0.057 | 0.354 | 0.021 | 0.000 | 0.902 | 0.099/0.057 | 0.000 |
| Movies | drop_T1 | 0.070 | 0.138 | 0.104 | 0.006 | 0.000 | 0.268 | 0.029/0.017 | 0.000 |
| Movies | drop_T2 | -0.090 | 0.031 | 0.238 | 0.015 | 0.000 | 0.631 | 0.068/0.039 | 0.000 |
| Movies | drop_V0 | -0.140 | 0.825 | 1.278 | 0.077 | 0.000 | 3.150 | 0.636/0.412 | 0.000 |
| Movies | drop_V1 | -0.100 | -0.346 | 0.186 | 0.009 | 0.000 | 0.530 | 0.124/0.075 | 0.000 |
| Movies | drop_V2 | -2.310 | -5.903 | 2.228 | 0.129 | 0.000 | 5.129 | 1.045/0.653 | 0.000 |
| Reddit-S | drop_T0 | -0.073 | -0.191 | 0.315 | 0.002 | 0.000 | 0.460 | 0.078/0.045 | 0.000 |
| Reddit-S | drop_T1 | -0.021 | -0.083 | 0.299 | 0.002 | 0.000 | 0.413 | 0.082/0.048 | 0.000 |
| Reddit-S | drop_T2 | -0.262 | -0.474 | 1.151 | 0.006 | 0.000 | 1.459 | 0.259/0.151 | 0.000 |
| Reddit-S | drop_V0 | -0.178 | -0.290 | 0.801 | 0.006 | 0.000 | 1.026 | 0.247/0.147 | 0.000 |
| Reddit-S | drop_V1 | -0.084 | -0.195 | 0.592 | 0.002 | 0.000 | 0.845 | 0.362/0.224 | 0.000 |
| Reddit-S | drop_V2 | -0.692 | -1.113 | 2.551 | 0.014 | 0.000 | 3.142 | 0.972/0.589 | 0.000 |

Token drops remove the requested memory item and renormalize attention over the remaining tokens. Changes are post-hoc diagnostics, not causal proof.
Q5 answer: V2 removal lowers accuracy by -1.980 pp on average and has the largest, consistent effect across all three datasets; T2 has a smaller, dataset-dependent effect (-0.394 pp), while the remaining tokens have weak or mixed effects. The pilot therefore shows a dominant useful visual state-2 contribution, not broad evidence that all histories are complementary.

## W. History-operation association

Spearman associations between history attention mass and operation intensity, state-transition norm, or incoming operation heterogeneity:

| Modality | state | association | mean Spearman |
|---|---:|---|---:|
| text | 1 | operation_intensity | -0.003 |
| text | 1 | state_transition_norm | 0.006 |
| text | 1 | operation_heterogeneity | -0.125 |
| text | 2 | operation_intensity | 0.018 |
| text | 2 | state_transition_norm | 0.176 |
| text | 2 | operation_heterogeneity | -0.034 |
| visual | 1 | operation_intensity | -0.045 |
| visual | 1 | state_transition_norm | -0.090 |
| visual | 1 | operation_heterogeneity | -0.020 |
| visual | 2 | operation_intensity | 0.059 |
| visual | 2 | state_transition_norm | 0.010 |
| visual | 2 | operation_heterogeneity | 0.293 |

P0 post-hoc sampled-node summaries, grouped by relation utility and similarity quintile, are in `p0_history_readout.csv`.

| Similarity | utility group | mean T/V state1 attention | mean T/V state2 attention | mean op heterogeneity (s1/s2) |
|---|---|---:|---:|---:|
| Q1 | beneficial | 0.063 | 0.326 | 0.175 |
| Q1 | harmful | 0.061 | 0.327 | 0.184 |
| Q5 | beneficial | 0.071 | 0.340 | 0.143 |
| Q5 | harmful | 0.047 | 0.269 | 0.145 |

## X. Q1: interaction-history complementarity conclusion

Q1 answer: state_history is -1.005 pp below terminal on mean accuracy, with 1/9 positive matched seeds. Yet turning the full history branch off in the trained full checkpoint costs -2.032 pp and flips 0.144 of validation predictions, so the full model uses its history branch. Usage within that checkpoint does not establish net complementary value over terminal; the cross-variant pilot favors terminal.

## Y. Q2: operation-profile conclusion

Q2 answer: operation_history recovers 0.192 pp over state_history (7/9 positive seed pairs), and replacing profiles with NULL_OPERATION costs -0.391 pp with measurable representation/attention changes. But swapping p0/p1 alignment changes accuracy by only -0.000 pp and flips 0.000 of predictions. The profile affects readout behavior, while evidence that it helps match each state to the operation that produced it is negligible.

## Z. Q3: relation-environment conclusion

Q3 answer: adding relation environments changes validation accuracy by only 0.069 pp over operation_history (5/9 positive pairs). Removing the environment from the trained full checkpoint costs -0.567 pp on average and changes its query/attention, so the query uses that signal; the between-variant comparison does not show additional task value from including it.

## AA. Q4: adaptive-preference conclusion

Q4 answer: the full readout is modality/stage-skewed (mean visual mass 0.792; stage-2 mass 0.626) and varies across nodes (preference variance 0.012, entropy 1.162 nats). However, removing intrinsic query terms costs only -0.258 pp on average; q0-only costs -0.993 pp but removes both node and relation terms. This supports nonuniform readout behavior, with modest isolated evidence for intrinsic node conditioning.

## AB. What is supported

The exact inherited Stage-I/II equivalence, zero-init terminal identity, Stage-III branch use, operation/relation-conditioned changes to the readout, and a dominant V2 token contribution are directly testable from committed CSVs and checkpoints/logs. The between-variant validation results still favor the terminal baseline.

## AC. What is not supported

Three validation seeds do not establish significance or universal ranking. Token-drop and P0 summaries are post-hoc and do not establish causal history utility. Attention differences alone do not prove that the model uses the producing operation correctly.

## AD. Recommendation for final M0 architecture

Recommendation: retain the frozen two-stage Interpret→Execute architecture for final M0 and do not include ROHC in the default model. The full history branch is used, and V2 has a clear token-drop effect, but every dataset's state_history accuracy is below terminal; operation profiles recover only a small part of that loss, correct operation-to-state alignment has essentially no effect, and relation environments do not improve the full-versus-operation validation comparison. This fails the predeclared full-ROHC task-value gate. Keep the Stage-III results as diagnostic evidence, stop for human review, and do not run LP, test evaluation, HPO, or deeper history modules.
