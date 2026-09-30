# M0 Final-Readout Audit: Prior-Retaining Context Injection

## A. Git provenance
Branch: exp/m0_prior_retention. Required parent: 108368f72c85441f580a1dd13b9720058915d57b (exp/m0_conditioner_v3). Stage-I/II implementation is frozen to the v3 absolute bilinear conditioner.

## B. Protocol audit
The pilot contains 36 unique completed checkpoints: three datasets (Movies, Grocery, Reddit-S), seeds 42/43/44, and four readout variants. Every run uses unified_full_graph_nc_v1, best validation accuracy checkpoint selection, and task.evaluate_test=false. Validation metrics were recomputed from each selected checkpoint. Test was not evaluated.

## C. Files added/modified
Added src/models/interaction_prior_retaining.py and src/models/prior_retention_components.py; configs/model/interaction_prior_retaining.yaml; tests/test_interaction_prior_retaining.py; scripts/run_m0_prior_retention.py, scripts/analyze_m0_prior_retention.py, and scripts/summarize_m0_prior_retention.py; docs/m0/prior_retention_design.md and this report; and the audit tables plus README under results/m0/prior_retention/. Full training logs/checkpoints are under outputs/m0/prior_retention/full/.

## D. Exact v3 Stage-I/II preservation
Frozen v3 preflight and trained checkpoint regressions contain 1215 tensor comparisons; all passed at absolute/relative tolerances 1e-5. Regressed tensors include H0/H1/H2, relation features and memory, base relation codes, execution codes, operator modulation, and messages. Stage health summaries are descriptive because the readout changes the objective path.

## E. Exact captured update-residual formulation
The update module computes one dropout residual U = Dropout(W_u GELU(M)), then returns LayerNorm(H + U) and captures that same tensor. No extra dropout call is used. The context summary is C_update = (U0 + U1)/2. Residual diagnostics are in update_residual_diagnostics.csv and update_residual_summary.csv.

## F. Prior-retaining readout formulation
For each modality, H0 is the topology-agnostic intrinsic semantic prior. A modality-specific one-layer ContextAdapter maps the accumulated update residual through Linear, LayerNorm, GELU, Dropout, Linear; its final projection is zero-initialized. The feature-wise gate is sigmoid(Linear([H0 || E])) with zero-initialized parameters. The final modality state is H0 + gate elementwise-multiplied by E, followed by the original v3 fusion without another LayerNorm.

## G. Four variants
terminal uses H2. prior_delta_gate uses H2-H0. prior_update_add uses the mean actual update residual without a gate. prior_update_gate uses the mean actual update residual with an independent text/visual feature gate. All share v3 Stage-I/II and fusion.

## H. Unit/regression tests
The requested unit/regression suite passed: 27 tests. It covers shared-weight v3 equivalence, exact terminal training RNG sequence, residual identity and single dropout, zero initialization, intervention routing, gradient paths, and NC output compatibility.

## I. Smoke
All four Movies seed-42 smoke jobs completed for five epochs with finite losses and complete gradient/context-growth traces. Active ContextAdapter outputs became nonzero; smoke runs are plumbing checks and are not used as evidence for architecture selection.

## J. 36-run validation pilot
All 36 validation-only runs completed. Selected checkpoints and validation reproductions are recorded in pilot_metrics.csv and validation_reproduction.csv. Accuracy and macro-F1 below are mean ± sample SD over the nine dataset-seed runs; these are descriptive summaries.

| Readout | Validation accuracy (%) | Validation macro-F1 | Active parameter range |
|---|---|---|---|
| terminal | 78.66 ± 17.63 | 0.7132 ± 0.2027 | 1031700-1031700 |
| prior_delta_gate | 78.28 ± 18.06 | 0.7056 ± 0.2148 | 1558548-1558548 |
| prior_update_add | 78.23 ± 17.95 | 0.7051 ± 0.2074 | 1295892-1295892 |
| prior_update_gate | 78.27 ± 17.94 | 0.7118 ± 0.2015 | 1558548-1558548 |

Paired validation-accuracy contrasts, in percentage points:

| Contrast | Mean ± SD | n |
|---|---|---|
| delta_gate_minus_terminal | -0.38 ± 0.51 | 9 |
| update_add_minus_delta_gate | -0.05 ± 0.43 | 9 |
| update_gate_minus_update_add | 0.04 ± 0.35 | 9 |
| update_gate_minus_terminal | -0.39 ± 0.55 | 9 |

## K. Stage-I/II regression health
All exact tensor regressions passed. The table compares nine matched frozen v3 absolute checkpoints with the nine trained prior_update_gate checkpoints. Variance and cosine definitions match the saved v3 diagnostics; the per-modality/per-step paired values are in stage12_v3_comparison.csv and its summary.

| Measure | v3 absolute mean | prior_update_gate mean | matched n |
|---|---|---|---|
| Relation feature variance | 0.6998 | 0.7537 | 18 |
| Cross-modal relation discrepancy (1-cosine) | 0.8570 | 0.9116 | 9 |
| Reverse-edge directionality (1-cosine) | 0.1913 | 0.2616 | 18 |
| Operator deviation ratio | 0.3569 | 0.1891 | 36 |
| Operation modulation variance | 0.4704 | 0.3931 | 36 |
| Execution-code variance | 0.9166 | 0.8938 | 36 |

The comparison is descriptive because the training objectives differ. Full four-variant health outputs are in stage12_health.csv and stage12_health_summary.csv.

## L. Update-residual health
Mean per-step residual norm is 3.82 for text and 3.86 for visual; mean U0/U1 cosine is 0.824 and 0.898, respectively. The ungated accumulated update-to-H0 norm ratio averages 1.084 for text and 0.754 for visual, while the final gated rho_context is lower. Feature/node variation and per-run values are in update_residual_diagnostics.csv and update_residual_summary.csv.

## M. Context-injection magnitude
rho_context is the norm of the gated injected residual divided by the H0 norm. Mean rho_context over full prior_update_gate checkpoints and both modalities: 0.4774. Raw distribution summaries are in prior_injection_diagnostics.csv. Compare rho to one when judging whether the correction remains subordinate to the intrinsic prior.

## N. Prior-retention / semantic-drift diagnostics
The analyzer reports cosine drift from H0 for the terminal state and the prior-retaining output. Mean retention gain (terminal drift minus prior-output drift) across modality-checkpoints: 0.1660. Per-checkpoint summaries and the fraction of nodes closer to H0 than H2 are in prior_retention_diagnostics.csv.

## O. Gate diagnostics
Mean checkpoint-level gate standard deviation is 0.1513. The text gate mean/std average 0.5208/0.1124; visual is 0.5392/0.1902. Saturation is rare; full distributions and residual alignment are in gate_diagnostics.csv and gate_summary.csv.

## P. Text/Visual asymmetry
Across nine full checkpoints, mean rho_context is 0.438 for text and 0.517 for visual; mean gate standard deviation is 0.112 for text and 0.190 for visual. Projector gradient RMS is finite throughout the trace for both modalities (nonzero fraction 1.0 in the aggregate trace). These differences are descriptive only; see update_residual_diagnostics.csv, prior_injection_diagnostics.csv, prior_retention_diagnostics.csv, gate_diagnostics.csv, projector_gradient_comparison.csv, and stage12_health_summary.csv.

## Q. Gradient-path diagnostics
Per-epoch gradient RMS by parameter group is preserved in gradient_trace.csv. The text and visual projector gradients are nonzero in every observed epoch for all four variants in the aggregate traces. gradient_summary.csv reports epoch-level RMS and per-run first-nonzero epoch ranges. This describes optimization flow and does not establish that gradient starvation was solved.

## R. Context-branch growth
Context adapter output and gate parameter norms by epoch are in context_branch_growth.csv. context_branch_growth_summary.csv aggregates parameter norms at each run's selected best-validation checkpoint and reports the per-run first-nonzero epoch range. Active variants open ContextAdapter W2 at epoch 1 and feature gates at epoch 2 in all nine runs; terminal keeps these unused branches at zero.

## S. Context-off intervention
Across nine prior_update_gate checkpoints, mean validation accuracy change -6.41 ± 2.48 pp; lower accuracy in 9/9 runs. See intervention_context_off.csv for individual outcomes. This is a same-checkpoint ablation of the injected context branch.

## T. Context-source substitution
The update-to-delta replacement is a post-hoc same-checkpoint substitution, not a retraining comparison or causal estimate. Outcomes are in intervention_context_source_swap.csv. The paired trained comparison prior_delta_gate versus prior_update_gate is reported separately in paired_comparisons.csv.

## U. Gate-one intervention
mean validation accuracy change -1.26 ± 0.72 pp; lower accuracy in 9/9 runs. Gate values and individual intervention effects are in gate_diagnostics.csv and intervention_gate_one.csv.

## V. Prior-off intervention
mean validation accuracy change -8.61 ± 4.63 pp; lower accuracy in 9/9 runs. This disables the direct H0 bypass while keeping the checkpoint and remaining branch fixed; it is a strong/OOD intervention, not a causal estimate of prior utility.

## W. Operator-off regression
mean validation accuracy change -0.46 ± 0.34 pp; lower accuracy in 8/9 runs. Operator-off lowers validation accuracy in 8/9 checkpoints. This tests whether the original relation-conditioned semantic operator remains functionally consequential.

## X. Relation-shuffle regression
mean validation accuracy change -0.17 ± 0.19 pp; lower accuracy in 7/9 runs. Within-target relation-memory shuffling lowers validation accuracy in 7/9 checkpoints. It is a same-checkpoint structural sensitivity test.

## Y. Q1 Prior-retention conclusion
prior_update_gate is lower than terminal in 7/9 paired comparisons, with mean validation accuracy difference -0.39 pp. Mean semantic drift is lower than terminal by 0.166 cosine-drift units and rho_context remains below one on average. Thus H0 retention is real and context is used, but this pilot does not show an accuracy improvement.

## Z. Q2 Delta-vs-update-context conclusion
The trained prior_delta_gate minus prior_update_gate validation accuracy difference is 0.01 pp, effectively tied descriptively. In the same-checkpoint source substitution, replacing the trained update context with H2-H0 changes accuracy by mean validation accuracy change -0.83 ± 0.72 pp; lower accuracy in 9/9 runs. That intervention is post-hoc and uses an adapter trained on update residuals, so it is not a retraining comparison. The explicit residual remains the cleaner signal by definition, but this pilot does not show a reliable task advantage over delta.

## AA. Q3 Gate necessity conclusion
prior_update_gate minus prior_update_add is 0.04 pp in validation accuracy and 0.67 pp in macro-F1. Forcing the gate to one changes accuracy by -1.26 pp on average (lower in 9/9 checkpoints), and learned gates vary across features/nodes. This supports keeping the gate in the PRCI candidate, while the trained gate-vs-add accuracy gap alone is negligible and parameter counts differ.

## AB. Q4 Graph-context necessity conclusion
Context-off: mean validation accuracy change -6.41 ± 2.48 pp; lower accuracy in 9/9 runs. Nonzero rho_context is 0.4774; interpret this alongside the intervention and prior-retention ratios.

## AC. Q5 Interpret→Execute preservation conclusion
All Stage-I/II tensors regress exactly to v3 under shared weights. Operator-off lowers accuracy in 8/9 checkpoints (mean -0.46 pp); relation shuffle lowers it in 7/9 (mean -0.17 pp). The operator-off hit rate is 8/9 here versus the prior v3 pilot's reported 9/9; relation-memory shuffle lowers accuracy in 7/9. This supports continued mechanism use, with weaker descriptive intervention consistency than the earlier v3 result, not a claim that the readout improves the mechanism.

## AD. What is supported
The code exposes the exact injected Stage-II residual without changing its update computation, keeps a direct H0 path in all prior variants, and audits the resulting readouts on validation only. The pilot supports descriptive comparisons across the prespecified three datasets and three seeds.

## AE. What is not supported
No test-set claim, statistical-significance claim, causal claim from post-hoc interventions, gradient-starvation resolution claim, or generalization beyond the evaluated datasets is made. Smoke results are not scientific evidence.

## AF. Recommended final architecture
Recommendation: keep the v3 terminal readout as the current default and do not freeze PRCI as the final default from this pilot. prior_update_gate preserves H0 better (mean cosine-drift gain 0.166, average rho_context 0.477); context-off lowers accuracy in 9/9, and mechanism checks remain active (operator-off 8/9, relation shuffle 7/9). However, prior_update_gate is lower than terminal in 7/9 comparisons and averages -0.39 pp accuracy; all three prior variants are about 0.38–0.43 pp below terminal on average. The protocol set no non-inferiority margin, so the evidence cannot establish Case A. Because these are descriptive results without significance tests, they also do not establish a stable degradation under Case D. Treat PRCI as a functionally validated alternative for human review, while retaining v3 terminal as the conservative architecture recommendation.

No extra datasets, test, link prediction, or hyperparameter search were run. Stop this experiment here pending human review.
