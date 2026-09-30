# M0 Prior Retention results

This directory contains validation-only outputs for the 36-run M0 Final-Readout Audit. All runs use best validation accuracy checkpoint selection and task.evaluate_test=false. No test split was accessed.

Primary files:
- pilot_metrics.csv: per-run validation metrics and parameter counts.
- paired_comparisons.csv and paired_comparison_summary.csv: paired descriptive contrasts by dataset and seed.
- stage12_preflight_regression.csv: frozen v3 preflight.
- stage12_regression.csv: frozen and trained Stage-I/II tensor regression.
- stage12_health.csv and stage12_health_summary.csv: relation/operator/execution health across four variants.
- stage12_v3_comparison.csv and stage12_v3_comparison_summary.csv: matched descriptive health comparison against frozen v3 absolute checkpoints.
- update_residual_diagnostics.csv and update_residual_summary.csv: actual injected residual diagnostics.
- prior_injection_diagnostics.csv and prior_injection_summary.csv: injection magnitude/alignment.
- prior_retention_diagnostics.csv: semantic drift and retention.
- gate_diagnostics.csv and gate_summary.csv: feature gate statistics.
- gradient_trace.csv and gradient_summary.csv: per-epoch gradients.
- context_branch_growth.csv and context_branch_growth_summary.csv: context branch parameter growth.
- intervention_context_off.csv, intervention_context_source_swap.csv, intervention_gate_one.csv, intervention_prior_off.csv, intervention_operator_off.csv, intervention_relation_shuffle.csv: same-checkpoint validation interventions.
- runtime_memory.csv: elapsed runtime and sampled GPU memory.
- pilot_summary.csv: descriptive mean and SD by variant.
- docs/m0/prior_retention_report.md: A-AF audit readout.

Metrics and interventions are descriptive, with no significance testing. Interventions are not causal estimates; prior-bypass-off is a strong/OOD perturbation. Full training checkpoints and logs are in outputs/m0/prior_retention/full and are not tracked in git.
