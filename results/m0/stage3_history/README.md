# M0-S3 — ROHC results

Validation-only `unified_full_graph_nc_v1`: 3 datasets × 3 seeds × 4 fixed variants (36 runs).
All run manifests set `task.evaluate_test=false`; checkpoint selection is best validation accuracy.
P0 is post-hoc only. No LP, test evaluation, HPO, History Transformer, operator bank, MoE, or extra propagation was run.
The Stage-I/II path is inherited from v3 `context_bilinear_absolute`; exact shared-weight regression is in `stage12_regression.csv`.
See `docs/m0/stage3_history_design.md` and `docs/m0/stage3_history_report.md`.

## Artifacts

- `pilot_metrics.csv`
- `pilot_summary.csv`
- `paired_comparisons.csv`
- `history_gradient_summary.csv`
- `stage12_regression.csv`
- `stage12_health.csv`
- `history_branch_diagnostics.csv`
- `history_gradient_trace.csv`
- `history_output_growth.csv`
- `history_attention.csv`
- `history_preference_heterogeneity.csv`
- `operation_profile_diagnostics.csv`
- `relation_environment_diagnostics.csv`
- `intervention_history_off.csv`
- `intervention_operation_profile_off.csv`
- `intervention_operation_alignment_swap.csv`
- `intervention_relation_env_off.csv`
- `intervention_node_conditioning_off.csv`
- `intervention_global_query.csv`
- `intervention_token_drop.csv`
- `history_operation_association.csv`
- `runtime_memory.csv`
- `p0_history_readout.csv`
- `smoke_stage12_regression.csv`
- `smoke_history_branch_diagnostics.csv`
- `smoke_history_attention.csv`
- `smoke_history_preference_heterogeneity.csv`
- `smoke_operation_profile_diagnostics.csv`
- `smoke_relation_environment_diagnostics.csv`
- `smoke_history_operation_association.csv`
- `smoke_stage12_health.csv`
