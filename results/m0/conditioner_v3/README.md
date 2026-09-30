# M0-Core v3 results

Validation-only full-graph node-classification pilot: 3 datasets × 3 seeds × 4 fixed conditioner variants (36 runs).
All runs use `unified_full_graph_nc_v1`, select checkpoints by best validation accuracy, and set `task.evaluate_test=false`.
Paired differences are descriptive across three matched seeds. P0 rows are post-hoc only and were not used for training or model selection.
No LP, test benchmark, HPO, Stage III, routing, operator bank, or additional propagation modules are part of this phase.
See `docs/m0/conditioner_v3_design.md` for the fixed specification and `docs/m0/conditioner_v3_report.md` for the evidence review.

## Artifacts

- `pilot_metrics.csv`
- `pilot_summary.csv`
- `paired_comparisons.csv`
- `stage1_health.csv`
- `base_relation_diagnostics.csv`
- `conditioner_diagnostics.csv`
- `training_gradient_trace.csv`
- `conditioner_growth_trace.csv`
- `variation_decomposition.csv`
- `dynamicity_diagnostics.csv`
- `operator_diagnostics.csv`
- `operation_geometry.csv`
- `intervention_conditioner_off.csv`
- `intervention_step1_dynamic_off.csv`
- `intervention_frozen_query_attn.csv`
- `intervention_relation_shuffle.csv`
- `intervention_context_shuffle.csv`
- `intervention_operator_off.csv`
- `p0_stagewise_hardcases.csv`
- `runtime_memory.csv`
- `smoke_base_relation_diagnostics.csv`
- `smoke_conditioner_diagnostics.csv`
- `smoke_conditioner_growth_trace.csv`
- `smoke_dynamicity_diagnostics.csv`
- `smoke_operation_geometry.csv`
- `smoke_operator_diagnostics.csv`
- `smoke_stage1_health.csv`
- `smoke_training_gradient_trace.csv`
- `smoke_variation_decomposition.csv`
