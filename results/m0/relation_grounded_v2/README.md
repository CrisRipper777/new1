# M0-Core v2 results

Validation-only full-graph NC experiment: 3 datasets × 3 seeds × 4 fixed variants (36 runs).
All checkpoints are selected by validation accuracy and all runs set `task.evaluate_test=false`.
Paired contrasts are descriptive across seeds; P0 operation summaries are post-hoc only and were not used in training or selection.
See `docs/m0/relation_grounded_v2_report.md` for the A–AA evidence review and limitations.

## Artifacts

- `pilot_metrics.csv`
- `pilot_summary.csv`
- `paired_comparisons.csv`
- `stage1_health.csv`
- `stage1_attention.csv`
- `training_gradient_trace.csv`
- `operator_growth_trace.csv`
- `execution_attention.csv`
- `operator_diagnostics.csv`
- `operation_geometry.csv`
- `dynamicity_diagnostics.csv`
- `intervention_relation_shuffle.csv`
- `intervention_context_shuffle.csv`
- `intervention_frozen_query.csv`
- `intervention_operator_off.csv`
- `p0_operation_hardcases.csv`
- `runtime_memory.csv`
