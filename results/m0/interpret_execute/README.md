# M0-Core pilot artifacts

Validation-only `unified_full_graph_nc_v1` pilot: Movies, Grocery, Reddit-S;
seeds 42/43/44; four fixed variants; 36 completed runs. Every run set
`task.evaluate_test=false`; best checkpoints were selected by validation accuracy.
No P0 artifact entered training or checkpoint selection. P0 operation rows are
post-hoc descriptions only. See `docs/m0/interpret_execute_report.md` for the
scientific interpretation and limits.

## Files

- `pilot_metrics.csv`
- `pilot_summary.csv`
- `paired_comparisons.csv`
- `operator_diagnostics.csv`
- `modulation_diagnostics.csv`
- `dynamicity_diagnostics.csv`
- `intervention_diagnostics.csv`
- `p0_operation_hardcases.csv`
- `runtime_memory.csv`
