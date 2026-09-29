# M0-S1 pilot artifacts

This directory contains the validation-only 36-run pilot for contextual relation
interpretation. It covers Movies, Grocery, and Reddit-S; seeds 42/43/44; and
`generic`, `pair`, `pair_compat`, and `pair_compat_context`. Every run used
`unified_full_graph_nc_v1`, `task.evaluate_test=false`, and a checkpoint selected
by best validation accuracy. The class list for Macro-F1 came from the declared
20-class dataset metadata; train plus validation labels cover all 20 classes for
all nine dataset/seed combinations. No test labels were indexed by the M0 run
wrapper and no test metrics or predictions were saved.

## Files

- `pilot_metrics.csv`: one validation-only row per run, including best epoch and
  model-plus-classifier parameter count.
- `pilot_summary.csv`: means and population standard deviations across the three
  seeds for each dataset and variant.
- `runtime_memory.csv`: average epoch time estimated from second-resolution log
  timestamps and peak assigned-device memory sampled with `nvidia-smi` at one
  second intervals, minus the idle baseline.
- `relation_diagnostics.csv`: relation-state magnitude/variance, cross-modal
  cosine, relation/source message ratio quantiles, context correction, and degree
  one rate.
- `p0_hardcase_diagnostics.csv`: Q1/Q5 beneficial/harmful relation-state centroid
  norms/radii and normalized centroid separation, using the frozen P0.2 relation
  population.
- `attention_diagnostics.csv`: descriptive mean-head attention summaries overall
  and by frozen P0 Q1/Q5 utility sign groups.
- `smoke_relation_diagnostics.csv`, `smoke_attention_diagnostics.csv`: engineering
  checks for the four Movies seed-42 five-epoch runs.

`NaN` context-correction values mean that a variant has no same-checkpoint
context-ablation comparison. Generic relation-state cosine/variance are zero by
construction. Attention is descriptive and is not a causal explanation.

## Runtime measurement note

The full pilot completed with the initial two-worker dispatcher. Its dynamic worker
scheduling allowed two runs to overlap on one physical GPU in some intervals, so
`peak_gpu_memory_mb` is the sampled device-level load during a run and can include
the concurrent sibling process. The maximum sampled baseline-subtracted device
load was 21,004 MB; neither RTX 3090 ran out of memory. The reusable runner was
subsequently corrected to keep one sequential queue per GPU, and the Movies smoke
was rerun with that queue. Epoch-time estimates have one-second timestamp
resolution and should be read as runtime summaries, not profiler-grade timings.

## Reproduction

From the repository root, with the `yhf_env` environment:

```bash
conda run -n yhf_env python -m pytest -q tests/test_interaction_m0.py
conda run -n yhf_env python scripts/run_m0_stage1.py --mode smoke --workers 2
conda run -n yhf_env python scripts/analyze_m0_stage1.py --mode smoke --device cuda:0
conda run -n yhf_env python scripts/run_m0_stage1.py --mode pilot --workers 2
conda run -n yhf_env python scripts/analyze_m0_stage1.py --mode pilot --device cuda:0
conda run -n yhf_env python scripts/summarize_m0_stage1.py
```

The post-hoc analyzer reads P0.2 outputs from
`/hdd1/DataInHere/YHF/mag_model/outputs/problem_validation` by default. It checks
the saved target/neighbor sequence against the frozen P0 split cache before mapping
relation states. P0 data are never inputs to training or checkpoint selection.

See [the stage report](../../../docs/m0/stage1_relation_report.md) for formulas,
tests, per-dataset results, interpretations, and limits.
