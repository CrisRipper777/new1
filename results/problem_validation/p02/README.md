# P0.2 Problem Validation

This directory contains an offline analysis of the fixed P0.1 validation-relation population. No model was trained, no relation was re-sampled, and no test labels or metrics were indexed or used.

## Analysis

- Input artifacts: the 15 `edge_analysis.pt`, `semantic_embeddings.pt`, and `splits/<dataset>.pt` files under `/hdd1/DataInHere/YHF/mag_model/outputs/problem_validation/`.
- Each split cache's stored target and directed relation arrays were checked for exact sequence equality against `edge_analysis.pt`.
- The current `new1` physical graph was checked against the frozen target degrees and directed relation endpoints.
- Recipient redundancy uses `cos(H_j^m, (d_i N_i^m - H_j^m)/(d_i-1))`; `d_i=1` rows are marked undefined and excluded only from this primary analysis. Coverage is in `p02_context_coverage.csv`.
- For each dataset, seed, and modality, P0.1+ stable-mergesort balanced quintiles of probe similarity are reused. Within each quintile, stable-mergesort balanced halves define low/high descriptor groups.
- Primary descriptor: recipient context redundancy. Secondary descriptor: absolute difference between text and visual probe similarities.
- CE utility is primary; margin utility is a robustness view. Differences are low descriptor minus high descriptor, with no assumed direction.
- Confidence intervals use 1,000 target-node bootstrap replicates (seed 42 by default); all sampled relations for a selected target are resampled together. Per-seed intervals are reported in `p02_per_seed.csv`.

## Files

- `p02_per_seed.csv`: Q1–Q5 and relation-count-weighted overall point estimates with per-seed node-bootstrap 95% CIs.
- `p02_cross_dataset.csv`: per-dataset means and population SDs over the three model seeds; these are not pooled inferential intervals.
- `p02_context_coverage.csv`: degree-one count/proportion and retained context-analysis coverage.
- `p02_counterexamples.csv`: descriptive redundancy and disagreement distributions in the predeclared Q1/Q5 sign subsets.
- `plots/`: per-dataset CE effect plots; error bars are across-seed SD.

## Reproduction

From the `new1` repository, run `PYTHONPATH=. python scripts/run_problem_validation_p02.py --input-root /hdd1/DataInHere/YHF/mag_model/outputs/problem_validation --split-root /hdd1/DataInHere/YHF/mag_model/outputs/problem_validation/splits`.
