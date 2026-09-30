# Final NC Architecture Freeze Audit

## A. Git provenance

- Branch: `exp/final_nc_freeze`.
- Required parent: `108368f72c85441f580a1dd13b9720058915d57b` (`exp/m0_conditioner_v3`).
- The final branch was checked out clean at that exact parent. No model structure was inherited from `exp/m0_stage3_history` or `exp/m0_prior_retention`.

## B. Architecture-freeze provenance

The paper-facing implementation is `src/models/final_interaction.py` plus `src/models/final_interaction_components.py`; its only benchmark configuration is `configs/model/final_interaction.yaml`. The locked equations and exclusions are recorded in `docs/final_model_spec.md` before smoke or final test evaluation. No hyperparameter search was run.

## C. NC protocol audit

Completed 75 runs for five datasets × three seeds × five variants. Protocol: `unified_full_graph_nc_v1`, full graph, AdamW, lr 0.001, weight decay 0.0001, maximum 300 epochs, validation each epoch, patience 30, minimum epoch 30, min delta 0.0001, gradient clip 1.0.

## D. Test sealing / no-leakage audit

Architecture and hyperparameters were frozen before final test evaluation. Every checkpoint uses `selection=best_val_accuracy`; the unchanged NC task restores that checkpoint before test evaluation. The dedicated runner fixes Macro-F1's class universe from dataset metadata and does not inspect test labels to choose it. Test metrics were reporting-only; they were not used to choose epochs, variants, seeds, or hyperparameters. No result-dependent rerun or seed cherry-picking occurred.

## E. Final model exact formulation

Stage I independently encodes text and visual features, constructs ordered pair evidence and recipient leave-one-out context, and uses cross-modal attention to form edge-specific relation memory. Stage II retrieves a grounded base relation code, applies an absolute bilinear relation × current-recipient-state conditioner, transforms source semantics through a rank-32 operator, mean-aggregates, and updates each modality independently for two shared-parameter steps. Terminal fusion reads only H2. Full equations and dimensions are in `docs/final_model_spec.md`.

## F. Five ablation definitions

`no_context` replaces LOO context with learned null tokens; `shared_relation` uses two learned global relation slots; `static_execution` sets ξ=r; `operator_off` sets Δ=0 while keeping z and the backbone. These are separate trained fits; same-checkpoint interventions are reported separately.

## G. V3→Final exact regression

The full variant maps all 70 shared parameters from `interaction_core_v3/context_bilinear_absolute`. All 28 required output checks passed at atol=rtol=1e-5 with maximum absolute error 0. The regression includes H0, pair/context evidence, relation outputs and memory, base codes, both-step execution codes/modulation/delta messages, H1/H2, and fused_z.

## H. Unit tests

`tests/test_final_interaction.py`: 3 passed. It checks v3 numerical equivalence, all five variant contracts/active sets, and gradient activation past the zero-initialized gates.

## I. Smoke

Movies seed 42, all five variants, five epochs each: complete. Losses and validation/test metrics were finite; all checkpoint audits passed. Smoke results are not scientific evidence.

## J. 75-run completion audit

Manifest contains 75 complete runs and the exact 5 × 3 × 5 grid. Each checkpoint records task, protocol, seed, validation-accuracy selection, selected epoch, validation/test accuracy and Macro-F1, model/head states, and data metadata.

## K. Validation reproduction

All 150 checkpoint validation metrics were recomputed after checkpoint reload. Maximum absolute error: 0.000e+00; tolerance statuses: {'pass_1e-6': 300}.

## L. Test reproduction

All 150 checkpoint test metrics were recomputed after checkpoint reload. Maximum absolute error: 0.000e+00; tolerance statuses: {'pass_1e-6': 300}. Test reproduction is an audit of saved predictions/metrics, not model selection.

## M. Full-model five-dataset validation results

See `docs/final_nc_main_table.md` for the supplementary validation table and `results/final_nc/final_nc_full_results.csv` for means and sample SDs.

| Dataset | Val Acc (%) | Val Macro-F1 (%) |
|---|---:|---:|
| Movies | 55.84 ± 0.53% | 46.18 ± 0.48% |
| Toys | 79.87 ± 0.21% | 76.72 ± 0.55% |
| Grocery | 83.43 ± 0.15% | 75.95 ± 1.37% |
| ele-fashion | 87.71 ± 0.19% | 69.05 ± 0.42% |
| Reddit-S | 96.51 ± 0.27% | 92.67 ± 0.63% |

## N. Full-model five-dataset test results

Test metrics are reported after restoring each best-validation checkpoint. The Average row is descriptive only.

| Dataset | Test Acc (%) | Test Macro-F1 (%) |
|---|---:|---:|
| Movies | 55.18 ± 0.48% | 45.92 ± 1.04% |
| Toys | 79.71 ± 0.25% | 76.57 ± 0.41% |
| Grocery | 81.73 ± 0.61% | 73.94 ± 1.48% |
| ele-fashion | 87.88 ± 0.22% | 70.10 ± 0.62% |
| Reddit-S | 96.61 ± 0.58% | 92.53 ± 1.48% |
| Average† | 80.22 ± 0.43% | 71.81 ± 1.01% |

† Descriptive macro-average across datasets; the SD shown is the average of within-dataset three-seed SDs, not pooled uncertainty.

## O. Trained ablation — validation

Per-dataset, per-variant validation metrics are in `docs/final_nc_ablation_table.md` and `results/final_nc/final_nc_ablation.csv`.

## P. Trained ablation — test

Per-dataset test accuracy and Macro-F1 mean ± sample SD for all five variants are in `docs/final_nc_ablation_table.md`.

## Q. Paired ablation deltas

`paired_ablation_deltas.csv` contains each matched dataset × seed delta and per-dataset plus all-15-pair mean, SD, and win/tie/loss. No Wilcoxon or t-test was used; no significance is claimed.

## R. Stage-I health

`stage1_health.csv` reports relation edge variance, cross-modal relation discrepancy, and directionality for 15 Full checkpoints × two modalities. Relation-variance collapse rows (≤1e-12): 0.

## S. Context health

`context_health.csv` compares real-context and null-context relation states per seed, modality, and quintile of |cos(text)-cos(visual)|. This is descriptive; no Q1>Q5 hypothesis is tested.

## T. Execution/operator health

`execution_health.csv` reports execution-code and modulation variance, bilinear correction, Δ/z, cos(z,Δ), and message rotation by modality and step. Zero-Δ rows after training: 0.

## U. Relation-specific variance decomposition

`variance_decomposition.csv` reports total, within-target, between-target, η_relation, and η_target for r, ξ, a, and Δ on each Full checkpoint. The primary within/between decomposition follows the population law of total variance.

## V. Relation-shuffle intervention

Within-target cyclic relation-memory shuffle results are in `intervention_relation_shuffle.csv`; the intervention preserves each recipient's relation-memory marginal and degree while disrupting source-edge alignment.

## W. Context-shuffle intervention

Degree-matched paired Text/Visual context shuffle results are in `intervention_context_shuffle.csv`; the intervention preserves paired modality context and exact recipient degree groups.

## X. Operator-off intervention

The same Full checkpoints were evaluated with Δ=0 while keeping base source messages, aggregation, and updates. Results are in `intervention_operator_off.csv`.

These three same-checkpoint perturbations are mechanism diagnostics, not causal estimates of module contribution. Their metrics are not used for model selection.

## Y. Parameter counts

`active_parameters.csv` lists model and classifier parameters by dataset/variant with active flags. `model_efficiency.csv` records active and total counts. Full has no inactive registered parameter modules; the expected inactive paths in ablations are explicitly marked.

| Variant | Active parameter range | Total parameter range |
|---|---:|---:|
| Full | 865,420–998,548 | 865,420–998,548 |
| w/o Context | 865,420–998,548 | 865,420–998,548 |
| w/o Relation Specificity | 757,260–890,388 | 865,548–998,676 |
| w/o State Conditioning | 840,844–973,972 | 865,420–998,548 |
| w/o Semantic Operator | 662,284–795,412 | 865,420–998,548 |

## Z. Runtime/memory

Full-graph epoch time includes per-epoch validation. Peak GPU memory is the assigned-device one-second `nvidia-smi` sample above baseline.

| Dataset | Mean epoch time (s) | Peak GPU memory mean / max (MiB) |
|---|---:|---:|
| Movies | 1.21 | 8593 / 8593 |
| Toys | 0.78 | 6907 / 6907 |
| Grocery | 0.89 | 7777 / 7777 |
| ele-fashion | 2.72 | 20605 / 20605 |
| Reddit-S | 1.70 | 13245 / 13245 |

## AA. Gradient-path sanity

Full training traces the seven required groups. Groups with no observed nonzero gradient in at least one Full dataset/seed run: [].

## AB. Movies conclusion

Movies: Full test Acc 55.18 ± 0.48%, Macro-F1 45.92 ± 1.04%; real-vs-null context relation change 1.714, mean Δ/z 0.1699, Δ η_relation 0.4211. Ablation Macro-F1 test-mean contrasts vs Full: w/o Context -0.16 pp, w/o Relation Specificity -1.59 pp, w/o State Conditioning +0.71 pp, w/o Semantic Operator -0.57 pp.

## AC. Toys conclusion

Toys: Full test Acc 79.71 ± 0.25%, Macro-F1 76.57 ± 0.41%; real-vs-null context relation change 1.474, mean Δ/z 0.1997, Δ η_relation 0.3231. Ablation Macro-F1 test-mean contrasts vs Full: w/o Context +0.07 pp, w/o Relation Specificity +0.77 pp, w/o State Conditioning +0.07 pp, w/o Semantic Operator +0.03 pp.

## AD. Grocery conclusion

Grocery: Full test Acc 81.73 ± 0.61%, Macro-F1 73.94 ± 1.48%; real-vs-null context relation change 1.674, mean Δ/z 0.434, Δ η_relation 0.3975. Ablation Macro-F1 test-mean contrasts vs Full: w/o Context +0.15 pp, w/o Relation Specificity -0.83 pp, w/o State Conditioning -0.37 pp, w/o Semantic Operator +0.84 pp.

## AE. ele-fashion conclusion

ele-fashion: Full test Acc 87.88 ± 0.22%, Macro-F1 70.10 ± 0.62%; real-vs-null context relation change 1.48, mean Δ/z 0.3376, Δ η_relation 0.2011. Ablation Macro-F1 test-mean contrasts vs Full: w/o Context +0.35 pp, w/o Relation Specificity +0.92 pp, w/o State Conditioning +0.15 pp, w/o Semantic Operator -0.42 pp.

## AF. Reddit-S conclusion

Reddit-S: Full test Acc 96.61 ± 0.58%, Macro-F1 92.53 ± 1.48%; real-vs-null context relation change 1.192, mean Δ/z 0.5116, Δ η_relation 0.124. Ablation Macro-F1 test-mean contrasts vs Full: w/o Context -0.20 pp, w/o Relation Specificity +0.13 pp, w/o State Conditioning -0.07 pp, w/o Semantic Operator +0.02 pp.

## AG. Which claims are supported

Evidence for these claims is descriptive: relation states vary across edges/recipients; real-context states differ from null-context states; trained Full checkpoints exhibit measured bilinear correction and source-operation magnitudes; the three same-checkpoint interventions quantify dependence on relation alignment, context alignment, and Δ. Dataset-specific support is reported in the linked CSV files. No claim of statistical significance is made.

## AH. Which claims are not supported

This benchmark does not establish statistical significance from three seeds, universal Full-model superiority, a causal percentage contribution for any module, or superiority over external baselines. A same-checkpoint conditioner-off intervention was not part of the frozen protocol. External-baseline values remain blank until their split/protocol sources are verified.

## AI. Final NC architecture-freeze verdict

**NC Architecture Freeze = FINAL.** The requested implementation, checkpoint, reproduction, and mechanism checks completed without a flagged path failure. Test results remain reporting-only.

## AJ. Readiness for LP

`docs/final_model_spec.md` is the architecture authority for a later LP phase. This round stops after the NC deliverables; LP training and new baseline runs were not started.

## Audit artifact index

Machine-readable results: `results/final_nc/README.md`, run manifest, metric/reproduction CSVs, tables, mechanism health and intervention CSVs, parameter/runtime records, and the blank external-baseline template.
