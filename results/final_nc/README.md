# Final NC Results README

This directory contains the locked five-dataset, three-seed, five-variant NC benchmark and final full-model mechanism audits. The final grid has 75 independent training runs. Full-graph training uses `unified_full_graph_nc_v1`; checkpoint selection is validation accuracy only. Each test metric was computed after restoring the selected checkpoint. No test result was used to select an epoch, variant, seed, or hyperparameter.

`final_nc_metrics.csv` contains all run metrics; `validation_reproduction.csv` and `test_reproduction.csv` independently reload every checkpoint. The full-model diagnostics cover all 15 Full checkpoints. Same-checkpoint interventions are diagnostic perturbations and are not causal contribution estimates. The external-baseline template is intentionally blank pending source-verified results on the same split and protocol.
