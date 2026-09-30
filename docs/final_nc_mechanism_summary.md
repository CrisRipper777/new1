# Final NC Mechanism Summary

All evidence below is descriptive. Trained-ablation contrasts compare separate validation-selected fits on matched seeds. Same-checkpoint interventions perturb a Full checkpoint and are not causal contribution estimates. Three seeds do not support a significance claim.

| Mechanism | Trained ablation: paired test delta vs Full (mean ± SD) | Win / tie / loss (15 pairs) | Same-checkpoint evidence |
|---|---:|---:|---|
| w/o Context | Acc 0.03% ± 0.61%; Macro-F1 0.04% ± 0.72% | Acc 7/1/7; F1 9/0/6 | context-health L2 change: mean logit L2 0.1186 ± 0.09124; test flip 0.57%; test Acc change 0.01% ± 0.17% |
| w/o Relation Specificity | Acc 0.11% ± 0.75%; Macro-F1 -0.12% ± 3.00% | Acc 12/0/3; F1 10/0/5 | within-target relation-memory shuffle: mean logit L2 0.1732 ± 0.1525; test flip 0.98%; test Acc change -0.06% ± 0.16% |
| w/o State Conditioning | Acc -0.04% ± 0.45%; Macro-F1 0.10% ± 0.91% | Acc 7/1/7; F1 7/0/8 | no direct same-checkpoint conditioner-off intervention was required; inspect measured bilinear correction and execution variation: Full mean ||xi-r||/||r|| 0.1305 ± 0.06617; execution-code variance 0.9018 ± 0.1417 |
| w/o Semantic Operator | Acc 0.05% ± 0.47%; Macro-F1 -0.02% ± 0.81% | Acc 8/2/5; F1 7/2/6 | same-checkpoint Delta=0 intervention: mean logit L2 1.399 ± 0.6062; test flip 4.90%; test Acc change -0.28% ± 0.26% |

Context evidence is based on degree-matched paired Text/Visual context shuffling and real-versus-null context relation changes. Relation-specificity evidence uses within-target cyclic relation-memory shuffling. Operator evidence compares Full with the same checkpoint evaluated at `Delta=0`. State conditioning has no separate same-checkpoint intervention in the frozen protocol; its mechanism audit is the measured bilinear correction, execution-code variation, and the trained `static_execution` contrast.

Per-dataset and per-seed results are in `paired_ablation_deltas.csv`, `context_health.csv`, `execution_health.csv`, `intervention_context_shuffle.csv`, `intervention_relation_shuffle.csv`, and `intervention_operator_off.csv`.
