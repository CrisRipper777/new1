# M0-Core v2 — Relation-Grounded Semantic Execution: Final Report

## A. Git provenance

The work was performed in `/hdd1/DataInHere/YHF/new1` on `exp/m0_relation_grounded_v2`, created from the verified v1 parent `cb83b6ea4654d58ade893e4e73e2edb359b47608`. The v1 implementation and its result directories remain unchanged. The completed v2 commit is pushed to the matching origin branch.

## B. Code/protocol audit

All 36 full runs used `unified_full_graph_nc_v1`, full-graph training, the existing validation-accuracy checkpoint selection, and `task.evaluate_test=false`. Every run record says test evaluation was false; every checkpoint says `selection=best_val_accuracy`; no checkpoint or run record contains test metrics. `src/tasks/nc.py`, `src/tasks/lp.py`, split definitions, and metric code were not modified. P0 artifacts were read only after checkpoint selection and only for post-hoc analysis.

The full audit checked all 36 checkpoints, selected epochs, and per-run trace lengths: 10 gradient rows and 2 operator-growth rows per observed epoch. All checks passed.

## C. Files added/modified

Added the requested standalone model, components, config, tests, runner, analyzer, summarizer, design document, and this report. Aggregated result tables are under `results/m0/relation_grounded_v2/`; checkpoints and per-run logs/traces are under `outputs/m0/relation_grounded_v2/`. No existing v1 or protocol files were modified.

## D. Stage-I exact formulation

Text and visual inputs are projected independently with `Linear → LayerNorm → GELU → Dropout` to `hidden_dim=256`. For each canonical directed arc `j → i`, each modality projects its intrinsic state to 64 dimensions. The pair encoder receives `[U_i, U_j, |U_i-U_j|, U_i*U_j, s_T, s_V, |s_T-s_V|]`, where `s_T` and `s_V` are modality-specific endpoint cosine similarities. The text and visual pair encoders are independent.

Recipient context is the exact leave-one-out incoming mean. Degree-one edges use `NO_CONTEXT_TEXT/ VISUAL`; pair-only runs use `NULL_CONTEXT_TEXT/VISUAL`. The text relation query attends over `[E_V,C_T,C_V]`, and the visual relation query attends over `[E_T,C_T,C_V]`, each with the v1 residual relation correction. This memory is built from `H0` once per forward pass and reused at both interaction steps. Relation dimension is 64.

## E. Relation-grounded Stage-II exact formulation

The dynamic query is `LN(W_q^m H_i,k^m + e_m)`. It consumes only the current recipient state. The source state does not enter query formation. A bias-free two-head retriever reads `[R_T,R_V]` and returns only `LN(AttnOut)`, with a non-affine output LayerNorm so zero relation memory produces exact zero execution code for every query. There is no query residual. Analysis returns the two memory attention weights; training requests no attention weights.

The source supplies only `z=W_msg H_j,k`. Execution modulation is `tanh(W_a xi)` and the rank-32 operator adds `W_up(modulation * W_down(z))`. `W_up.weight` and `W_up.bias` start at zero, so initialization gives `m=z`. Text and visual aggregation and residual state updates are independent. There are two shared-parameter interaction steps, followed by fusion of only the terminal text and visual states.

## F. Four variants

| Variant | Relation memory | Execution query |
|---|---|---|
| `global_grounded_dynamic` | Two learned global slots shared by all edges | Recipient-state dynamic |
| `pair_grounded_dynamic` | Pair relation with learned null recipient context | Recipient-state dynamic |
| `context_grounded_static` | Pair relation plus real recipient LOO context | Learned static query |
| `context_grounded_dynamic` | Pair relation plus real recipient LOO context | Recipient-state dynamic |

Primary paired contrasts are pair minus global (edge-specific relation memory), context-dynamic minus pair (recipient context), and context-dynamic minus context-static (recipient-state-conditioned execution). Each contrast uses three matched seeds per dataset.

## G. Unit/regression tests

`tests/test_interaction_core_v2.py`: **32 passed**. Coverage includes modality splitting, canonical directed edges, pair-feature formula, LOO and degree-one context, pair/null versus contextual memory, memory construction once, shared two-step retrieval, target-only and static queries, zero-memory grounding, global same-target invariance, edge-specific retrieval, source-only messages, zero-up initialization and gradients, two-step gradient recovery, aggregation, zero-degree safety, inference contract, and no label/split access.

Relevant M0/protocol regression subset: **87 passed**. The broader repository suite has two pre-existing issues outside this change: collection of `tests/test_multi_order_analyzer.py` fails because `scripts/summarize_multi_order_bank_nc.py` is absent on the parent branch; excluding that module yields 161 passed and one failure in `tests/test_problem_validation_p02.py` (`KeyError: delta_mean_margin` in the existing P0 bootstrap code). Neither file was modified.

## H. Smoke

All four Movies seed-42 five-epoch runs completed. Every run reached epoch 5, produced finite training/validation values and nonzero trained `W_up`; the gradient and growth traces had all five epochs. These are engineering checks, not performance evidence. Smoke validation accuracy was 39.922% for `global_grounded_dynamic` and 39.982% for the other three variants; validation Macro-F1 was 7.383% and 7.454%, respectively. See smoke logs and CSVs for exact values.

## I. 36-run validation pilot

All 36 runs completed: Movies, Grocery, and Reddit-S × seeds 42/43/44 × four variants. The table gives mean ± population SD over seeds, in percent. No test split was evaluated.

| Dataset | Variant | Val Acc | Val Macro-F1 |
|---|---|---:|---:|
| Movies | global | 56.26 ± 0.67 | 46.29 ± 2.06 |
| Movies | pair | 55.88 ± 0.36 | 44.66 ± 3.41 |
| Movies | context static | 56.11 ± 0.48 | 46.68 ± 1.17 |
| Movies | context dynamic | 56.11 ± 0.41 | 45.76 ± 3.69 |
| Grocery | global | 83.64 ± 0.55 | 76.52 ± 1.16 |
| Grocery | pair | 82.76 ± 0.20 | 74.87 ± 1.25 |
| Grocery | context static | 83.18 ± 0.36 | 75.93 ± 0.83 |
| Grocery | context dynamic | 82.79 ± 0.12 | 74.87 ± 1.80 |
| Reddit-S | global | 96.41 ± 0.14 | 92.12 ± 0.29 |
| Reddit-S | pair | 96.47 ± 0.22 | 92.29 ± 0.27 |
| Reddit-S | context static | 96.49 ± 0.20 | 92.47 ± 0.33 |
| Reddit-S | context dynamic | 96.51 ± 0.27 | 92.73 ± 0.12 |

Paired validation-accuracy differences were small and inconsistent. Pair minus global was −0.38 pp on Movies (1/3 seed pairs positive), −0.88 pp on Grocery (0/3), and +0.05 pp on Reddit-S (2/3). Context-dynamic minus pair was +0.23, +0.03, and +0.04 pp (positive in 3/3, 2/3, and 2/3 pairs). Context-dynamic minus static was 0.00, −0.39, and +0.02 pp (positive in 1/3, 1/3, and 1/3 pairs). These three-seed differences are descriptive, not significance tests.

## J. Stage-I health

For `context_grounded_dynamic`, feature-wise mean relation variance was nonzero in both modalities: Movies 0.289 text / 0.724 visual; Grocery 0.707 / 0.855; Reddit-S 0.496 / 0.846. Mean `1-cos(R_T,R_V)` was 1.03–1.24 across datasets, showing distinct modality views. Reverse-edge `1-cos` was nonzero in every dataset and modality (means 0.052–0.297); reverse-edge L2 differences were also nonzero (means about 1.52–5.95). The learned relation states are therefore non-collapsed, modality-dependent, and directed.

The same-checkpoint full-context versus null-context relation sensitivity averaged 1.48 at compatibility Q1 and 1.61 at Q5. Q5 was larger in all three datasets (Movies 1.78 vs 1.77; Grocery 1.56 vs 1.39; Reddit-S 1.47 vs 1.29). This is a sensitivity diagnostic using the same checkpoint, not a causal effect; the null context was not trained by the contextual variant and may be out of distribution.

Stage-I attention entropy was about 1.01–1.08 nats over three evidence keys, near the maximum `ln(3)=1.10`, while key masses varied by modality and compatibility quintile. For example, the text relation query assigned roughly 0.63–0.73 mean mass to the visual relation view. Attention weights are descriptive and do not establish causal evidence use. The global control's pair/context encoders are unused during training; its Stage-I diagnostic-only rows are explicitly marked.

## K. Stage-I gradient health

Raw gradients were captured by parameter hooks before clipping. Across the nine contextual-dynamic runs, Stage-I pair and context-attention groups first became nonzero at epoch 2; the same epoch-2 transition occurred for retrieval and `W_a`. The first epoch's zero gradients are expected from zero-initialized `W_up`. At best epochs, median RMS was `4.18e-4` for Stage-I pair, `7.31e-5` for Stage-I context attention, `1.40e-4` for retrieval, and `1.37e-4` for `W_a`. All 10 groups and every observed epoch are recorded in the full CSV. The global control's context-attention group correctly remained unused and had no gradient.

## L. Operator growth trajectory

Median `W_up` Frobenius norms across runs grew from approximately 0.09 after epoch 1 to 0.23–0.25 at epoch 5, 0.34–0.44 at epoch 10, and 0.47–0.77 at epoch 20, depending on variant and modality. At selected best epochs, medians ranged from 1.03–1.67 for text and 1.22–2.05 for visual across variants. `operator_growth_trace.csv` contains every epoch and identifies the selected best epoch; it supports the expected delayed gradient flow into Stage I.

## M. Execution attention / grounding

The two-view execution attention was non-uniform and differed by modality. For contextual-dynamic checkpoints, text queries placed mean mass of about 0.63–0.73 on `R_V`; visual queries ranged from about 0.43 to 0.61 on `R_V`, depending on dataset. Entropy was about 0.45–0.54 nats over two views (maximum `ln(2)=0.693`). The unit tests verify exact zero code under zero memory, independence from target query in that case, and same-target code invariance for the global control.

## N. Operator deviation

For contextual-dynamic models, mean `||Delta m||/(||z||+eps)` by text/visual and step 0/1 was: Movies 0.154/0.151 and 0.164/0.188; Grocery 0.374/0.336 and 0.411/0.530; Reddit-S 0.650/0.734 and 0.335/0.354. The distribution is heterogeneous across dataset, modality, and step. Global-control means were smaller (about 0.03–0.18), while edge-specific pair/context variants had larger ratios. Full mean, median, p10, and p90 are in `operator_diagnostics.csv`.

## O. Operation geometry

Contextual-dynamic message scale means exceeded 1 in every dataset/modality/step, from 1.02–1.03 on Movies to 1.09–1.25 on Grocery and 1.16–1.36 on Reddit-S. Mean message rotation `1-cos(z,z+Delta)` ranged from 0.017–0.023 on Movies, 0.053–0.077 on Grocery, and 0.033–0.156 on Reddit-S. Mean `cos(z,Delta)` was often near zero; descriptive redirect proportions were high for text and for Movies visual. Reddit-S visual had a larger reinforcement share (about 53–62% at the stated `cos>0.3` threshold). These thresholds describe geometry; they are not ground-truth semantic roles.

## P. Structured dynamicity

Recipient states changed substantially between the two steps: mean target-level `D_H` was 8.59–9.61 across modalities/datasets. Mean target-level `D_A` was 0.31–0.72. However, target-level Spearman correlation between `D_H` and `D_A` was weak (−0.029 to 0.107). Thus operation adaptation is nonzero and heterogeneous, but this diagnostic does not support a strong monotonic relationship between state-change magnitude and operation-change magnitude.

## Q. Relation-shuffle intervention

On the same contextual-dynamic checkpoint, a cyclic within-recipient relation-memory permutation changed 97.0–98.3% of sampled edges while preserving target, modality, degree, and each recipient's relation-memory marginal; degree-one edges were unchanged. Mean execution-code L2 changes were 1.7–5.0 and modulation L2 changes 0.7–2.4, depending on dataset/modality/step. Validation accuracy changes were small and mixed: Movies −0.23 pp, Grocery −0.08 pp, Reddit-S +0.06 pp. Prediction flips were 1.84%, 1.74%, and 0.09%. Relation-memory alignment clearly changes the executed code, while task predictions were comparatively insensitive in this pilot.

## R. Context-shuffle intervention

The deterministic shuffle moved paired text/visual LOO contexts to a different recipient with the same exact degree within the requested buckets; it preserved bucket marginals and excluded degree-one edges. About 97.0–98.3% of edges changed context. Mean relation-state L2 change was 0.86 on Movies, 0.90 on Grocery, and 0.095 on Reddit-S; modulation changes were about 0.36–0.59 on Movies/Grocery and 0.03–0.04 on Reddit-S. Validation accuracy changes were −0.28, +0.08, and approximately 0.00 pp; flip rates were 1.12%, 1.00%, and 0.06%. Correct recipient-context alignment affects relation state and operation, with small and inconsistent validation effects.

## S. Frozen-query intervention

At step 1, the intervention reused `q(H_i,0)` while retaining source content `H_j,1` and the original relation memory. Step-1 execution-code L2 changed by 0.58–1.39 and modulation L2 by 0.29–0.74, while step 0 was unchanged by construction. Validation accuracy changes were +0.08 pp on Movies, +0.08 pp on Grocery, and −0.03 pp on Reddit-S; flip rates were 1.09%, 0.89%, and 0.14%. The trained model uses the evolved recipient to alter step-1 execution codes, but this intervention had little task-level effect here.

## T. Operator-off intervention

Disabling the conditional operator reduced validation accuracy by 0.62 pp on Movies, 0.64 pp on Grocery, and 0.43 pp on Reddit-S. Macro-F1 fell by 2.14, 0.96, and 0.92 pp, respectively; prediction flips were 10.62%, 7.03%, and 1.87%. The conditional path contributes to validation behavior, though the effect size is modest and varies by dataset.

## U. P0 hard-case operation analysis

P0 was used only post-hoc. For beneficial-versus-harmful modulation-centroid separation, the mean normalized separation was 0.211 in Q1 and 0.532 in Q5. Q5 exceeded Q1 across each dataset/modality/step group. Mean operator deviation ratio was nearly identical overall (0.348 in Q1, 0.348 in Q5); mean `cos(z,Delta)` was 0.036 vs 0.132, and mean message rotation was 0.063 vs 0.055. This does not support the hypothesis that contextual operation differentiation is strongest in low-similarity Q1. P0 data did not enter training, checkpoint selection, or validation evaluation.

## V. Q1 Stage-I conclusion

Stage-I relation states are non-collapsed, directed, modality-dependent, context-sensitive, and receive task gradients after `W_up` updates. Context correction was measurable but larger at Q5 than Q1 in all datasets, contrary to the low-compatibility emphasis. Global-control encoder diagnostics are not trained and should not be read as evidence about learned edge-specific Stage I.

## W. Q2 Interface-grounding conclusion

The bypass is structurally removed and unit-tested: the source has no query path, the query has no residual into execution code, and zero relation memory gives exact zero code. The global control is invariant to source identity for a fixed target. Within-recipient relation shuffle substantially changed execution codes and modulation but had only small, mixed validation changes. Interface grounding is supported as a computational property; strong task dependence on exact relation-edge alignment is not established.

## X. Q3 Dynamic-execution conclusion

Recipient-state evolution changed step-1 execution code and modulation; frozen-query ablation preserved source step-1 content and relation memory. Yet validation changes and flips were small, and the `D_H`–`D_A` association was weak. Dynamic re-query is active in the forward computation, with limited demonstrated task reliance in this pilot.

## Y. Q4 Context conclusion

Real LOO context changes Stage-I relation states and downstream modulation, and degree-matched context shuffle changes most edge contexts. The task response is small and inconsistent across datasets. The operator also learns heterogeneous, nontrivial transformations, especially for edge-specific variants and on Reddit-S; message scaling and rotation vary by modality. Mechanism usage is clearer than a general task-performance advantage for context or dynamic queries.

## Z. What is supported / not supported

Supported: the exact relation-grounded interface; delayed Stage-I gradient flow after zero-up initialization; non-collapsed/directed relation representations; relation and context interventions changing execution state; nontrivial, modality- and dataset-dependent operator geometry; and a measurable contribution from the conditional operator.

Not supported: consistent accuracy gains from edge specificity, context, or dynamic queries; a large validation dependence on within-recipient relation/context alignment; a strong target-level `D_H`–`D_A` association; or a stronger Q1 than Q5 P0 operation separation. Three seeds and validation-only outcomes limit inference; the intervention metrics are descriptive, not causal estimates.

## AA. Recommendation for next stage

Freeze the M0-Core v2 interface for human review. The mechanism is implemented and its computational invariants are verified, while validation evidence for task-level relation/context dependence remains modest. Review the Q1/Q5 reversal and weak dynamicity association before making broader claims. This phase stops here; no Stage III, operator bank, routing, or history readout was implemented.
