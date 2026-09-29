# P0.2: Context-Conditioned Relation Utility

## Summary

P0.2 asks whether recipient-side local context further stratifies the utility of a physical relation after conditioning on task-aware endpoint similarity. The analysis reuses the fixed P0.1 validation-relation population and frozen embeddings for Movies, Toys, Grocery, ele-fashion, and Reddit-S, with seeds 42, 43, and 44.

The answer depends on which aspect of utility is considered:

- **Utility sign is consistently stratified.** Within probe-similarity quintiles, the beneficial rate is higher in the high-recipient-redundancy half than in the low-redundancy half. The low-minus-high difference in `P(U_CE > 0)` is negative in all 30 dataset × modality × seed weighted-overall rows, and every run-specific target-node bootstrap 95% CI is below zero. The average difference is `-0.080` (8.0 percentage points). This observed direction is opposite to a simple “more redundancy is less useful” expectation; no direction was assumed in the analysis.
- **Continuous utility magnitude is weak and inconsistent.** The weighted-overall low-minus-high `E[U_CE]` difference averages `+0.000019` across the 30 rows; only 9/30 run-specific CIs exclude zero. Its sign varies across datasets and modalities. Across similarity quintiles, the mean CE difference moves from `-0.00433` in Q1 to `+0.00201` in Q5, so these conditional differences partly cancel in the weighted aggregate. Mean `U_Margin` differences also average close to zero (`-0.000663`) with mixed directions.

Thus, local recipient context adds a stable association with the **sign propensity** of relation utility at comparable semantic compatibility. This run does not establish a stable global change in expected CE utility magnitude, a causal effect, or a learned relation interpretation mechanism. The result supports retaining local context as a problem-level diagnostic for a later stage; it does not decide any model architecture.

## Historical protocol and interpretation

P0.0 trained task-aware semantic projectors from text and visual features into `H_text` and `H_visual`. Projector training and checkpoint selection used only the inner `probe_train` and `probe_calib` subsets of original training nodes. Original validation was used only for held-out analysis after checkpoint selection; test labels and test metrics were not used.

P0.1 formed

```text
F_i = [H_i^T, H_i^V, N_i^T, N_i^V]
N_i^m = mean_{u in N(i)} H_u^m
```

and trained a linear `MessageProbe`. For each sampled relation `j -> i`, its text and visual message contributions are analytically separable because the neighbor-message blocks enter a linear head. P0.1 also audited analytic removal against explicit feature removal; the existing audit enforces maximum absolute error below `1e-5`.

The P0.1/P0.1+ conclusion remains: task-aware semantic compatibility is informative about the sign of utility, but only weakly tracks utility magnitude; counterexamples remain at extreme similarities; utility is modality dependent. P0.2 does not repeat that test. It asks whether recipient context adds another conditional stratification. See [P0.1 report](p0_p1_report.md) and [P0.1+ report](p01plus_report.md).

## Data and procedure

- The 15 P0.1 `edge_analysis.pt`, `semantic_embeddings.pt`, and five fixed split-cache files were reused from `/hdd1/DataInHere/YHF/mag_model/outputs/problem_validation/`. Target-node, directed-edge, and analysis-target arrays were checked for exact sequence equality between `edge_analysis.pt` and the corresponding split cache.
- Physical graph degrees and every frozen directed relation endpoint were checked against the graph loaded from the `new1` dataset configuration. The analysis code used graph edges and node counts only; it did not index or use test labels, test metrics, or test splits.
- The primary descriptor is

  ```text
  C_{i,-j}^m = (d_i N_i^m - H_j^m) / (d_i - 1)
  R_{j->i}^m = cosine(H_j^m, C_{i,-j}^m)
  ```

  computed from the existing full mean-neighbor message. This is a lightweight diagnostic of overlap with other neighbors, not ground-truth redundancy.
- For `d_i = 1`, `context_defined = False`; no value was imputed. The primary context analysis retains only `d_i > 1` relations. Coverage and degree-one counts are reported below and in `p02_context_coverage.csv`.
- For each dataset × seed × modality, P0.1+ stable-mergesort balanced probe-similarity quintiles were reused. Within each quintile, a stable sort and balanced split created low/high descriptor groups. The reported effect is always low minus high; direction was not prescribed.
- Primary utility is `utility_ce_text` / `utility_ce_visual`; margin utility is retained as a robustness view. The secondary descriptor is `D = |s_probe^T - s_probe^V|` and is analyzed on all sampled relations, including degree-one rows.
- Confidence intervals use 1,000 bootstrap replicates with seed 42. The resampling unit is the target node; all of a selected node’s sampled relations move together. CIs are per dataset × seed × modality × analysis and are not pooled across model seeds.

No model or new predictive probe was trained, no targets or relations were re-sampled, and no NC/LP protocol or dataset split was changed. The analysis is offline; it did not require GPU computation.

## Context coverage

| Dataset | Sampled relations | Context-defined (`d_i > 1`) | Retained coverage | Degree-one relations |
|---|---:|---:|---:|---:|
| Movies | 30,552 | 29,848 | 97.70% | 704 (2.30%) |
| Toys | 22,528 | 21,371 | 94.86% | 1,157 (5.14%) |
| Grocery | 26,053 | 25,193 | 96.70% | 860 (3.30%) |
| ele-fashion | 37,264 | 32,678 | 87.69% | 4,586 (12.31%) |
| Reddit-S | 39,613 | 38,686 | 97.66% | 927 (2.34%) |

Counts are identical across the three model seeds because the P0.1 relation population is fixed independently of model seed.

## Primary recipient-context results

The table reports mean ± population SD over the three model seeds. `Δ` means low redundancy minus high redundancy. For the beneficial-rate contrast, the 95% target-node bootstrap interval was below zero in all three seeds for every dataset × modality row.

| Dataset | Modality | Δ mean CE utility | Δ beneficial rate, CE | Δ mean margin utility | Benefit CIs below zero |
|---|---|---:|---:|---:|---:|
| Movies | Text | -0.00107 ± 0.00027 | -0.072 ± 0.006 | -0.00460 ± 0.00047 | 3/3 |
| Movies | Visual | -0.00215 ± 0.00031 | -0.077 ± 0.004 | -0.00631 ± 0.00049 | 3/3 |
| Toys | Text | +0.00107 ± 0.00044 | -0.085 ± 0.008 | -0.00805 ± 0.00219 | 3/3 |
| Toys | Visual | -0.00204 ± 0.00044 | -0.113 ± 0.002 | -0.02175 ± 0.00203 | 3/3 |
| Grocery | Text | +0.00056 ± 0.00014 | -0.093 ± 0.002 | -0.01075 ± 0.00165 | 3/3 |
| Grocery | Visual | -0.00030 ± 0.00045 | -0.129 ± 0.006 | -0.01545 ± 0.00169 | 3/3 |
| ele-fashion | Text | +0.00066 ± 0.00041 | -0.060 ± 0.008 | +0.02250 ± 0.00416 | 3/3 |
| ele-fashion | Visual | +0.00039 ± 0.00033 | -0.069 ± 0.019 | -0.00101 ± 0.01075 | 3/3 |
| Reddit-S | Text | +0.00108 ± 0.00049 | -0.053 ± 0.010 | +0.00472 ± 0.00222 | 3/3 |
| Reddit-S | Visual | +0.00199 ± 0.00110 | -0.050 ± 0.010 | +0.03407 ± 0.00121 | 3/3 |

The beneficial-rate difference remains visible within every similarity quintile. The values below are averages over the 30 dataset × modality × seed rows per quintile; the last column counts run-specific CIs entirely below zero.

| Similarity bin | Mean Δ `P(U_CE > 0)` | CIs below zero |
|---|---:|---:|
| Q1 | -0.167 | 30/30 |
| Q2 | -0.076 | 25/30 |
| Q3 | -0.062 | 26/30 |
| Q4 | -0.051 | 27/30 |
| Q5 | -0.046 | 21/30 |

This pattern is not a uniform shift in mean utility. Across all 30 weighted-overall contrasts, the mean CE difference is near zero, only 9/30 CIs exclude zero, and 18/30 point estimates are positive. The mean CE difference is negative in low-similarity Q1 and positive in high-similarity Q5. The existing per-relation CE utility SD spans approximately `0.049–0.126` across dataset × seed × modality artifacts, substantially larger than the weighted-overall mean contrast. Margin mean differences likewise have mixed direction, so the magnitude result is not robustly similar between CE and margin.

All Q1–Q5 group means, beneficial/harmful rates, per-seed CIs, and weighted-overall values are retained in [p02_per_seed.csv](../../results/problem_validation/p02/p02_per_seed.csv). The per-seed intervals are not multiple-comparison adjusted; the Q-bin profiles are diagnostic conditional summaries.

## Secondary disagreement and counterexamples

The low-minus-high disagreement split is less consistent than recipient redundancy. Across the 15 dataset × seed combinations per modality, mean weighted-overall `Δ P(U_CE > 0)` is `-0.017` for Text and `-0.027` for Visual; the run-specific intervals exclude zero in 9/15 and 8/15 cases, respectively. Mean continuous CE contrasts are `-0.00288` and `-0.00326`. Margin results do not share a consistent direction. This remains secondary evidence and is not a direct measurement of utility variance.

Counterexample summaries are descriptive only. Averaging run-level group means equally across dataset × seed × modality rows:

| Subset comparison | Recipient redundancy, mean | Cross-modal disagreement, mean |
|---|---:|---:|
| Q5 harmful vs Q5 expected-sign | 0.830 vs 0.882 | 0.195 vs 0.201 |
| Q1 beneficial vs Q1 expected-sign | 0.576 vs 0.492 | 0.282 vs 0.238 |

`Expected-sign` means Q5 with `U_CE > 0` or Q1 with `U_CE < 0`. Redundancy summaries use only degree-greater-than-one relations. These comparisons do not use a counterexample classifier or a separate hypothesis test.

## Interpretation and limits

P0.2 supports a limited extension of the P0.1 claim: endpoint compatibility does not fully stratify the sign of relation utility, and the recipient’s other-neighbor semantic context adds a stable sign-level association within similarity quintiles. The observed association is opposite the intuitive monotonic redundancy penalty. Continuous CE and margin magnitudes do not show a stable overall low-versus-high context effect; any claim should preserve that distinction.

`R` is an observational cosine statistic over frozen task-aware embeddings and an unweighted physical-neighbor mean. It is not ground-truth redundancy, does not establish causality, and does not prove that any relation encoder, operator, expert, or history module is necessary. P0.2 is sufficient to motivate keeping context in the subsequent problem/model discussion, but not to choose an architecture or to claim that context improves a trained model.

## Outputs

- [Per-seed Q1–Q5 results and bootstrap intervals](../../results/problem_validation/p02/p02_per_seed.csv)
- [Cross-dataset means and seed SDs](../../results/problem_validation/p02/p02_cross_dataset.csv)
- [Context coverage and degree-one accounting](../../results/problem_validation/p02/p02_context_coverage.csv)
- [Descriptive counterexample subsets](../../results/problem_validation/p02/p02_counterexamples.csv)
- [P0.2 output README](../../results/problem_validation/p02/README.md)
- [Movies conditional-effect plot](../../results/problem_validation/p02/plots/p02_Movies_conditional_effects.png)
- [Toys conditional-effect plot](../../results/problem_validation/p02/plots/p02_Toys_conditional_effects.png)
- [Grocery conditional-effect plot](../../results/problem_validation/p02/plots/p02_Grocery_conditional_effects.png)
- [ele-fashion conditional-effect plot](../../results/problem_validation/p02/plots/p02_ele-fashion_conditional_effects.png)
- [Reddit-S conditional-effect plot](../../results/problem_validation/p02/plots/p02_Reddit-S_conditional_effects.png)

