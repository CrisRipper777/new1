# Final NC Ablation Results

Values are mean ± sample SD across three matched seeds. The trained variants are separate fits; their differences are not same-checkpoint intervention effects.

## Test metrics

| Dataset | Variant | Test Acc (%) | Test Macro-F1 (%) |
|---|---|---:|---:|
| Movies | Full | 55.18 ± 0.48 | 45.92 ± 1.04 |
| Movies | w/o Context | 54.92 ± 1.15 | 45.76 ± 2.37 |
| Movies | w/o Relation Specificity | 54.49 ± 0.92 | 44.32 ± 6.71 |
| Movies | w/o State Conditioning | 54.68 ± 0.98 | 46.63 ± 2.51 |
| Movies | w/o Semantic Operator | 54.78 ± 0.81 | 45.35 ± 1.99 |
| Toys | Full | 79.71 ± 0.25 | 76.57 ± 0.41 |
| Toys | w/o Context | 79.68 ± 0.41 | 76.64 ± 0.68 |
| Toys | w/o Relation Specificity | 80.20 ± 0.25 | 77.34 ± 0.55 |
| Toys | w/o State Conditioning | 79.74 ± 0.48 | 76.64 ± 0.65 |
| Toys | w/o Semantic Operator | 79.73 ± 0.23 | 76.60 ± 0.38 |
| Grocery | Full | 81.73 ± 0.61 | 73.94 ± 1.48 |
| Grocery | w/o Context | 82.18 ± 0.07 | 74.09 ± 0.88 |
| Grocery | w/o Relation Specificity | 82.25 ± 1.04 | 73.11 ± 1.19 |
| Grocery | w/o State Conditioning | 82.01 ± 0.23 | 73.57 ± 0.17 |
| Grocery | w/o Semantic Operator | 82.39 ± 0.80 | 74.77 ± 1.28 |
| ele-fashion | Full | 87.88 ± 0.22 | 70.10 ± 0.62 |
| ele-fashion | w/o Context | 87.93 ± 0.13 | 70.44 ± 0.44 |
| ele-fashion | w/o Relation Specificity | 88.11 ± 0.10 | 71.02 ± 0.33 |
| ele-fashion | w/o State Conditioning | 87.87 ± 0.13 | 70.24 ± 0.77 |
| ele-fashion | w/o Semantic Operator | 87.85 ± 0.11 | 69.67 ± 0.29 |
| Reddit-S | Full | 96.61 ± 0.58 | 92.53 ± 1.48 |
| Reddit-S | w/o Context | 96.56 ± 0.49 | 92.33 ± 1.10 |
| Reddit-S | w/o Relation Specificity | 96.62 ± 0.24 | 92.65 ± 0.39 |
| Reddit-S | w/o State Conditioning | 96.61 ± 0.56 | 92.46 ± 1.40 |
| Reddit-S | w/o Semantic Operator | 96.62 ± 0.58 | 92.55 ± 1.49 |

## Validation supplementary metrics

| Dataset | Variant | Val Acc (%) | Val Macro-F1 (%) |
|---|---|---:|---:|
| Movies | Full | 55.84 ± 0.53 | 46.18 ± 0.48 |
| Movies | w/o Context | 55.85 ± 0.65 | 45.62 ± 1.25 |
| Movies | w/o Relation Specificity | 56.14 ± 0.67 | 45.16 ± 4.81 |
| Movies | w/o State Conditioning | 55.69 ± 0.45 | 47.35 ± 0.16 |
| Movies | w/o Semantic Operator | 55.77 ± 0.46 | 45.87 ± 0.49 |
| Toys | Full | 79.87 ± 0.21 | 76.72 ± 0.55 |
| Toys | w/o Context | 79.77 ± 0.20 | 76.70 ± 0.27 |
| Toys | w/o Relation Specificity | 79.89 ± 0.11 | 77.47 ± 0.31 |
| Toys | w/o State Conditioning | 79.92 ± 0.31 | 76.86 ± 0.92 |
| Toys | w/o Semantic Operator | 79.86 ± 0.21 | 76.73 ± 0.56 |
| Grocery | Full | 83.43 ± 0.15 | 75.95 ± 1.37 |
| Grocery | w/o Context | 83.02 ± 0.09 | 74.69 ± 2.70 |
| Grocery | w/o Relation Specificity | 83.35 ± 0.51 | 74.85 ± 1.09 |
| Grocery | w/o State Conditioning | 83.29 ± 0.18 | 75.28 ± 1.69 |
| Grocery | w/o Semantic Operator | 83.34 ± 0.13 | 76.02 ± 0.98 |
| ele-fashion | Full | 87.71 ± 0.19 | 69.05 ± 0.42 |
| ele-fashion | w/o Context | 87.85 ± 0.15 | 69.52 ± 0.63 |
| ele-fashion | w/o Relation Specificity | 87.95 ± 0.11 | 70.32 ± 0.76 |
| ele-fashion | w/o State Conditioning | 87.73 ± 0.10 | 69.33 ± 0.41 |
| ele-fashion | w/o Semantic Operator | 87.71 ± 0.18 | 68.77 ± 0.44 |
| Reddit-S | Full | 96.51 ± 0.27 | 92.67 ± 0.63 |
| Reddit-S | w/o Context | 96.54 ± 0.13 | 92.40 ± 0.24 |
| Reddit-S | w/o Relation Specificity | 96.62 ± 0.24 | 92.70 ± 0.50 |
| Reddit-S | w/o State Conditioning | 96.51 ± 0.27 | 92.53 ± 0.38 |
| Reddit-S | w/o Semantic Operator | 96.52 ± 0.28 | 92.70 ± 0.66 |
