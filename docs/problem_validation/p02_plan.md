# P0.2 Minimal Problem Validation Plan

## Goal

This phase is intentionally limited to **problem-existence validation before model design**.

Existing P0.1/P0.1+ already establishes:

1. task-aware semantic compatibility stratifies the **sign** of message utility;
2. compatibility is only weakly associated with utility **magnitude**;
3. counterexamples persist even at extreme compatibility;
4. relation utility is modality dependent.

P0.2 therefore asks only one additional question:

> **Among relations with comparable semantic compatibility, does the recipient's local structure-semantic context still associate with different message utility?**

If yes, this supports the paper-level premise that a physical relation should be interpreted in its current structure-semantic environment rather than by endpoint compatibility alone.

P0.2 is **not** intended to:
- prove a continuous relation manifold;
- compare scalar vs MoE/operator architectures;
- train relation encoders;
- perform causal neighborhood surgery;
- decide low-rank/FiLM/MoE implementations;
- validate every proposed future module.

Those belong to later model-design/analysis phases.

## Source and protocol freeze

- Source branch: `exp/problem_validation_p01plus`
- Source HEAD: `f7d68272038e5d866075e91d81707ac42e6a2a17`
- New branch: `exp/problem_validation_p02`
- Reuse the existing 15 P0.1 artifacts whenever possible.
- Datasets: Movies, Toys, Grocery, ele-fashion, Reddit-S.
- Seeds: 42, 43, 44.
- Analysis population: exactly the existing sampled validation relation population.
- Do not regenerate target/edge samples.
- Do not access test labels, test metrics, or test checkpoints.
- Keep CE message-removal utility as the primary utility; margin utility is a robustness view.

## P0.2-A: Context-conditioned utility stratification

For every sampled directed relation `j -> i` and modality `m`, reuse:

- probe similarity `s_ij^m`;
- CE utility `U_ij^m`;
- frozen semantic embeddings `H^m`;
- target degree and physical graph.

Construct only two lightweight, predeclared context descriptors.

### 1. Recipient context redundancy

Let `N(i) \ {j}` denote the other physical neighbors of the recipient.

Compute the mean semantic context excluding source `j`:

```
C_{i,-j}^m = mean_{u in N(i) \ {j}} H_u^m
```

For degree > 1 this should be computed efficiently from the already available full mean-neighbor message:

```
C_{i,-j}^m = (d_i * N_i^m - H_j^m) / (d_i - 1)
```

Define:

```
R_{j->i}^m = cosine(H_j^m, C_{i,-j}^m)
```

Interpretation: whether the semantic evidence supplied by `j` is already redundant with the recipient's other neighborhood context.

Degree-1 targets have no other-neighbor context and must be reported separately rather than assigned an arbitrary value.

### 2. Cross-modal relation disagreement

Reuse:

```
D_ij = |s_ij^T - s_ij^V|
```

This asks whether the same physical relation is interpreted differently across modality spaces.

## Analysis rule

Do **not** train a new probe.

For each dataset x seed x modality:

1. partition relations into the same five probe-similarity quintiles used in P0.1+;
2. inside each similarity quintile, split recipient redundancy into low/high groups using the within-bin median (or balanced halves);
3. compare:
   - mean CE utility;
   - beneficial rate `P(U > 0)`;
   - harmful rate `P(U < 0)`;
4. aggregate the within-similarity-bin low-vs-high context differences using relation-count weighting;
5. report target-node bootstrap 95% CIs, preserving the existing cluster-bootstrap convention.

Repeat the same conditional stratification for low/high cross-modal disagreement.

The key quantity is therefore not a global correlation but:

> **At comparable semantic compatibility, does local context still stratify utility?**

## Primary evidence

P0.2 supports contextual relation interpretation if the context-conditioned utility difference:

- has a consistent direction across seeds;
- appears in multiple datasets/modalities;
- is non-negligible relative to the existing P0.1 utility scale;
- remains qualitatively similar under CE and margin utility.

No universal hard threshold is required.

A weak overall effect can still be retained as a supporting observation if it is stable in specific regimes (e.g. counterexample subsets), but it should not be elevated to the main problem claim.

## Counterexample subset view

As a secondary descriptive analysis only:

- high-sim harmful: Q5 similarity and U < 0;
- low-sim beneficial: Q1 similarity and U > 0.

Compare their recipient redundancy and cross-modal disagreement against similarity-matched expected-sign relations.

This is for interpretation; it is not a separate hypothesis test.

## Stop rule after P0.2

If P0.2 shows stable context-conditioned differences, the pre-model problem-validation stage is considered sufficient.

Then proceed to model architecture design around:

1. contextual relation interpretation;
2. dynamic relation-conditioned interaction;
3. lightweight interaction-history consolidation.

Do **not** automatically proceed to residual-MLP attribution, causal recipient intervention, relation-manifold analysis, or operator competition before an initial model is designed.

If P0.2 is weak/inconsistent, retain the already established P0.1 findings and reconsider whether edge-local context should be a core claim or only an optional supporting mechanism.

## Later analyses (not part of current P0.2)

The following are intentionally deferred:

- learned residual-utility prediction;
- richer endpoint pair encoders;
- pure-structure feature blocks;
- causal removal of redundant neighbors;
- stage-specific edge interventions;
- scalar vs fixed-role vs latent-operator competition;
- operator specialization/collapse analysis.

These can be introduced after the first model prototype if needed to choose between competing architectures or strengthen mechanism analysis.
