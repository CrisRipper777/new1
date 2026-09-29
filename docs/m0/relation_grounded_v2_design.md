# M0-Core v2 — Relation-Grounded Semantic Execution

## Research question

This phase tests whether M0-Core has a clean Stage-I to Stage-II interface. Stage I encodes what a directed physical relation may mean. Stage II uses that relation memory together with the current recipient state to choose an execution code. The source state supplies only the semantic content being transformed.

The v1 bypass came from putting both endpoints in the execution query and retaining that query in the retrieval output. V2 removes both paths. This phase stops at the two-step core; it adds no Stage III, operator bank, routing, or history readout.

## Frozen protocol and scope

- Project: `/hdd1/DataInHere/YHF/new1`; branch: `exp/m0_relation_grounded_v2`.
- Task: `unified_full_graph_nc_v1`, with validation-accuracy checkpoint selection and `task.evaluate_test=false` for every run.
- Grid: Movies, Grocery, Reddit-S × seeds 42/43/44 × four variants; 36 full runs after the four-run, five-epoch Movies seed-42 smoke.
- `src/tasks/nc.py`, split definitions, and earlier v1 results are untouched. P0 artifacts are loaded only after checkpoint selection for post-hoc diagnostics.

## Stage I: intrinsic relation memory

Text and visual projectors remain separate:

\[
P_i^m=\operatorname{Dropout}(\operatorname{GELU}(\operatorname{LN}(W_x^mX_i^m))),\quad m\in\{T,V\}.
\]

For directed arc \(j\to i\), projected modality states produce within-modality cosine scores \(s_T,s_V\) and discrepancy \(d=|s_T-s_V|\). Each modality-specific pair encoder consumes

\[
[U_i^m,U_j^m,|U_i^m-U_j^m|,U_i^m\odot U_j^m,s_T,s_V,d].
\]

Recipient context is the exact leave-one-out incoming mean. Degree-one edges use modality-specific `NO_CONTEXT` vectors. Pair-only training uses modality-specific `NULL_CONTEXT` vectors. Contextual relation attention preserves the v1 residual correction:

\[
R_T=\operatorname{LN}(E_T+\operatorname{CrossAttn}(E_T,[E_V,C_T,C_V])),
\]

with the symmetric visual expression. This is computed once from intrinsic \(H_0\) per forward pass and reused at both interaction steps.

## Stage II: relation-grounded execution

Dynamic recipient query:

\[
q_{i,k}^m=\operatorname{LN}(W_q^mH_{i,k}^m+e_m).
\]

The query has no source-state input. `RelationGroundedRetriever` uses bias-free two-head attention over `[R_T,R_V]`, then returns only a non-affine LayerNorm of the attention output:

\[
\xi_{ji,k}^m=\operatorname{LN}(\operatorname{MHA}(q_{i,k}^m,[R_{ji}^T,R_{ji}^V])).
\]

There is no query residual. Zero relation memory therefore yields exact zero execution code for every query. The static variant substitutes a learned modality-specific query but uses the same retriever.

Source-only semantic content and the conditional rank-32 operation are:

\[
z_{j,k}^m=W_{msg}^mH_{j,k}^m,\quad
a_{ji,k}^m=\tanh(W_a^m\xi_{ji,k}^m),
\]

\[
m_{ji,k}^m=z_{j,k}^m+W_{up}^m(a_{ji,k}^m\odot W_{down}^mz_{j,k}^m).
\]

`W_up.weight` and `W_up.bias` start at zero, so the first forward has \(m=z\). Incoming mean aggregation and residual state update remain modality-specific. Only terminal text and visual states are fused.

## Fixed variants and primary contrasts

| Variant | Relation memory | Query |
|---|---|---|
| `global_grounded_dynamic` | Two learned global slots shared by all edges | Recipient state |
| `pair_grounded_dynamic` | Pair evidence with learned null context | Recipient state |
| `context_grounded_static` | Pair evidence plus real recipient LOO context | Learned static vector |
| `context_grounded_dynamic` | Pair evidence plus real recipient LOO context | Recipient state |

The paired contrasts are pair vs global (edge-specific relation memory), context-dynamic vs pair (recipient contextual interpretation), and context-dynamic vs context-static (state-conditioned execution). Validation metrics are supporting evidence; mechanism diagnostics and same-checkpoint interventions are primary.

## Diagnostics and interventions

The analysis records Stage-I feature-wise relation variance, cross-modal discrepancy, reverse-arc cosine and L2 differences, full-vs-null contextual sensitivity by compatibility quintile, and evidence-key attention. Parameter hooks record raw pre-clipping gradient RMS for ten groups; `set_epoch` flushes a complete row after the optimizer step at the subsequent validation forward. Registered buffers retain the trajectory in validation-selected state dictionaries, and per-run CSVs preserve every epoch. The same flush records both `W_up` Frobenius norms.

Stage-II analysis includes attention over both relation views and entropy; operator deviation \(\|\Delta m\|/(\|z\|+\epsilon)\); `cos(z, delta)`, message scale, and message rotation; edge and target-level dynamicity; and selected-epoch markers. Same-checkpoint tests include within-recipient cyclic relation-memory shuffle (degree-one unchanged), degree-bucket recipient-context shuffle (Text/Visual contexts move together; degree one excluded), frozen step-one query with the step-one source state retained, and operator-off. P0 beneficial/harmful Q1/Q5 operation summaries are post-hoc only.

The first backward can have zero Stage-I/retrieval gradients because `W_up=0`; this is expected. The diagnostic question is whether task gradients reach Stage I after `W_up` takes its first optimizer step. All mechanism correlations and interventions are interpreted descriptively, with no claim of causal identification.

## Required outputs

Checkpoints and per-run traces live under `outputs/m0/relation_grounded_v2/`. Aggregated CSVs, `README.md`, and the final A–AA evidence report live under `results/m0/relation_grounded_v2/` and `docs/m0/relation_grounded_v2_report.md`. The phase ends after validation, report, commit, and push to the v2 branch.
