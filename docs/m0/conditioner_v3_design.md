# M0-Core v3 — Relation–State Conditional Execution Audit

## Objective and scope

This phase tests whether recipient state should dynamically condition relation execution, and whether a feature-wise relation–state interaction is more expressive and more useful than the v2 two-slot attention retriever. Stage I, the source-only low-rank semantic operator, incoming-mean aggregation, residual state update, fusion, and two interaction steps are frozen to v2. The only intended change across four variants is the construction of the execution code from contextual relation memory and recipient state.

No Stage III, history readout, operator bank, MoE, multi-track path, routing, extra graph propagation, topology change, multi-hop context, recurrent unit, auxiliary loss, relation label, LP, robustness run, HPO, or test evaluation is in scope. P0 artifacts are read only after checkpoint selection.

## Frozen Stage I

Text and visual features use independent projectors and relation projections. For directed edge `j → i`, each modality's pair encoder receives `[U_i, U_j, |U_i-U_j|, U_i*U_j, s_text, s_visual, |s_text-s_visual|]`. Recipient context is the exact incoming leave-one-out mean; degree-one recipients use a learned `NO_CONTEXT` token. The contextual relation states remain

`R_T = LN(E_T + CrossAttn(E_T, [E_V, C_T, C_V]))`

and the symmetric visual expression. The directed relation memory is `[R_T, R_V]`. The code copies the v2 components and retains their formulas; it does not add a context encoder or cross-modal state propagation.

## Frozen source operator and update

For each modality, `z = W_msg H_j`, `v = W_down z`, `a = tanh(W_a xi)`, `Delta = W_up(a * v)`, and `m = z + Delta`. Rank is 32; `W_up` weight and bias start at zero. Incoming messages are mean aggregated, and state updates are `LN(H + Dropout(W_u GELU(M)))`. There are two interaction steps, with modality-specific parameters and shared Stage-II parameters across steps.

## Shared base relation code

For each modality, a learned static query retrieves `r` from `[R_T,R_V]` with the v2 pure relation-grounded retriever. It has no query residual, its attention projections are bias-free, and its output normalization is non-affine LayerNorm. Thus zero relation memory gives exactly `r=0`. The same modality retriever is shared with the attention-dynamic query so that static and dynamic codes use the same relation-memory read function.

## Four fixed variants

- `context_static`: `xi_k = r` for both steps.
- `context_attn_dynamic`: v2 query `LN(W_q H_i,k + e_m)` retrieves `xi` directly from `[R_T,R_V]`; there is no query residual.
- `context_bilinear_absolute`: `xi = r + W_c(tanh(W_r r) * tanh(W_h H_i,k))`.
- `context_bilinear_delta`: `xi = r + W_c(tanh(W_r r) * tanh(W_h(H_i,k-H_i,0)))`.

The bilinear rank is 32. `W_r`, `W_h`, and `W_c` are bias-free; `W_c` starts at zero. There is no scalar gate or extra output normalization. Therefore both bilinear variants initialize exactly at `xi=r`, and the delta variant has exact `xi_0=r` for every checkpoint. A zero `r` makes the correction and `xi` exactly zero for every state.

## Protocol and contrasts

The run grid is Movies, Grocery, and Reddit-S × seeds 42/43/44 × the four variants (36 runs). Four Movies seed-42 five-epoch smokes must finish first. All runs use `unified_full_graph_nc_v1`, validation accuracy checkpoint selection, and `task.evaluate_test=false`. No task runner, split, early-stopping, metric, or NC/LP source is modified.

Primary paired contrasts: attention dynamic vs static; bilinear absolute vs attention dynamic; bilinear delta vs bilinear absolute; and bilinear delta vs static. Accuracy is one part of the evidence, together with conditioner usage, same-checkpoint interventions, relation/target variance decomposition, and operation geometry.

## Required diagnostics

For `r`, `xi`, modulation `a`, and operator message correction `Delta`, report directed-edge total, within-target, and between-target variance, with the exact decomposition check and `eta_relation`/`eta_target`. Also report a node-balanced secondary descriptive decomposition. Bilinear checkpoints include correction ratios, latent statistics, and `W_c` growth. Dynamicity includes `D_H`, `D_A`, `D_XI`, and their rank correlations. Same-checkpoint interventions disable only the bilinear correction, disable delta step 1, freeze the attention query at `H0`, shuffle relation alignment, shuffle degree-matched paired context, or set `Delta=0`.

P0 beneficial/harmful groups are post-hoc only for the selected `context_bilinear_delta` checkpoint. They are stratified by Q1/Q5 probe similarity and compared at relation, execution, and modulation stages; operator deviation, `cos(base, delta)`, and message rotation are reported descriptively.

## Stop point

After 36 runs, diagnostics, and the evidence report, stop for human review. No Stage III or other follow-on architecture work is included.
