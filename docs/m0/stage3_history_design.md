# M0-S3 Design — ROHC

## Question

Stage I and II produce successive relation-conditioned interaction states. ROHC tests whether the intrinsic state, later interaction states, and the operations that produced them should be selectively consolidated for node classification. This is a relation-conditioned interaction-history question, not a multi-order or multi-hop aggregation design.

## Frozen Stage I and II

The model subclasses the v3 absolute conditioner and calls its modality projectors, directed pair/relation encoder, leave-one-out recipient contexts, relation cross-attention, static grounded retriever, absolute bilinear conditioner, rank-32 semantic operator, incoming mean aggregator, state update, and terminal fusion. Stage-II uses `context_bilinear_absolute`; dimensions are fixed to hidden 256, relation 64, operator/conditioner rank 32, and two interaction steps. The terminal variant delegates to the v3 forward method. A shared-weight regression gate compares both models on identical features and edges before smoke/pilot work.

## Operation profiles and state transitions

For modality `m`, target `i`, and interaction step `k`, let `a^m_{ji,k}` be the rank-32 modulation emitted by the existing operator. ROHC builds

`p^m_{i,k} = concat(mean_j(a^m_{ji,k}), std_j(a^m_{ji,k})) ∈ R^64`.

The standard deviation is the population, featurewise incoming-edge standard deviation with `sqrt(var + eps)`. Nodes with no incoming edges receive zero mean and zero standard deviation. Profile construction uses a differentiable indexed reduction and no neighbor attention.

The histories are `H^m=[H^m_0,H^m_1,H^m_2]`, with `D^m_0=0`, `D^m_1=H^m_1−H^m_0`, and `D^m_2=H^m_2−H^m_1`. These are interaction-induced transitions.

## Tokens

One shared token encoder produces three tokens per modality:

- `tau_0 = LN(W_H H_0 + W_D 0 + W_O NO_OPERATION_m + e_m + e_stage0)`
- `tau_1 = LN(W_H H_1 + W_D D_1 + W_O p_0 + e_m + e_stage1)`
- `tau_2 = LN(W_H H_2 + W_D D_2 + W_O p_1 + e_m + e_stage2)`

The six-token memory is ordered `T0,T1,T2,V0,V1,V2`. `state_history` supplies learned NULL_OPERATION vectors for p0/p1. All learned null vectors and encoder parameters are present across the three Stage-III history variants.

## Relation environment and readout

For each modality, incoming base relation codes `r^m_{ji} ∈ R^64` are summarized as `env^m_i=concat(mean_j(r^m_{ji}), std_j(r^m_{ji})) ∈ R^128`; degree-zero nodes receive zeros. This reuses Stage-II relation codes and adds no graph encoder.

The query is `q_i=LN(q0 + W_PT H^T_{i,0} + W_PV H^V_{i,0} + W_RT env^T_i + W_RV env^V_i)`. State-only and operation-only variants substitute learned NULL_REL_ENV vectors. A single four-head cross-attention reads the query against the six tokens, followed by LayerNorm. It has no query residual, self-attention, or FFN stack.

The terminal representation is exactly the existing v3 fused embedding `z_term`. The learned linear `W_hist` is zero initialized and the output is `Z=z_term+W_hist h_hist`, with no extra terminal norm. Thus initialization is exactly terminal-equivalent.

## Fixed variants and contrasts

1. `terminal`: exact v3 terminal output.
2. `state_history`: histories/transitions, NULL_OPERATION and NULL_REL_ENV.
3. `operation_history`: real operation profiles, NULL_REL_ENV.
4. `relation_operation_history`: real profiles and relation environments.

The three paired validation contrasts are state-history minus terminal, operation-history minus state-history, and relation-operation-history minus operation-history. Same-checkpoint interventions turn off the history branch, replace profiles by NULL_OPERATION, swap the two state/profile alignments, replace relation environments by NULL_REL_ENV, remove intrinsic query terms, use q0 alone, or remove one attention-memory token at a time.

## Execution protocol and limits

The pilot uses `unified_full_graph_nc_v1`, validation-accuracy checkpoint selection, and `task.evaluate_test=false`: Movies/Grocery/Reddit-S × seeds 42/43/44 × the four variants. The smoke is Movies seed 42 for five epochs. P0 is post-hoc only. No HPO, LP, test-set evaluation, auxiliary loss, history Transformer, routing, operator bank, MoE, multi-track propagation, topology rewrite, extra graph encoder, or extra propagation module is part of this phase.
