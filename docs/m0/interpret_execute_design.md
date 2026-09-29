# M0-Core: Contextual Relation Interpretation + Dynamic Semantic Execution

## Research question

The physical graph identifies where two nodes interact. It does not specify how a
relation should transform the source node's text or visual semantics. This stage
implements that missing interface as **Interpret → Execute**. It does not add a
history readout or another propagation stage.

## Inputs and intrinsic states

The loader supplies `x=[text|visual]`. Independent projector stacks produce
`H0_text` and `H0_visual` in 256 dimensions. Independent linear projections map
those intrinsic states to 64 dimensional relation features. The modalities remain
separate until the final terminal-state fusion.

## Physical relations and interpretation

The internal graph removes self-loops, symmetrizes, and coalesces edges. An edge is
stored as `src=j, dst=i` and means `j → i`. For each modality, its endpoint feature
contains `[U_i,U_j,|U_i-U_j|,U_i*U_j]`; both modalities' cosine similarities and
their absolute difference are appended as explicit pair evidence. Each encoder maps
`4*64+3` inputs to 64 dimensions.

Recipient context for edge `j → i` is the exact one-hop leave-one-out mean of the
other relation projections incoming to `i`. A learnable modality-specific
`NO_CONTEXT` token is used when `degree(i)=1`. The pair-only variant supplies
learnable `NULL_CONTEXT` tokens. Contextual interpretation uses one two-head
cross-attention block per modality: `E_text` queries `[E_visual,C_text,C_visual]`
and, symmetrically, `E_visual` queries `[E_text,C_text,C_visual]`. Residual
addition, dropout, and LayerNorm produce the two relation slots. This memory is
computed once from `H0` and reused for both interaction steps.

The global control instead broadcasts two learned relation slots to all edges. Its
normal forward does not run the pair/context encoders.

## State-conditioned execution

For dynamic variants, each modality's edge query is
`LN(W_t H_i,k + W_s H_j,k + e_m)`. `context_static` uses a learned per-modality
query vector. One two-head cross-attention block reads the two relation slots
`[R_text,R_visual]` and returns execution code `xi`.

The source base message is `z=W_msg H_j,k`. A rank-32 bottleneck makes a
relation-conditioned deviation:

```text
v = W_down z
a = tanh(W_a xi)
Delta m = W_up(a ⊙ v)
m = z + Delta m
```

`W_up.weight` and its bias start at zero, so initialization is exactly the base
message. Relation state never enters the message additively. Messages are mean
aggregated by recipient, then each modality performs a residual linear update on
`GELU(M)` followed by dropout and LayerNorm. The update, execution attention, and
operator parameters are shared between the two steps.

The final representation is a `Linear(512,256) → LayerNorm → GELU → Dropout`
fusion of the two terminal states.

## Fixed variants

| Variant | Relation memory | Execution query | Comparison purpose |
|---|---|---|---|
| `global_dynamic` | Two learned global slots | Current source and target states | Control for edge-specific relation interpretation |
| `pair_dynamic` | Endpoint pair evidence with NULL context | Current source and target states | Edge-specific relation without recipient context |
| `context_static` | Pair evidence and real recipient LOO context | Learned static vector | Context with no state-conditioned query |
| `context_dynamic` | Pair evidence and real recipient LOO context | Current source and target states | Full M0-Core |

All use the same relation dimensions, rank, two interaction steps, and Stage-II
operator design. Cross-modal graph message passing, topology changes, source-side
context, auxiliary relation losses, and additional propagation/readout stages are
outside this experiment.

## Training and diagnostics boundary

The experiment uses the existing `unified_full_graph_nc_v1` NC loader and trainer,
with its split, early-stopping, and metric definitions unchanged. Every run disables
test evaluation. Best checkpoints are selected by validation accuracy. The only
variant comparisons are pair-vs-global, contextual-vs-pair, and dynamic-vs-static.

`analyze()` is a separate diagnostics path. It exposes edge-level pair/context and
relation tensors, both steps' execution codes and modulation, message norms, and
operator-deviation ratios. The runner checks the zero initialization and the
smoke checkpoint checks that training makes the operator nonzero. Post-hoc
interventions use the validation-selected `context_dynamic` checkpoint only:
replace context by NULL tokens, broadcast mean relation slots, or set the
conditional deviation to zero. None of these interventions select checkpoints.

Frozen P0 Q1/Q5 utility cases are mapped to physical directed edges after training.
Their operation centroids, radii, normalized separations, and deviation ratios are
descriptive only; P0 cases are not used for training, loss, model selection, or
hyperparameter selection.
