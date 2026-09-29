# M0-S1: Contextual Relation Interpretation

## Scope

This stage asks whether multimodal endpoint evidence, a coarse compatibility prior,
and recipient-side local context help represent the interaction potential carried
by a physical relation. It implements M0.0, M0.1-P, M0.1-PC, and M0.1-PCC under
the existing unified full-graph node-classification training loop. It does not
implement an operator bank, expert routing, cross-modal propagation, history
readout, auxiliary supervision, or link prediction.

## Encoder and physical graph

Text and visual inputs are sliced from the existing `[text | visual]` tensor and
encoded by independent `Linear → LayerNorm → GELU → Dropout(0.2)` projectors into
256-dimensional states. The relation graph removes self-loops, is made undirected,
and is coalesced. Every stored arc uses `src=j, dst=i` for the interaction
`j → i`; self-information is retained through the residual state update.

Each modality projects the initial state to a 64-dimensional relation space. For
each directed relation, its ordered pair encoder receives
`[U_i, U_j, |U_i-U_j|, U_i⊙U_j]`. Compatibility features are the text and visual
cosines and their absolute difference, embedded by a separate MLP. Neither cosine
is converted to an edge weight.

## Recipient context and typed mixer

For each modality, node sums and degrees are accumulated over incoming physical
arcs. For `j → i`, the exact context is `(sum_i - U_j)/(degree_i - 1)` when the
recipient degree exceeds one. At degree one, a learned modality-specific
`NO_CONTEXT` vector is used. Degree-zero nodes have no relation edges and receive
zero incoming aggregate.

Every interpreted edge uses five typed tokens in fixed order: text pair, visual
pair, compatibility, text context, visual context. One shared one-layer,
two-head
Transformer-style mixer uses learned type embeddings, residual LayerNorm blocks,
and an FFN with ratio two. P, PC, and PCC instantiate identical parameters and
replace unavailable evidence with learned null tokens. M0.0 skips relation
interpretation and sends zero relation states through the shared executor.

## Generic interaction executor

At each of two steps, shared weights are reused within each modality:

```text
m_ji^m = W_s^m H_j^m + W_r^m R_ji^m
M_i^m  = mean over incoming j of m_ji^m
H_i^m  = LayerNorm(H_i^m + Dropout(W_u^m GELU(M_i^m)))
```

There is no text-to-visual or visual-to-text state propagation. A sparse incoming
mean operator and relation-space mean aggregation compute the same additive
message average without materializing a hidden-dimension message for every edge.
The terminal text and visual states are concatenated and fused to a 256-dimensional
node representation.

## Validation and test boundary

The task runner sets `task.evaluate_test=false`; checkpoint selection remains best
validation accuracy. The M0-specific entry point supplies the Macro-F1 class list
from the dataset's declared `num_classes`, avoiding the existing helper's otherwise
unnecessary read of labels at `test_idx`. No task implementation or split file is
changed. P0 artifacts are only read by the post-training analyzer, after a
validation-selected checkpoint exists.

## Diagnostics

`Model.analyze()` exposes initial states, relation-space states, pair and
compatibility evidence, exact leave-one-out context, relation states, message
ratios, terminal states, fused node embeddings, and mean-head attention weights.
Normal training forward requests no attention weights. PCC also evaluates a
same-checkpoint pair-plus-compatibility masking pass so its context correction is
computed within one model rather than by subtracting states from separately
trained checkpoints.

Hard-case summaries reuse the frozen P0.2 directed relation population. Per
modality, P0 probe-similarity quintiles are reconstructed by stable mergesort;
Q1/Q5 beneficial/harmful centroid radii and normalized centroid separation are
descriptive only and never affect optimization or checkpoint choice.
