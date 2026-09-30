# Final MAG Interaction Model Specification

This document is the architecture authority for the frozen NC benchmark and the subsequent LP implementation. The model has two reasoning stages followed by terminal multimodal fusion:

**contextual relation interpretation → relation-conditioned semantic execution → terminal fusion**

The scientific question is: *A physical relation specifies where interaction occurs, but under-specifies how multimodal semantics should interact.* Stage I estimates the interaction potential of an observed relation under recipient context. Stage II uses that relation state to determine how source semantics act on the current recipient.

## Stage I: contextual relation interpretation

Text and visual features are encoded independently:

\[
H^T_0=f_T(X^T),\qquad H^V_0=f_V(X^V),\qquad H^m_0\in\mathbb{R}^{N\times256}.
\]

Each stream is a linear projection, LayerNorm, GELU, and dropout. The streams are not fused before message passing. Each is projected to a 64-dimensional relation space, \(U^m=W^m_{rel}H^m_0\).

For each canonical directed physical arc \(j\to i\), let \(s_T=\cos(U_i^T,U_j^T)\), \(s_V=\cos(U_i^V,U_j^V)\), and \(d=|s_T-s_V|\). The ordered pair feature for modality \(m\) is

\[
[U_i^m,U_j^m,|U_i^m-U_j^m|,U_i^m\odot U_j^m,s_T,s_V,d]\in\mathbb{R}^{259}.
\]

Separate text and visual pair encoders map 259 to 64 dimensions using Linear–GELU–LayerNorm–Dropout–Linear.

Recipient context excludes the current source neighbor:

\[
C^m_{i\setminus j}=\frac{\sum_{u\in N(i)}U^m_u-U^m_j}{d_i-1},\quad d_i>1.
\]

For degree-one recipients, a learned modality-specific `NO_CONTEXT` vector is used. Each modality then queries three tokens: the other modality's pair evidence, text context, and visual context. Two-head cross-attention, residual dropout, and LayerNorm produce \(R^T_{ji},R^V_{ji}\in\mathbb{R}^{64}\). Relation memory is \([R^T_{ji},R^V_{ji}]\in\mathbb{R}^{2\times64}\), calculated once per forward pass from \(H_0\).

## Stage II: relation-conditioned semantic execution

For each modality, a learned static query reads the two-token relation memory with two-head, bias-free attention. There is no query residual; the output LayerNorm is non-affine. This gives grounded base code \(r^m_{ji}\in\mathbb{R}^{64}\). An all-zero memory gives an all-zero code.

At interaction step \(k\), the absolute bilinear conditioner is

\[
u_r=\tanh(W_r^m r^m_{ji}),\quad u_h=\tanh(W_h^m H^m_{i,k}),\\
\xi^m_{ji,k}=r^m_{ji}+W_c^m(u_r\odot u_h).
\]

The rank is 32 and \(W_c\) is zero-initialized, so execution starts at \(\xi=r\). The same Stage-II parameters are shared across both steps.

Source content and its relation-conditioned operation are

\[
z^m_{j,k}=W^m_{msg}H^m_{j,k},\quad v^m_{j,k}=W^m_{down}z^m_{j,k},\\
a^m_{ji,k}=\tanh(W^m_a\xi^m_{ji,k}),\quad
\Delta^m_{ji,k}=W^m_{up}(a^m_{ji,k}\odot v^m_{j,k}),\\
m^m_{ji,k}=z^m_{j,k}+\Delta^m_{ji,k}.
\]

Operator rank is 32. `W_up` weight and bias are zero-initialized. Incoming messages use a simple mean. Each modality updates independently:

\[
H^m_{i,k+1}=\operatorname{LN}\left(H^m_{i,k}+\operatorname{Dropout}\left(W^m_u\operatorname{GELU}(M^m_{i,k})\right)\right),\quad k=0,1.
\]

The terminal representation is `Fusion([H2_text || H2_visual])`, implemented as Linear(512,256), LayerNorm, GELU, and dropout. There is no H0 bypass or intermediate-state readout.

## Dimensions, initialization, and sharing

| Component | Frozen setting |
|---|---:|
| Hidden width | 256 per modality |
| Relation width | 64 |
| Pair encoder input | 259 |
| Relation cross-attention | 2 heads |
| Base relation retrieval | 2 heads, bias-free |
| Conditioner rank | 32 |
| Operator rank | 32 |
| Interaction steps | 2 |
| Feature dropout | 0.2 |
| Relation dropout | 0.1 |
| Edge chunk size | 16,384 |
| Shared across steps | Retriever, conditioner, operator, update modules |

Learned null contexts and static retrieval queries use `normal_(mean=0, std=0.02)`. The full model contains no Stage III, PRCI, dynamic attention query, or history-readout parameters. Zero-initialized `W_c` and `W_up` are retained as specified above.

## Final NC variants

| Variant | Single change from full |
|---|---|
| `full` | Real recipient LOO contexts, edge-specific relation memory, absolute bilinear conditioning, low-rank operator. |
| `no_context` | Replace both LOO contexts on every edge with learned modality-specific null context tokens; retain pair evidence, cross-attention, and Stage II. |
| `shared_relation` | Replace each edge's relation memory with two learned global 64-dimensional slots initialized with `normal_(0, 0.02)`; retain retrieval and all of Stage II. |
| `static_execution` | Set \(\xi=r\) at both steps; retain relation memory, source operator, aggregation, and updates. |
| `operator_off` | Set \(\Delta=0\) and keep \(z\), aggregation, and updates. Its relation branch is intentionally output-inactive. |

## NC training and reporting protocol

The only benchmark protocol is `unified_full_graph_nc_v1`: full-graph training; AdamW; 300 maximum epochs; learning rate 0.001; weight decay 0.0001; evaluate every epoch; patience 30; minimum epoch 30; minimum validation-accuracy improvement 0.0001; gradient clip 1.0. Seeds are 42, 43, and 44. Datasets are Movies, Toys, Grocery, ele-fashion, and Reddit-S. There is no hyperparameter search.

Every run selects its checkpoint by validation accuracy. Test metrics are calculated once, after restoring that checkpoint. Macro-F1 uses the class universe declared by dataset metadata, so the NC runner does not inspect test labels to choose the F1 label set before training. The architecture and hyperparameters are frozen before these final test evaluations.

The final benchmark contains the five variants above only. Results use per-dataset mean and sample SD over three seeds, paired seed-level descriptive deltas, and win/tie/loss counts. Cross-dataset averages are descriptive macro-averages, not statistical tests. No significance claim is based on three seeds.

## Explicitly rejected components

- Stage III relation-operation history consolidation.
- PRCI and prior-retaining readout.
- Delta-state conditioning.
- Two-slot attention dynamic conditioner.
- `multi_order_bank` and GPR/multi-order framing.
- H0 final bypass and history/intermediate readouts.
- Additional NC variants beyond the five controls above.

These exclusions are part of the freeze. NC test results do not reopen architecture search; only a documented implementation bug can require a corrective rerun.
