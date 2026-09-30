# PRCI design

## Scope

Prior-Retaining Context Injection (PRCI) changes only the final readout of interaction_core_v3. Stage I and Stage II remain context_bilinear_absolute with hidden dimension 256, relation dimension 64, operator/conditioner rank 32, and two interaction steps. The existing multimodal fusion is reused.

## Intrinsic prior and contextual residual

For modality m, the intrinsic semantic prior is P_i^m = H_i,0^m, the modality projector output before graph interaction. During each of K=2 Stage-II updates, the model computes the actual residual U_i,k^m = Dropout(W_u^m GELU(M_i,k^m)) and applies H_i,k+1^m = LayerNorm(H_i,k^m + U_i,k^m). Instrumentation returns that very residual from the update call; it does not invoke dropout again.

The full candidate summarizes the actual injected updates as C_i^m = (U_i,0^m + U_i,1^m)/2. A control uses the effective state delta H_i,2^m - H_i,0^m, which includes repeated LayerNorm effects.

## Adapter, gate, and output

Each modality has an independent one-layer adapter: Linear(256,256), LayerNorm, GELU, Dropout(0.2), Linear(256,256). The last projection is zero-initialized, so E_i^m = 0 at initialization. The gated candidate computes g_i^m = sigmoid(W_g^m[H_i,0^m || E_i^m]) with independent feature-wise 256-dimensional gates initialized to 0.5. It returns Htilde_i^m = H_i,0^m + g_i^m elementwise-multiplied by E_i^m. There is no post-injection LayerNorm. The original v3 fusion consumes the concatenated modality representations.

## Prespecified variants

- terminal: use H2 for each modality.
- prior_delta_gate: adapt H2-H0 and inject it through the feature gate.
- prior_update_add: adapt mean actual update residual and add it without a gate.
- prior_update_gate: adapt mean actual update residual and inject it through the feature gate.

All variants share the same Stage I/II, classifier protocol, and fusion. Full-checkpoint interventions disable context, substitute delta for update context, force the gate to one, remove the prior bypass, disable the operator, or cyclically shuffle relation memory within each target group. All NC experiments use validation-only checkpoint selection and task.evaluate_test=false.
