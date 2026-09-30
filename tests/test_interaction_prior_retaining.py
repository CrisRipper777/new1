from __future__ import annotations

import inspect

import pytest
import torch
from omegaconf import OmegaConf

from src.models.interaction_core_v3 import Model as V3Model
from src.models.interaction_core_v3_components import StateUpdate, canonicalize_physical_edges
from src.models.interaction_prior_retaining import Model, READOUT_VARIANTS, TRACE_GROUPS
from src.models.prior_retention_components import (
    ContextAdapter,
    FeatureContextGate,
    PriorRetainingStateUpdate,
)


def _cfg(readout_variant="prior_update_gate", hidden_dim=16, dropout=0.0):
    return OmegaConf.create({"model": {
        "name": "interaction_prior_retaining",
        "hidden_dim": hidden_dim, "relation_dim": 64, "operator_rank": 32,
        "conditioner_rank": 32, "num_interaction_steps": 2,
        "dropout": dropout, "relation_dropout": 0.0, "edge_chunk_size": 64,
        "variant": "context_bilinear_absolute", "readout_variant": readout_variant,
        "relation": {"num_heads": 2}, "base_relation_retrieval": {"num_heads": 2},
        "attn_dynamic": {"num_heads": 2},
    }})


def _v3_cfg(hidden_dim=16, dropout=0.0):
    cfg = _cfg("terminal", hidden_dim, dropout)
    cfg.model.name = "interaction_core_v3"
    return cfg


def _graph():
    return canonicalize_physical_edges(
        torch.tensor([[0, 0, 1, 2, 2, 3, 3, 4, 5], [1, 2, 2, 3, 4, 4, 5, 5, 6]]), 7
    )


def _model(variant="prior_update_gate", dropout=0.0):
    torch.manual_seed(7)
    return Model(_cfg(variant, dropout=dropout), {"input_dim": 7, "text_dim": 3, "visual_dim": 4})


def _copy_shared(v3, prior):
    source, target = v3.state_dict(), prior.state_dict()
    shared = {k: v for k, v in source.items() if k in target and target[k].shape == v.shape}
    assert len(shared) > 80
    prior.load_state_dict(shared, strict=False)
    return shared


def test_exact_four_readout_variants_and_fixed_v3_absolute_stage2():
    assert READOUT_VARIANTS == (
        "terminal", "prior_delta_gate", "prior_update_add", "prior_update_gate"
    )
    for variant in READOUT_VARIANTS:
        model = _model(variant)
        assert model.variant == "context_bilinear_absolute"
        assert model.readout_variant == variant


def test_terminal_shared_weight_mapping_matches_v3_h0_h1_h2_and_fused_embedding():
    torch.manual_seed(1)
    v3 = V3Model(_v3_cfg(), {"input_dim": 7, "text_dim": 3, "visual_dim": 4}).eval()
    prior = _model("terminal").eval()
    _copy_shared(v3, prior)
    x, edges = torch.randn(7, 7), _graph()
    expected, actual = v3.analyze(x, edges), prior.analyze(x, edges)
    expected["H2_text"], expected["H2_visual"] = expected["final_text"], expected["final_visual"]
    for key in (
        "H0_text", "H0_visual", "H1_text", "H1_visual", "H2_text", "H2_visual",
        "relation_memory", "base_relation_text", "base_relation_visual",
    ):
        assert torch.allclose(actual[key], expected[key], atol=1e-7, rtol=1e-7), key
    for modality in ("text", "visual"):
        for step in (0, 1):
            for suffix in ("modulation", "execution_code", "delta_execution_code", "delta_message"):
                key = f"{suffix}_{modality}_step{step}"
                assert torch.allclose(actual[key], expected[key], atol=1e-7, rtol=1e-7), key
    assert torch.allclose(actual["fused_z"], expected["fused_z"], atol=1e-7, rtol=1e-7)
    assert torch.equal(actual["final_text"], actual["H2_text"])
    assert torch.equal(actual["final_visual"], actual["H2_visual"])


def test_terminal_training_forward_preserves_v3_random_call_sequence():
    torch.manual_seed(2)
    v3 = V3Model(_v3_cfg(dropout=0.3), {"input_dim": 7, "text_dim": 3, "visual_dim": 4})
    prior = _model("terminal", dropout=0.3)
    _copy_shared(v3, prior)
    v3.train(); prior.train()
    x, edges = torch.randn(7, 7), _graph()
    torch.manual_seed(77)
    z_v3 = v3(x, edges)[0]
    rng_v3 = torch.random.get_rng_state()
    torch.manual_seed(77)
    z_prior = prior(x, edges)[0]
    rng_prior = torch.random.get_rng_state()
    assert torch.equal(z_prior, z_v3)
    assert torch.equal(rng_prior, rng_v3)


def test_prior_state_update_uses_exactly_one_v3_dropout_residual():
    base = StateUpdate(16, 0.4)
    captured = PriorRetainingStateUpdate(16, 0.4)
    captured.load_state_dict(base.state_dict())
    base.train(); captured.train()
    state, aggregate = torch.randn(7, 16), torch.randn(7, 16)
    torch.manual_seed(41)
    expected = base(state, aggregate)
    base_rng = torch.random.get_rng_state()
    torch.manual_seed(41)
    captured.capture_residual = True
    actual = captured(state, aggregate)
    prior_rng = torch.random.get_rng_state()
    residual = captured.take_residual()
    assert torch.equal(actual, expected)
    assert torch.equal(actual, captured.norm(state + residual))
    assert torch.equal(base_rng, prior_rng)


def test_state_update_residual_is_captured_without_duplicate_dropout():
    updater = PriorRetainingStateUpdate(8, 0.5).train()
    updater.capture_residual = True
    torch.manual_seed(4)
    state, agg = torch.randn(5, 8), torch.randn(5, 8)
    actual = updater(state, agg)
    residual = updater.take_residual()
    assert torch.equal(actual, updater.norm(state + residual))
    assert updater._last_residual is None


@pytest.mark.parametrize("variant", ["prior_delta_gate", "prior_update_add", "prior_update_gate"])
def test_zero_output_adapter_makes_each_prior_readout_start_at_h0(variant):
    model = _model(variant).eval()
    result = model.analyze(torch.randn(7, 7), _graph())
    assert torch.count_nonzero(model.context_adapter_text.output.weight) == 0
    assert torch.count_nonzero(model.context_adapter_visual.output.weight) == 0
    assert torch.equal(result["E_text"], torch.zeros_like(result["E_text"]))
    assert torch.equal(result["E_visual"], torch.zeros_like(result["E_visual"]))
    assert torch.equal(result["final_text"], result["H0_text"])
    assert torch.equal(result["final_visual"], result["H0_visual"])


def test_gate_initialization_is_exact_half_and_separate_by_modality():
    text_gate, visual_gate = FeatureContextGate(16), FeatureContextGate(16)
    p, e = torch.randn(7, 16), torch.randn(7, 16)
    assert torch.equal(text_gate(p, e), torch.full_like(p, 0.5))
    assert torch.equal(visual_gate(p, e), torch.full_like(p, 0.5))
    assert text_gate is not visual_gate


def test_prior_delta_context_is_h2_minus_h0_and_update_context_is_mean_of_actual_updates():
    for variant in ("prior_delta_gate", "prior_update_add", "prior_update_gate"):
        model = _model(variant).eval()
        result = model.analyze(torch.randn(7, 7), _graph())
        assert torch.equal(result["C_delta_text"], result["H2_text"] - result["H0_text"])
        assert torch.equal(result["C_delta_visual"], result["H2_visual"] - result["H0_visual"])
        expected_t = 0.5 * (result["U_text_step0"] + result["U_text_step1"])
        expected_v = 0.5 * (result["U_visual_step0"] + result["U_visual_step1"])
        assert torch.equal(result["C_update_text"], expected_t)
        assert torch.equal(result["C_update_visual"], expected_v)


def test_update_residuals_are_the_actual_state_updates_in_eval_mode():
    model = _model().eval()
    result = model.analyze(torch.randn(7, 7), _graph())
    for modality in ("text", "visual"):
        for step in (0, 1):
            previous = result[f"H{step}_{modality}"]
            next_state = result[f"H{step + 1}_{modality}"]
            residual = result[f"U_{modality}_step{step}"]
            updater = getattr(model, f"{modality}_state_update")
            assert torch.allclose(next_state, updater.norm(previous + residual), atol=1e-7, rtol=1e-7)


def test_context_adapter_is_shallow_and_zero_output():
    adapter = ContextAdapter(16, 0.0)
    assert isinstance(adapter.input, torch.nn.Linear)
    assert isinstance(adapter.output, torch.nn.Linear)
    assert torch.count_nonzero(adapter.output.weight) == 0
    assert torch.count_nonzero(adapter.output.bias) == 0
    assert torch.equal(adapter(torch.randn(5, 16)), torch.zeros(5, 16))


def test_gate_and_add_variants_follow_distinct_formulas():
    x, edges = torch.randn(7, 7), _graph()
    gate = _model("prior_update_gate").eval().analyze(x, edges)
    add = _model("prior_update_add").eval().analyze(x, edges)
    assert torch.equal(gate["final_text"], gate["H0_text"] + gate["gate_text"] * gate["E_text"])
    assert torch.equal(add["gate_text"], torch.ones_like(add["E_text"]))
    assert torch.equal(add["final_visual"], add["H0_visual"] + add["E_visual"])


def test_context_off_reproduces_exact_prior_while_other_computation_stays_available():
    model = _model("prior_update_gate").eval()
    x, edges = torch.randn(7, 7), _graph()
    baseline = model.analyze(x, edges)
    off = model.analyze(x, edges, context_off=True)
    assert torch.equal(off["final_text"], off["H0_text"])
    assert torch.equal(off["final_visual"], off["H0_visual"])
    assert torch.equal(off["E_text"], baseline["E_text"])
    assert torch.equal(off["gate_text"], baseline["gate_text"])


def test_gate_one_changes_only_allocation_not_adapter_context():
    model = _model("prior_update_gate").eval()
    x, edges = torch.randn(7, 7), _graph()
    baseline, full = model.analyze(x, edges), model.analyze(x, edges, force_gate_one=True)
    assert torch.equal(full["E_text"], baseline["E_text"])
    assert torch.equal(full["C_update_text"], baseline["C_update_text"])
    assert torch.equal(full["gate_text"], torch.ones_like(full["gate_text"]))
    assert torch.equal(full["final_text"], full["H0_text"] + full["E_text"])


def test_prior_off_only_removes_direct_h0_bypass():
    model = _model("prior_update_gate").eval()
    x, edges = torch.randn(7, 7), _graph()
    baseline, off = model.analyze(x, edges), model.analyze(x, edges, prior_bypass_off=True)
    assert torch.equal(off["final_text"], off["R_text"])
    assert torch.equal(off["final_visual"], off["R_visual"])
    assert torch.equal(off["E_text"], baseline["E_text"])
    assert torch.equal(off["gate_text"], baseline["gate_text"])


def test_context_source_override_changes_only_source_summary_before_adapter():
    model = _model("prior_update_gate").eval()
    x, edges = torch.randn(7, 7), _graph()
    with torch.no_grad():
        model.context_adapter_text.output.weight.normal_(std=0.05)
        model.context_adapter_visual.output.weight.normal_(std=0.05)
    update, delta = model.analyze(x, edges), model.analyze(x, edges, context_source_override="delta")
    assert torch.equal(update["C_update_text"], delta["C_update_text"])
    assert not torch.equal(update["E_text"], delta["E_text"])
    assert torch.equal(update["gate_text"], delta["gate_text"])


def test_context_source_override_is_validated():
    model = _model("prior_update_gate").eval()
    with pytest.raises(ValueError):
        model.analyze(torch.randn(7, 7), _graph(), context_source_override="invalid")


def test_operator_off_changes_stage2_operator_only():
    model = _model("prior_update_gate").eval()
    x, edges = torch.randn(7, 7), _graph()
    with torch.no_grad():
        for modality in ("text", "visual"):
            getattr(model, f"{modality}_operator").up.weight.normal_(std=0.05)
            getattr(model, f"{modality}_operator").up.bias.normal_(std=0.01)
            getattr(model, f"{modality}_conditioner").output.weight.normal_(std=0.05)
    full, off = model.analyze(x, edges), model.analyze(x, edges, operator_enabled=False)
    assert torch.equal(full["relation_memory"], off["relation_memory"])
    for modality in ("text", "visual"):
        assert torch.equal(full[f"H0_{modality}"], off[f"H0_{modality}"])
        assert torch.equal(off[f"delta_message_{modality}_step0"],
                           torch.zeros_like(off[f"delta_message_{modality}_step0"]))
        assert not torch.equal(full[f"H2_{modality}"], off[f"H2_{modality}"])


def test_relation_shuffle_preserves_edges_and_reassigns_memory_within_target():
    model = _model("prior_update_gate").eval()
    x, edges = torch.randn(7, 7), _graph()
    full, shuffled = model.analyze(x, edges), model.analyze(x, edges, relation_shuffle=True)
    assert torch.equal(full["canonical_edge_index"], shuffled["canonical_edge_index"])
    assert torch.equal(full["degree"], shuffled["degree"])
    assert not torch.equal(full["relation_memory"], shuffled["relation_memory"])


def test_interventions_preserve_final_fusion_and_classification_contract():
    model = _model("prior_update_gate").eval()
    x, edges = torch.randn(7, 7), _graph()
    result = model.analyze(x, edges)
    expected = model.fusion(torch.cat((result["final_text"], result["final_visual"]), -1))
    actual = model(x, edges)[0]
    assert torch.equal(result["fused_z"], expected)
    assert torch.allclose(actual, result["fused_z"])
    z, aux1, aux2, aux_loss, extra = model(x, edges)
    assert z.shape == (7, 16)
    assert aux1 is None and aux2 is None and float(aux_loss) == 0.0 and extra == {}


def test_context_branch_and_gate_receive_finite_gradients_after_zero_init_opens():
    model = _model("prior_update_gate").train()
    with torch.no_grad():
        model.context_adapter_text.output.weight.normal_(std=0.02)
        model.context_adapter_visual.output.weight.normal_(std=0.02)
    x = torch.randn(7, 7)
    model(x, _graph())[0].square().mean().backward()
    for module in (model.context_adapter_text, model.context_adapter_visual,
                   model.context_gate_text, model.context_gate_visual):
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(float(g.norm()) for g in grads) > 0


def test_v3_stage1_and_stage2_still_receive_finite_gradients_after_adapter_opens():
    model = _model("prior_update_gate").train()
    with torch.no_grad():
        model.context_adapter_text.output.weight.normal_(std=0.02)
        model.context_adapter_visual.output.weight.normal_(std=0.02)
    model(torch.randn(7, 7), _graph())[0].square().mean().backward()
    params = (model.pair_text_encoder.network[0].weight, model.text_operator.up.weight,
              model.text_state_update.update.weight)
    for parameter in params:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_gradient_trace_has_readout_groups_and_training_hooks_are_finite():
    model = _model("prior_update_gate").train()
    assert "prior_context_adapter_text" in TRACE_GROUPS
    assert "prior_context_adapter_visual" in TRACE_GROUPS
    model.set_epoch(1)
    model(torch.randn(7, 7), _graph())[0].square().mean().backward()
    assert any(model._grad_numels)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_terminal_has_only_v3_active_parameters_and_add_bypasses_gates():
    terminal, add = _model("terminal"), _model("prior_update_add")
    assert not any(n.startswith(("context_adapter_", "context_gate_")) for n in terminal.active_parameter_names())
    names = add.active_parameter_names()
    assert any(n.startswith("context_adapter_text") for n in names)
    assert not any(n.startswith("context_gate_") for n in names)


def test_no_auxiliary_or_test_or_split_access_in_model_source():
    source = inspect.getsource(Model)
    assert "aux_loss" not in source
    assert "test_idx" not in source
    assert "train_idx" not in source
    assert "val_idx" not in source


def test_zero_degree_has_zero_update_context_and_finite_outputs():
    model = _model("prior_update_gate").eval()
    result = model.analyze(torch.randn(7, 7), torch.empty((2, 0), dtype=torch.long))
    assert torch.isfinite(result["fused_z"]).all()
    for modality in ("text", "visual"):
        assert torch.isfinite(result[f"C_update_{modality}"]).all()
