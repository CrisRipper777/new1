from __future__ import annotations

import inspect
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.models.interaction_core_v3 import Model as V3Model
from src.models.interaction_core_v3_components import canonicalize_physical_edges
from src.models.interaction_full_s3 import Model, STAGE3_VARIANTS
from src.models.interaction_history_components import (
    HISTORY_TOKEN_LABELS,
    HistoryReadout,
    IncomingMeanStd,
)


def _cfg(stage3_variant="relation_operation_history"):
    return OmegaConf.create({"model": {
        "hidden_dim": 256, "relation_dim": 64, "operator_rank": 32,
        "conditioner_rank": 32, "num_interaction_steps": 2,
        "dropout": 0.0, "relation_dropout": 0.0, "edge_chunk_size": 32,
        "variant": "context_bilinear_absolute", "stage3_variant": stage3_variant,
        "relation": {"num_heads": 2}, "base_relation_retrieval": {"num_heads": 2},
        "attn_dynamic": {"num_heads": 2},
        "history": {"num_heads": 4, "operation_profile": "mean_std", "zero_init_output": True},
    }})


def _graph():
    return canonicalize_physical_edges(
        torch.tensor([[0, 0, 1, 2, 3, 4, 4], [1, 2, 2, 3, 4, 5, 5]]), 7,
    )


def _data_info():
    return {"input_dim": 9, "text_dim": 4, "visual_dim": 5, "num_nodes": 7}


def _model(variant="relation_operation_history"):
    torch.manual_seed(91)
    return Model(_cfg(variant), _data_info())


def _load_shared(v3, s3):
    missing, unexpected = s3.load_state_dict(v3.state_dict(), strict=False)
    expected = {name for name in s3.state_dict() if name not in v3.state_dict()}
    assert set(missing) == expected
    assert not unexpected
    return expected


def test_stage3_variant_set_is_fixed():
    assert STAGE3_VARIANTS == ("terminal", "state_history", "operation_history", "relation_operation_history")


def test_shared_stage1_stage2_weights_and_terminal_match_v3_exactly():
    torch.manual_seed(7)
    v3 = V3Model(_cfg("terminal"), _data_info()).eval()
    s3 = _model("terminal").eval()
    new_keys = _load_shared(v3, s3)
    assert new_keys and all(name.startswith(("history_token_encoder.", "history_query.", "history_readout.", "history_output.")) for name in new_keys)
    x, edge = torch.randn(7, 9), _graph()
    with torch.no_grad():
        old = v3.analyze(x, edge)
        new = s3.analyze(x, edge)
    for modality in ("text", "visual"):
        for old_name, new_name in (("H0", "H0"), ("H1", "H1"), ("final", "H2")):
            assert torch.equal(old[f"{old_name}_{modality}"], new[f"{new_name}_{modality}"])
        assert torch.equal(old[f"base_relation_{modality}"], new[f"base_relation_{modality}"])
        for step in (0, 1):
            assert torch.equal(old[f"modulation_{modality}_step{step}"], new["operation_modulation"][f"{modality}_step{step}"])
    assert torch.equal(old["fused_z"], new["fused_z"])


def test_stage1_formula_and_relation_attention_are_inherited_unchanged():
    model = _model("terminal").eval()
    values = model._pair_and_context(torch.randn(7, 256), torch.randn(7, 256), *_graph(), False, True)
    keys = torch.stack((values["pair_visual"][:1], values["context_text"][:1], values["context_visual"][:1]), 1)
    expected, _ = model.relation_text_cross_attention(values["pair_text"][:1], keys, True)
    assert torch.allclose(values["relation_text"][:1], expected, atol=1e-6, rtol=1e-6)


def test_stage2_uses_absolute_bilinear_formula_and_is_fixed_by_config():
    model = _model("terminal")
    assert model.variant == "context_bilinear_absolute"
    r, h = torch.randn(5, 64), torch.randn(5, 256)
    code, delta, ur, uh, latent = model.text_conditioner(r, h)
    expected_latent = torch.tanh(model.text_conditioner.relation(r)) * torch.tanh(model.text_conditioner.state(h))
    assert torch.equal(latent, expected_latent)
    assert torch.equal(delta, model.text_conditioner.output(latent))
    assert torch.equal(code, r + delta)


def test_operation_profile_mean_std_and_degree_zero():
    values = torch.tensor([[1., 2.], [3., 4.], [5., 0.]])
    targets = torch.tensor([0, 0, 2])
    mean, std = IncomingMeanStd(eps=1e-8)(values, targets, 4)
    assert torch.allclose(mean[0], torch.tensor([2., 3.]))
    assert torch.allclose(std[0], torch.tensor([1., 1.]), atol=1e-7)
    assert torch.equal(mean[1], torch.zeros(2)) and torch.equal(std[1], torch.zeros(2))
    assert torch.equal(mean[3], torch.zeros(2)) and torch.equal(std[3], torch.zeros(2))
    assert torch.equal(mean[2], values[2])


def test_operation_profile_is_differentiable():
    values = torch.randn(4, 3, requires_grad=True)
    _, std = IncomingMeanStd()(values, torch.tensor([0, 0, 1, 1]), 3)
    (std.sum()).backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()


def test_relation_environment_mean_std_and_degree_zero_are_safe():
    model = _model("relation_operation_history")
    relations = torch.tensor([[1., 3.], [3., 1.], [9., 7.]])
    env = model._profile(relations, torch.tensor([1, 1, 3]), 5)
    assert torch.allclose(env[1, :2], torch.tensor([2., 2.]))
    assert torch.allclose(env[1, 2:], torch.tensor([1., 1.]), atol=1e-7)
    assert torch.equal(env[0], torch.zeros(4)) and torch.equal(env[2], torch.zeros(4))
    assert torch.equal(env[4], torch.zeros(4))


def test_transition_definitions_and_intrinsic_zero_operation_token():
    model = _model("relation_operation_history").eval()
    result = model.analyze(torch.randn(7, 9), _graph())
    assert torch.equal(result["transition_text_step0"], torch.zeros_like(result["H0_text"]))
    assert torch.equal(result["transition_text_step1"], result["H1_text"] - result["H0_text"])
    assert torch.equal(result["transition_text_step2"], result["H2_text"] - result["H1_text"])
    enc = model.history_token_encoder
    expected = enc.norm(enc.W_H(result["H0_text"]) + enc.W_D(torch.zeros_like(result["H0_text"]))
                        + enc.W_O(enc.NO_OPERATION_TEXT.expand(7, -1)) + enc.modality_text + enc.stage_0)
    assert torch.equal(result["history_tokens"][:, 0], expected)


def test_history_memory_token_order_is_t0_t1_t2_v0_v1_v2():
    model = _model().eval()
    result = model.analyze(torch.randn(7, 9), _graph())
    assert HISTORY_TOKEN_LABELS == ("T0", "T1", "T2", "V0", "V1", "V2")
    assert result["history_token_labels"] == HISTORY_TOKEN_LABELS
    enc = model.history_token_encoder
    h2v = enc.norm(enc.W_H(result["H2_visual"]) + enc.W_D(result["H2_visual"] - result["H1_visual"])
                   + enc.W_O(result["operation_profile_visual_step1"]) + enc.modality_visual + enc.stage_2)
    assert torch.equal(result["history_tokens"][:, 5], h2v)


def test_state_history_uses_null_operation_profiles():
    model = _model("state_history").eval()
    result = model.analyze(torch.randn(7, 9), _graph())
    null_text = model.history_token_encoder.NULL_OPERATION_TEXT.expand(7, -1)
    null_visual = model.history_token_encoder.NULL_OPERATION_VISUAL.expand(7, -1)
    enc = model.history_token_encoder
    expected = enc.norm(enc.W_H(result["H1_text"]) + enc.W_D(result["H1_text"] - result["H0_text"])
                        + enc.W_O(null_text) + enc.modality_text + enc.stage_1)
    assert torch.equal(result["history_tokens"][:, 1], expected)
    assert not torch.equal(null_text, result["operation_profile_text_step0"])
    assert null_visual.shape == result["operation_profile_visual_step0"].shape


def test_operation_history_uses_real_profiles_and_null_relation_environment():
    model = _model("operation_history").eval()
    result = model.analyze(torch.randn(7, 9), _graph())
    enc = model.history_token_encoder
    expected = enc.norm(enc.W_H(result["H1_text"]) + enc.W_D(result["transition_text_step1"])
                        + enc.W_O(result["operation_profile_text_step0"])
                        + enc.modality_text + enc.stage_1)
    assert torch.equal(result["history_tokens"][:, 1], expected)
    assert torch.equal(result["query_relation_environment_text"], model.history_query.NULL_REL_ENV_TEXT.expand(7, -1))


def test_full_variant_uses_real_relation_environment():
    model = _model("relation_operation_history").eval()
    result = model.analyze(torch.randn(7, 9), _graph())
    assert torch.equal(result["query_relation_environment_text"], result["relation_environment_text"])
    assert torch.equal(result["query_relation_environment_visual"], result["relation_environment_visual"])


def test_history_readout_has_no_query_residual():
    class ZeroAttention(nn.Module):
        def forward(self, query, key, value, **kwargs):
            weights = query.new_full((query.size(0), 4, 1, key.size(1)), 1.0 / key.size(1))
            return torch.zeros_like(query), weights
    readout = HistoryReadout().eval()
    readout.attention = ZeroAttention()
    query, history = torch.randn(3, 256), torch.randn(3, 6, 256)
    output, weights = readout(query, history, need_weights=True)
    assert torch.equal(output, torch.zeros_like(output))
    assert torch.allclose(weights.sum(-1), torch.ones(3, 4))


def test_zero_initialized_history_branch_preserves_terminal_for_all_variants():
    x, edge = torch.randn(7, 9), _graph()
    baseline = _model("terminal").eval()
    outputs = []
    for variant in STAGE3_VARIANTS:
        model = _model(variant).eval()
        model.load_state_dict(baseline.state_dict(), strict=False)
        with torch.no_grad():
            outputs.append(model(x, edge)[0])
        assert torch.count_nonzero(model.history_output.weight) == 0
        assert torch.count_nonzero(model.history_output.bias) == 0
    for output in outputs:
        assert torch.equal(outputs[0], output)


def test_history_branch_opens_then_upstream_parameters_receive_gradient():
    model = _model("relation_operation_history").train()
    x, edge = torch.randn(7, 9), _graph()
    model.set_epoch(1)
    output = model(x, edge)[0]
    output.square().mean().backward()
    first = model.history_output.weight.grad
    assert first is not None and torch.isfinite(first).all() and first.norm() > 0
    assert model.history_token_encoder.W_H.weight.grad is None or torch.count_nonzero(model.history_token_encoder.W_H.weight.grad) == 0
    optimizer = torch.optim.SGD([model.history_output.weight, model.history_output.bias], lr=0.1)
    optimizer.step()
    assert torch.count_nonzero(model.history_output.weight) > 0
    model.zero_grad(set_to_none=True)
    model(x, edge)[0].square().mean().backward()
    for parameter in (model.history_token_encoder.W_H.weight, model.history_token_encoder.W_D.weight,
                      model.history_token_encoder.W_O.weight, model.history_query.W_PT.weight,
                      model.history_query.W_RT.weight, model.history_readout.attention.in_proj_weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0


def test_history_off_reproduces_terminal_from_same_checkpoint_path():
    model = _model("relation_operation_history").eval()
    x = torch.randn(7, 9)
    result = model.analyze(x, _graph())
    off = model.analyze(x, _graph(), history_off=True)
    assert torch.equal(result["terminal_z"], off["fused_z"])
    assert torch.equal(off["history_branch"], torch.zeros_like(off["history_branch"]))


def test_operation_profile_off_changes_tokens_only_not_states_or_transitions():
    model = _model("relation_operation_history").eval()
    x, edge = torch.randn(7, 9), _graph()
    full = model.analyze(x, edge)
    off = model.analyze(x, edge, operation_profile_off=True)
    for key in ("H0_text", "H1_text", "H2_text", "H0_visual", "H1_visual", "H2_visual",
                "transition_text_step1", "transition_text_step2", "readout_query"):
        assert torch.equal(full[key], off[key])
    assert torch.equal(full["history_tokens"][:, 0], off["history_tokens"][:, 0])
    assert not torch.equal(full["history_tokens"][:, 1], off["history_tokens"][:, 1])


def test_operation_alignment_swap_preserves_states_and_transitions():
    model = _model("relation_operation_history").eval()
    with torch.no_grad():
        model.text_conditioner.output.weight.normal_(std=0.05)
        model.visual_conditioner.output.weight.normal_(std=0.05)
    x, edge = torch.randn(7, 9), _graph()
    full = model.analyze(x, edge)
    swap = model.analyze(x, edge, operation_alignment_swap=True)
    for key in ("H0_text", "H1_text", "H2_text", "H0_visual", "H1_visual", "H2_visual",
                "transition_text_step1", "transition_text_step2", "readout_query"):
        assert torch.equal(full[key], swap[key])
    assert torch.equal(full["history_tokens"][:, 0], swap["history_tokens"][:, 0])
    assert not torch.equal(full["history_tokens"][:, 1], swap["history_tokens"][:, 1])
    assert not torch.equal(full["history_tokens"][:, 2], swap["history_tokens"][:, 2])


def test_relation_environment_off_changes_query_not_tokens():
    model = _model("relation_operation_history").eval()
    x, edge = torch.randn(7, 9), _graph()
    full = model.analyze(x, edge)
    off = model.analyze(x, edge, relation_env_off=True)
    assert torch.equal(full["history_tokens"], off["history_tokens"])
    assert not torch.equal(full["readout_query"], off["readout_query"])
    assert torch.equal(off["query_relation_environment_text"], model.history_query.NULL_REL_ENV_TEXT.expand(7, -1))


def test_node_conditioning_off_removes_intrinsic_terms_only():
    model = _model("relation_operation_history").eval()
    x, edge = torch.randn(7, 9), _graph()
    full = model.analyze(x, edge)
    off = model.analyze(x, edge, node_conditioning_off=True)
    assert torch.equal(full["history_tokens"], off["history_tokens"])
    assert torch.equal(full["query_relation_environment_text"], off["query_relation_environment_text"])
    assert not torch.equal(full["readout_query"], off["readout_query"])


def test_global_query_uses_only_dataset_preference():
    model = _model("relation_operation_history").eval()
    x, edge = torch.randn(7, 9), _graph()
    full = model.analyze(x, edge, global_query=True)
    expected = model.history_query.norm(model.history_query.global_readout_query.expand(7, -1))
    assert torch.equal(full["readout_query"], expected)


def test_token_drop_removes_only_token_and_renormalizes_available_attention():
    model = _model("relation_operation_history").eval()
    x, edge = torch.randn(7, 9), _graph()
    full = model.analyze(x, edge)
    dropped = model.analyze(x, edge, drop_token=2)
    weights = dropped["readout_attention_heads"]
    assert torch.equal(weights[:, :, 2], torch.zeros_like(weights[:, :, 2]))
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-6)
    assert not torch.equal(full["readout_attention"], dropped["readout_attention"])


def test_attention_weights_are_finite_and_normalized():
    result = _model().eval().analyze(torch.randn(7, 9), _graph())
    alpha = result["readout_attention_heads"]
    assert alpha.shape == (7, 4, 6)
    assert torch.isfinite(alpha).all()
    assert torch.allclose(alpha.sum(-1), torch.ones(7, 4), atol=1e-6)


def test_forward_nc_contract_and_zero_aux_loss():
    model = _model("relation_operation_history").eval()
    result = model(torch.randn(7, 9), _graph())
    assert len(result) == 5
    z, embedding, aux, loss, metadata = result
    assert z.shape == (7, 256) and embedding is None and aux is None
    assert loss.item() == 0.0 and metadata == {}


def test_analyze_fused_embedding_matches_forward():
    model = _model("relation_operation_history").eval()
    x, edge = torch.randn(7, 9), _graph()
    with torch.no_grad():
        forward = model(x, edge)[0]
        analyzed = model.analyze(x, edge)["fused_z"]
    assert torch.allclose(forward, analyzed, atol=2e-6, rtol=2e-6)


def test_full_model_gradients_are_finite():
    model = _model("relation_operation_history").train()
    model.set_epoch(1)
    model(torch.randn(7, 9), _graph())[0].square().mean().backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_new_training_driver_disables_test_evaluation_and_has_no_label_split_access():
    root = Path(__file__).resolve().parents[1]
    driver = (root / "scripts" / "run_m0_stage3.py").read_text(encoding="utf-8")
    assert '"task.evaluate_test=false"' in driver
    source = inspect.getsource(Model)
    for forbidden in ("test_idx", "train_idx", "val_idx", "data.y", "test_y"):
        assert forbidden not in source
