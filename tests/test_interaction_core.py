from __future__ import annotations

import inspect

import pytest
import torch
from omegaconf import OmegaConf

from src.models.interaction_core import Model
from src.models.interaction_core_components import (
    ConditionalLowRankOperator,
    PairRelationEncoder,
    canonicalize_physical_edges,
    incoming_mean,
    leave_one_out_context,
    pair_relation_features,
)


VARIANTS = ("global_dynamic", "pair_dynamic", "context_static", "context_dynamic")


def _cfg(variant: str = "context_dynamic"):
    return OmegaConf.create({"model": {
        "name": "interaction_core", "hidden_dim": 16, "relation_dim": 64,
        "operator_rank": 32, "num_interaction_steps": 2, "dropout": 0.0,
        "relation_dropout": 0.0, "edge_chunk_size": 2, "variant": variant,
        "relation": {"num_heads": 2},
        "execution": {"num_heads": 2, "zero_init_up": True},
    }})


def _info():
    return {"input_dim": 7, "text_dim": 3, "visual_dim": 4, "num_nodes": 6}


def _x():
    torch.manual_seed(17)
    return torch.randn(6, 7)


def _edges():
    # 0<->1<->2, 3<->4; node 5 is isolated.
    return torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])


@pytest.mark.parametrize("variant", VARIANTS)
def test_model_output_contract_all_variants(variant):
    model = Model(_cfg(variant), _info()).eval()
    z, first, second, aux_loss, extra = model(_x(), _edges())
    assert z.shape == (6, 16)
    assert first is None and second is None
    assert aux_loss.item() == 0.0 and extra == {}
    assert torch.isfinite(z).all()


def test_modality_split_is_text_then_visual_without_early_fusion():
    model = Model(_cfg(), _info())
    text, visual = model._split_features(_x())
    assert torch.equal(text, _x()[:, :3])
    assert torch.equal(visual, _x()[:, 3:])
    assert model.text_projector is not model.visual_projector


def test_src_dst_direction_is_source_j_to_recipient_i():
    canonical = canonicalize_physical_edges(torch.tensor([[0], [1]]), 2)
    src, dst = canonical
    assert {(int(s), int(d)) for s, d in zip(src, dst)} == {(0, 1), (1, 0)}
    # For 0 -> 1, edge-aligned endpoint order is U_i followed by U_j.
    u = torch.tensor([[1., 2.], [8., 9.]])
    pair = torch.cat((u[dst[:1]], u[src[:1]]), dim=-1)
    assert torch.equal(pair[0], torch.tensor([8., 9., 1., 2.]))


def test_canonical_edges_remove_loops_undirect_and_coalesce():
    edges = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 1, 0, 2, 2, 1]])
    actual = canonicalize_physical_edges(edges, 3)
    assert torch.equal(actual, torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]))
    assert not (actual[0] == actual[1]).any()


def test_pair_features_have_expected_shape_and_explicit_compatibility():
    target, source = torch.randn(5, 64), torch.randn(5, 64)
    s_t, s_v = torch.randn(5), torch.randn(5)
    features = pair_relation_features(target, source, s_t, s_v)
    assert features.shape == (5, 4 * 64 + 3)
    assert torch.equal(features[:, :64], target)
    assert torch.equal(features[:, 64:128], source)
    assert torch.equal(features[:, -3], s_t)
    assert torch.equal(features[:, -2], s_v)
    assert torch.equal(features[:, -1], (s_t - s_v).abs())
    assert PairRelationEncoder(64).input_dim == 259


def test_exact_leave_one_out_and_degree_one_no_context():
    u = torch.tensor([[1., 0.], [0., 2.], [9., 9.], [3., 4.]])
    src = torch.tensor([0, 1, 3])
    dst = torch.tensor([2, 2, 0])
    no_context = torch.tensor([7., 8.])
    context, defined, degree = leave_one_out_context(u, src, dst, 4, no_context)
    assert degree.tolist() == [1, 0, 2, 0]
    assert defined.tolist() == [True, True, False]
    assert torch.equal(context[0], u[1])
    assert torch.equal(context[1], u[0])
    assert torch.equal(context[2], no_context)


def test_context_variant_uses_real_recipient_loo_and_no_context_on_degree_one():
    model = Model(_cfg(), _info()).eval()
    details = model.analyze(_x(), _edges())
    src, dst = details["canonical_edge_index"]
    for modality in ("text", "visual"):
        u = details[f"U_{modality}"]
        raw, defined, _ = leave_one_out_context(
            u, src, dst, u.size(0), getattr(model, f"no_context_{modality}")
        )
        assert torch.allclose(details[f"context_{modality}"], raw)
        assert defined.any()
        assert not torch.allclose(details[f"context_{modality}"], getattr(model, f"null_context_{modality}").expand_as(raw))


def test_global_relation_slots_broadcast_to_every_edge():
    model = Model(_cfg("global_dynamic"), _info()).eval()
    details = model.analyze(_x(), _edges())
    assert torch.allclose(details["relation_text"], model.global_relation_text.expand_as(details["relation_text"]))
    assert torch.allclose(details["relation_visual"], model.global_relation_visual.expand_as(details["relation_visual"]))


def test_pair_variant_uses_null_context_tokens():
    model = Model(_cfg("pair_dynamic"), _info()).eval()
    details = model.analyze(_x(), _edges())
    assert torch.equal(details["context_text"], model.null_context_text.expand_as(details["context_text"]))
    assert torch.equal(details["context_visual"], model.null_context_visual.expand_as(details["context_visual"]))


def test_static_query_does_not_depend_on_node_states():
    model = Model(_cfg("context_static"), _info()).eval()
    q1 = model.build_execution_query("text", torch.randn(4, 16), torch.randn(4, 16))
    q2 = model.build_execution_query("text", torch.randn(4, 16) * 100, torch.randn(4, 16) * -37)
    assert torch.equal(q1, q2)


def test_dynamic_query_changes_with_current_source_and_target_states():
    model = Model(_cfg("context_dynamic"), _info()).eval()
    target, source = torch.randn(4, 16), torch.randn(4, 16)
    q1 = model.build_execution_query("text", target, source)
    q2 = model.build_execution_query("text", target + 1.0, source)
    assert not torch.allclose(q1, q2)


def test_relation_memory_is_computed_once_and_execution_weights_are_shared():
    model = Model(_cfg(), _info()).eval()
    relation_calls, execution_calls = [], []
    h1 = model.relation_text_cross_attention.register_forward_hook(lambda *_: relation_calls.append(1))
    h2 = model.text_execution_attention.register_forward_hook(lambda *_: execution_calls.append(id(model.text_execution_attention)))
    with torch.no_grad():
        model(_x(), _edges())
    h1.remove(); h2.remove()
    edge_count = canonicalize_physical_edges(_edges(), 6).size(1)
    assert len(relation_calls) == (edge_count + model.edge_chunk_size - 1) // model.edge_chunk_size
    assert len(execution_calls) == 2 * len(relation_calls)
    assert len(set(execution_calls)) == 1


def test_text_and_visual_graph_execution_streams_are_independent():
    model = Model(_cfg(), _info()).eval()
    details = model.analyze(_x(), _edges())
    edges = details["canonical_edge_index"]
    src, dst = edges
    memory = details["relation_memory"]
    text, visual = details["H0_text"], details["H0_visual"]
    with torch.no_grad():
        before, _ = model._execution_step(visual, memory, src, dst, "visual", True, False)
        _changed_text, _ = model._execution_step(text + 100.0, memory, src, dst, "text", True, False)
        after, _ = model._execution_step(visual, memory, src, dst, "visual", True, False)
    assert torch.equal(before, after)
    assert model.text_operator is not model.visual_operator
    assert model.text_state_update is not model.visual_state_update
    assert text.shape == visual.shape


def test_low_rank_operator_shape_zero_initialization_and_initial_identity():
    operator = ConditionalLowRankOperator(16, 64, 32, zero_init_up=True)
    source, code = torch.randn(7, 16), torch.randn(7, 64)
    base, delta, modulation = operator(source, code)
    assert base.shape == delta.shape == (7, 16)
    assert modulation.shape == (7, 32)
    assert torch.count_nonzero(operator.up.weight) == 0
    assert torch.count_nonzero(operator.up.bias) == 0
    assert torch.equal(base + delta, base)


def test_conditional_operator_branch_receives_finite_gradient():
    model = Model(_cfg(), _info()).train()
    z, _, _, _, _ = model(_x(), _edges())
    z.square().sum().backward()
    grad = model.text_operator.up.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_incoming_mean_correct_and_zero_degree_safe():
    values = torch.tensor([[2.], [4.], [9.]])
    dst = torch.tensor([1, 1, 2])
    actual = incoming_mean(values, dst, 4)
    assert torch.equal(actual, torch.tensor([[0.], [3.], [9.], [0.]]))


def test_empty_graph_and_isolated_nodes_are_safe():
    model = Model(_cfg(), _info()).eval()
    empty_edges = torch.empty((2, 0), dtype=torch.long)
    with torch.no_grad():
        z = model(_x(), empty_edges)[0]
        details = model.analyze(_x(), empty_edges)
    assert z.shape == (6, 16) and torch.isfinite(z).all()
    assert details["relation_memory"].shape == (0, 2, 64)
    assert torch.isfinite(details["fused_z"]).all()


@pytest.mark.parametrize("variant", VARIANTS)
def test_analysis_contains_required_step_diagnostics_and_matches_forward(variant):
    model = Model(_cfg(variant), _info()).eval()
    with torch.no_grad():
        forward = model(_x(), _edges())[0]
        details = model.analyze(_x(), _edges())
    for key in ("H0_text", "H0_visual", "pair_text", "pair_visual", "context_text", "context_visual",
                "relation_text", "relation_visual", "final_text", "final_visual", "fused_z",
                "base_message_norm", "delta_message_norm", "operator_deviation_ratio"):
        assert key in details
    for modality in ("text", "visual"):
        for step in (0, 1):
            assert details[f"execution_code_{modality}_step{step}"].shape == (details["canonical_edge_index"].size(1), 64)
            assert details[f"modulation_{modality}_step{step}"].shape == (details["canonical_edge_index"].size(1), 32)
    assert torch.allclose(details["fused_z"], forward)


def test_operator_off_returns_base_messages_and_zero_delta():
    model = Model(_cfg(), _info()).eval()
    details = model.analyze(_x(), _edges(), operator_enabled=False)
    for key, ratio in details["operator_deviation_ratio"].items():
        assert torch.count_nonzero(details["delta_message_norm"][key]) == 0
        assert torch.count_nonzero(ratio) == 0


def test_intervention_modes_keep_prediction_api_separate_from_labels_and_splits():
    signature = inspect.signature(Model.forward)
    assert "context_mode" in signature.parameters
    assert "relation_mode" in signature.parameters
    assert "operator_enabled" in signature.parameters
    assert not ({"y", "labels", "train_idx", "val_idx", "test_idx"} & set(signature.parameters))
    model = Model(_cfg(), _info()).eval()
    full = model(_x(), _edges())[0]
    no_context = model(_x(), _edges(), context_mode="null")[0]
    mean_relation = model(_x(), _edges(), relation_mode="mean")[0]
    operator_off = model(_x(), _edges(), operator_enabled=False)[0]
    assert full.shape == no_context.shape == mean_relation.shape == operator_off.shape
