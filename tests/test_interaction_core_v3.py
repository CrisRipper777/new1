from __future__ import annotations

import inspect

import torch
from omegaconf import OmegaConf

from src.models.interaction_core_v3 import Model, TRACE_GROUPS, VARIANTS
from src.models.interaction_core_v3_components import (
    RelationGroundedRetriever,
    RelationStateBilinearConditioner,
    canonicalize_physical_edges,
    operation_variance_decomposition,
    pair_relation_features,
    incoming_degree,
    leave_one_out_context,
)
from src.models.interaction_components import incoming_mean


def _cfg(variant="context_bilinear_delta"):
    return OmegaConf.create({"model": {
        "hidden_dim": 16, "relation_dim": 64, "operator_rank": 32,
        "conditioner_rank": 32, "num_interaction_steps": 2,
        "dropout": 0.0, "relation_dropout": 0.0, "edge_chunk_size": 64,
        "variant": variant, "relation": {"num_heads": 2},
        "base_relation_retrieval": {"num_heads": 2}, "attn_dynamic": {"num_heads": 2},
    }})


def _graph():
    return canonicalize_physical_edges(
        torch.tensor([[0, 0, 1, 3, 3, 4, 5], [1, 2, 2, 4, 5, 5, 6]]), 7
    )


def _model(variant="context_bilinear_delta"):
    torch.manual_seed(11)
    return Model(_cfg(variant), {"input_dim": 7, "text_dim": 3, "visual_dim": 4})


def _edge_inputs(model, variant=None):
    model = model or _model(variant)
    src = torch.tensor([0, 2], dtype=torch.long)
    dst = torch.tensor([1, 1], dtype=torch.long)
    state = torch.randn(4, model.hidden_dim)
    initial = torch.randn_like(state)
    memory = torch.randn(2, 2, model.relation_dim)
    r = model._base_relation_code("text", memory)
    return model, state, initial, memory, r, src, dst


def test_variant_set_is_exactly_the_four_registered_conditioners():
    assert VARIANTS == ("context_static", "context_attn_dynamic",
                        "context_bilinear_absolute", "context_bilinear_delta")


def test_v2_stage1_relation_formula_is_preserved():
    model = _model().eval()
    h_t, h_v = torch.randn(7, 16), torch.randn(7, 16)
    edges = _graph(); src, dst = edges
    values = model._pair_and_context(h_t, h_v, src, dst, False, True)
    keys = torch.stack((values["pair_visual"][:2], values["context_text"][:2], values["context_visual"][:2]), 1)
    expected, _ = model.relation_text_cross_attention(values["pair_text"][:2], keys, True)
    assert torch.allclose(values["relation_text"][:2], expected, atol=1e-7)


def test_exact_incoming_loo_context_and_degree_one_no_context_token():
    edges = _graph(); src, dst = edges
    values = torch.arange(7, dtype=torch.float32).unsqueeze(-1)
    no_context = torch.tensor([-99.0])
    context, defined, degree = leave_one_out_context(values, src, dst, 7, no_context)
    assert torch.equal(degree, incoming_degree(dst, 7))
    for k, (source, target) in enumerate(edges.t().tolist()):
        other = values[src[(dst == target) & (src != source)]].mean(0) if degree[target] > 1 else no_context
        assert torch.equal(context[k], other)
        assert bool(defined[k]) == (int(degree[target]) > 1)


def test_directed_pair_encoding_and_cross_modal_feature_formula():
    a, b = torch.tensor([[1., 2.]]), torch.tensor([[3., 5.]])
    st, sv = torch.tensor([.25]), torch.tensor([-.5])
    expected = torch.tensor([[1., 2., 3., 5., 2., 3., 3., 10., .25, -.5, .75]])
    assert torch.equal(pair_relation_features(a, b, st, sv), expected)
    assert not torch.equal(pair_relation_features(a, b, st, sv), pair_relation_features(b, a, st, sv))


def test_contextual_relation_memory_shape_and_two_modality_streams():
    model = _model().eval()
    result = model.analyze(torch.randn(7, 7), _graph())
    assert result["relation_text"].shape == (14, 64)
    assert result["relation_visual"].shape == (14, 64)
    assert result["relation_memory"].shape == (14, 2, 64)
    assert model.text_projector is not model.visual_projector
    assert model.text_state_update is not model.visual_state_update
    assert model.text_operator is not model.visual_operator


def test_base_relation_code_is_purely_grounded_and_query_is_static():
    model = _model().eval()
    memory = torch.randn(3, 2, 64)
    code = model._base_relation_code("text", memory)
    assert torch.equal(code, model._base_relation_code("text", memory))
    assert code.shape == (3, 64)


def test_zero_relation_memory_gives_zero_base_code():
    retriever = RelationGroundedRetriever(64, 2, 0.0).eval()
    code, _ = retriever(torch.randn(5, 64), torch.zeros(5, 2, 64))
    assert torch.equal(code, torch.zeros_like(code))
    assert retriever.attention.in_proj_bias is None and retriever.attention.out_proj.bias is None


def test_static_variant_execution_code_equals_base_relation_code():
    model, state, initial, memory, r, src, dst = _edge_inputs(_model("context_static"))
    _, details = model._execution_step(state, initial, memory, r, src, dst, "text", True, True, 0)
    assert torch.equal(details["execution_code"], r)


def test_attention_dynamic_code_uses_target_state_not_source_state():
    model, state, initial, memory, r, src, dst = _edge_inputs(_model("context_attn_dynamic"))
    model.eval()
    _, original = model._execution_step(state, initial, memory, r, src, dst, "text", True, True, 0)
    changed_target = state.clone(); changed_target[1] += torch.randn_like(changed_target[1]) * 3
    _, target_result = model._execution_step(changed_target, initial, memory, r, src, dst, "text", True, True, 0)
    changed_source = state.clone(); changed_source[0] += torch.randn_like(changed_source[0]) * 3
    _, source_result = model._execution_step(changed_source, initial, memory, r, src, dst, "text", True, True, 0)
    assert not torch.allclose(original["execution_code"][0], target_result["execution_code"][0])
    assert torch.equal(original["execution_code"], source_result["execution_code"])


def test_bilinear_absolute_target_state_changes_delta_but_source_state_does_not():
    model, state, initial, memory, r, src, dst = _edge_inputs(_model("context_bilinear_absolute"))
    model.eval()
    with torch.no_grad():
        model.text_conditioner.output.weight.normal_(std=.1)
    _, original = model._execution_step(state, initial, memory, r, src, dst, "text", True, True, 0)
    changed_target = state.clone(); changed_target[1] += torch.randn_like(changed_target[1]) * 2
    _, target_result = model._execution_step(changed_target, initial, memory, r, src, dst, "text", True, True, 0)
    changed_source = state.clone(); changed_source[0] += torch.randn_like(changed_source[0]) * 2
    _, source_result = model._execution_step(changed_source, initial, memory, r, src, dst, "text", True, True, 0)
    assert not torch.allclose(original["delta_execution_code"][0], target_result["delta_execution_code"][0])
    assert torch.equal(original["execution_code"], source_result["execution_code"])


def test_bilinear_zero_relation_memory_is_grounded_for_any_target_state():
    model = _model("context_bilinear_absolute").eval()
    memory = torch.zeros(1, 2, 64)
    r = model._base_relation_code("text", memory)
    state = torch.randn(3, 16); src = torch.tensor([0]); dst = torch.tensor([1])
    _, result = model._execution_step(state, state, memory, r, src, dst, "text", True, True, 1)
    assert torch.equal(r, torch.zeros_like(r))
    assert torch.equal(result["delta_execution_code"], torch.zeros_like(r))
    assert torch.equal(result["execution_code"], torch.zeros_like(r))


def test_w_c_is_zero_initialized_and_bilinear_init_is_static_for_absolute_and_delta():
    for variant in ("context_bilinear_absolute", "context_bilinear_delta"):
        model, state, initial, memory, r, src, dst = _edge_inputs(_model(variant))
        conditioner = model.text_conditioner
        assert torch.count_nonzero(conditioner.output.weight) == 0
        _, details = model._execution_step(state, initial, memory, r, src, dst, "text", True, True, 1)
        assert torch.equal(details["execution_code"], r)
        assert torch.equal(details["delta_execution_code"], torch.zeros_like(r))


def test_delta_step0_is_exactly_static_even_when_w_c_is_nonzero():
    model, state, _, memory, r, src, dst = _edge_inputs(_model("context_bilinear_delta"))
    model.eval()
    with torch.no_grad():
        model.text_conditioner.output.weight.normal_(std=.2)
    _, details = model._execution_step(state, state, memory, r, src, dst, "text", True, True, 0)
    assert torch.equal(details["delta_execution_code"], torch.zeros_like(r))
    assert torch.equal(details["execution_code"], r)


def test_delta_step1_state_change_can_change_execution_code():
    model, state, initial, memory, r, src, dst = _edge_inputs(_model("context_bilinear_delta"))
    model.eval()
    with torch.no_grad():
        model.text_conditioner.output.weight.normal_(std=.2)
    changed = state.clone(); changed[1] += torch.randn_like(changed[1])
    _, details = model._execution_step(changed, initial, memory, r, src, dst, "text", True, True, 1)
    assert details["delta_execution_code"].norm() > 0


def test_source_h_changes_base_semantic_message_but_not_execution_code():
    model, state, initial, memory, r, src, dst = _edge_inputs(_model("context_bilinear_absolute"))
    model.eval()
    with torch.no_grad():
        model.text_conditioner.output.weight.normal_(std=.1)
    _, original = model._execution_step(state, initial, memory, r, torch.tensor([0]), torch.tensor([1]), "text", True, True, 0)
    changed = state.clone(); changed[0] += 1
    _, result = model._execution_step(changed, initial, memory, r, torch.tensor([0]), torch.tensor([1]), "text", True, True, 0)
    assert torch.equal(original["execution_code"], result["execution_code"])
    original_base = model.text_operator.message(state[torch.tensor([0])])
    changed_base = model.text_operator.message(changed[torch.tensor([0])])
    assert not torch.equal(original_base, changed_base)


def test_target_h_changes_execution_code_but_not_source_base_message():
    model, state, initial, memory, r, src, dst = _edge_inputs(_model("context_bilinear_absolute"))
    model.eval()
    with torch.no_grad():
        model.text_conditioner.output.weight.normal_(std=.1)
    src, dst = torch.tensor([0]), torch.tensor([1])
    _, original = model._execution_step(state, initial, memory, r[:1], src, dst, "text", True, True, 0)
    changed = state.clone(); changed[1] += 1
    _, result = model._execution_step(changed, initial, memory, r[:1], src, dst, "text", True, True, 0)
    assert not torch.equal(original["execution_code"], result["execution_code"])
    original_base = model.text_operator.message(state[torch.tensor([0])])
    changed_base = model.text_operator.message(changed[torch.tensor([0])])
    assert torch.equal(original_base, changed_base)


def test_operator_w_up_is_zero_and_initial_message_equals_z():
    model = _model()
    for operator in (model.text_operator, model.visual_operator):
        assert torch.count_nonzero(operator.up.weight) == 0
        assert torch.count_nonzero(operator.up.bias) == 0
        source, code = torch.randn(4, 16), torch.randn(4, 64)
        z, delta, _, message = operator(source, code)
        assert torch.equal(delta, torch.zeros_like(delta))
        assert torch.equal(message, z)


def test_conditioner_output_receives_gradient_after_operator_path_opens():
    model = _model("context_bilinear_absolute").train()
    with torch.no_grad():
        model.text_operator.up.weight.normal_(std=.03)
        model.visual_operator.up.weight.normal_(std=.03)
    z = model(torch.randn(7, 7), _graph())[0]
    z.square().mean().backward()
    grad = model.text_conditioner.output.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0


def test_stage1_receives_task_gradients_after_operator_path_opens():
    model = _model("context_bilinear_delta").train()
    with torch.no_grad():
        model.text_operator.up.weight.normal_(std=.03)
        model.visual_operator.up.weight.normal_(std=.03)
    model(torch.randn(7, 7), _graph())[0].square().mean().backward()
    grad = model.pair_text_encoder.network[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0


def test_operation_variance_decomposition_identity_and_eta_ranges():
    x = torch.tensor([[1., 0.], [3., 2.], [-1., 4.], [2., 3.]])
    target = torch.tensor([0, 0, 1, 2])
    result = operation_variance_decomposition(x, target, 4)
    assert torch.allclose(result["V_total"], result["V_within_target"] + result["V_between_target"], atol=1e-7)
    assert abs(float(result["decomposition_error"])) < 1e-7
    assert 0 <= result["eta_relation"] <= 1
    assert 0 <= result["eta_target"] <= 1
    assert result["node_balanced_within"] >= 0 and result["node_mean_variance"] >= 0


def test_incoming_mean_aggregation_and_degree_zero_are_safe():
    messages = torch.tensor([[2., 0.], [4., 2.], [7., 1.]])
    dst = torch.tensor([1, 1, 2])
    aggregate = incoming_mean(messages, dst, 4)
    assert torch.equal(aggregate[1], torch.tensor([3., 1.]))
    assert torch.equal(aggregate[2], torch.tensor([7., 1.]))
    assert torch.equal(aggregate[[0, 3]], torch.zeros(2, 2))
    model = _model().eval()
    memory = torch.empty(0, 2, 64); state = torch.randn(4, 16); r = torch.empty(0, 64)
    out, _ = model._execution_step(state, state, memory, r, torch.empty(0, dtype=torch.long),
                                  torch.empty(0, dtype=torch.long), "text", True, True, 0)
    assert torch.isfinite(out).all()


def test_no_explicit_cross_modal_graph_propagation_and_modality_projectors_are_independent():
    model = _model().eval()
    assert not hasattr(model, "text_to_visual_graph") and not hasattr(model, "visual_to_text_graph")
    x = torch.randn(7, 7)
    before = model.analyze(x, _graph())
    changed = x.clone(); changed[:, :3] += 2
    after = model.analyze(changed, _graph())
    assert not torch.equal(before["H0_text"], after["H0_text"])
    assert torch.equal(before["H0_visual"], after["H0_visual"])


def test_forward_output_contract_aux_loss_zero_and_analyze_matches_forward():
    model = _model().eval(); x, edges = torch.randn(7, 7), _graph()
    result = model(x, edges)
    analyzed = model.analyze(x, edges)
    assert len(result) == 5 and result[0].shape == (7, 16)
    assert result[1] is None and result[2] is None and result[3].item() == 0 and result[4] == {}
    assert torch.allclose(result[0], analyzed["fused_z"], atol=0, rtol=0)


def test_gradients_are_finite_and_model_does_not_read_labels_or_splits():
    model = _model().train()
    z = model(torch.randn(7, 7), _graph())[0]
    z.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    signature = inspect.signature(Model.forward)
    assert "labels" not in signature.parameters and "split" not in signature.parameters
    source = inspect.getsource(Model.forward)
    assert "data.y" not in source and "train_idx" not in source and "test_idx" not in source


def test_gradient_trace_has_requested_conditioner_groups():
    assert "stage2_conditioner_relation" in TRACE_GROUPS
    assert "stage2_conditioner_state" in TRACE_GROUPS
    assert "stage2_conditioner_output" in TRACE_GROUPS
    model = _model()
    assert model._group_for_parameter("text_conditioner.relation.weight") == "stage2_conditioner_relation"
    assert model._group_for_parameter("visual_conditioner.state.weight") == "stage2_conditioner_state"
    assert model._group_for_parameter("text_conditioner.output.weight") == "stage2_conditioner_output"


def test_attn_dynamic_frozen_query_hook_changes_only_step1_query():
    model = _model("context_attn_dynamic").eval()
    values = model.analyze(torch.randn(7, 7), _graph())
    h0, h1 = values["H0_text"], values["H1_text"]
    q0 = model.build_execution_query("text", h0)
    q1 = model.build_execution_query("text", h1)
    assert not torch.allclose(q0, q1)
    result = model.analyze(torch.randn(7, 7), _graph(), frozen_query=True)
    assert result["execution_code_text_step0"].shape == values["execution_code_text_step0"].shape
