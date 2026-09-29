from __future__ import annotations

import inspect

import torch
from omegaconf import OmegaConf

from src.models.interaction_core_v2 import Model, TRACE_GROUPS
from src.models.interaction_core_v2_components import (
    RelationConditionedOperator,
    RelationGroundedRetriever,
    canonicalize_physical_edges,
    pair_relation_features,
    incoming_degree,
    leave_one_out_context,
)
from src.models.interaction_components import incoming_mean


def _cfg(variant="context_grounded_dynamic"):
    return OmegaConf.create({"model": {
        "hidden_dim": 16, "relation_dim": 64, "operator_rank": 32,
        "num_interaction_steps": 2, "dropout": 0.0, "relation_dropout": 0.0,
        "edge_chunk_size": 64, "variant": variant,
        "relation": {"num_heads": 2}, "retrieval": {"num_heads": 2},
    }})


def _graph():
    # Two triangles provide repeated degree-two recipient groups plus an isolate.
    edge_index = torch.tensor([[0, 0, 1, 3, 3, 4, 5], [1, 2, 2, 4, 5, 5, 6]])
    return canonicalize_physical_edges(edge_index, 7)


def _model(variant="context_grounded_dynamic"):
    torch.manual_seed(11)
    return Model(_cfg(variant), {"input_dim": 7, "text_dim": 3, "visual_dim": 4})


def test_zero_relation_memory_produces_exact_zero_code_independent_of_query():
    retriever = RelationGroundedRetriever(64, 2, 0.0).eval()
    query = torch.randn(5, 64)
    memory = torch.zeros(5, 2, 64)
    code, weights = retriever(query, memory, need_weights=True)
    other, _ = retriever(torch.randn_like(query), memory, need_weights=True)
    assert torch.equal(code, torch.zeros_like(code))
    assert torch.equal(code, other)
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights.sum(-1), torch.ones(5))
    assert retriever.attention.in_proj_bias is None
    assert retriever.attention.out_proj.bias is None


def test_retriever_has_no_query_residual_and_attention_is_retrieval_only():
    retriever = RelationGroundedRetriever(64, 2, 0.0).eval()
    query = torch.randn(3, 64)
    memory = torch.randn(3, 2, 64)
    code, _ = retriever(query, memory)
    assert not torch.allclose(code, query)
    assert code.shape == query.shape


def test_dynamic_execution_query_uses_target_only_and_global_memory_is_edge_invariant():
    model = _model("global_grounded_dynamic").eval()
    signature = inspect.signature(model.build_execution_query)
    assert list(signature.parameters) == ["modality", "target"]
    target = torch.randn(1, model.hidden_dim)
    q = model.build_execution_query("text", target)
    memory = torch.stack((model.global_relation_text, model.global_relation_visual), dim=0)
    memory = memory.unsqueeze(0).expand(2, -1, -1)
    query = q.expand(2, -1)
    code, _ = model.text_retriever(query, memory)
    assert torch.allclose(code[0], code[1], atol=0.0, rtol=0.0)


def test_operator_uses_source_content_and_relation_code_to_transform_it():
    torch.manual_seed(3)
    operator = RelationConditionedOperator(16, 64, 32)
    with torch.no_grad():
        operator.up.weight.normal_(std=0.05)
    source = torch.randn(4, 16)
    other_source = source + 0.7
    code = torch.randn(4, 64)
    base, delta, modulation, message = operator(source, code)
    base2, delta2, modulation2, message2 = operator(other_source, code)
    assert torch.allclose(message, base + delta)
    assert not torch.allclose(base, base2)
    assert not torch.allclose(delta, delta2)
    assert torch.equal(modulation, modulation2)  # code alone determines the operation
    assert not torch.allclose(message, message2)


def test_context_is_leave_one_out_and_degree_one_uses_no_context():
    edges = _graph()
    src, dst = edges
    values = torch.arange(7, dtype=torch.float32).unsqueeze(-1)
    no_context = torch.tensor([-99.0])
    context, defined, degree = leave_one_out_context(values, src, dst, 7, no_context)
    assert torch.equal(degree, incoming_degree(dst, 7))
    for idx in range(src.numel()):
        recipient, source = int(dst[idx]), int(src[idx])
        incoming = values[src[dst == recipient]].reshape(-1)
        remaining = incoming[incoming != values[source]]
        if degree[recipient] > 1:
            assert defined[idx]
            assert torch.allclose(context[idx, 0], remaining.mean())
        else:
            assert not defined[idx]
            assert context[idx, 0] == no_context[0]


def test_context_shuffle_pairs_modalities_and_preserves_each_degree_bucket_marginal():
    model = _model()
    edges = _graph()
    src, dst = edges
    degree = incoming_degree(dst, 7)
    text = torch.arange(src.numel(), dtype=torch.float32).unsqueeze(-1)
    visual = text + 100
    out_t, out_v = model._shuffle_context(text, visual, src, dst, degree)
    again_t, again_v = model._shuffle_context(text, visual, src, dst, degree)
    assert torch.equal(out_t, again_t) and torch.equal(out_v, again_v)
    assert torch.equal(out_v - out_t, torch.full_like(out_t, 100))
    for d in (1, 2):
        mask = degree[dst] == d
        assert torch.equal(torch.sort(out_t[mask, 0]).values, torch.sort(text[mask, 0]).values)
    assert torch.equal(out_t[degree[dst] == 1], text[degree[dst] == 1])
    assert not torch.equal(out_t[degree[dst] == 2], text[degree[dst] == 2])


def test_model_forward_and_analysis_contract_and_relation_shuffle_marginal():
    model = _model().eval()
    x = torch.randn(7, 7)
    edges = _graph()
    output = model(x, edges)
    assert len(output) == 5 and output[0].shape == (7, 16)
    full = model.analyze(x, edges)
    shuffled = model.analyze(x, edges, relation_shuffle=True)
    assert full["relation_memory"].shape == (edges.size(1), 2, 64)
    assert full["stage1_attention_text"].shape == (edges.size(1), 3)
    assert full["execution_attention_text_step0"].shape == (edges.size(1), 2)
    assert torch.allclose(
        torch.sort(full["relation_memory"][:, 0, 0])[0],
        torch.sort(shuffled["relation_memory"][:, 0, 0])[0],
    )
    src, dst = edges
    for target in torch.unique(dst):
        positions = torch.nonzero(dst == target, as_tuple=False).reshape(-1)
        if positions.numel() == 1:
            assert torch.equal(full["relation_memory"][positions], shuffled["relation_memory"][positions])
        else:
            for modality in range(2):
                assert torch.allclose(
                    torch.sort(full["relation_memory"][positions, modality, 0])[0],
                    torch.sort(shuffled["relation_memory"][positions, modality, 0])[0],
                )


def test_variant_queries_and_zero_up_initialization():
    dynamic = _model().eval()
    static = _model("context_grounded_static").eval()
    target = torch.randn(2, 16)
    q0 = dynamic.build_execution_query("text", target)
    with torch.no_grad():
        dynamic.text_dynamic_embedding.add_(1.0)
    q1 = dynamic.build_execution_query("text", target)
    assert not torch.allclose(q0, q1)
    static_q = static.build_execution_query("text", target)
    assert torch.equal(static_q[0], static_q[1])
    assert torch.count_nonzero(dynamic.text_operator.up.weight) == 0
    assert torch.count_nonzero(dynamic.visual_operator.up.weight) == 0
    assert len(TRACE_GROUPS) == 10


def test_gradient_hook_records_raw_preclip_group_rms():
    model = _model().train()
    model.set_epoch(1)
    x = torch.randn(7, 7)
    edges = _graph()
    z = model(x, edges)[0]
    (z.square().mean()).backward()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()
    # Simulate the subsequent validation pass used by unified_full_graph_nc_v1.
    model.eval()
    with torch.no_grad():
        model(x, edges)
    assert model.trace_epoch[0].item() == 1
    assert model.gradient_numel_trace[0].sum().item() > 0
    assert model.gradient_rms_trace[0].isfinite().all()
    assert model.operator_up_norm_trace[0].min().item() > 0


# The additional tests below exercise the v2 scientific invariants explicitly.
def test_text_visual_feature_split_is_ordered_and_projectors_are_independent():
    model = _model().eval()
    x = torch.randn(7, 7)
    with torch.no_grad():
        h0 = model.analyze(x, _graph())
        changed = x.clone(); changed[:, :3] += 2.0
        h1 = model.analyze(changed, _graph())
    assert not torch.allclose(h0["H0_text"], h1["H0_text"])
    assert torch.equal(h0["H0_visual"], h1["H0_visual"])
    assert model.text_projector is not model.visual_projector
    assert model.text_operator is not model.visual_operator
    assert model.text_state_update is not model.visual_state_update


def test_canonical_physical_edges_are_bidirectional_unique_and_loop_free():
    edge = torch.tensor([[0, 1, 1], [1, 0, 1]])
    canonical = canonicalize_physical_edges(edge, 3)
    assert canonical.tolist() == [[0, 1], [1, 0]]


def test_pair_feature_formula_contains_cross_modal_compatibility():
    target = torch.tensor([[1.0, 2.0]])
    source = torch.tensor([[3.0, 5.0]])
    st, sv = torch.tensor([0.25]), torch.tensor([-0.5])
    features = pair_relation_features(target, source, st, sv)
    expected = torch.tensor([[1., 2., 3., 5., 2., 3., 3., 10., .25, -.5, .75]])
    assert torch.equal(features, expected)
    assert torch.equal(features[0, -3:], torch.tensor([.25, -.5, .75]))


def test_pair_variant_uses_null_context_and_context_variant_uses_real_loo():
    edges = _graph(); x = torch.randn(7, 7)
    pair = _model("pair_grounded_dynamic").eval()
    ctx = _model("context_grounded_dynamic").eval()
    with torch.no_grad():
        pair_values = pair.analyze(x, edges)
        ctx_values = ctx.analyze(x, edges)
        assert torch.equal(pair_values["context_text"], pair.null_context_text.expand(edges.size(1), -1))
        u = ctx.text_relation_projection(ctx_values["H0_text"])
        src, dst = ctx_values["canonical_edge_index"]
        expected, _, _ = leave_one_out_context(u, src, dst, x.size(0), ctx.no_context_text)
        assert torch.allclose(ctx_values["context_text"], expected)
        assert not torch.equal(ctx_values["context_text"], ctx.null_context_text.expand_as(expected))


def test_relation_memory_is_built_once_and_stage_two_parameters_are_shared_between_steps():
    model = _model().eval()
    calls, retrieval_calls, need_weights = [], [], []
    h1 = model.pair_text_encoder.register_forward_hook(lambda *_: calls.append(1))
    h2 = model.text_retriever.register_forward_hook(lambda *_: retrieval_calls.append(1))
    h3 = model.text_retriever.attention.register_forward_pre_hook(
        lambda module, args, kwargs: need_weights.append(kwargs.get("need_weights")), with_kwargs=True
    )
    with torch.no_grad():
        model(torch.randn(7, 7), _graph())
    h1.remove(); h2.remove(); h3.remove()
    assert len(calls) == 1
    assert len(retrieval_calls) == 2
    assert need_weights == [False, False]
    assert model.text_retriever is model.text_retriever
    assert model.visual_retriever is model.visual_retriever
    assert model.text_operator is model.text_operator


def test_dynamic_query_changes_with_target_but_api_has_no_source_argument():
    model = _model().eval()
    a, b = torch.randn(2, 16), torch.randn(2, 16)
    q_a = model.build_execution_query("text", a)
    q_b = model.build_execution_query("text", b)
    assert not torch.allclose(q_a, q_b)
    assert list(inspect.signature(model.build_execution_query).parameters) == ["modality", "target"]


def test_static_query_is_independent_of_node_states():
    model = _model("context_grounded_static").eval()
    targets = [torch.randn(5, 16), torch.randn(5, 16) * 100]
    q0 = model.build_execution_query("visual", targets[0])
    q1 = model.build_execution_query("visual", targets[1])
    assert torch.equal(q0, q1)


def test_edge_specific_memory_changes_code_for_fixed_recipient_query():
    retriever = RelationGroundedRetriever(64, 2, 0.0).eval()
    torch.manual_seed(29)
    query = torch.randn(2, 64)
    memories = torch.randn(2, 2, 64)
    codes, _ = retriever(query, memories)
    assert not torch.allclose(codes[0], codes[1])


def test_source_content_changes_message_but_never_retrieval_code():
    model = _model().eval()
    query = torch.randn(1, 64)
    memory = torch.randn(1, 2, 64)
    source_a, source_b = torch.randn(1, 16), torch.randn(1, 16)
    with torch.no_grad():
        code_a, _ = model.text_retriever(query, memory)
        code_b, _ = model.text_retriever(query, memory)
        base_a, _, _, _ = model.text_operator(source_a, code_a)
        base_b, _, _, _ = model.text_operator(source_b, code_b)
    assert torch.equal(code_a, code_b)
    assert not torch.allclose(base_a, base_b)


def test_target_changes_execution_code_without_changing_source_base_message():
    model = _model().eval()
    torch.manual_seed(17)
    target_a, target_b = torch.randn(1, 16), torch.randn(1, 16)
    source = torch.randn(1, 16)
    memory = torch.randn(1, 2, 64)
    with torch.no_grad():
        code_a, _ = model.text_retriever(model.build_execution_query("text", target_a), memory)
        code_b, _ = model.text_retriever(model.build_execution_query("text", target_b), memory)
        base_a, _, _, _ = model.text_operator(source, code_a)
        base_b, _, _, _ = model.text_operator(source, code_b)
    assert not torch.allclose(code_a, code_b)
    assert torch.equal(base_a, base_b)


def test_zero_up_initialization_makes_operator_message_equal_source_base():
    model = _model().eval()
    source, code = torch.randn(4, 16), torch.randn(4, 64)
    base, delta, _, message = model.text_operator(source, code)
    assert torch.count_nonzero(model.text_operator.up.weight) == 0
    assert torch.count_nonzero(model.text_operator.up.bias) == 0
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(message, base)


def test_w_up_receives_nonzero_first_backward_gradient():
    model = _model().train()
    z = model(torch.randn(7, 7), _graph())[0]
    (z.square().mean()).backward()
    assert model.text_operator.up.weight.grad is not None
    assert model.text_operator.up.weight.grad.norm().item() > 0
    assert torch.isfinite(model.text_operator.up.weight.grad).all()


def test_zero_w_up_then_update_unblocks_finite_stage_one_gradients():
    torch.manual_seed(37)
    model = _model().train()
    head = torch.nn.Linear(16, 3)
    optim = torch.optim.SGD(list(model.parameters()) + list(head.parameters()), lr=0.05)
    x, edges = torch.randn(7, 7), _graph()
    target = torch.tensor([0, 1, 2, 0, 1, 2, 1])
    model.set_epoch(1); optim.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(head(model(x, edges)[0]), target)
    loss.backward()
    p = model.text_relation_projection.weight
    assert p.grad is None or torch.count_nonzero(p.grad) == 0
    optim.step()
    model.set_epoch(2); optim.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(head(model(x, edges)[0]), target)
    loss.backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert p.grad.norm().item() > 0


def test_incoming_aggregation_is_correct_and_zero_degree_nodes_are_safe():
    src = torch.tensor([0, 2, 1]); dst = torch.tensor([1, 1, 2])
    messages = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    aggregated = incoming_mean(messages, dst, 4)
    assert torch.equal(aggregated[1], torch.tensor([2., 3.]))
    assert torch.equal(aggregated[2], torch.tensor([5., 6.]))
    assert torch.equal(aggregated[0], torch.zeros(2))
    assert torch.equal(aggregated[3], torch.zeros(2))


def test_two_modalities_have_separate_stage_two_graph_operators():
    model = _model()
    text_names = {n for n, _ in model.named_parameters() if n.startswith("text_operator.")}
    visual_names = {n for n, _ in model.named_parameters() if n.startswith("visual_operator.")}
    assert text_names and visual_names
    assert {n.split(".", 1)[1] for n in text_names} == {n.split(".", 1)[1] for n in visual_names}
    assert model.text_operator.message is not model.visual_operator.message
    assert model.text_retriever is not model.visual_retriever


def test_stage_two_aggregation_has_no_cross_modal_state_message():
    class IdentityUpdate(torch.nn.Module):
        def forward(self, state, aggregate):
            return state + aggregate
    class SourceOnlyOperator(torch.nn.Module):
        def forward(self, source, code, enabled=True):
            zero = torch.zeros_like(source)
            return source, zero, code.new_zeros((source.size(0), 32)), source
    model = _model().eval()
    model.text_state_update = IdentityUpdate()
    model.text_operator = SourceOnlyOperator()
    edges = _graph(); src, dst = edges
    state = torch.randn(7, 16); memory = torch.randn(edges.size(1), 2, 64)
    with torch.no_grad():
        next_state, _ = model._execution_step(state, state, memory, src, dst,
                                               "text", True, False)
    expected = state + incoming_mean(state[src], dst, 7)
    assert torch.allclose(next_state, expected, atol=1e-6)


def test_analyze_forward_equivalence_aux_zero_contract_and_finite_gradients():
    model = _model().train()
    x, edges = torch.randn(7, 7), _graph()
    z, _, _, aux, _ = model(x, edges)
    assert aux.item() == 0.0
    loss = z.square().mean(); loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    model.eval()
    with torch.no_grad():
        forward = model(x, edges)[0]
        analyzed = model.analyze(x, edges)["fused_z"]
    assert torch.allclose(forward, analyzed)


def test_model_has_no_label_or_split_access():
    model = _model()
    forbidden = ("label", "split", "train_idx", "val_idx", "test_idx")
    assert not any(any(term in name.lower() for term in forbidden) for name in vars(model))
    assert not hasattr(model, "y")



def test_relation_shuffle_preserves_graph_targets_degree_and_per_target_marginal():
    model = _model().eval(); x = torch.randn(7, 7); edges = _graph()
    with torch.no_grad():
        a = model.analyze(x, edges)
        b = model.analyze(x, edges, relation_shuffle=True)
    assert torch.equal(a["canonical_edge_index"], b["canonical_edge_index"])
    assert torch.equal(a["degree"], b["degree"])
    src, dst = a["canonical_edge_index"]
    for target in torch.unique(dst):
        positions = torch.where(dst == target)[0]
        assert a["degree"][target] == b["degree"][target]
        assert torch.allclose(torch.sort(a["relation_memory"][positions].reshape(-1))[0],
                              torch.sort(b["relation_memory"][positions].reshape(-1))[0])


def test_execution_attention_weights_cover_both_relation_views():
    model = _model().eval()
    with torch.no_grad():
        values = model.analyze(torch.randn(7, 7), _graph())
    for modality in ("text", "visual"):
        for step in (0, 1):
            attn = values[f"execution_attention_{modality}_step{step}"]
            assert attn.shape == (values["canonical_edge_index"].size(1), 2)
            assert torch.allclose(attn.sum(-1), torch.ones(attn.size(0)), atol=1e-6)


def test_shared_stage_two_weights_have_no_step_specific_parameter_copies():
    model = _model()
    names = [name for name, _ in model.named_parameters()]
    assert not any("step0" in name or "step1" in name for name in names)
    assert sum(name == "text_operator.up.weight" for name in names) == 1
    assert sum(name == "visual_operator.up.weight" for name in names) == 1


def test_degree_one_context_records_are_excluded_from_context_shuffle():
    model = _model(); edges = _graph(); src, dst = edges
    degree = incoming_degree(dst, 7)
    text = torch.arange(src.numel(), dtype=torch.float32).unsqueeze(-1)
    visual = text + 50
    shuffled_t, shuffled_v = model._shuffle_context(text, visual, src, dst, degree)
    one = degree[dst] == 1
    assert torch.equal(shuffled_t[one], text[one])
    assert torch.equal(shuffled_v[one], visual[one])
    assert torch.equal(shuffled_v - shuffled_t, torch.full_like(text, 50))


def test_zero_degree_node_remains_finite_in_complete_model_forward():
    model = _model().eval()
    # Node 7 is a zero-degree isolated node.
    edges = canonicalize_physical_edges(_graph(), 8)
    with torch.no_grad():
        z = model(torch.randn(8, 7), edges)[0]
    assert z.shape == (8, 16)
    assert torch.isfinite(z).all()
