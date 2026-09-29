from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.models.interaction_components import (
    PairEvidenceEncoder,
    canonicalize_physical_edges,
    incoming_mean,
    leave_one_out_context,
    ordered_pair_features,
)
from src.models.interaction_m0 import Model


def _cfg(variant: str = "pair_compat_context"):
    return OmegaConf.create(
        {
            "model": {
                "name": "interaction_m0",
                "hidden_dim": 16,
                "relation_dim": 8,
                "num_interaction_steps": 2,
                "dropout": 0.0,
                "relation_dropout": 0.0,
                "edge_chunk_size": 4,
                "stage1_variant": variant,
                "relation_mixer": {"num_heads": 2, "ffn_ratio": 2},
                "context": {"type": "mean_loo"},
            }
        }
    )


def _info():
    return {"input_dim": 7, "text_dim": 3, "visual_dim": 4, "num_nodes": 6}


def _graph(device="cpu"):
    return torch.tensor(
        [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long, device=device
    )


def _features(device="cpu"):
    return torch.arange(42, dtype=torch.float32, device=device).reshape(6, 7) / 10


def test_text_visual_feature_split_is_explicit_and_ordered() -> None:
    model = Model(_cfg(), _info())
    x = _features()
    text, visual = model._split_features(x)
    assert torch.equal(text, x[:, :3])
    assert torch.equal(visual, x[:, 3:])


def test_physical_graph_removes_loops_undirects_and_coalesces() -> None:
    edges = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 1, 0, 2, 2, 1]])
    actual = canonicalize_physical_edges(edges, 3)
    expected = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    assert torch.equal(actual, expected)
    assert not bool((actual[0] == actual[1]).any())


def test_src_dst_convention_is_source_j_to_recipient_i() -> None:
    u = torch.tensor([[1.0, 2.0], [4.0, 7.0]])
    edge_index = torch.tensor([[0], [1]])  # 0 -> 1, so src=j=0 and dst=i=1.
    src, dst = edge_index
    pair = ordered_pair_features(u[dst], u[src])
    assert torch.equal(pair[0, :2], u[1])
    assert torch.equal(pair[0, 2:4], u[0])
    reverse = ordered_pair_features(u[src], u[dst])
    assert not torch.equal(pair, reverse)


def test_pair_encoder_has_edge_aligned_relation_shape_and_is_direction_sensitive() -> None:
    torch.manual_seed(4)
    encoder = PairEvidenceEncoder(8, dropout=0.0).eval()
    u = torch.randn(3, 8)
    src = torch.tensor([0, 1, 2])
    dst = torch.tensor([1, 0, 1])
    encoded = encoder(u[dst], u[src])
    assert encoded.shape == (3, 8)
    assert not torch.allclose(encoded[0], encoded[1])


def test_compatibility_cosines_and_token_are_finite() -> None:
    model = Model(_cfg(), _info()).eval()
    values = model.analyze(_features(), _graph())
    assert values["compatibility_text"].shape == (6,)
    assert values["compatibility_visual"].shape == (6,)
    assert values["compatibility_token"].shape == (6, 8)
    assert torch.isfinite(values["compatibility_text"]).all()
    assert torch.isfinite(values["compatibility_visual"]).all()
    assert torch.isfinite(values["compatibility_token"]).all()


def test_exact_recipient_leave_one_out_context_and_degree_one_token() -> None:
    # Node 2 receives from 0 and 1; node 0 receives only from 3.
    u = torch.tensor([[1.0, 0.0], [0.0, 2.0], [9.0, 9.0], [3.0, 4.0]])
    src = torch.tensor([0, 1, 3])
    dst = torch.tensor([2, 2, 0])
    no_context = torch.tensor([7.0, 8.0])
    context, defined, degree = leave_one_out_context(u, src, dst, 4, no_context)
    assert degree.tolist() == [1, 0, 2, 0]
    assert defined.tolist() == [True, True, False]
    assert torch.equal(context[0], u[1])
    assert torch.equal(context[1], u[0])
    assert torch.equal(context[2], no_context)


def test_model_degree_one_uses_learned_no_context_vector() -> None:
    model = Model(_cfg(), _info()).eval()
    values = model.analyze(_features(), torch.tensor([[0], [1]]))
    for idx in range(2):
        assert not values["context_defined_mask"][idx]
        assert torch.equal(values["context_text"][idx], model.no_context_text)
        assert torch.equal(values["context_visual"][idx], model.no_context_visual)


def test_degree_zero_nodes_forward_with_zero_aggregate_and_finite_output() -> None:
    model = Model(_cfg("generic"), _info()).eval()
    x = _features()
    edge_index = torch.tensor([[0], [1]])
    with torch.no_grad():
        z, _, _, aux_loss, _ = model(x, edge_index)
    assert z.shape == (6, 16)
    assert torch.isfinite(z).all()
    assert aux_loss.item() == 0.0


class _CaptureMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = None

    def forward(self, tokens, need_weights=False):
        self.tokens = tokens
        attention = tokens.new_zeros((tokens.size(0), 5, 5)) if need_weights else None
        return tokens, attention


@pytest.mark.parametrize(
    "variant,masked_positions",
    [("pair", {2, 3, 4}), ("pair_compat", {3, 4}), ("pair_compat_context", set())],
)
def test_variant_evidence_masking_is_exact(variant: str, masked_positions: set[int]) -> None:
    model = Model(_cfg(variant), _info())
    capture = _CaptureMixer()
    model.relation_mixer = capture
    pair_t, pair_v, comp, ctx_t, ctx_v = [torch.randn(3, 8) for _ in range(5)]
    model._mix_chunk(pair_t, pair_v, comp, ctx_t, ctx_v, need_weights=False)
    assert capture.tokens.shape == (3, 5, 8)
    expected = {
        0: pair_t,
        1: pair_v,
        2: model.null_compatibility.expand(3, -1),
        3: model.null_context_text.expand(3, -1),
        4: model.null_context_visual.expand(3, -1),
    }
    actuals = [pair_t, pair_v, comp, ctx_t, ctx_v]
    for pos, expected_value in expected.items():
        if pos in masked_positions:
            assert torch.equal(capture.tokens[:, pos], expected_value)
        else:
            assert torch.equal(capture.tokens[:, pos], actuals[pos])


def test_pair_pc_pcc_share_identical_parameter_structure() -> None:
    models = [Model(_cfg(name), _info()) for name in ("pair", "pair_compat", "pair_compat_context")]
    assert [sum(p.numel() for p in m.parameters()) for m in models].count(
        sum(p.numel() for p in models[0].parameters())
    ) == 3
    assert [list(m.state_dict()) for m in models].count(list(models[0].state_dict())) == 3


def test_generic_relation_state_and_relation_contribution_are_zero() -> None:
    model = Model(_cfg("generic"), _info()).eval()
    values = model.analyze(_features(), _graph())
    assert torch.count_nonzero(values["relation_text"]) == 0
    assert torch.count_nonzero(values["relation_visual"]) == 0
    assert torch.count_nonzero(values["relation_message_ratio_text"]) == 0
    assert torch.count_nonzero(values["relation_message_ratio_visual"]) == 0


def test_incoming_mean_aggregation_matches_manual_destination_means() -> None:
    edge_values = torch.tensor([[2.0], [4.0], [9.0]])
    dst = torch.tensor([1, 1, 2])
    actual = incoming_mean(edge_values, dst, 4)
    expected = torch.tensor([[0.0], [3.0], [9.0], [0.0]])
    assert torch.equal(actual, expected)


def test_two_interaction_steps_reuse_the_same_parameters() -> None:
    model = Model(_cfg(), _info()).eval()
    calls = []
    handle = model.text_message_source.register_forward_hook(lambda *_: calls.append(1))
    with torch.no_grad():
        model(_features(), _graph())
    handle.remove()
    assert model.num_interaction_steps == 2
    assert calls == [1, 1]
    assert model.text_message_source is model._modules["text_message_source"]


def test_text_visual_execution_streams_have_independent_parameters_and_messages() -> None:
    model = Model(_cfg(), _info()).eval()
    assert model.text_message_source is not model.visual_message_source
    assert model.text_message_relation is not model.visual_message_relation
    assert model.text_message_update is not model.visual_message_update
    edge = canonicalize_physical_edges(_graph(), 6)
    src, dst = edge
    adjacency, degree = __import__(
        "src.models.interaction_components", fromlist=["build_incoming_mean_operator"]
    ).build_incoming_mean_operator(src, dst, 6, torch.float32)
    ht = torch.randn(6, 16)
    hv = torch.randn(6, 16)
    rt = torch.randn(edge.size(1), 8)
    rv = torch.randn(edge.size(1), 8)
    with torch.no_grad():
        base_t, base_v = model._interaction_step(ht, hv, rt, rv, src, dst, adjacency, degree)
        changed_t, changed_v = model._interaction_step(ht + 1.0, hv, rt, rv, src, dst, adjacency, degree)
    assert not torch.allclose(base_t, changed_t)
    assert torch.allclose(base_v, changed_v)


def test_forward_interface_shape_zero_aux_loss_and_finite_backward() -> None:
    model = Model(_cfg(), _info())
    z, first, second, aux_loss, extra = model(_features(), _graph())
    assert z.shape == (6, 16)
    assert first is None and second is None
    assert aux_loss.ndim == 0 and aux_loss.item() == 0.0
    assert extra == {}
    z.square().mean().backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_analyze_does_not_change_eval_forward_result() -> None:
    torch.manual_seed(13)
    model = Model(_cfg(), _info()).eval()
    with torch.no_grad():
        before = model(_features(), _graph())[0]
        diagnostics = model.analyze(_features(), _graph())
        after = model(_features(), _graph())[0]
    assert torch.allclose(before, after, atol=1e-6, rtol=1e-6)
    assert diagnostics["fused_z"].shape == before.shape
    assert diagnostics["relation_attention"].shape == (6, 5, 5)


def test_model_interface_does_not_store_or_read_labels() -> None:
    model = Model(_cfg(), _info())
    assert not hasattr(model, "y")
    assert not hasattr(model, "train_idx")
    assert not hasattr(model, "test_idx")


def test_pcc_context_correction_is_within_the_same_checkpoint() -> None:
    model = Model(_cfg("pair_compat_context"), _info()).eval()
    values = model.analyze(_features(), _graph())
    for key in ("context_correction_text", "context_correction_visual"):
        assert values[key].shape == (6,)
        assert torch.isfinite(values[key]).all()
