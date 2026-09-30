from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.models.final_interaction import Model as FinalModel
from src.models.interaction_core_v3 import Model as V3Model


ROOT = Path(__file__).resolve().parents[1]
DATA_INFO = {"input_dim": 8, "text_dim": 5, "visual_dim": 3,
             "num_nodes": 13, "num_classes": 4}
EDGE_INDEX = torch.tensor([
    [0, 0, 1, 2, 2, 3, 4, 4, 5, 7, 8, 10],
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 8, 9, 11],
], dtype=torch.long)


def _cfg(filename: str, variant: str):
    model = OmegaConf.load(ROOT / "configs/model" / filename)
    model.variant = variant
    return OmegaConf.create({"model": model})


def _x():
    return torch.linspace(-1.0, 1.0, DATA_INFO["num_nodes"] * DATA_INFO["input_dim"]).reshape(
        DATA_INFO["num_nodes"], DATA_INFO["input_dim"]
    )


def _load_shared_v3_weights(reference, final):
    state = reference.state_dict()
    final_keys = final.state_dict()
    mapped = {key: state[key] for key in final_keys if key in state}
    assert set(mapped) == set(final_keys)
    final.load_state_dict(mapped, strict=True)


def test_full_exactly_matches_v3_context_bilinear_absolute():
    torch.manual_seed(31415)
    reference = V3Model(
        _cfg("interaction_core_v3.yaml", "context_bilinear_absolute"), DATA_INFO
    ).eval()
    final = FinalModel(_cfg("final_interaction.yaml", "full"), DATA_INFO).eval()
    _load_shared_v3_weights(reference, final)
    with torch.no_grad():
        old = reference.analyze(_x(), EDGE_INDEX)
        new = final.analyze(_x(), EDGE_INDEX)
    keys = [
        "H0_text", "H0_visual", "pair_text", "pair_visual", "context_text", "context_visual",
        "relation_text", "relation_visual", "relation_memory", "base_relation_text",
        "base_relation_visual", "H1_text", "H1_visual", "H2_text", "H2_visual", "fused_z",
    ]
    for modality in ("text", "visual"):
        for step in (0, 1):
            keys.extend((f"execution_code_{modality}_step{step}",
                         f"modulation_{modality}_step{step}",
                         f"delta_message_{modality}_step{step}"))
    for key in keys:
        old_value = old["final_text"] if key == "H2_text" else (
            old["final_visual"] if key == "H2_visual" else old[key]
        )
        torch.testing.assert_close(new[key], old_value, atol=1e-5, rtol=1e-5)


def test_five_variants_keep_their_frozen_semantics_and_active_sets():
    x = _x()
    edge = EDGE_INDEX
    models = {}
    outputs = {}
    for i, variant in enumerate(("full", "no_context", "shared_relation", "static_execution", "operator_off")):
        torch.manual_seed(800 + i)
        model = FinalModel(_cfg("final_interaction.yaml", variant), DATA_INFO).eval()
        with torch.no_grad():
            outputs[variant] = model.analyze(x, edge)
        models[variant] = model
        assert outputs[variant]["fused_z"].shape == (DATA_INFO["num_nodes"], 256)
        assert torch.isfinite(outputs[variant]["fused_z"]).all()

    null = outputs["no_context"]
    torch.testing.assert_close(
        null["context_text"], models["no_context"].no_context_text.expand_as(null["context_text"])
    )
    torch.testing.assert_close(
        null["context_visual"], models["no_context"].no_context_visual.expand_as(null["context_visual"])
    )
    assert not torch.allclose(null["relation_memory"], outputs["full"]["relation_memory"])

    shared = outputs["shared_relation"]["relation_memory"]
    expected = torch.stack((models["shared_relation"].shared_relation_text,
                            models["shared_relation"].shared_relation_visual), dim=0)
    torch.testing.assert_close(shared, expected.unsqueeze(0).expand_as(shared))
    assert torch.count_nonzero(expected) > 0
    assert not any(name.startswith(("pair_text_encoder.", "pair_visual_encoder."))
                   for name in models["shared_relation"].active_parameter_names())

    static = outputs["static_execution"]
    for modality in ("text", "visual"):
        for step in (0, 1):
            torch.testing.assert_close(
                static[f"execution_code_{modality}_step{step}"], static[f"base_relation_{modality}"]
            )
        assert not any(name.startswith(f"{modality}_conditioner.")
                       for name in models["static_execution"].active_parameter_names())

    off = outputs["operator_off"]
    for modality in ("text", "visual"):
        for step in (0, 1):
            assert torch.count_nonzero(off[f"delta_message_{modality}_step{step}"]) == 0
        names = models["operator_off"].active_parameter_names()
        assert not any(name.startswith((f"{modality}_retriever.", f"{modality}_conditioner.",
                                       f"{modality}_operator.up.", f"{modality}_operator.down.",
                                       f"{modality}_operator.modulation.")) for name in names)
        assert f"{modality}_operator.message.weight" in names

    with torch.no_grad():
        off_plain = models["operator_off"](_x(), EDGE_INDEX)[0]
        off_shuffled = models["operator_off"](_x(), EDGE_INDEX, relation_shuffle=True)[0]
    torch.testing.assert_close(off_plain, off_shuffled, atol=0.0, rtol=0.0)

    assert models["full"].active_parameter_names() == {name for name, _ in models["full"].named_parameters()}


def test_full_gradient_paths_wake_up_after_zero_initialized_gates():
    torch.manual_seed(90210)
    model = FinalModel(_cfg("final_interaction.yaml", "full"), DATA_INFO).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(1, 4):
        model.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        z, *_ = model(_x(), EDGE_INDEX)
        z.square().mean().backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        model(_x(), EDGE_INDEX)
    assert model._epoch_open is False
    assert torch.count_nonzero(model.text_operator.up.weight) > 0
    assert torch.count_nonzero(model.visual_operator.up.weight) > 0
    assert torch.count_nonzero(model.text_conditioner.output.weight) > 0
    assert torch.count_nonzero(model.visual_conditioner.output.weight) > 0
