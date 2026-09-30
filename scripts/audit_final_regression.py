from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from src.models.final_interaction import Model as FinalModel
from src.models.interaction_core_v3 import Model as V3Model


def _config(path: Path, variant: str):
    model = OmegaConf.load(path)
    model.variant = variant
    return OmegaConf.create({"model": model})


def main() -> None:
    data_info = {
        "input_dim": 8, "text_dim": 5, "visual_dim": 3,
        "num_nodes": 13, "num_classes": 4,
    }
    torch.manual_seed(20260930)
    reference = V3Model(
        _config(ROOT / "configs/model/interaction_core_v3.yaml", "context_bilinear_absolute"),
        data_info,
    ).eval()
    final = FinalModel(_config(ROOT / "configs/model/final_interaction.yaml", "full"), data_info).eval()
    reference_state = reference.state_dict()
    final_state = final.state_dict()
    shared = {key: reference_state[key] for key in final_state if key in reference_state}
    missing = sorted(set(final_state) - set(shared))
    if missing:
        raise AssertionError(f"final parameters missing from v3 state: {missing}")
    final.load_state_dict(shared, strict=True)

    x = torch.linspace(-1.0, 1.0, data_info["num_nodes"] * data_info["input_dim"]).reshape(
        data_info["num_nodes"], data_info["input_dim"]
    )
    edge_index = torch.tensor([
        [0, 0, 1, 2, 2, 3, 4, 4, 5, 7, 8, 10],
        [1, 2, 2, 3, 4, 4, 5, 6, 6, 8, 9, 11],
    ], dtype=torch.long)
    with torch.no_grad():
        old = reference.analyze(x, edge_index)
        new = final.analyze(x, edge_index)

    checks = [
        "H0_text", "H0_visual", "pair_text", "pair_visual",
        "context_text", "context_visual", "relation_text", "relation_visual",
        "relation_memory", "base_relation_text", "base_relation_visual",
        "H1_text", "H1_visual", "H2_text", "H2_visual", "fused_z",
    ]
    for modality in ("text", "visual"):
        for step in (0, 1):
            checks.extend((
                f"execution_code_{modality}_step{step}",
                f"modulation_{modality}_step{step}",
                f"delta_message_{modality}_step{step}",
            ))
    for key in checks:
        old_value = old["final_text"] if key == "H2_text" else (
            old["final_visual"] if key == "H2_visual" else old[key]
        )
        torch.testing.assert_close(new[key], old_value, atol=1e-5, rtol=1e-5, msg=lambda msg: f"{key}: {msg}")
        max_abs = float((new[key] - old_value).abs().max().item()) if new[key].numel() else 0.0
        print(f"PASS {key}: max_abs={max_abs:.3e}")
    print(f"PASS mapped_parameters={len(shared)}/{len(final_state)} checks={len(checks)}")


if __name__ == "__main__":
    main()
