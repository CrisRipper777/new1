from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402


OUTPUT_ROOT = ROOT / "outputs" / "m0" / "stage1_relation"
RESULT_ROOT = ROOT / "results" / "m0" / "stage1_relation"
VARIANTS = ("generic", "pair", "pair_compat", "pair_compat_context")
DATASETS = ("Movies", "Grocery", "Reddit-S")
SEEDS = (42, 43, 44)
DEFAULT_P0_ROOT = Path("/hdd1/DataInHere/YHF/mag_model/outputs/problem_validation")
ATTENTION_PAIRS = {
    "Text": {"VisualPair": (0, 1), "Compatibility": (0, 2), "TextContext": (0, 3), "VisualContext": (0, 4)},
    "Visual": {"TextPair": (1, 0), "Compatibility": (1, 2), "TextContext": (1, 3), "VisualContext": (1, 4)},
}


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _compose(dataset: str, seed: int, variant: str):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(
            config_name="config",
            overrides=[
                f"dataset={dataset}",
                "task=nc",
                "model=interaction_m0",
                f"model.stage1_variant={variant}",
                f"seed={seed}",
                "num_runs=1",
                "task.evaluate_test=false",
            ],
        )


def _checkpoint_path(mode: str, dataset: str, variant: str, seed: int) -> Path:
    if mode == "smoke":
        return OUTPUT_ROOT / "smoke" / "checkpoints" / dataset / variant / f"seed{seed}.pt"
    return OUTPUT_ROOT / dataset / variant / f"seed{seed}.pt"


def _summary(values: torch.Tensor) -> tuple[float, float, float]:
    if values.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(values.mean().item()),
        float(values.std(unbiased=False).item()),
        float(values.var(dim=0, unbiased=False).mean().item()) if values.ndim == 2 else float("nan"),
    )


def _quantile_masks(values: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(values, kind="mergesort")
    masks = []
    for group in np.array_split(order, 5):
        mask = np.zeros(values.shape[0], dtype=bool)
        mask[group] = True
        masks.append(mask)
    return masks


def _load_p0_artifacts(p0_root: Path, dataset: str, seed: int):
    diag_path = p0_root / "p02" / dataset / f"seed{seed}" / "conditional_feature" / "edge_diagnostics.pt"
    split_path = p0_root / "splits" / f"{dataset}.pt"
    diagnostics = torch.load(diag_path, map_location="cpu", weights_only=False)
    split = torch.load(split_path, map_location="cpu", weights_only=False)
    if diagnostics.get("test_evaluation") is not False or diagnostics.get("test_labels_accessed") is not False:
        raise RuntimeError(f"P0 artifact is not marked test-sealed: {diag_path}")
    if not torch.equal(diagnostics["target_node"], split["sampled_edge_target_node"]):
        raise RuntimeError(f"Frozen P0 target relation sequence differs from split cache: {dataset}/{seed}")
    if not torch.equal(diagnostics["neighbor_node"], split["sampled_edge_neighbor_node"]):
        raise RuntimeError(f"Frozen P0 neighbor relation sequence differs from split cache: {dataset}/{seed}")
    return diagnostics


def _hardcase_rows(
    p0: dict,
    relation_text: torch.Tensor,
    relation_visual: torch.Tensor,
    canonical_edge_index: torch.Tensor,
    num_nodes: int,
    dataset: str,
    seed: int,
    variant: str,
) -> list[dict]:
    src, dst = canonical_edge_index.detach().cpu()
    graph_keys = src * int(num_nodes) + dst
    p0_src = p0["neighbor_node"].long()
    p0_dst = p0["target_node"].long()
    p0_keys = p0_src * int(num_nodes) + p0_dst
    indices = torch.searchsorted(graph_keys, p0_keys)
    safe_indices = indices.clamp(max=max(graph_keys.numel() - 1, 0))
    if graph_keys.numel() == 0 or not torch.equal(graph_keys[safe_indices], p0_keys):
        raise RuntimeError(f"A frozen P0 directed relation was not found in the physical graph: {dataset}/{seed}")
    indices = indices.to(relation_text.device)

    rows = []
    for modality, representations, similarity_key, utility_key in (
        ("Text", relation_text, "probe_sim_text", "utility_ce_text"),
        ("Visual", relation_visual, "probe_sim_visual", "utility_ce_visual"),
    ):
        similarity = p0[similarity_key].numpy()
        utility = p0[utility_key].numpy()
        q_masks = _quantile_masks(similarity)
        representation = representations[indices].detach().float().cpu()
        for regime, q_index in (("Q1", 0), ("Q5", 4)):
            regime_mask = q_masks[q_index]
            group_stats: dict[str, tuple[int, float, float, torch.Tensor]] = {}
            for group_name, group_mask in (
                ("beneficial", regime_mask & (utility > 0)),
                ("harmful", regime_mask & (utility < 0)),
            ):
                selected = representation[torch.from_numpy(group_mask)]
                if selected.numel() == 0:
                    centroid = representation.new_full((representation.size(1),), float("nan"))
                    centroid_norm = radius = float("nan")
                else:
                    centroid = selected.mean(dim=0)
                    centroid_norm = float(centroid.norm().item())
                    radius = float((selected - centroid).norm(dim=-1).mean().item())
                group_stats[group_name] = (int(selected.size(0)), centroid_norm, radius, centroid)
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "variant": variant,
                        "modality": modality,
                        "similarity_regime": regime,
                        "utility_group": group_name,
                        "n_relations": int(selected.size(0)),
                        "centroid_norm": centroid_norm,
                        "within_group_radius": radius,
                        "normalized_centroid_separation": float("nan"),
                    }
                )
            beneficial = group_stats["beneficial"]
            harmful = group_stats["harmful"]
            if beneficial[0] and harmful[0] and torch.isfinite(beneficial[3]).all() and torch.isfinite(harmful[3]).all():
                separation = float((beneficial[3] - harmful[3]).norm().item())
                normalized = separation / (0.5 * (beneficial[2] + harmful[2]) + 1e-12)
            else:
                normalized = float("nan")
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "variant": variant,
                    "modality": modality,
                    "similarity_regime": regime,
                    "utility_group": "beneficial_vs_harmful",
                    "n_relations": beneficial[0] + harmful[0],
                    "centroid_norm": float("nan"),
                    "within_group_radius": float("nan"),
                    "normalized_centroid_separation": normalized,
                }
            )
    return rows


def _attention_rows(
    values: dict,
    p0: dict | None,
    canonical_edge_index: torch.Tensor,
    num_nodes: int,
    dataset: str,
    seed: int,
    variant: str,
) -> list[dict]:
    attention = values.get("relation_attention")
    if attention is None:
        return []
    attention = attention.detach().float().cpu()
    if p0 is None:
        sample_attention = attention
        groups = [("all", "all", torch.ones(attention.size(0), dtype=torch.bool))]
    else:
        src, dst = canonical_edge_index.detach().cpu()
        graph_keys = src * int(num_nodes) + dst
        p0_keys = p0["neighbor_node"].long() * int(num_nodes) + p0["target_node"].long()
        p0_indices = torch.searchsorted(graph_keys, p0_keys)
        sample_attention = attention[p0_indices]
        groups = [("all", "all", torch.ones(p0_keys.numel(), dtype=torch.bool))]
        for modality, similarity_key, utility_key in (
            ("Text", "probe_sim_text", "utility_ce_text"),
            ("Visual", "probe_sim_visual", "utility_ce_visual"),
        ):
            similarity = p0[similarity_key].numpy()
            utility = p0[utility_key].numpy()
            masks = _quantile_masks(similarity)
            for regime, q in (("Q1", 0), ("Q5", 4)):
                groups.append((regime, f"{modality}_beneficial", torch.from_numpy(masks[q] & (utility > 0))))
                groups.append((regime, f"{modality}_harmful", torch.from_numpy(masks[q] & (utility < 0))))

    rows = []
    for regime, utility_group, mask in groups:
        if not bool(mask.any()):
            continue
        selected = sample_attention[mask]
        for query_name, key_map in ATTENTION_PAIRS.items():
            for evidence_name, (query_index, key_index) in key_map.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "variant": variant,
                        "query_token": query_name,
                        "evidence_key": evidence_name,
                        "similarity_regime": regime,
                        "utility_group": utility_group,
                        "mean_attention": float(selected[:, query_index, key_index].mean().item()),
                        "n_relations": int(selected.size(0)),
                    }
                )
    return rows


def _diagnostic_rows(values: dict, dataset: str, seed: int, variant: str) -> list[dict]:
    r_text = values["relation_text"]
    r_visual = values["relation_visual"]
    rows = []
    if r_text.numel():
        cross_modal = torch.nn.functional.cosine_similarity(r_text, r_visual, dim=-1, eps=1e-8).mean()
        cross_modal_value = float(cross_modal.item())
    else:
        cross_modal_value = float("nan")
    for modality, relation, ratio_key, correction_key in (
        ("Text", r_text, "relation_message_ratio_text", "context_correction_text"),
        ("Visual", r_visual, "relation_message_ratio_visual", "context_correction_visual"),
    ):
        if relation.size(0):
            norm = relation.norm(dim=-1)
            norm_mean = float(norm.mean().item())
            norm_std = float(norm.std(unbiased=False).item())
            variance = float(relation.var(dim=0, unbiased=False).mean().item())
        else:
            norm_mean = norm_std = variance = float("nan")
        ratio = values[ratio_key]
        if ratio is not None and ratio.numel():
            ratio_q = torch.quantile(ratio.float(), torch.tensor([0.1, 0.5, 0.9], device=ratio.device))
            ratio_mean = float(ratio.mean().item())
            ratio_median = float(ratio_q[1].item())
            ratio_p10 = float(ratio_q[0].item())
            ratio_p90 = float(ratio_q[2].item())
        else:
            ratio_mean = ratio_median = ratio_p10 = ratio_p90 = float("nan")
        correction = values.get(correction_key)
        if correction is not None and correction.numel():
            correction_mean = float(correction.mean().item())
            correction_std = float(correction.std(unbiased=False).item())
        else:
            correction_mean = correction_std = float("nan")
        defined = values.get("context_defined_mask")
        degree1_rate = float((~defined).float().mean().item()) if defined is not None and defined.numel() else float("nan")
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "variant": variant,
                "modality": modality,
                "relation_norm_mean": norm_mean,
                "relation_norm_std": norm_std,
                "relation_feature_variance": variance,
                "cross_modal_relation_cosine": cross_modal_value,
                "relation_message_ratio_mean": ratio_mean,
                "relation_message_ratio_median": ratio_median,
                "relation_message_ratio_p10": ratio_p10,
                "relation_message_ratio_p90": ratio_p90,
                "context_correction_norm_mean": correction_mean,
                "context_correction_norm_std": correction_std,
                "degree1_context_rate": degree1_rate,
            }
        )
    return rows


def _selected_runs(mode: str, args):
    if mode == "smoke":
        return [("Movies", 42, variant) for variant in args.variants]
    return [(dataset, seed, variant) for dataset in args.datasets for seed in args.seeds for variant in args.variants]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze validation-selected M0-S1 checkpoints.")
    parser.add_argument("--mode", choices=("smoke", "pilot"), default="pilot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--p0-root", type=Path, default=DEFAULT_P0_ROOT)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")

    relation_rows: list[dict] = []
    hardcase_rows: list[dict] = []
    attention_rows: list[dict] = []
    for dataset, seed, variant in _selected_runs(args.mode, args):
        checkpoint_path = _checkpoint_path(args.mode, dataset, variant, seed)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("selection") != "best_val_accuracy":
            raise RuntimeError(f"Checkpoint is not selected by validation accuracy: {checkpoint_path}")
        if any(name.startswith("test_") for name in checkpoint.get("metrics", {})):
            raise RuntimeError(f"Test metric found in checkpoint: {checkpoint_path}")
        cfg = _compose(dataset, seed, variant)
        data = load_mag_data(cfg, "nc", seed)
        data_info = {
            "input_dim": data.input_dim,
            "num_nodes": data.num_nodes,
            "num_classes": data.num_classes,
            "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
            "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
        }
        model = build_model(cfg, data_info).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        with torch.no_grad():
            values = model.analyze(data.x.to(device), data.edge_index.to(device))
            forward_z = model(data.x.to(device), data.edge_index.to(device))[0]
        assert torch.allclose(values["fused_z"].cpu(), forward_z.cpu(), atol=2e-5, rtol=2e-5)
        relation_rows.extend(_diagnostic_rows(values, dataset, seed, variant))

        p0 = None
        if args.mode == "pilot":
            p0 = _load_p0_artifacts(args.p0_root, dataset, seed)
            hardcase_rows.extend(
                _hardcase_rows(
                    p0,
                    values["relation_text"],
                    values["relation_visual"],
                    values["canonical_edge_index"],
                    data.num_nodes,
                    dataset,
                    seed,
                    variant,
                )
            )
        attention_rows.extend(
            _attention_rows(
                values,
                p0,
                values["canonical_edge_index"],
                data.num_nodes,
                dataset,
                seed,
                variant,
            )
        )
        print(f"ANALYZED {args.mode} {dataset}/{variant}/seed{seed} edges={values['canonical_edge_index'].size(1)}", flush=True)
        del model, data, values
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fields = [
        "dataset", "seed", "variant", "modality", "relation_norm_mean", "relation_norm_std",
        "relation_feature_variance", "cross_modal_relation_cosine", "relation_message_ratio_mean",
        "relation_message_ratio_median", "relation_message_ratio_p10", "relation_message_ratio_p90",
        "context_correction_norm_mean", "context_correction_norm_std", "degree1_context_rate",
    ]
    if args.mode == "smoke":
        _write_csv(RESULT_ROOT / "smoke_relation_diagnostics.csv", relation_rows, fields)
        _write_csv(
            RESULT_ROOT / "smoke_attention_diagnostics.csv",
            attention_rows,
            ["dataset", "seed", "variant", "query_token", "evidence_key", "similarity_regime", "utility_group", "mean_attention", "n_relations"],
        )
    else:
        _write_csv(RESULT_ROOT / "relation_diagnostics.csv", relation_rows, fields)
        _write_csv(
            RESULT_ROOT / "p0_hardcase_diagnostics.csv",
            hardcase_rows,
            ["dataset", "seed", "variant", "modality", "similarity_regime", "utility_group", "n_relations", "centroid_norm", "within_group_radius", "normalized_centroid_separation"],
        )
        _write_csv(
            RESULT_ROOT / "attention_diagnostics.csv",
            attention_rows,
            ["dataset", "seed", "variant", "query_token", "evidence_key", "similarity_regime", "utility_group", "mean_attention", "n_relations"],
        )


if __name__ == "__main__":
    main()
