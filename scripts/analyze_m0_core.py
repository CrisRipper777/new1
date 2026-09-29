from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402


OUTPUT_ROOT = ROOT / "outputs" / "m0" / "interpret_execute"
RESULT_ROOT = ROOT / "results" / "m0" / "interpret_execute"
VARIANTS = ("global_dynamic", "pair_dynamic", "context_static", "context_dynamic")
DATASETS = ("Movies", "Grocery", "Reddit-S")
SEEDS = (42, 43, 44)
DEFAULT_P0_ROOT = Path("/hdd1/DataInHere/YHF/mag_model/outputs/problem_validation")


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _compose(dataset: str, seed: int, variant: str):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(
            config_name="config",
            overrides=[
                f"dataset={dataset}", "task=nc", "model=interaction_core",
                f"model.variant={variant}", f"seed={seed}", "num_runs=1",
                "task.evaluate_test=false",
            ],
        )


def _checkpoint_path(mode: str, dataset: str, variant: str, seed: int) -> Path:
    if mode == "smoke":
        return OUTPUT_ROOT / "smoke" / "checkpoints" / dataset / variant / f"seed{seed}.pt"
    return OUTPUT_ROOT / "checkpoints" / dataset / variant / f"seed{seed}.pt"


def _quantile_masks(values: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(values, kind="mergesort")
    masks = []
    for group in np.array_split(order, 5):
        mask = np.zeros(values.shape[0], dtype=bool)
        mask[group] = True
        masks.append(mask)
    return masks


def _load_p0(p0_root: Path, dataset: str, seed: int) -> dict:
    diag_path = p0_root / "p02" / dataset / f"seed{seed}" / "conditional_feature" / "edge_diagnostics.pt"
    split_path = p0_root / "splits" / f"{dataset}.pt"
    diagnostics = torch.load(diag_path, map_location="cpu", weights_only=False)
    split = torch.load(split_path, map_location="cpu", weights_only=False)
    if diagnostics.get("test_evaluation") is not False or diagnostics.get("test_labels_accessed") is not False:
        raise RuntimeError(f"P0 artifact is not test-sealed: {diag_path}")
    if not torch.equal(diagnostics["target_node"], split["sampled_edge_target_node"]):
        raise RuntimeError(f"P0 target sequence differs from frozen split: {dataset}/{seed}")
    if not torch.equal(diagnostics["neighbor_node"], split["sampled_edge_neighbor_node"]):
        raise RuntimeError(f"P0 neighbor sequence differs from frozen split: {dataset}/{seed}")
    return diagnostics


def _p0_edge_indices(p0: dict, canonical_edges: torch.Tensor, num_nodes: int) -> torch.Tensor:
    src, dst = canonical_edges.detach().cpu()
    graph_keys = src * int(num_nodes) + dst
    p0_src = p0["neighbor_node"].long()
    p0_dst = p0["target_node"].long()
    p0_keys = p0_src * int(num_nodes) + p0_dst
    positions = torch.searchsorted(graph_keys, p0_keys)
    if graph_keys.numel() == 0:
        raise RuntimeError("P0 has edges but the model graph is empty")
    safe = positions.clamp(max=graph_keys.numel() - 1)
    if not torch.equal(graph_keys[safe], p0_keys):
        raise RuntimeError("A frozen P0 directed relation was not found in the M0-Core graph")
    return positions


def _stats(values: torch.Tensor) -> tuple[float, float, float, float]:
    values = values.detach().float().reshape(-1).cpu()
    if values.numel() == 0:
        return (float("nan"),) * 4
    q = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9]))
    return float(values.mean()), float(q[1]), float(q[0]), float(q[2])


def _operator_rows(values: dict, dataset: str, seed: int, variant: str) -> list[dict]:
    rows = []
    for modality in ("text", "visual"):
        for step in (0, 1):
            key = f"{modality}_step{step}"
            ratio = values["operator_deviation_ratio"][key]
            r_mean, r_median, r_p10, r_p90 = _stats(ratio)
            rows.append({
                "dataset": dataset, "seed": seed, "variant": variant,
                "modality": modality, "interaction_step": step,
                "operator_deviation_ratio_mean": r_mean,
                "operator_deviation_ratio_median": r_median,
                "operator_deviation_ratio_p10": r_p10,
                "operator_deviation_ratio_p90": r_p90,
                "base_message_norm_mean": float(values["base_message_norm"][key].float().mean().item()) if ratio.numel() else float("nan"),
                "delta_message_norm_mean": float(values["delta_message_norm"][key].float().mean().item()) if ratio.numel() else float("nan"),
                "num_edges": int(ratio.numel()),
            })
    return rows


def _modulation_rows(values: dict, dataset: str, seed: int, variant: str) -> list[dict]:
    rows = []
    for modality in ("text", "visual"):
        for step in (0, 1):
            a = values[f"modulation_{modality}_step{step}"].detach().float()
            if a.numel():
                norms = a.norm(dim=-1)
                feature_variance = float(a.var(dim=0, unbiased=False).mean().item())
                edge_variance = float(a.var(dim=1, unbiased=False).mean().item())
                norm_mean = float(norms.mean().item())
                norm_std = float(norms.std(unbiased=False).item())
                saturated = float((a.abs() > 0.95).float().mean().item())
            else:
                feature_variance = edge_variance = norm_mean = norm_std = saturated = float("nan")
            rows.append({
                "dataset": dataset, "seed": seed, "variant": variant,
                "modality": modality, "interaction_step": step,
                "feature_variance": feature_variance, "edge_variance": edge_variance,
                "norm_mean": norm_mean, "norm_std": norm_std,
                "tanh_saturation_fraction_abs_gt_0_95": saturated,
                "num_edges": int(a.size(0)),
            })
    return rows


def _dynamicity_rows(values: dict, dataset: str, seed: int, variant: str) -> list[dict]:
    if variant == "context_static":
        return []
    rows = []
    for modality in ("text", "visual"):
        a0 = values[f"modulation_{modality}_step0"].detach().float()
        a1 = values[f"modulation_{modality}_step1"].detach().float()
        if a0.numel():
            cosine_change = 1.0 - F_cosine(a0, a1)
            norm_change = (a1 - a0).norm(dim=-1)
            c_mean, c_median, _, c_p90 = _stats(cosine_change)
            n_mean, n_median, _, n_p90 = _stats(norm_change)
        else:
            c_mean = c_median = c_p90 = n_mean = n_median = n_p90 = float("nan")
        rows.append({
            "dataset": dataset, "seed": seed, "variant": variant, "modality": modality,
            "cosine_change_mean": c_mean, "cosine_change_median": c_median,
            "cosine_change_p90": c_p90, "l2_change_mean": n_mean,
            "l2_change_median": n_median, "l2_change_p90": n_p90,
            "num_edges": int(a0.size(0)),
        })
    return rows


def F_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(left, right, dim=-1, eps=1e-8)


def _hardcase_rows(
    p0: dict, values: dict, edge_indices: torch.Tensor,
    dataset: str, seed: int, variant: str,
) -> list[dict]:
    rows = []
    for modality, sim_key, utility_key in (
        ("text", "probe_sim_text", "utility_ce_text"),
        ("visual", "probe_sim_visual", "utility_ce_visual"),
    ):
        similarity = p0[sim_key].detach().cpu().numpy()
        utility = p0[utility_key].detach().cpu().numpy()
        quantiles = _quantile_masks(similarity)
        for regime, q_index in (("Q1", 0), ("Q5", 4)):
            qmask = quantiles[q_index]
            for step in (0, 1):
                operation = values[f"modulation_{modality}_step{step}"][edge_indices.to(values[f"modulation_{modality}_step{step}"].device)].detach().float().cpu()
                deviation = values["operator_deviation_ratio"][f"{modality}_step{step}"][edge_indices.to(values["operator_deviation_ratio"][f"{modality}_step{step}"].device)].detach().float().cpu()
                groups: dict[str, tuple[torch.Tensor, int, float, float]] = {}
                for group, mask in (("beneficial", qmask & (utility > 0)), ("harmful", qmask & (utility < 0))):
                    selected = operation[torch.from_numpy(mask)]
                    selected_ratio = deviation[torch.from_numpy(mask)]
                    if selected.size(0):
                        centroid = selected.mean(dim=0)
                        radius = float((selected - centroid).norm(dim=-1).mean().item())
                        centroid_norm = float(centroid.norm().item())
                        ratio_mean = float(selected_ratio.mean().item())
                    else:
                        centroid = operation.new_full((operation.size(-1),), float("nan"))
                        radius = centroid_norm = ratio_mean = float("nan")
                    groups[group] = (centroid, int(selected.size(0)), radius, ratio_mean)
                    rows.append({
                        "dataset": dataset, "seed": seed, "variant": variant,
                        "modality": modality, "similarity_regime": regime,
                        "utility_group": group, "interaction_step": step,
                        "n_relations": int(selected.size(0)), "centroid_norm": centroid_norm,
                        "within_group_radius": radius,
                        "normalized_centroid_separation": float("nan"),
                        "operator_deviation_ratio_mean": ratio_mean,
                    })
                beneficial, harmful = groups["beneficial"], groups["harmful"]
                if beneficial[1] and harmful[1] and torch.isfinite(beneficial[0]).all() and torch.isfinite(harmful[0]).all():
                    separation = float((beneficial[0] - harmful[0]).norm().item())
                    normalized = separation / (0.5 * (beneficial[2] + harmful[2]) + 1e-12)
                else:
                    normalized = float("nan")
                rows.append({
                    "dataset": dataset, "seed": seed, "variant": variant,
                    "modality": modality, "similarity_regime": regime,
                    "utility_group": "beneficial_vs_harmful", "interaction_step": step,
                    "n_relations": beneficial[1] + harmful[1], "centroid_norm": float("nan"),
                    "within_group_radius": float("nan"),
                    "normalized_centroid_separation": normalized,
                    "operator_deviation_ratio_mean": float(np.nanmean([beneficial[3], harmful[3]])),
                })
    return rows


def _val_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict:
    pred = logits.argmax(dim=-1).detach().cpu()
    target = labels.detach().cpu()
    return {
        "val_acc": float((pred == target).float().mean().item()),
        "val_macro_f1": float(f1_score(
            target.numpy(), pred.numpy(), labels=list(range(int(num_classes))),
            average="macro", zero_division=0,
        )),
        "pred": pred,
    }


def _intervention_rows(model, head, data, x, edges, details, dataset: str, seed: int) -> list[dict]:
    rows = []
    val_idx = data.val_idx.to(x.device)
    val_labels = data.y[data.val_idx].cpu()
    num_classes = int(data.num_classes)
    with torch.no_grad():
        full_logits = head(details["fused_z"][val_idx])
        full = _val_metrics(full_logits, val_labels, num_classes)

        cases = []
        noctx = model.analyze(x, edges, context_mode="null")
        cases.append(("no_context", noctx, noctx["fused_z"], None))
        mean_relation = model.analyze(x, edges, relation_mode="mean")
        cases.append(("mean_relation", mean_relation, mean_relation["fused_z"], None))
        operator_off_z = model(x, edges, operator_enabled=False)[0]
        cases.append(("operator_off", None, operator_off_z, None))

        for name, analysis, z, _ in cases:
            logits = head(z[val_idx])
            metrics = _val_metrics(logits, val_labels, num_classes)
            logit_l2 = (logits - full_logits).norm(dim=-1)
            flips = float((metrics["pred"] != full["pred"]).float().mean().item())
            common = {
                "dataset": dataset, "seed": seed, "variant": "context_dynamic",
                "intervention": name,
                "val_acc_full": full["val_acc"], "val_acc_intervened": metrics["val_acc"],
                "val_acc_change": metrics["val_acc"] - full["val_acc"],
                "val_macro_f1_full": full["val_macro_f1"],
                "val_macro_f1_intervened": metrics["val_macro_f1"],
                "val_macro_f1_change": metrics["val_macro_f1"] - full["val_macro_f1"],
                "val_logit_l2_mean": float(logit_l2.mean().item()),
                "val_logit_l2_p90": float(torch.quantile(logit_l2, 0.9).item()),
                "val_prediction_flip_rate": flips,
            }
            if analysis is None:
                rows.append({**common, "modality": "all", "interaction_step": "all",
                             "operator_modulation_change_mean": float("nan"),
                             "operator_modulation_change_median": float("nan"),
                             "operator_modulation_change_p90": float("nan")})
            else:
                for modality in ("text", "visual"):
                    for step in (0, 1):
                        key = f"modulation_{modality}_step{step}"
                        difference = (details[key] - analysis[key]).norm(dim=-1)
                        mean, median, _, p90 = _stats(difference)
                        rows.append({**common, "modality": modality, "interaction_step": step,
                                     "operator_modulation_change_mean": mean,
                                     "operator_modulation_change_median": median,
                                     "operator_modulation_change_p90": p90})
            del analysis, z, logits
        del noctx, mean_relation, operator_off_z
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze validation-selected M0-Core checkpoints.")
    parser.add_argument("--mode", choices=("smoke", "pilot"), default="pilot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--p0-root", type=Path, default=DEFAULT_P0_ROOT)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    selected = [("Movies", 42, v) for v in args.variants] if args.mode == "smoke" else [
        (d, s, v) for d in args.datasets for s in args.seeds for v in args.variants
    ]
    operator_rows, modulation_rows, dynamicity_rows = [], [], []
    intervention_rows, hardcase_rows = [], []

    for dataset, seed, variant in selected:
        checkpoint_path = _checkpoint_path(args.mode, dataset, variant, seed)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("selection") != "best_val_accuracy":
            raise RuntimeError(f"Unexpected checkpoint selection: {checkpoint_path}")
        if any(key.startswith("test_") for key in checkpoint.get("metrics", {})):
            raise RuntimeError(f"Test metrics are present in a validation-only checkpoint: {checkpoint_path}")
        cfg = _compose(dataset, seed, variant)
        data = load_mag_data(cfg, "nc", seed)
        info = {
            "input_dim": data.input_dim, "num_nodes": data.num_nodes,
            "num_classes": data.num_classes,
            "text_dim": int(data.x_t.size(1)), "visual_dim": int(data.x_i.size(1)),
        }
        model = build_model(cfg, info).to(device)
        model.load_state_dict(checkpoint["model_state"])
        head = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
        head.load_state_dict(checkpoint["head_state"])
        model.eval(); head.eval()
        x = data.x.to(device)
        edges = data.edge_index.to(device)
        with torch.no_grad():
            values = model.analyze(x, edges)
            val_logits = head(values["fused_z"][data.val_idx.to(device)])
            baseline = _val_metrics(val_logits, data.y[data.val_idx], int(data.num_classes))
        # Baseline reproduction is checked using validation only.
        if abs(baseline["val_acc"] - float(checkpoint["metrics"]["val_acc"])) > 1e-6:
            raise RuntimeError(f"Validation checkpoint metric did not reproduce: {checkpoint_path}")
        operator_rows.extend(_operator_rows(values, dataset, seed, variant))
        modulation_rows.extend(_modulation_rows(values, dataset, seed, variant))
        dynamicity_rows.extend(_dynamicity_rows(values, dataset, seed, variant))

        if args.mode == "pilot":
            if variant == "context_dynamic":
                intervention_rows.extend(
                    _intervention_rows(model, head, data, x, edges, values, dataset, seed)
                )
            p0 = _load_p0(args.p0_root, dataset, seed)
            edge_indices = _p0_edge_indices(p0, values["canonical_edge_index"], data.num_nodes)
            hardcase_rows.extend(_hardcase_rows(p0, values, edge_indices, dataset, seed, variant))

        edge_count = int(values["canonical_edge_index"].size(1))
        print(f"ANALYZED {args.mode} {dataset}/{variant}/seed{seed} edges={edge_count}", flush=True)
        del model, head, data, values, x, edges
        if device.type == "cuda":
            torch.cuda.empty_cache()

    operator_fields = [
        "dataset", "seed", "variant", "modality", "interaction_step",
        "operator_deviation_ratio_mean", "operator_deviation_ratio_median",
        "operator_deviation_ratio_p10", "operator_deviation_ratio_p90",
        "base_message_norm_mean", "delta_message_norm_mean", "num_edges",
    ]
    modulation_fields = [
        "dataset", "seed", "variant", "modality", "interaction_step", "feature_variance",
        "edge_variance", "norm_mean", "norm_std", "tanh_saturation_fraction_abs_gt_0_95", "num_edges",
    ]
    dynamicity_fields = [
        "dataset", "seed", "variant", "modality", "cosine_change_mean", "cosine_change_median",
        "cosine_change_p90", "l2_change_mean", "l2_change_median", "l2_change_p90", "num_edges",
    ]
    if args.mode == "smoke":
        _write_csv(RESULT_ROOT / "smoke_operator_diagnostics.csv", operator_rows, operator_fields)
        _write_csv(RESULT_ROOT / "smoke_modulation_diagnostics.csv", modulation_rows, modulation_fields)
        _write_csv(RESULT_ROOT / "smoke_dynamicity_diagnostics.csv", dynamicity_rows, dynamicity_fields)
    else:
        _write_csv(RESULT_ROOT / "operator_diagnostics.csv", operator_rows, operator_fields)
        _write_csv(RESULT_ROOT / "modulation_diagnostics.csv", modulation_rows, modulation_fields)
        _write_csv(RESULT_ROOT / "dynamicity_diagnostics.csv", dynamicity_rows, dynamicity_fields)
        _write_csv(RESULT_ROOT / "intervention_diagnostics.csv", intervention_rows, [
            "dataset", "seed", "variant", "intervention", "modality", "interaction_step",
            "val_acc_full", "val_acc_intervened", "val_acc_change", "val_macro_f1_full",
            "val_macro_f1_intervened", "val_macro_f1_change", "val_logit_l2_mean",
            "val_logit_l2_p90", "val_prediction_flip_rate", "operator_modulation_change_mean",
            "operator_modulation_change_median", "operator_modulation_change_p90",
        ])
        _write_csv(RESULT_ROOT / "p0_operation_hardcases.csv", hardcase_rows, [
            "dataset", "seed", "variant", "modality", "similarity_regime", "utility_group",
            "interaction_step", "n_relations", "centroid_norm", "within_group_radius",
            "normalized_centroid_separation", "operator_deviation_ratio_mean",
        ])


if __name__ == "__main__":
    main()
