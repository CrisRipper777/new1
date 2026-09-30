from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

from src.data import load_mag_data
from src.models.final_interaction import Model
from src.models.final_interaction_components import operation_variance_decomposition


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
MODALITIES = ("text", "visual")
STEPS = (0, 1)
RESULTS = ROOT / "results/final_nc"
EPS = 1e-12


def _write(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _data_cfg(dataset: str, seed: int):
    cfg = compose(config_name="config", overrides=[
        f"dataset={dataset}", "task=nc", "model=final_interaction",
        f"seed={seed}", "num_runs=1", "task.epochs=1", "task.evaluate_test=true",
    ])
    OmegaConf.resolve(cfg)
    return cfg


def _load_model(data, dataset: str, seed: int, device: torch.device):
    info = {
        "input_dim": data.input_dim, "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]), "visual_dim": int(data.x_i.shape[1]),
    }
    checkpoint_path = RESULTS.parent.parent / "outputs/final_nc/checkpoints" / dataset / "full" / f"seed{seed}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = OmegaConf.load(ROOT / "configs/model/final_interaction.yaml")
    config.variant = "full"
    model = Model(OmegaConf.create({"model": config}), info).to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device).eval()
    classifier.load_state_dict(checkpoint["head_state"], strict=True)
    return model, classifier, checkpoint


def _cpu_values(values: dict) -> dict:
    output = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.detach().cpu()
        elif isinstance(value, dict):
            output[key] = {subkey: tensor.detach().cpu() for subkey, tensor in value.items()}
    return output


def _metric_pair(logits: torch.Tensor, data, indices: torch.Tensor) -> tuple[float, float]:
    pred = logits[indices].argmax(dim=-1).cpu()
    target = data.y[indices].cpu()
    labels = list(range(int(data.num_classes)))
    return float((pred == target).float().mean()), float(f1_score(
        target.numpy(), pred.numpy(), labels=labels, average="macro", zero_division=0,
    ))


def _metric_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).norm(dim=-1).mean())


def _mean(x: torch.Tensor) -> float:
    return float(x.float().mean()) if x.numel() else float("nan")


def _safe_cos(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left.float(), right.float(), dim=-1, eps=1e-8)


def _stage1_rows(base: dict, dataset: str, seed: int) -> list[dict]:
    rows = []
    edge = base["canonical_edge_index"]
    src, dst = edge
    edge_lookup = {(int(a), int(b)): index for index, (a, b) in enumerate(zip(src.tolist(), dst.tolist()))}
    reverse_pairs = [(idx, edge_lookup[(int(b), int(a))]) for idx, (a, b) in enumerate(zip(src.tolist(), dst.tolist()))
                     if int(a) < int(b) and (int(b), int(a)) in edge_lookup]
    for modality in MODALITIES:
        relation = base[f"relation_{modality}"]
        variance = float(relation.float().var(dim=0, unbiased=False).mean()) if relation.size(0) else 0.0
        other = base[f"relation_{'visual' if modality == 'text' else 'text'}"]
        discrepancy = _mean(1.0 - _safe_cos(relation, other))
        direction = float("nan")
        if reverse_pairs:
            left_idx = torch.tensor([pair[0] for pair in reverse_pairs], dtype=torch.long)
            right_idx = torch.tensor([pair[1] for pair in reverse_pairs], dtype=torch.long)
            direction = _mean(1.0 - _safe_cos(relation[left_idx], relation[right_idx]))
        degree_one_edges = int((base["degree"][dst] == 1).sum()) if dst.numel() else 0
        rows.append({
            "dataset": dataset, "seed": seed, "modality": modality,
            "relation_edge_variance": variance,
            "cross_modal_relation_discrepancy_1_minus_cos": discrepancy,
            "directionality_1_minus_cos": direction,
            "directed_edges": int(relation.size(0)), "undirected_pairs": len(reverse_pairs),
            "degree_one_edges": degree_one_edges,
        })
    return rows


def _context_rows(real: dict, null: dict, dataset: str, seed: int) -> list[dict]:
    gap = real["compatibility_abs_difference"].float()
    if not gap.numel():
        return []
    quantiles = torch.quantile(gap, torch.tensor([0.2, 0.4, 0.6, 0.8]))
    quintile = torch.bucketize(gap.contiguous(), quantiles, right=True)
    rows = []
    for modality in MODALITIES:
        difference = (real[f"relation_{modality}"].float() - null[f"relation_{modality}"].float()).norm(dim=-1)
        rows.append({
            "dataset": dataset, "seed": seed, "modality": modality, "compatibility_gap_quintile": "all",
            "edge_count": int(difference.numel()), "mean_relation_change_l2": _mean(difference),
            "compatibility_gap_definition": "abs(cosine_text-cosine_visual)",
        })
        for q in range(5):
            selected = quintile == q
            rows.append({
                "dataset": dataset, "seed": seed, "modality": modality,
                "compatibility_gap_quintile": f"Q{q + 1}", "edge_count": int(selected.sum()),
                "mean_relation_change_l2": _mean(difference[selected]) if bool(selected.any()) else "",
                "compatibility_gap_definition": "abs(cosine_text-cosine_visual)",
            })
    return rows


def _execution_rows(base: dict, dataset: str, seed: int, num_nodes: int) -> tuple[list[dict], list[dict]]:
    health, variance_rows = [], []
    targets = base["canonical_edge_index"][1]
    for modality in MODALITIES:
        base_code = base[f"base_relation_{modality}"]
        for step in STEPS:
            suffix = f"{modality}_step{step}"
            code = base["execution_code"][suffix]
            modulation = base["modulation"][suffix]
            delta = base["delta_message"][suffix]
            message = base["base_message"][suffix]
            correction = code.float() - base_code.float()
            code_var = float(code.float().var(dim=0, unbiased=False).mean()) if code.size(0) else 0.0
            modulation_var = float(modulation.float().var(dim=0, unbiased=False).mean()) if modulation.size(0) else 0.0
            correction_ratio = _mean(correction.norm(dim=-1) / (base_code.float().norm(dim=-1) + EPS))
            delta_ratio = _mean(delta.float().norm(dim=-1) / (message.float().norm(dim=-1) + EPS))
            delta_norm = delta.float().norm(dim=-1)
            valid = delta_norm > EPS
            cos_z_delta = _mean(_safe_cos(message[valid], delta[valid])) if bool(valid.any()) else float("nan")
            rotation = _mean(1.0 - _safe_cos(message, message + delta))
            health.append({
                "dataset": dataset, "seed": seed, "modality": modality, "step": step,
                "execution_code_variance": code_var,
                "bilinear_correction_relative_norm": correction_ratio,
                "operator_modulation_variance": modulation_var,
                "operator_deviation_relative_norm": delta_ratio,
                "cosine_source_delta": cos_z_delta,
                "message_rotation_1_minus_cos": rotation,
                "mean_delta_l2": _mean(delta_norm), "mean_source_l2": _mean(message.float().norm(dim=-1)),
            })
            for name, vector in (("r", base_code), ("xi", code), ("a", modulation), ("Delta", delta)):
                values = operation_variance_decomposition(vector, targets, num_nodes)
                variance_rows.append({
                    "dataset": dataset, "seed": seed, "modality": modality, "step": step,
                    "representation": name,
                    **{key: float(value) for key, value in values.items()},
                })
    return health, variance_rows


def _delta_stats(reference: torch.Tensor, altered: torch.Tensor) -> tuple[float, float]:
    difference = (reference.float() - altered.float()).norm(dim=-1)
    relative = difference / (reference.float().norm(dim=-1) + EPS)
    return _mean(difference), _mean(relative)


def _intervention_rows(base: dict, altered: dict, baseline_logits: torch.Tensor,
                       altered_logits: torch.Tensor, data, dataset: str, seed: int,
                       intervention: str) -> list[dict]:
    val_base = _metric_pair(baseline_logits, data, data.val_idx)
    val_alt = _metric_pair(altered_logits, data, data.val_idx)
    test_base = _metric_pair(baseline_logits, data, data.test_idx)
    test_alt = _metric_pair(altered_logits, data, data.test_idx)
    pred_base = baseline_logits.argmax(dim=-1)
    pred_alt = altered_logits.argmax(dim=-1)
    all_node_logit_l2 = _metric_l2(baseline_logits, altered_logits)
    relation_memory_change, relation_memory_relative = _delta_stats(
        base["relation_memory"].reshape(-1, base["relation_memory"].size(-1)),
        altered["relation_memory"].reshape(-1, altered["relation_memory"].size(-1)),
    )
    common = {
        "dataset": dataset, "seed": seed, "intervention": intervention,
        "val_acc_baseline": val_base[0], "val_acc_intervention": val_alt[0],
        "val_acc_change": val_alt[0] - val_base[0],
        "val_macro_f1_baseline": val_base[1], "val_macro_f1_intervention": val_alt[1],
        "val_macro_f1_change": val_alt[1] - val_base[1],
        "test_acc_baseline": test_base[0], "test_acc_intervention": test_alt[0],
        "test_acc_change": test_alt[0] - test_base[0],
        "test_macro_f1_baseline": test_base[1], "test_macro_f1_intervention": test_alt[1],
        "test_macro_f1_change": test_alt[1] - test_base[1],
        "all_node_logit_l2_mean": all_node_logit_l2,
        "val_prediction_flip_rate": float((pred_base[data.val_idx] != pred_alt[data.val_idx]).float().mean()),
        "test_prediction_flip_rate": float((pred_base[data.test_idx] != pred_alt[data.test_idx]).float().mean()),
        "interpretation": "same-checkpoint mechanism intervention; descriptive, not a causal contribution estimate",
    }
    rows = []
    for modality in MODALITIES:
        relation_change, relation_relative = _delta_stats(
            base[f"relation_{modality}"], altered[f"relation_{modality}"]
        )
        context_change, context_relative = _delta_stats(
            base[f"context_{modality}"], altered[f"context_{modality}"]
        )
        for step in STEPS:
            suffix = f"{modality}_step{step}"
            code_change, code_relative = _delta_stats(
                base["execution_code"][suffix], altered["execution_code"][suffix]
            )
            modulation_change, modulation_relative = _delta_stats(
                base["modulation"][suffix], altered["modulation"][suffix]
            )
            operation_change, operation_relative = _delta_stats(
                base["delta_message"][suffix], altered["delta_message"][suffix]
            )
            rows.append({
                **common, "modality": modality, "step": step,
                "relation_memory_change_l2_mean": relation_memory_change,
                "relation_memory_change_relative_mean": relation_memory_relative,
                "relation_change_l2_mean": relation_change,
                "relation_change_relative_mean": relation_relative,
                "context_change_l2_mean": context_change,
                "context_change_relative_mean": context_relative,
                "execution_change_l2_mean": code_change,
                "execution_change_relative_mean": code_relative,
                "modulation_change_l2_mean": modulation_change,
                "modulation_change_relative_mean": modulation_relative,
                "operation_delta_change_l2_mean": operation_change,
                "operation_delta_change_relative_mean": operation_relative,
            })
    return rows


def main() -> None:
    parser_device = sys.argv[1] if len(sys.argv) > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(parser_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for final mechanism audit but unavailable")
    torch.set_num_threads(4)
    stage_rows, context_rows, execution_rows, variance_rows = [], [], [], []
    relation_interventions, context_interventions, operator_interventions = [], [], []
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        for dataset in DATASETS:
            for seed in SEEDS:
                data = load_mag_data(_data_cfg(dataset, seed), "nc", seed)
                model, classifier, checkpoint = _load_model(data, dataset, seed, device)
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                with torch.no_grad():
                    baseline = _cpu_values(model.analyze(x, edge_index))
                    baseline_logits = classifier(baseline["fused_z"].to(device)).detach().cpu()
                    null_context = _cpu_values(model.analyze(x, edge_index, null_context=True))
                    stage_rows.extend(_stage1_rows(baseline, dataset, seed))
                    context_rows.extend(_context_rows(baseline, null_context, dataset, seed))
                    health, decomposition = _execution_rows(baseline, dataset, seed, data.num_nodes)
                    execution_rows.extend(health)
                    variance_rows.extend(decomposition)
                    del null_context

                    for name, kwargs, destination in (
                        ("relation_shuffle", {"relation_shuffle": True}, relation_interventions),
                        ("context_shuffle", {"context_shuffle": True}, context_interventions),
                        ("operator_off", {"operator_enabled": False}, operator_interventions),
                    ):
                        changed = _cpu_values(model.analyze(x, edge_index, **kwargs))
                        changed_logits = classifier(changed["fused_z"].to(device)).detach().cpu()
                        destination.extend(_intervention_rows(
                            baseline, changed, baseline_logits, changed_logits,
                            data, dataset, seed, name,
                        ))
                        del changed, changed_logits
                print(f"AUDITED {dataset} seed={seed} best_epoch={checkpoint['epoch']}", flush=True)
                del model, classifier, checkpoint, data, x, edge_index, baseline, baseline_logits
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    _write(RESULTS / "stage1_health.csv", stage_rows,
           ["dataset", "seed", "modality", "relation_edge_variance",
            "cross_modal_relation_discrepancy_1_minus_cos", "directionality_1_minus_cos",
            "directed_edges", "undirected_pairs", "degree_one_edges"])
    _write(RESULTS / "context_health.csv", context_rows,
           ["dataset", "seed", "modality", "compatibility_gap_quintile", "edge_count",
            "mean_relation_change_l2", "compatibility_gap_definition"])
    _write(RESULTS / "execution_health.csv", execution_rows,
           ["dataset", "seed", "modality", "step", "execution_code_variance",
            "bilinear_correction_relative_norm", "operator_modulation_variance",
            "operator_deviation_relative_norm", "cosine_source_delta", "message_rotation_1_minus_cos",
            "mean_delta_l2", "mean_source_l2"])
    _write(RESULTS / "variance_decomposition.csv", variance_rows,
           ["dataset", "seed", "modality", "step", "representation", "V_total",
            "V_within_target", "V_between_target", "eta_relation", "eta_target",
            "node_balanced_within", "node_mean_variance", "decomposition_error"])
    intervention_fields = [
        "dataset", "seed", "intervention", "modality", "step",
        "val_acc_baseline", "val_acc_intervention", "val_acc_change",
        "val_macro_f1_baseline", "val_macro_f1_intervention", "val_macro_f1_change",
        "test_acc_baseline", "test_acc_intervention", "test_acc_change",
        "test_macro_f1_baseline", "test_macro_f1_intervention", "test_macro_f1_change",
        "all_node_logit_l2_mean", "val_prediction_flip_rate", "test_prediction_flip_rate",
        "relation_memory_change_l2_mean", "relation_memory_change_relative_mean",
        "relation_change_l2_mean", "relation_change_relative_mean",
        "context_change_l2_mean", "context_change_relative_mean",
        "execution_change_l2_mean", "execution_change_relative_mean",
        "modulation_change_l2_mean", "modulation_change_relative_mean",
        "operation_delta_change_l2_mean", "operation_delta_change_relative_mean", "interpretation",
    ]
    _write(RESULTS / "intervention_relation_shuffle.csv", relation_interventions, intervention_fields)
    _write(RESULTS / "intervention_context_shuffle.csv", context_interventions, intervention_fields)
    _write(RESULTS / "intervention_operator_off.csv", operator_interventions, intervention_fields)
    print("FINAL FULL-MODEL MECHANISM AUDIT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
