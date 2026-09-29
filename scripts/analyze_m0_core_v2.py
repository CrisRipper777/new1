from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_mag_data  # noqa: E402
from src.models.interaction_core_v2 import Model, TRACE_GROUPS, VARIANTS  # noqa: E402

OUTPUT_ROOT = ROOT / "outputs" / "m0" / "relation_grounded_v2"
RESULT_ROOT = ROOT / "results" / "m0" / "relation_grounded_v2"
DATASETS = ("Movies", "Grocery", "Reddit-S")
SEEDS = (42, 43, 44)
P0_ROOT = Path("/hdd1/DataInHere/YHF/mag_model/outputs/problem_validation")


def _write(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _compose(dataset, seed, variant):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", "model=interaction_core_v2",
            f"model.variant={variant}", f"seed={seed}", "num_runs=1", "task.evaluate_test=false",
        ])


def _checkpoint(mode, dataset, variant, seed):
    base = OUTPUT_ROOT / "smoke" if mode == "smoke" else OUTPUT_ROOT
    return base / "checkpoints" / dataset / variant / f"seed{seed}.pt"


def _metrics(logits, labels, num_classes):
    pred, target = logits.argmax(-1).detach().cpu(), labels.detach().cpu()
    return {
        "val_acc": float((pred == target).float().mean().item()),
        "val_macro_f1": float(f1_score(target.numpy(), pred.numpy(), labels=list(range(int(num_classes))),
                                        average="macro", zero_division=0)),
        "pred": pred,
    }


def _stats(tensor):
    value = tensor.detach().float().reshape(-1).cpu()
    value = value[torch.isfinite(value)]
    if not value.numel():
        return {"mean": float("nan"), "median": float("nan"), "p10": float("nan"),
                "p90": float("nan"), "p95": float("nan"), "n": 0}
    q = torch.quantile(value, torch.tensor([0.1, 0.5, 0.9, 0.95]))
    return {"mean": float(value.mean()), "median": float(q[1]), "p10": float(q[0]),
            "p90": float(q[2]), "p95": float(q[3]), "n": int(value.numel())}


def _quantiles(values: torch.Tensor):
    cpu = values.detach().float().cpu().numpy()
    order = np.argsort(cpu, kind="mergesort")
    out = []
    for group in np.array_split(order, 5):
        mask = np.zeros(cpu.shape[0], dtype=bool)
        mask[group] = True
        out.append(mask)
    return out


def _quantile_rows(dataset, seed, variant, score, values, kind, keys):
    rows = []
    quantiles = _quantiles(score)
    for qi, mask_np in enumerate(quantiles, 1):
        mask = torch.as_tensor(mask_np, device=score.device)
        for key in keys:
            selected = values[key][mask]
            stat = _stats(selected)
            rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                         "quintile": f"Q{qi}", "measure": kind, "key": key,
                         **stat})
    return rows


@torch.no_grad()
def _stage1_rows(model, x, edges, full, dataset, seed, variant):
    rows, attn_rows = [], []
    score = 0.5 * (full["compatibility_text"] + full["compatibility_visual"])
    for modality in ("text", "visual"):
        rel = full[f"relation_{modality}"].detach().float()
        stat = _stats(rel.var(dim=0, unbiased=False).mean().reshape(1))
        rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                     "measure": "relation_edge_variance_mean_feature_variance",
                     "modality": modality, **stat})
    cross = torch.nn.functional.cosine_similarity(full["relation_text"], full["relation_visual"], dim=-1, eps=1e-8)
    rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                 "measure": "text_visual_relation_1_minus_cosine", "modality": "both", **_stats(1 - cross)})
    edges_cpu = full["canonical_edge_index"].detach().cpu()
    n = int(x.size(0))
    index = {(int(s), int(d)): i for i, (s, d) in enumerate(edges_cpu.t().tolist())}
    forward, reverse = [], []
    for i, (s, d) in enumerate(edges_cpu.t().tolist()):
        j = index.get((d, s))
        if j is not None and s < d:
            forward.append(i); reverse.append(j)
    for modality in ("text", "visual"):
        if forward:
            a = full[f"relation_{modality}"][torch.tensor(forward, device=x.device)]
            b = full[f"relation_{modality}"][torch.tensor(reverse, device=x.device)]
            direction = 1 - torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-8)
            stat = _stats(direction)
        else:
            stat = _stats(torch.empty(0))
        if forward:
            l2_stat = _stats((a - b).norm(dim=-1))
        else:
            l2_stat = _stats(torch.empty(0))
        rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                     "measure": "reverse_edge_directional_1_minus_cosine", "modality": modality,
                     **stat, "num_undirected_pairs": len(forward)})
        rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                     "measure": "reverse_edge_directional_l2", "modality": modality,
                     **l2_stat, "num_undirected_pairs": len(forward)})

    # Context correction is a sensitivity diagnostic on the full v2 checkpoint, not a causal effect.
    if variant == "context_grounded_dynamic":
        null = model.analyze(x, edges, context_mode="null")
        correction = torch.stack((full["relation_text"] - null["relation_text"],
                                  full["relation_visual"] - null["relation_visual"]), dim=1).norm(dim=-1).mean(-1)
        correction_rows = _quantile_rows(dataset, seed, variant, score,
                                         {"context_correction_l2": correction},
                                         "full_minus_pair_null_relation_l2", ["context_correction_l2"])
        for row in correction_rows:
            row["diagnostic_scope"] = "same-checkpoint sensitivity; not causal"
        rows.extend(correction_rows)
    for modality in ("text", "visual"):
        attn = full[f"stage1_attention_{modality}"]
        for key_idx, key_name in enumerate(("pair_other_modality", "context_text", "context_visual")):
            entropy = -(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(-1)
            attn_rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                              "modality": modality, "evidence_key": key_name,
                              "quintile": "ALL", **_stats(attn[:, key_idx]),
                              "attention_entropy_mean": _stats(entropy)["mean"]})
            for qi, mask_np in enumerate(_quantiles(score), 1):
                selected_mask = torch.as_tensor(mask_np, device=attn.device)
                selected = attn[selected_mask, key_idx]
                attn_rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                                  "modality": modality, "evidence_key": key_name,
                                  "quintile": f"Q{qi}", **_stats(selected),
                                  "attention_entropy_mean": _stats(entropy[selected_mask])["mean"]})
    stage1_used = variant != "global_grounded_dynamic"
    for row in rows:
        row["pair_context_encoder_used_in_training"] = stage1_used
    for row in attn_rows:
        row["pair_context_encoder_used_in_training"] = stage1_used
    return rows, attn_rows


def _execution_attention_rows(values, dataset, seed, variant):
    rows = []
    for modality in ("text", "visual"):
        for step in (0, 1):
            attn = values[f"execution_attention_{modality}_step{step}"].float()
            if not attn.numel():
                continue
            entropy = -(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(-1)
            for slot, name in enumerate(("relation_text", "relation_visual")):
                stat = _stats(attn[:, slot])
                rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                             "modality": modality, "interaction_step": step,
                             "memory_slot": name, **stat,
                             "edge_variance": float(attn[:, slot].var(unbiased=False).item())})
            rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                         "modality": modality, "interaction_step": step,
                         "memory_slot": "attention_entropy", **_stats(entropy),
                         "edge_variance": float(entropy.var(unbiased=False).item())})
    return rows


def _operator_rows(values, dataset, seed, variant):
    ratio_rows, geometry_rows = [], []
    for modality in ("text", "visual"):
        for step in (0, 1):
            key = f"{modality}_step{step}"
            ratio = values["operator_deviation_ratio"][key]
            norm = _stats(ratio)
            ratio_rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                               "modality": modality, "interaction_step": step,
                               **{f"ratio_{k}": v for k, v in norm.items()},
                               "base_message_norm_mean": _stats(values["base_message_norm"][key])["mean"],
                               "delta_message_norm_mean": _stats(values["delta_message_norm"][key])["mean"]})
            cosine = values[f"cosine_base_delta_{modality}_step{step}"]
            angle = values[f"message_rotation_degrees_{modality}_step{step}"]
            row = {"dataset": dataset, "seed": seed, "variant": variant,
                   "modality": modality, "interaction_step": step}
            row.update({f"cos_base_delta_{k}": v for k, v in _stats(cosine).items()})
            row.update({f"rotation_degrees_{k}": v for k, v in _stats(angle).items()})
            row.update({f"message_scale_{k}": v for k, v in _stats(values[f"message_scale_{modality}_step{step}"]).items()})
            row.update({f"message_rotation_{k}": v for k, v in _stats(values[f"message_rotation_{modality}_step{step}"]).items()})
            finite_cos = cosine[torch.isfinite(cosine)]
            row["fraction_reinforce_cos_gt_0_3"] = float((finite_cos > 0.3).float().mean()) if finite_cos.numel() else float("nan")
            row["fraction_suppress_cos_lt_minus_0_3"] = float((finite_cos < -0.3).float().mean()) if finite_cos.numel() else float("nan")
            row["fraction_redirect_abs_cos_le_0_3"] = float((finite_cos.abs() <= 0.3).float().mean()) if finite_cos.numel() else float("nan")
            for threshold in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0):
                row[f"fraction_ratio_gt_{threshold:g}"] = float((ratio > threshold).float().mean().item()) if ratio.numel() else float("nan")
            row["fraction_delta_nonzero"] = float((values["delta_message_norm"][key] > 1e-12).float().mean().item()) if ratio.numel() else float("nan")
            geometry_rows.append(row)
    return ratio_rows, geometry_rows


def _dynamicity_rows(values, dataset, seed, variant):
    rows = []
    dst = values["canonical_edge_index"][1]
    degree = values["degree"].float()
    valid_nodes = degree > 0
    for modality in ("text", "visual"):
        h0, h1 = values[f"H0_{modality}"].float(), values[f"H1_{modality}"].float()
        d_h = (h1 - h0).norm(dim=-1)
        a0 = values[f"modulation_{modality}_step0"].float()
        a1 = values[f"modulation_{modality}_step1"].float()
        node0 = a0.new_zeros((degree.numel(), a0.size(-1)))
        node1 = a1.new_zeros((degree.numel(), a1.size(-1)))
        if dst.numel():
            node0.index_add_(0, dst, a0); node1.index_add_(0, dst, a1)
            denom = degree.clamp_min(1).unsqueeze(-1)
            node0, node1 = node0 / denom, node1 / denom
        d_a = (node1 - node0).norm(dim=-1)
        selection = valid_nodes.detach().cpu().numpy()
        rho_h = float(spearmanr(degree[valid_nodes].cpu().numpy(), d_h[valid_nodes].cpu().numpy()).statistic) if selection.sum() > 1 else float("nan")
        rho_a = float(spearmanr(degree[valid_nodes].cpu().numpy(), d_a[valid_nodes].cpu().numpy()).statistic) if selection.sum() > 1 else float("nan")
        rho_ha = float(spearmanr(d_h[valid_nodes].cpu().numpy(), d_a[valid_nodes].cpu().numpy()).statistic) if selection.sum() > 1 else float("nan")
        for measure, tensor, rho in (("D_H_state_change", d_h, rho_h), ("D_A_target_modulation_change", d_a[valid_nodes], rho_a)):
            rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                         "modality": modality, "measure": measure, **_stats(tensor),
                         "spearman_degree": rho, "spearman_D_H_D_A": rho_ha,
                         "n_nodes_nonzero_degree": int(selection.sum())})
        edge_cos = 1 - torch.nn.functional.cosine_similarity(a0, a1, dim=-1, eps=1e-8)
        edge_l2 = (a1 - a0).norm(dim=-1)
        for measure, tensor in (("edge_modulation_1_minus_cosine", edge_cos), ("edge_modulation_l2", edge_l2)):
            rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                         "modality": modality, "measure": measure, **_stats(tensor),
                         "spearman_degree": float("nan"), "spearman_D_H_D_A": rho_ha,
                         "n_nodes_nonzero_degree": int(selection.sum())})
    return rows


def _intervention_rows(model, head, data, x, edges, base, dataset, seed, intervention, **kwargs):
    with torch.no_grad():
        changed = model.analyze(x, edges, **kwargs)
        val_idx = data.val_idx.to(x.device)
        labels = data.y[data.val_idx]
        full_logits = head(base["fused_z"][val_idx])
        alt_logits = head(changed["fused_z"][val_idx])
        full = _metrics(full_logits, labels, int(data.num_classes))
        alt = _metrics(alt_logits, labels, int(data.num_classes))
        logit_delta = (alt_logits - full_logits).norm(dim=-1)
        common = {
            "dataset": dataset, "seed": seed, "variant": "context_grounded_dynamic",
            "intervention": intervention, "val_acc_full": full["val_acc"],
            "val_acc_intervened": alt["val_acc"], "val_acc_change": alt["val_acc"] - full["val_acc"],
            "val_macro_f1_full": full["val_macro_f1"], "val_macro_f1_intervened": alt["val_macro_f1"],
            "val_macro_f1_change": alt["val_macro_f1"] - full["val_macro_f1"],
            "val_logit_l2_mean": _stats(logit_delta)["mean"], "val_logit_l2_p90": _stats(logit_delta)["p90"],
            "val_prediction_flip_rate": float((alt["pred"] != full["pred"]).float().mean().item()),
            "relation_state_change_mean": _stats((base["relation_memory"] - changed["relation_memory"]).norm(dim=-1).mean(-1))["mean"],
        }
        if intervention == "degree_matched_context_pair_shuffle":
            context_change = (base["context_text"] - changed["context_text"]).norm(dim=-1) + \
                             (base["context_visual"] - changed["context_visual"]).norm(dim=-1)
            dst = base["canonical_edge_index"][1]
            degree = base["degree"][dst]
            common["context_changed_edge_fraction"] = float((context_change > 1e-12).float().mean().item())
            common["degree1_unchanged_edges"] = int(((degree == 1) & (context_change <= 1e-12)).sum().item())
        elif intervention == "within_target_relation_cyclic":
            memory_change = (base["relation_memory"] - changed["relation_memory"]).norm(dim=(1, 2))
            common["relation_alignment_changed_edge_fraction"] = float((memory_change > 1e-12).float().mean().item())
            dst = base["canonical_edge_index"][1]
            degree = base["degree"][dst]
            common["degree1_unchanged_edges"] = int(((degree == 1) & (memory_change <= 1e-12)).sum().item())
        rows = []
        if intervention == "operator_off":
            rows.append({**common, "modality": "all", "interaction_step": "all",
                         "execution_code_change_mean": float("nan"), "modulation_change_mean": float("nan")})
        else:
            for modality in ("text", "visual"):
                for step in (0, 1):
                    mod_delta = (base[f"modulation_{modality}_step{step}"] - changed[f"modulation_{modality}_step{step}"]).norm(dim=-1)
                    code_delta = (base[f"execution_code_{modality}_step{step}"] - changed[f"execution_code_{modality}_step{step}"]).norm(dim=-1)
                    rows.append({**common, "modality": modality, "interaction_step": step,
                                 "execution_code_change_mean": _stats(code_delta)["mean"],
                                 "modulation_change_mean": _stats(mod_delta)["mean"]})
        return rows, changed


def _load_p0(dataset, seed):
    path = P0_ROOT / "p02" / dataset / f"seed{seed}" / "conditional_feature" / "edge_diagnostics.pt"
    split_path = P0_ROOT / "splits" / f"{dataset}.pt"
    p0 = torch.load(path, map_location="cpu", weights_only=False)
    split = torch.load(split_path, map_location="cpu", weights_only=False)
    if p0.get("test_evaluation") is not False or p0.get("test_labels_accessed") is not False:
        raise RuntimeError(f"P0 artifact is not test sealed: {path}")
    if not torch.equal(p0["target_node"], split["sampled_edge_target_node"]):
        raise RuntimeError(f"P0 edge targets mismatch frozen split: {dataset}/{seed}")
    if not torch.equal(p0["neighbor_node"], split["sampled_edge_neighbor_node"]):
        raise RuntimeError(f"P0 edge sources mismatch frozen split: {dataset}/{seed}")
    return p0


def _p0_indices(p0, edges, num_nodes):
    graph_keys = edges[0].detach().cpu() * int(num_nodes) + edges[1].detach().cpu()
    keys = p0["neighbor_node"].long() * int(num_nodes) + p0["target_node"].long()
    pos = torch.searchsorted(graph_keys, keys)
    safe = pos.clamp(max=max(graph_keys.numel() - 1, 0))
    if not graph_keys.numel() or not torch.equal(graph_keys[safe], keys):
        raise RuntimeError("P0 sampled directed relation was not found in the canonical model graph")
    return pos


def _p0_rows(p0, values, indices, dataset, seed):
    rows = []
    for modality, simkey, utilitykey in (("text", "probe_sim_text", "utility_ce_text"),
                                         ("visual", "probe_sim_visual", "utility_ce_visual")):
        sim = p0[simkey].float().numpy(); utility = p0[utilitykey].float().numpy()
        for qi, group in enumerate(np.array_split(np.argsort(sim, kind="mergesort"), 5), 1):
            if qi not in (1, 5):
                continue
            selected_by_group = {}
            for label, utility_mask in (("beneficial", utility > 0), ("harmful", utility < 0)):
                selected_mask = np.zeros(len(sim), dtype=bool); selected_mask[group] = True
                selected_mask &= utility_mask
                mask = torch.as_tensor(selected_mask)
                for step in (0, 1):
                    edge_ids = indices.to(values[f"modulation_{modality}_step{step}"].device)
                    op = values[f"modulation_{modality}_step{step}"][edge_ids].detach().float().cpu()[mask]
                    ratio = values["operator_deviation_ratio"][f"{modality}_step{step}"][edge_ids].detach().float().cpu()[mask]
                    cos = values[f"cosine_base_delta_{modality}_step{step}"][edge_ids].detach().float().cpu()[mask]
                    rot = values[f"message_rotation_{modality}_step{step}"][edge_ids].detach().float().cpu()[mask]
                    centroid = op.mean(0) if op.numel() else torch.full((32,), float("nan"))
                    radius = float((op - centroid).norm(dim=-1).mean()) if op.numel() else float("nan")
                    selected_by_group[(label, step)] = {
                        "centroid": centroid, "count": int(op.size(0)), "radius": radius,
                        "ratio": _stats(ratio)["mean"], "cos": _stats(cos)["mean"],
                        "rotation": _stats(rot)["mean"],
                    }
                    rows.append({"dataset": dataset, "seed": seed, "variant": "context_grounded_dynamic",
                                 "modality": modality, "similarity_quintile": f"Q{qi}", "utility_group": label,
                                 "interaction_step": step, "n_relations": int(op.size(0)),
                                 "modulation_centroid_norm": float(centroid.norm()) if op.numel() else float("nan"),
                                 "within_group_radius": radius, "normalized_centroid_separation": float("nan"),
                                 "operator_deviation_ratio_mean": _stats(ratio)["mean"],
                                 "cos_base_delta_mean": _stats(cos)["mean"],
                                 "message_rotation_mean": _stats(rot)["mean"]})
            for step in (0, 1):
                b = selected_by_group[("beneficial", step)]
                h = selected_by_group[("harmful", step)]
                if b["count"] and h["count"] and torch.isfinite(b["centroid"]).all() and torch.isfinite(h["centroid"]).all():
                    separation = float((b["centroid"] - h["centroid"]).norm()) / (0.5 * (b["radius"] + h["radius"]) + 1e-12)
                else:
                    separation = float("nan")
                rows.append({"dataset": dataset, "seed": seed, "variant": "context_grounded_dynamic",
                             "modality": modality, "similarity_quintile": f"Q{qi}",
                             "utility_group": "beneficial_vs_harmful", "interaction_step": step,
                             "n_relations": b["count"] + h["count"], "modulation_centroid_norm": float("nan"),
                             "within_group_radius": float("nan"),
                             "normalized_centroid_separation": separation,
                             "operator_deviation_ratio_mean": float(np.nanmean([b["ratio"], h["ratio"]])),
                             "cos_base_delta_mean": float(np.nanmean([b["cos"], h["cos"]])),
                             "message_rotation_mean": float(np.nanmean([b["rotation"], h["rotation"]]))})
    return rows


def _trace_rows(mode, selected):
    grad_rows, growth_rows = [], []
    root = OUTPUT_ROOT / "smoke" if mode == "smoke" else OUTPUT_ROOT
    for dataset, seed, variant, best_epoch in selected:
        run_dir = root / "logs" / dataset / variant / f"seed{seed}"
        for file_name, output in (("training_gradient_trace.csv", grad_rows), ("operator_growth_trace.csv", growth_rows)):
            path = run_dir / file_name
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row.update({"dataset": dataset, "seed": seed, "variant": variant,
                            "is_best_epoch": int(row["epoch"]) == int(best_epoch),
                            "selected_best_epoch": int(best_epoch)})
                if file_name == "training_gradient_trace.csv":
                    row["parameter_group"] = row["group"]
                    row["grad_rms"] = row["gradient_rms_preclip"]
                    row["parameter_norm"] = row["parameter_l2_norm"]
                else:
                    row["up_weight_norm"] = row["W_up_frobenius_norm"]
                output.append(row)
    return grad_rows, growth_rows


def main():
    parser = argparse.ArgumentParser(description="Analyze validation-selected M0-Core v2 checkpoints only.")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    selected = [("Movies", 42, v) for v in VARIANTS] if args.mode == "smoke" else [
        (d, s, v) for d in DATASETS for s in SEEDS for v in VARIANTS
    ]
    stage1_rows, stage1_attn_rows, exec_attn_rows = [], [], []
    operator_rows, geometry_rows, dynamicity_rows = [], [], []
    relation_interventions, context_interventions, frozen_interventions, off_interventions = [], [], [], []
    p0_rows, trace_selected = [], []
    for dataset, seed, variant in selected:
        cp_path = _checkpoint(args.mode, dataset, variant, seed)
        checkpoint = torch.load(cp_path, map_location="cpu", weights_only=False)
        if checkpoint.get("selection") != "best_val_accuracy" or any(k.startswith("test_") for k in checkpoint.get("metrics", {})):
            raise RuntimeError(f"Checkpoint includes forbidden selection/metrics: {cp_path}")
        cfg = _compose(dataset, seed, variant)
        data = load_mag_data(cfg, "nc", seed)
        data_info = {"input_dim": data.input_dim, "num_nodes": data.num_nodes,
                     "num_classes": data.num_classes, "text_dim": int(data.x_t.size(1)),
                     "visual_dim": int(data.x_i.size(1))}
        model = Model(cfg, data_info).to(device)
        model.load_state_dict(checkpoint["model_state"])
        head = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
        head.load_state_dict(checkpoint["head_state"])
        model.eval(); head.eval()
        x, edges = data.x.to(device), data.edge_index.to(device)
        with torch.no_grad():
            full = model.analyze(x, edges)
            logits = head(full["fused_z"][data.val_idx.to(device)])
            metrics = _metrics(logits, data.y[data.val_idx], int(data.num_classes))
        if abs(metrics["val_acc"] - float(checkpoint["metrics"]["val_acc"])) > 1e-6:
            raise RuntimeError(f"Validation metric did not reproduce: {cp_path}")
        stage, attentions = _stage1_rows(model, x, edges, full, dataset, seed, variant)
        stage1_rows.extend(stage); stage1_attn_rows.extend(attentions)
        exec_attn_rows.extend(_execution_attention_rows(full, dataset, seed, variant))
        ops, geoms = _operator_rows(full, dataset, seed, variant)
        operator_rows.extend(ops); geometry_rows.extend(geoms)
        dynamicity_rows.extend(_dynamicity_rows(full, dataset, seed, variant))
        trace_selected.append((dataset, seed, variant, int(checkpoint["epoch"])))
        if args.mode == "full":
            if variant == "context_grounded_dynamic":
                rows, changed = _intervention_rows(model, head, data, x, edges, full, dataset, seed,
                                                   "within_target_relation_cyclic", relation_shuffle=True)
                relation_interventions.extend(rows)
                # Context interventions keep the paired text/visual contexts together and degree-matched.
                rows, changed_context = _intervention_rows(model, head, data, x, edges, full, dataset, seed,
                                                           "degree_matched_context_pair_shuffle", context_shuffle=True)
                context_interventions.extend(rows)
                rows, changed_frozen = _intervention_rows(model, head, data, x, edges, full, dataset, seed,
                                                          "step1_frozen_query_H0", frozen_query=True)
                frozen_interventions.extend(rows)
                rows, changed_off = _intervention_rows(model, head, data, x, edges, full, dataset, seed,
                                                       "operator_off", operator_enabled=False)
                off_interventions.extend(rows)
                del changed, changed_context, changed_frozen, changed_off
                p0 = _load_p0(dataset, seed)
                indices = _p0_indices(p0, full["canonical_edge_index"], data.num_nodes)
                p0_rows.extend(_p0_rows(p0, full, indices, dataset, seed))
        print(f"ANALYZED {args.mode} {dataset}/{variant}/seed{seed} edges={full['canonical_edge_index'].size(1)}", flush=True)
        del model, head, data, full, x, edges
        if device.type == "cuda":
            torch.cuda.empty_cache()

    grad_rows, growth_rows = _trace_rows(args.mode, trace_selected)
    # Add per-run first-nonzero and selected-epoch gradient summaries to every epoch row.
    grouped = defaultdict(list)
    for row in grad_rows:
        grouped[(row["dataset"], row["seed"], row["variant"], row["group"])].append(row)
    for key, values in grouped.items():
        nonzero = [int(r["epoch"]) for r in values if float(r["gradient_rms_preclip"]) > 0.0]
        median = float(np.median([float(r["gradient_rms_preclip"]) for r in values]))
        best = next((float(r["gradient_rms_preclip"]) for r in values if r["is_best_epoch"]), float("nan"))
        for row in values:
            row["first_nonzero_epoch"] = min(nonzero) if nonzero else ""
            row["trajectory_median_gradient_rms"] = median
            row["best_epoch_gradient_rms"] = best
    prefix = "smoke_" if args.mode == "smoke" else ""
    _write(RESULT_ROOT / f"{prefix}stage1_health.csv", stage1_rows)
    _write(RESULT_ROOT / f"{prefix}stage1_attention.csv", stage1_attn_rows)
    _write(RESULT_ROOT / f"{prefix}training_gradient_trace.csv", grad_rows)
    _write(RESULT_ROOT / f"{prefix}operator_growth_trace.csv", growth_rows)
    _write(RESULT_ROOT / f"{prefix}execution_attention.csv", exec_attn_rows)
    _write(RESULT_ROOT / f"{prefix}operator_diagnostics.csv", operator_rows)
    _write(RESULT_ROOT / f"{prefix}operation_geometry.csv", geometry_rows)
    _write(RESULT_ROOT / f"{prefix}dynamicity_diagnostics.csv", dynamicity_rows)
    if args.mode == "full":
        _write(RESULT_ROOT / "intervention_relation_shuffle.csv", relation_interventions)
        _write(RESULT_ROOT / "intervention_context_shuffle.csv", context_interventions)
        _write(RESULT_ROOT / "intervention_frozen_query.csv", frozen_interventions)
        _write(RESULT_ROOT / "intervention_operator_off.csv", off_interventions)
        _write(RESULT_ROOT / "p0_operation_hardcases.csv", p0_rows)
    print(f"WROTE {len(stage1_rows)} Stage-I, {len(grad_rows)} gradient, {len(operator_rows)} operator rows", flush=True)


if __name__ == "__main__":
    main()
