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
from src.models.interaction_core_v3 import Model, VARIANTS  # noqa: E402
from src.models.interaction_core_v3_components import operation_variance_decomposition  # noqa: E402

OUTPUT_ROOT = ROOT / "outputs" / "m0" / "conditioner_v3"
RESULT_ROOT = ROOT / "results" / "m0" / "conditioner_v3"
DATASETS = ("Movies", "Grocery", "Reddit-S")
SEEDS = (42, 43, 44)
P0_ROOT = Path("/hdd1/DataInHere/YHF/mag_model/outputs/problem_validation")


def _write(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields, seen = [], set()
        for row in rows:
            for key in row:
                if key not in seen: seen.add(key); fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _compose(dataset, seed, variant):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", "model=interaction_core_v3",
            f"model.variant={variant}", f"seed={seed}", "num_runs=1", "task.evaluate_test=false",
        ])


def _checkpoint(mode, dataset, variant, seed):
    base = OUTPUT_ROOT / "smoke" if mode == "smoke" else OUTPUT_ROOT
    return base / "checkpoints" / dataset / variant / f"seed{seed}.pt"


def _metrics(logits, labels, num_classes):
    pred, target = logits.argmax(-1).detach().cpu(), labels.detach().cpu()
    return {"val_acc": float((pred == target).float().mean()),
            "val_macro_f1": float(f1_score(target.numpy(), pred.numpy(), labels=list(range(int(num_classes))),
                                           average="macro", zero_division=0)),
            "pred": pred}


def _stats(tensor):
    value = torch.as_tensor(tensor).detach().float().reshape(-1).cpu()
    value = value[torch.isfinite(value)]
    if not value.numel():
        return {"mean": float("nan"), "median": float("nan"), "p10": float("nan"),
                "p90": float("nan"), "p95": float("nan"), "n": 0}
    q = torch.quantile(value, torch.tensor([.1, .5, .9, .95]))
    return {"mean": float(value.mean()), "median": float(q[1]), "p10": float(q[0]),
            "p90": float(q[2]), "p95": float(q[3]), "n": int(value.numel())}


def _safe_spearman(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 2 or np.unique(a[keep]).size < 2 or np.unique(b[keep]).size < 2: return float("nan")
    return float(spearmanr(a[keep], b[keep]).statistic)


@torch.no_grad()
def _stage1_rows(model, x, full, dataset, seed, variant):
    rows = []
    edges = full["canonical_edge_index"]; src, dst = edges
    for modality in ("text", "visual"):
        rel = full[f"relation_{modality}"].float()
        rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                     "measure": "relation_feature_variance", "modality": modality,
                     **_stats(rel.var(0, unbiased=False).mean().reshape(1))})
    cross = 1 - torch.nn.functional.cosine_similarity(full["relation_text"], full["relation_visual"], dim=-1, eps=1e-8)
    rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                 "measure": "R_T_R_V_1_minus_cosine", "modality": "both", **_stats(cross)})
    edge_cpu = edges.detach().cpu(); edge_index = {(int(s), int(d)): i for i, (s, d) in enumerate(edge_cpu.t().tolist())}
    forward, reverse = [], []
    for i, (s, d) in enumerate(edge_cpu.t().tolist()):
        j = edge_index.get((d, s))
        if j is not None and s < d: forward.append(i); reverse.append(j)
    for modality in ("text", "visual"):
        if forward:
            a = full[f"relation_{modality}"][torch.tensor(forward, device=x.device)]
            b = full[f"relation_{modality}"][torch.tensor(reverse, device=x.device)]
            direction = 1 - torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-8)
            dist = (a - b).norm(dim=-1)
        else: direction, dist = x.new_empty(0), x.new_empty(0)
        rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                     "measure": "directed_reverse_1_minus_cosine", "modality": modality,
                     **_stats(direction), "num_undirected_pairs": len(forward)})
        rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                     "measure": "directed_reverse_l2", "modality": modality,
                     **_stats(dist), "num_undirected_pairs": len(forward)})
    # Same-checkpoint context correction: replace every recipient context by
    # the learned degree-one NO_CONTEXT token, retaining pair encodings and weights.
    count = src.numel()
    no_relation_t_parts, no_relation_v_parts = [], []
    for start in range(0, count, model.edge_chunk_size):
        end = min(start + model.edge_chunk_size, count)
        no_t = model.no_context_text.to(full["H0_text"]).expand(end - start, -1)
        no_v = model.no_context_visual.to(full["H0_visual"]).expand(end - start, -1)
        keys_t = torch.stack((full["pair_visual"][start:end], no_t, no_v), dim=1)
        keys_v = torch.stack((full["pair_text"][start:end], no_t, no_v), dim=1)
        no_relation_t_parts.append(model.relation_text_cross_attention(
            full["pair_text"][start:end], keys_t)[0])
        no_relation_v_parts.append(model.relation_visual_cross_attention(
            full["pair_visual"][start:end], keys_v)[0])
    no_relation_t = torch.cat(no_relation_t_parts) if no_relation_t_parts else full["relation_text"].new_empty((0, model.relation_dim))
    no_relation_v = torch.cat(no_relation_v_parts) if no_relation_v_parts else full["relation_visual"].new_empty((0, model.relation_dim))
    for modality, no_relation in (("text", no_relation_t), ("visual", no_relation_v)):
        change = (full[f"relation_{modality}"] - no_relation).norm(dim=-1)
        rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                     "measure": "full_vs_NO_CONTEXT_relation_correction_l2", "modality": modality,
                     **_stats(change), "interpretation": "same-checkpoint context ablation sensitivity"})
    # Alignment sensitivity uses the v2 degree-matched paired shuffle.
    shuffled = model._pair_and_context(full["H0_text"], full["H0_visual"], src, dst, True, False)
    for modality in ("text", "visual"):
        change = (full[f"relation_{modality}"] - shuffled[f"relation_{modality}"]).norm(dim=-1)
        rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                     "measure": "full_vs_degree_matched_context_shuffle_l2", "modality": modality,
                     **_stats(change), "interpretation": "same-checkpoint context alignment sensitivity"})
    return rows


def _base_rows(full, dataset, seed, variant):
    rows = []
    for modality in ("text", "visual"):
        r = full[f"base_relation_{modality}"]
        rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                     "modality": modality, "relation_code_l2": _stats(r.norm(dim=-1))["mean"],
                     "feature_variance_mean": float(r.var(0, unbiased=False).mean()),
                     "edge_variance_mean": float(r.var(unbiased=False))})
    return rows


def _conditioner_rows(model, full, dataset, seed, variant):
    rows = []
    for modality in ("text", "visual"):
        wc = getattr(model, f"{modality}_conditioner").output.weight.detach()
        for step in (0, 1):
            r = full[f"base_relation_{modality}"]
            correction = full[f"delta_execution_code_{modality}_step{step}"]
            latent = full[f"conditioner_latent_{modality}_step{step}"]
            ratio = correction.norm(dim=-1) / (r.norm(dim=-1) + 1e-12)
            row = {"dataset": dataset, "seed": seed, "variant": variant, "modality": modality,
                   "interaction_step": step, **{f"rho_xi_{k}": v for k, v in _stats(ratio).items()},
                   "latent_norm_mean": _stats(latent.norm(dim=-1))["mean"],
                   "latent_feature_variance": float(latent.var(0, unbiased=False).mean()) if latent.numel() else 0.0,
                   "latent_edge_variance": float(latent.var(unbiased=False)) if latent.numel() else 0.0,
                   "latent_fraction_near_zero": float((latent.norm(dim=-1) < 1e-8).float().mean()) if latent.numel() else 1.0,
                   "W_c_frobenius_norm": float(wc.float().norm())}
            row.update({f"delta_xi_l2_{k}": v for k, v in _stats(correction.norm(dim=-1)).items()})
            rows.append(row)
    return rows


def _variance_rows(full, dataset, seed, variant):
    rows = []; dst = full["canonical_edge_index"][1]
    for modality in ("text", "visual"):
        entries = [("base_relation_r", full[f"base_relation_{modality}"])]
        entries += [(f"execution_xi_step{step}", full[f"execution_code_{modality}_step{step}"]) for step in (0, 1)]
        entries += [(f"operator_modulation_a_step{step}", full[f"modulation_{modality}_step{step}"]) for step in (0, 1)]
        entries += [(f"operator_delta_message_step{step}", full[f"delta_message_{modality}_step{step}"]) for step in (0, 1)]
        for object_name, vectors in entries:
            result = operation_variance_decomposition(vectors, dst, int(full["degree"].numel()))
            total = float(result["V_total"])
            error = abs(float(result["decomposition_error"]))
            if error > max(1e-5, 1e-5 * abs(total)):
                raise RuntimeError(f"operation variance identity failed for {dataset}/{seed}/{variant}/{modality}/{object_name}: {error}")
            for eta_name in ("eta_relation", "eta_target"):
                eta = float(result[eta_name])
                if eta < -1e-5 or eta > 1.00001:
                    raise RuntimeError(f"{eta_name} outside [0,1] for {dataset}/{seed}/{variant}/{modality}/{object_name}: {eta}")
            rows.append({"dataset": dataset, "seed": seed, "variant": variant, "modality": modality,
                         "object": object_name, **{k: float(v) for k, v in result.items()},
                         "node_balanced_scope": "secondary descriptive measure; excluded from strict variance identity"})
    return rows


def _operator_rows(full, dataset, seed, variant):
    diag, geometry = [], []
    for modality in ("text", "visual"):
        for step in (0, 1):
            key = f"{modality}_step{step}"
            ratio = full["operator_deviation_ratio"][key]
            row = {"dataset": dataset, "seed": seed, "variant": variant,
                   "modality": modality, "interaction_step": step}
            row.update({f"operator_deviation_ratio_{k}": v for k, v in _stats(ratio).items()})
            for measure in ("base_message_norm", "delta_message_norm"):
                row[f"{measure}_mean"] = _stats(full[measure][key])["mean"]
            diag.append(row)
            cosine = full[f"cosine_base_delta_{modality}_step{step}"]
            scale = full[f"message_scale_{modality}_step{step}"]
            rotation = full[f"message_rotation_{modality}_step{step}"]
            geo = {"dataset": dataset, "seed": seed, "variant": variant,
                   "modality": modality, "interaction_step": step}
            geo.update({f"cos_base_delta_{k}": v for k, v in _stats(cosine).items()})
            geo.update({f"operator_deviation_ratio_{k}": v for k, v in _stats(ratio).items()})
            geo.update({f"message_scale_{k}": v for k, v in _stats(scale).items()})
            geo.update({f"message_rotation_{k}": v for k, v in _stats(rotation).items()})
            finite = cosine[torch.isfinite(cosine)]
            geo["reinforce_fraction_cos_gt_0_3"] = float((finite > .3).float().mean()) if finite.numel() else float("nan")
            geo["suppress_fraction_cos_lt_minus_0_3"] = float((finite < -.3).float().mean()) if finite.numel() else float("nan")
            geo["redirect_fraction_abs_cos_le_0_3"] = float((finite.abs() <= .3).float().mean()) if finite.numel() else float("nan")
            geometry.append(geo)
    return diag, geometry


def _dynamicity_rows(full, dataset, seed, variant):
    rows = []; dst = full["canonical_edge_index"][1]; degree = full["degree"].float(); valid = degree > 0
    for modality in ("text", "visual"):
        h0, h1 = full[f"H0_{modality}"].float(), full[f"H1_{modality}"].float()
        d_h = (h1 - h0).norm(dim=-1)
        a0, a1 = full[f"modulation_{modality}_step0"].float(), full[f"modulation_{modality}_step1"].float()
        edge_da = (a1 - a0).norm(dim=-1)
        node_da = degree.new_zeros(degree.numel())
        node_xi = degree.new_zeros(degree.numel())
        if dst.numel():
            node_da.index_add_(0, dst, edge_da)
            node_xi.index_add_(0, dst, full[f"delta_execution_code_{modality}_step1"].float().norm(dim=-1))
        d_a = node_da / degree.clamp_min(1)
        d_xi = node_xi / degree.clamp_min(1)
        dh_np, da_np, dxi_np = d_h[valid].cpu().numpy(), d_a[valid].cpu().numpy(), d_xi[valid].cpu().numpy()
        rho_ha, rho_hxi = _safe_spearman(dh_np, da_np), _safe_spearman(dh_np, dxi_np)
        for measure, values, rho in (("D_H_recipient_state_change", d_h[valid], float("nan")),
                                     ("D_A_target_mean_modulation_change", d_a[valid], rho_ha),
                                     ("D_XI_target_mean_dynamic_correction", d_xi[valid], rho_hxi)):
            rows.append({"dataset": dataset, "seed": seed, "variant": variant, "modality": modality,
                         "measure": measure, **_stats(values), "spearman_D_H_D_A": rho_ha,
                         "spearman_D_H_D_XI": rho_hxi})
        for measure, values in (("edge_execution_code_step_change_l2",
                                 (full[f"execution_code_{modality}_step1"] - full[f"execution_code_{modality}_step0"]).norm(dim=-1)),
                                ("edge_operator_modulation_step_change_l2", (a1 - a0).norm(dim=-1)),
                                ("edge_operator_modulation_step_change_1_minus_cosine",
                                 1 - torch.nn.functional.cosine_similarity(a0, a1, dim=-1, eps=1e-8))):
            rows.append({"dataset": dataset, "seed": seed, "variant": variant, "modality": modality,
                         "measure": measure, **_stats(values), "spearman_D_H_D_A": rho_ha,
                         "spearman_D_H_D_XI": rho_hxi})
    return rows


def _intervention_rows(model, head, data, x, edges, base, dataset, seed, name, **kwargs):
    with torch.no_grad():
        changed = model.analyze(x, edges, **kwargs)
        val_idx = data.val_idx.to(x.device)
        labels = data.y[data.val_idx]
        logits = head(base["fused_z"][val_idx]); alt_logits = head(changed["fused_z"][val_idx])
        full_metrics = _metrics(logits, labels, int(data.num_classes)); alt_metrics = _metrics(alt_logits, labels, int(data.num_classes))
        diff_logits = (alt_logits - logits).norm(dim=-1)
        common = {"dataset": dataset, "seed": seed, "variant": model.variant, "intervention": name,
                  "val_acc_full": full_metrics["val_acc"], "val_acc_intervened": alt_metrics["val_acc"],
                  "val_acc_change": alt_metrics["val_acc"] - full_metrics["val_acc"],
                  "val_macro_f1_full": full_metrics["val_macro_f1"], "val_macro_f1_intervened": alt_metrics["val_macro_f1"],
                  "val_macro_f1_change": alt_metrics["val_macro_f1"] - full_metrics["val_macro_f1"],
                  "val_logit_l2_mean": _stats(diff_logits)["mean"], "val_logit_l2_p90": _stats(diff_logits)["p90"],
                  "val_prediction_flip_rate": float((full_metrics["pred"] != alt_metrics["pred"]).float().mean()),
                  "relation_memory_change_mean": _stats((base["relation_memory"] - changed["relation_memory"]).norm(dim=(1, 2)))["mean"]}
        dst = base["canonical_edge_index"][1]; deg_e = base["degree"][dst]
        if name == "within_target_relation_cyclic":
            delta = (base["relation_memory"] - changed["relation_memory"]).norm(dim=(1, 2))
            common["alignment_changed_edge_fraction"] = float((delta > 1e-12).float().mean())
            common["degree1_unchanged_edges"] = int(((deg_e == 1) & (delta <= 1e-12)).sum())
        if name == "degree_matched_paired_context_shuffle":
            delta = (base["context_text"] - changed["context_text"]).norm(dim=-1) + (base["context_visual"] - changed["context_visual"]).norm(dim=-1)
            common["alignment_changed_edge_fraction"] = float((delta > 1e-12).float().mean())
            common["degree1_unchanged_edges"] = int(((deg_e == 1) & (delta <= 1e-12)).sum())
        rows = []
        for modality in ("text", "visual"):
            for step in (0, 1):
                xi_delta = (base[f"execution_code_{modality}_step{step}"] - changed[f"execution_code_{modality}_step{step}"]).norm(dim=-1)
                a_delta = (base[f"modulation_{modality}_step{step}"] - changed[f"modulation_{modality}_step{step}"]).norm(dim=-1)
                msg_delta = (base[f"delta_message_{modality}_step{step}"] - changed[f"delta_message_{modality}_step{step}"]).norm(dim=-1)
                rows.append({**common, "modality": modality, "interaction_step": step,
                             "execution_code_change_l2_mean": _stats(xi_delta)["mean"],
                             "modulation_change_l2_mean": _stats(a_delta)["mean"],
                             "delta_message_change_l2_mean": _stats(msg_delta)["mean"]})
        return rows, changed


def _load_p0(dataset, seed):
    path = P0_ROOT / "p02" / dataset / f"seed{seed}" / "conditional_feature" / "edge_diagnostics.pt"
    split_path = P0_ROOT / "splits" / f"{dataset}.pt"
    p0 = torch.load(path, map_location="cpu", weights_only=False)
    split = torch.load(split_path, map_location="cpu", weights_only=False)
    if p0.get("test_evaluation") is not False or p0.get("test_labels_accessed") is not False:
        raise RuntimeError(f"P0 artifact is not test sealed: {path}")
    if not torch.equal(p0["target_node"], split["sampled_edge_target_node"]):
        raise RuntimeError(f"P0 target mismatch frozen split: {dataset}/{seed}")
    if not torch.equal(p0["neighbor_node"], split["sampled_edge_neighbor_node"]):
        raise RuntimeError(f"P0 source mismatch frozen split: {dataset}/{seed}")
    return p0


def _p0_indices(p0, edges, num_nodes):
    graph_keys = edges[0].detach().cpu() * int(num_nodes) + edges[1].detach().cpu()
    keys = p0["neighbor_node"].long() * int(num_nodes) + p0["target_node"].long()
    pos = torch.searchsorted(graph_keys, keys); safe = pos.clamp(max=max(graph_keys.numel() - 1, 0))
    if not graph_keys.numel() or not torch.equal(graph_keys[safe], keys):
        raise RuntimeError("P0 sampled directed edge was not found in canonical model graph")
    return pos


def _p0_rows(p0, full, indices, dataset, seed):
    rows = []
    for modality in ("text", "visual"):
        sim = p0[f"probe_sim_{modality}"].float().numpy()
        utility = p0[f"utility_ce_{modality}"].float().numpy()
        for qi, group in enumerate(np.array_split(np.argsort(sim, kind="mergesort"), 5), 1):
            if qi not in (1, 5): continue
            group_mask = np.zeros(len(sim), dtype=bool); group_mask[group] = True
            for step in (0, 1):
                edge_ids = indices.to(full[f"base_relation_{modality}"].device)
                vectors = {
                    "relation_r": full[f"base_relation_{modality}"][edge_ids].detach().float().cpu(),
                    "execution_xi": full[f"execution_code_{modality}_step{step}"][edge_ids].detach().float().cpu(),
                    "operator_modulation_a": full[f"modulation_{modality}_step{step}"][edge_ids].detach().float().cpu(),
                }
                groups = {}
                for utility_group, mask_utility in (("beneficial", utility > 0), ("harmful", utility < 0)):
                    mask = torch.as_tensor(group_mask & mask_utility)
                    groups[utility_group] = {}
                    for stage, tensor in vectors.items():
                        selected = tensor[mask]
                        centroid = selected.mean(0) if selected.numel() else torch.full((tensor.size(-1),), float("nan"))
                        groups[utility_group][stage] = (centroid, int(selected.size(0)),
                            float((selected - centroid).norm(dim=-1).mean()) if selected.numel() else float("nan"))
                        rows.append({"dataset": dataset, "seed": seed, "variant": "context_bilinear_delta",
                                     "modality": modality, "similarity_quintile": f"Q{qi}",
                                     "utility_group": utility_group, "interaction_step": step, "stage": stage,
                                     "n_relations": int(selected.size(0)), "centroid_norm": float(centroid.norm()) if selected.numel() else float("nan"),
                                     "within_group_radius": groups[utility_group][stage][2],
                                     "normalized_centroid_separation": float("nan")})
                for stage in vectors:
                    b, h = groups["beneficial"][stage], groups["harmful"][stage]
                    sep = float((b[0] - h[0]).norm()) / (0.5 * (b[2] + h[2]) + 1e-12) if b[1] and h[1] else float("nan")
                    rows.append({"dataset": dataset, "seed": seed, "variant": "context_bilinear_delta",
                                 "modality": modality, "similarity_quintile": f"Q{qi}",
                                 "utility_group": "beneficial_vs_harmful", "interaction_step": step,
                                 "stage": stage, "n_relations": b[1] + h[1], "centroid_norm": float("nan"),
                                 "within_group_radius": float("nan"), "normalized_centroid_separation": sep})
                for label, utility_mask in (("beneficial", utility > 0), ("harmful", utility < 0)):
                    mask = torch.as_tensor(group_mask & utility_mask)
                    edge_ids = indices.to(full["operator_deviation_ratio"][f"{modality}_step{step}"].device)
                    ratio = full["operator_deviation_ratio"][f"{modality}_step{step}"][edge_ids].detach().float().cpu()[mask]
                    cos = full[f"cosine_base_delta_{modality}_step{step}"][edge_ids].detach().float().cpu()[mask]
                    rotation = full[f"message_rotation_{modality}_step{step}"][edge_ids].detach().float().cpu()[mask]
                    rows.append({"dataset": dataset, "seed": seed, "variant": "context_bilinear_delta",
                                 "modality": modality, "similarity_quintile": f"Q{qi}", "utility_group": label,
                                 "interaction_step": step, "stage": "semantic_operation_geometry",
                                 "operator_deviation_mean": _stats(ratio)["mean"],
                                 "cos_base_delta_mean": _stats(cos)["mean"],
                                 "message_rotation_mean": _stats(rotation)["mean"]})
    return rows


def _trace_rows(mode, selected):
    gradient, growth = [], []
    base = OUTPUT_ROOT / "smoke" if mode == "smoke" else OUTPUT_ROOT
    for dataset, seed, variant, best_epoch in selected:
        run_dir = base / "logs" / dataset / variant / f"seed{seed}"
        gpath = run_dir / "training_gradient_trace.csv"
        wpath = run_dir / "operator_growth_trace.csv"
        for path, output, is_gradient in ((gpath, gradient, True), (wpath, growth, False)):
            with path.open(newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
            for row in rows:
                row.update({"dataset": dataset, "seed": seed, "variant": variant,
                            "is_best_epoch": int(row["epoch"]) == int(best_epoch), "selected_best_epoch": int(best_epoch)})
                if is_gradient:
                    row["parameter_group"] = row["group"]; row["grad_rms"] = row["gradient_rms_preclip"]
                    row["parameter_norm"] = row["parameter_l2_norm"]
                else:
                    row["up_weight_norm"] = row["W_up_frobenius_norm"]
                    row["W_c_norm"] = row["W_c_frobenius_norm"]
                output.append(row)
    grouped = defaultdict(list)
    for row in gradient: grouped[(row["dataset"], row["seed"], row["variant"], row["group"])].append(row)
    for rows in grouped.values():
        nonzero = [int(r["epoch"]) for r in rows if float(r["gradient_rms_preclip"]) > 0]
        median = float(np.median([float(r["gradient_rms_preclip"]) for r in rows]))
        best = next((float(r["gradient_rms_preclip"]) for r in rows if r["is_best_epoch"]), float("nan"))
        for row in rows:
            row["first_nonzero_epoch"] = min(nonzero) if nonzero else ""
            row["median_grad_rms"] = median; row["best_epoch_grad_rms"] = best
    return gradient, growth


def main():
    parser = argparse.ArgumentParser(description="Analyze validation-selected M0-Core v3 checkpoints only.")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    selected = [("Movies", 42, v) for v in VARIANTS] if args.mode == "smoke" else [
        (d, s, v) for d in DATASETS for s in SEEDS for v in VARIANTS]
    stage1, base_rows, conditioner_rows, variation = [], [], [], []
    dynamicity, operator, geometry = [], [], []
    conditioner_off, step1_off, frozen_query = [], [], []
    relation_shuffle, context_shuffle, operator_off, p0_rows = [], [], [], []
    trace_selected = []
    for dataset, seed, variant in selected:
        path = _checkpoint(args.mode, dataset, variant, seed)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("selection") != "best_val_accuracy" or any(k.startswith("test_") for k in checkpoint.get("metrics", {})):
            raise RuntimeError(f"Checkpoint is not validation-only: {path}")
        cfg = _compose(dataset, seed, variant); data = load_mag_data(cfg, "nc", seed)
        info = {"input_dim": data.input_dim, "num_nodes": data.num_nodes, "num_classes": data.num_classes,
                "text_dim": int(data.x_t.size(1)), "visual_dim": int(data.x_i.size(1))}
        model = Model(cfg, info).to(device); model.load_state_dict(checkpoint["model_state"])
        head = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device); head.load_state_dict(checkpoint["head_state"])
        model.eval(); head.eval(); x, edges = data.x.to(device), data.edge_index.to(device)
        with torch.no_grad():
            full = model.analyze(x, edges)
            val_idx = data.val_idx.to(device)
            measured = _metrics(head(full["fused_z"][val_idx]), data.y[data.val_idx], int(data.num_classes))
        if abs(measured["val_acc"] - float(checkpoint["metrics"]["val_acc"])) > 1e-6:
            raise RuntimeError(f"Validation metric did not reproduce: {path}")
        if variant == "context_bilinear_delta":
            for modality in ("text", "visual"):
                if not torch.equal(full[f"delta_execution_code_{modality}_step0"],
                                   torch.zeros_like(full[f"delta_execution_code_{modality}_step0"])):
                    raise RuntimeError("trained delta conditioner violated exact step0 correction invariant")
                if not torch.equal(full[f"execution_code_{modality}_step0"], full[f"base_relation_{modality}"]):
                    raise RuntimeError("trained delta conditioner violated xi_step0 == r")
        stage1.extend(_stage1_rows(model, x, full, dataset, seed, variant))
        base_rows.extend(_base_rows(full, dataset, seed, variant))
        conditioner_rows.extend(_conditioner_rows(model, full, dataset, seed, variant))
        variation.extend(_variance_rows(full, dataset, seed, variant))
        dynamicity.extend(_dynamicity_rows(full, dataset, seed, variant))
        op, geo = _operator_rows(full, dataset, seed, variant); operator.extend(op); geometry.extend(geo)
        trace_selected.append((dataset, seed, variant, int(checkpoint["epoch"])))
        if args.mode == "full":
            if variant.startswith("context_bilinear"):
                rows, _ = _intervention_rows(model, head, data, x, edges, full, dataset, seed,
                                              "conditioner_off", conditioner_enabled=False)
                conditioner_off.extend(rows)
            if variant == "context_bilinear_delta":
                rows, changed = _intervention_rows(model, head, data, x, edges, full, dataset, seed,
                                                    "step1_recipient_state_change_off", step1_dynamic_off=True)
                step1_off.extend(rows); del changed
                rows, changed = _intervention_rows(model, head, data, x, edges, full, dataset, seed,
                                                    "within_target_relation_cyclic", relation_shuffle=True)
                relation_shuffle.extend(rows); del changed
                rows, changed = _intervention_rows(model, head, data, x, edges, full, dataset, seed,
                                                    "degree_matched_paired_context_shuffle", context_shuffle=True)
                context_shuffle.extend(rows); del changed
                rows, changed = _intervention_rows(model, head, data, x, edges, full, dataset, seed,
                                                    "operator_off", operator_enabled=False)
                operator_off.extend(rows); del changed
                p0 = _load_p0(dataset, seed); indices = _p0_indices(p0, full["canonical_edge_index"], data.num_nodes)
                p0_rows.extend(_p0_rows(p0, full, indices, dataset, seed))
            if variant == "context_attn_dynamic":
                rows, changed = _intervention_rows(model, head, data, x, edges, full, dataset, seed,
                                                    "step1_frozen_query_H0", frozen_query=True)
                frozen_query.extend(rows); del changed
        print(f"ANALYZED {args.mode} {dataset}/{variant}/seed{seed} edges={full['canonical_edge_index'].size(1)}", flush=True)
        del model, head, data, full, x, edges
        if device.type == "cuda": torch.cuda.empty_cache()
    gradients, growth = _trace_rows(args.mode, trace_selected)
    prefix = "smoke_" if args.mode == "smoke" else ""
    _write(RESULT_ROOT / f"{prefix}stage1_health.csv", stage1)
    _write(RESULT_ROOT / f"{prefix}base_relation_diagnostics.csv", base_rows)
    _write(RESULT_ROOT / f"{prefix}conditioner_diagnostics.csv", conditioner_rows)
    _write(RESULT_ROOT / f"{prefix}training_gradient_trace.csv", gradients)
    _write(RESULT_ROOT / f"{prefix}conditioner_growth_trace.csv", growth)
    _write(RESULT_ROOT / f"{prefix}variation_decomposition.csv", variation)
    _write(RESULT_ROOT / f"{prefix}dynamicity_diagnostics.csv", dynamicity)
    _write(RESULT_ROOT / f"{prefix}operator_diagnostics.csv", operator)
    _write(RESULT_ROOT / f"{prefix}operation_geometry.csv", geometry)
    if args.mode == "full":
        _write(RESULT_ROOT / "intervention_conditioner_off.csv", conditioner_off)
        _write(RESULT_ROOT / "intervention_step1_dynamic_off.csv", step1_off)
        _write(RESULT_ROOT / "intervention_frozen_query_attn.csv", frozen_query)
        _write(RESULT_ROOT / "intervention_relation_shuffle.csv", relation_shuffle)
        _write(RESULT_ROOT / "intervention_context_shuffle.csv", context_shuffle)
        _write(RESULT_ROOT / "intervention_operator_off.csv", operator_off)
        _write(RESULT_ROOT / "p0_stagewise_hardcases.csv", p0_rows)
    print(f"WROTE {len(stage1)} Stage-I, {len(gradients)} gradient, {len(variation)} variance rows", flush=True)


if __name__ == "__main__": main()
