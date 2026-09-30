from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT_ROOT = ROOT / "outputs" / "m0" / "stage3_history"
V3_OUTPUT_ROOT = ROOT / "outputs" / "m0" / "conditioner_v3"
RESULT_ROOT = ROOT / "results" / "m0" / "stage3_history"
P0_ROOT = Path("/hdd1/DataInHere/YHF/mag_model/outputs/problem_validation")
DATASETS = ("Movies", "Grocery", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = ("terminal", "state_history", "operation_history", "relation_operation_history")
TOKEN_LABELS = ("T0", "T1", "T2", "V0", "V1", "V2")
DEGREE_BINS = ((1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 8, "5-8"),
               (9, 16, "9-16"), (17, math.inf, "17+"))


def _write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields, seen = [], set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key); fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _stats(value) -> dict:
    x = torch.as_tensor(value).detach().float().reshape(-1).cpu()
    x = x[torch.isfinite(x)]
    if not x.numel():
        return {"mean": float("nan"), "median": float("nan"), "p10": float("nan"),
                "p90": float("nan"), "p95": float("nan"), "n": 0}
    q = torch.quantile(x, torch.tensor([0.1, 0.5, 0.9, 0.95]))
    return {"mean": float(x.mean()), "median": float(q[1]), "p10": float(q[0]),
            "p90": float(q[2]), "p95": float(q[3]), "n": int(x.numel())}


def _spearman(a, b) -> float:
    a = torch.as_tensor(a).detach().float().cpu().numpy().reshape(-1)
    b = torch.as_tensor(b).detach().float().cpu().numpy().reshape(-1)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.unique(a[keep]).size < 2 or np.unique(b[keep]).size < 2:
        return float("nan")
    return float(spearmanr(a[keep], b[keep]).statistic)


def _compose(dataset: str, seed: int, model_name: str, variant: str | None = None):
    overrides = [f"dataset={dataset}", "task=nc", f"model={model_name}", f"seed={seed}",
                 "num_runs=1", "task.evaluate_test=false"]
    if model_name == "interaction_full_s3":
        overrides.extend(["model.variant=context_bilinear_absolute", f"model.stage3_variant={variant or 'terminal'}"])
    elif variant is not None:
        overrides.append(f"model.variant={variant}")
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def _load_data(dataset: str, seed: int):
    from src.data import load_mag_data
    cfg = _compose(dataset, seed, "interaction_full_s3", "terminal")
    return cfg, load_mag_data(cfg, "nc", seed)


def _info(data) -> dict:
    return {"input_dim": data.input_dim, "num_nodes": data.num_nodes, "num_classes": data.num_classes,
            "text_dim": int(data.x_t.size(1)), "visual_dim": int(data.x_i.size(1))}


def _model_for_s3(dataset: str, seed: int, variant: str, data_info: dict, device: torch.device):
    from src.models.interaction_full_s3 import Model
    cfg = _compose(dataset, seed, "interaction_full_s3", variant)
    return Model(cfg, data_info).to(device)


def _v3_model(dataset: str, seed: int, data_info: dict, device: torch.device):
    from src.models.interaction_core_v3 import Model
    cfg = _compose(dataset, seed, "interaction_core_v3", "context_bilinear_absolute")
    return Model(cfg, data_info).to(device)


def _checkpoint(mode: str, dataset: str, variant: str, seed: int) -> Path:
    stage = "smoke" if mode == "smoke" else "full"
    return OUTPUT_ROOT / stage / "checkpoints" / dataset / variant / f"seed{seed}.pt"


def _v3_checkpoint(dataset: str, seed: int) -> Path:
    return V3_OUTPUT_ROOT / "checkpoints" / dataset / "context_bilinear_absolute" / f"seed{seed}.pt"


def _metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict:
    pred, target = logits.argmax(-1).detach().cpu(), labels.detach().cpu()
    return {"val_acc": float((pred == target).float().mean()),
            "val_macro_f1": float(f1_score(target.numpy(), pred.numpy(), labels=list(range(num_classes)),
                                           average="macro", zero_division=0)), "pred": pred}


def _compare_stage12(v3_values: dict, s3_values: dict, dataset: str, seed: int, variant: str, mode: str) -> list[dict]:
    pairs = []
    for modality in ("text", "visual"):
        for old, new in ((f"H0_{modality}", f"H0_{modality}"),
                         (f"H1_{modality}", f"H1_{modality}"),
                         (f"final_{modality}", f"H2_{modality}"),
                         (f"base_relation_{modality}", f"base_relation_{modality}"),
                         (f"relation_{modality}", f"relation_{modality}")):
            pairs.append((old, new, old))
        for step in (0, 1):
            pairs.extend([
                (f"modulation.{modality}_step{step}", f"operation_modulation.{modality}_step{step}", f"a_{modality}_step{step}"),
                (f"execution_code.{modality}_step{step}", f"execution_code.{modality}_step{step}", f"xi_{modality}_step{step}"),
            ])
    pairs.append(("fused_z", "terminal_z", "terminal_fused_z"))
    rows = []
    for old_key, new_key, label in pairs:
        old_value, new_value = v3_values, s3_values
        for key in old_key.split("."):
            old_value = old_value[key]
        for key in new_key.split("."):
            new_value = new_value[key]
        diff = (old_value.detach().float() - new_value.detach().float()).abs()
        maximum = float(diff.max()) if diff.numel() else 0.0
        rows.append({"mode": mode, "dataset": dataset, "seed": seed, "variant": variant,
                     "measure": label, "max_abs_difference": maximum,
                     "allclose_atol_1e-5_rtol_1e-5": bool(torch.allclose(old_value, new_value, atol=1e-5, rtol=1e-5)),
                     "passed": bool(torch.allclose(old_value, new_value, atol=1e-5, rtol=1e-5))})
    return rows


@torch.no_grad()
def _preflight(device: torch.device) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            path = _v3_checkpoint(dataset, seed)
            if not path.exists():
                raise FileNotFoundError(f"Frozen v3 absolute checkpoint required for regression gate: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("selection") != "best_val_accuracy" or any(k.startswith("test_") for k in payload.get("metrics", {})):
                raise RuntimeError(f"Frozen parent checkpoint is not validation-only: {path}")
            cfg, data = _load_data(dataset, seed)
            info = payload["data_info"]
            v3 = _v3_model(dataset, seed, info, device)
            s3 = _model_for_s3(dataset, seed, "terminal", info, device)
            v3.load_state_dict(payload["model_state"])
            missing, unexpected = s3.load_state_dict(payload["model_state"], strict=False)
            expected_missing = {k for k in s3.state_dict() if k not in payload["model_state"]}
            if set(missing) != expected_missing or unexpected:
                raise RuntimeError(f"Could not map all shared v3 weights into terminal S3: {path}")
            v3.eval(); s3.eval()
            x, edge = data.x.to(device), data.edge_index.to(device)
            old = v3.analyze(x, edge)
            new = s3.analyze(x, edge)
            run_rows = _compare_stage12(old, new, dataset, seed, "terminal", "preflight")
            rows.extend(run_rows)
            if any(not row["passed"] for row in run_rows):
                raise RuntimeError(f"Stage-I/II exact-equivalence gate failed on {dataset} seed {seed}")
            print(f"REGRESSION PASS preflight {dataset}/seed{seed}", flush=True)
            del v3, s3, data, old, new
            if device.type == "cuda":
                torch.cuda.empty_cache()
    _write(RESULT_ROOT / "stage12_regression.csv", rows)
    return rows


def _evaluate(model, head, data, x, edges, val_idx):
    values = model.analyze(x, edges)
    logits = head(values["fused_z"][val_idx])
    val_labels = data.y[val_idx.detach().cpu()].to(logits.device)
    metrics = _metrics(logits, val_labels, int(data.num_classes))
    return values, logits, metrics


def _history_branch_rows(full: dict, model, dataset: str, seed: int, variant: str) -> list[dict]:
    rho = full["history_branch"].norm(dim=-1) / (full["terminal_z"].norm(dim=-1) + 1e-12)
    stats = _stats(rho)
    return [{"dataset": dataset, "seed": seed, "variant": variant, **{f"rho_hist_{k}": v for k, v in stats.items()},
             "history_branch_l2_mean": float(full["history_branch"].norm(dim=-1).mean()),
             "terminal_l2_mean": float(full["terminal_z"].norm(dim=-1).mean()),
             "W_hist_frobenius_norm": float(model.history_output.weight.detach().float().norm())}]


def _attention_rows(full: dict, dataset: str, seed: int, variant: str) -> list[dict]:
    alpha_heads = full.get("readout_attention_heads")
    if alpha_heads is None:
        return []
    rows = []
    alpha = alpha_heads.detach().float()
    for head in range(alpha.size(1)):
        for token, label in enumerate(TOKEN_LABELS):
            stats = _stats(alpha[:, head, token])
            rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                         "head": f"head{head}", "token": label,
                         "across_node_std": float(alpha[:, head, token].std(unbiased=False)), **stats})
    mean_alpha = alpha.mean(dim=1)
    for token, label in enumerate(TOKEN_LABELS):
        stats = _stats(mean_alpha[:, token])
        rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                     "head": "mean", "token": label,
                     "across_node_std": float(mean_alpha[:, token].std(unbiased=False)), **stats})
    return rows


def _preference_rows(full: dict, dataset: str, seed: int, variant: str) -> list[dict]:
    if "readout_attention" not in full:
        return []
    alpha = full["readout_attention"].detach().float()
    entropy = -(alpha.clamp_min(1e-12) * alpha.clamp_min(1e-12).log()).sum(-1)
    degree = full["degree"].detach().cpu()
    groups = [("all_positive_degree", (degree > 0))]
    for low, high, label in DEGREE_BINS:
        groups.append((label, (degree >= low) & (degree <= high)))
    rows = []
    for label, mask_cpu in groups:
        mask = mask_cpu.to(alpha.device)
        selected = alpha[mask]
        selected_entropy = entropy[mask]
        if selected.size(0):
            stage = (selected[:, 0] + selected[:, 3], selected[:, 1] + selected[:, 4], selected[:, 2] + selected[:, 5])
            modality = (selected[:, :3].sum(-1), selected[:, 3:].sum(-1))
            token_std = selected.std(dim=0, unbiased=False)
            pref_var = float(selected.var(dim=0, unbiased=False).mean())
            mean_entropy = float(selected_entropy.mean())
            masses = [float(v.mean()) for v in stage + modality]
            std_tokens = [float(v) for v in token_std]
        else:
            masses = [float("nan")] * 5
            std_tokens = [float("nan")] * 6
            pref_var = mean_entropy = float("nan")
        rows.append({"dataset": dataset, "seed": seed, "variant": variant, "degree_bin": label,
                     "num_nodes": int(mask.sum()), "stage0_mass": masses[0], "stage1_mass": masses[1],
                     "stage2_mass": masses[2], "text_mass": masses[3], "visual_mass": masses[4],
                     "attention_entropy_mean": mean_entropy, "preference_variance": pref_var,
                     "token_std_T0": std_tokens[0], "token_std_T1": std_tokens[1], "token_std_T2": std_tokens[2],
                     "token_std_V0": std_tokens[3], "token_std_V1": std_tokens[4], "token_std_V2": std_tokens[5]})
    return rows


def _profile_rows(full: dict, dataset: str, seed: int, variant: str) -> tuple[list[dict], list[dict]]:
    op_rows, env_rows = [], []
    for modality in ("text", "visual"):
        for step in (0, 1):
            profile = full[f"operation_profile_{modality}_step{step}"].detach().float()
            mu, sigma = profile[:, :32], profile[:, 32:]
            degree = full["degree"].to(profile.device)
            active = degree > 0
            op_rows.append({"dataset": dataset, "seed": seed, "variant": variant, "modality": modality,
                            "interaction_state": step + 1, "profile_dim": int(profile.size(-1)),
                            "mean_operation_norm": float(mu.norm(dim=-1).mean()),
                            "mean_operation_std_norm": float(sigma.norm(dim=-1).mean()),
                            "mean_incoming_operation_heterogeneity": float(sigma[active].mean()) if active.any() else float("nan"),
                            "zero_degree_nodes": int((~active).sum())})
        env = full[f"relation_environment_{modality}"].detach().float()
        mu, sigma = env[:, :64], env[:, 64:]
        degree = full["degree"].to(env.device); active = degree > 0
        env_rows.append({"dataset": dataset, "seed": seed, "variant": variant, "modality": modality,
                         "environment_dim": int(env.size(-1)), "mean_relation_environment_norm": float(mu.norm(dim=-1).mean()),
                         "mean_relation_environment_std_norm": float(sigma.norm(dim=-1).mean()),
                         "mean_incoming_relation_heterogeneity": float(sigma[active].mean()) if active.any() else float("nan"),
                         "zero_degree_nodes": int((~active).sum())})
    return op_rows, env_rows


def _association_rows(full: dict, dataset: str, seed: int, variant: str) -> list[dict]:
    if "readout_attention" not in full:
        return []
    alpha = full["readout_attention"].detach().float()
    rows = []
    dst = full["canonical_edge_index"][1]
    n = alpha.size(0)
    for modality, offset in (("text", 0), ("visual", 3)):
        for step in (0, 1):
            stage = step + 1
            token_mass = alpha[:, offset + stage]
            transition = full[f"transition_{modality}_step{stage}"].detach().norm(dim=-1)
            profile = full[f"operation_profile_{modality}_step{step}"].detach()
            operation_heterogeneity = profile[:, 32:].mean(-1)
            edge_intensity = full["operator_deviation_ratio"][f"{modality}_step{step}"].detach()
            sums = edge_intensity.new_zeros(n).index_add(0, dst, edge_intensity)
            counts = torch.bincount(dst, minlength=n).to(edge_intensity.dtype)
            intensity = sums / counts.clamp_min(1)
            for metric, values in (("operation_intensity", intensity),
                                   ("state_transition_norm", transition),
                                   ("operation_heterogeneity", operation_heterogeneity)):
                rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                             "modality": modality, "interaction_state": stage,
                             "association": metric, "spearman_attention_mass": _spearman(token_mass, values),
                             "n_nodes": n})
    return rows


def _stage12_health(full: dict, dataset: str, seed: int, variant: str) -> list[dict]:
    rows = []
    rt, rv = full["relation_text"].detach(), full["relation_visual"].detach()
    rows.append({"dataset": dataset, "seed": seed, "variant": variant, "stage": "stage1",
                 "measure": "relation_feature_variance_text", "value": float(rt.float().var(0, unbiased=False).mean())})
    rows.append({"dataset": dataset, "seed": seed, "variant": variant, "stage": "stage1",
                 "measure": "relation_feature_variance_visual", "value": float(rv.float().var(0, unbiased=False).mean())})
    rows.append({"dataset": dataset, "seed": seed, "variant": variant, "stage": "stage1",
                 "measure": "modality_relation_1_minus_cosine_mean",
                 "value": float((1 - torch.nn.functional.cosine_similarity(rt, rv, dim=-1, eps=1e-8)).mean())})
    edges = full["canonical_edge_index"].detach().cpu()
    edge_map = {(int(s), int(d)): i for i, (s, d) in enumerate(edges.t().tolist())}
    forward, reverse = [], []
    for i, (s, d) in enumerate(edges.t().tolist()):
        j = edge_map.get((d, s))
        if j is not None and s < d:
            forward.append(i); reverse.append(j)
    for modality in ("text", "visual"):
        rel = full[f"relation_{modality}"]
        if forward:
            a, b = rel[torch.tensor(forward, device=rel.device)], rel[torch.tensor(reverse, device=rel.device)]
            value = float((1 - torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-8)).mean())
        else:
            value = float("nan")
        rows.append({"dataset": dataset, "seed": seed, "variant": variant, "stage": "stage1",
                     "measure": f"directed_reverse_1_minus_cosine_{modality}", "value": value})
        for step in (0, 1):
            ratio = full["operator_deviation_ratio"][f"{modality}_step{step}"]
            rotation = full["message_rotation"][f"{modality}_step{step}"]
            rows.append({"dataset": dataset, "seed": seed, "variant": variant, "stage": "stage2",
                         "measure": f"operator_deviation_ratio_{modality}_step{step}", "value": float(ratio.mean())})
            rows.append({"dataset": dataset, "seed": seed, "variant": variant, "stage": "stage2",
                         "measure": f"message_rotation_{modality}_step{step}", "value": float(rotation.mean())})
    return rows


def _intervention_rows(model, head, data, x, edges, val_idx, baseline_values, baseline_logits, baseline_metrics,
                       dataset: str, seed: int, intervention: str, kwargs: dict) -> dict:
    changed = model.analyze(x, edges, **kwargs)
    changed_logits = head(changed["fused_z"][val_idx])
    val_labels = data.y[val_idx.detach().cpu()].to(changed_logits.device)
    metrics = _metrics(changed_logits, val_labels, int(data.num_classes))
    row = {"dataset": dataset, "seed": seed, "variant": "relation_operation_history",
           "intervention": intervention,
           "val_acc_full": baseline_metrics["val_acc"], "val_acc_intervened": metrics["val_acc"],
           "val_acc_change_pp": 100 * (metrics["val_acc"] - baseline_metrics["val_acc"]),
           "val_macro_f1_full": baseline_metrics["val_macro_f1"], "val_macro_f1_intervened": metrics["val_macro_f1"],
           "val_macro_f1_change_pp": 100 * (metrics["val_macro_f1"] - baseline_metrics["val_macro_f1"]),
           "logit_l2_mean": float((baseline_logits - changed_logits).norm(dim=-1).mean()),
           "prediction_flip_rate": float((baseline_metrics["pred"] != metrics["pred"]).float().mean())}
    for key, label in (("readout_attention", "attention"), ("history_representation", "history_representation"),
                       ("history_tokens", "history_tokens"), ("readout_query", "query")):
        if key in changed and key in baseline_values:
            delta = (baseline_values[key].detach().float() - changed[key].detach().float())
            row[f"{label}_l2_mean"] = float(delta.reshape(delta.size(0), -1).norm(dim=-1).mean())
            if key == "readout_attention":
                row["attention_l1_mean"] = float(delta.abs().sum(dim=-1).mean())
    return row


def _p0_rows(dataset: str, seed: int, p0: dict, full: dict) -> list[dict]:
    if p0.get("test_evaluation") is not False or p0.get("test_labels_accessed") is not False:
        raise RuntimeError("P0 artifact was not sealed against test evaluation/labels")
    targets = p0["target_node"].long().cpu()
    rows = []
    alpha = full["readout_attention"].detach().float().cpu()
    for modality, offset in (("text", 0), ("visual", 3)):
        similarity = p0[f"probe_sim_{modality}"].float().cpu()
        utility = p0[f"utility_ce_{modality}"].float().cpu()
        for quintile, edge_ids in (("Q1", torch.argsort(similarity, stable=True)[:len(similarity)//5]),
                                  ("Q5", torch.argsort(similarity, stable=True)[-len(similarity)//5:])):
            for group, mask_u in (("beneficial", utility > 0), ("harmful", utility < 0)):
                chosen = edge_ids[mask_u[edge_ids]]
                nodes = targets[chosen].unique()
                nodes = nodes[(nodes >= 0) & (nodes < alpha.size(0))]
                if not nodes.numel():
                    continue
                profile0 = full[f"operation_profile_{modality}_step0"].detach().cpu()[nodes]
                profile1 = full[f"operation_profile_{modality}_step1"].detach().cpu()[nodes]
                rows.append({"dataset": dataset, "seed": seed, "variant": "relation_operation_history",
                             "modality": modality, "similarity_quintile": quintile, "utility_group": group,
                             "n_sampled_target_nodes": int(nodes.numel()),
                             "attention_intrinsic": float(alpha[nodes, offset].mean()),
                             "attention_state1": float(alpha[nodes, offset + 1].mean()),
                             "attention_state2": float(alpha[nodes, offset + 2].mean()),
                             "operation_profile_mean_norm_step1": float(profile0[:, :32].norm(dim=-1).mean()),
                             "operation_heterogeneity_step1": float(profile0[:, 32:].mean()),
                             "operation_profile_mean_norm_step2": float(profile1[:, :32].norm(dim=-1).mean()),
                             "operation_heterogeneity_step2": float(profile1[:, 32:].mean())})
    return rows


@torch.no_grad()
def _run_analysis(mode: str, device: torch.device) -> None:
    selected = [("Movies", 42, v) for v in VARIANTS] if mode == "smoke" else [
        (d, s, v) for d in DATASETS for s in SEEDS for v in VARIANTS]
    branch_rows, attention_rows, preference_rows = [], [], []
    operation_rows, environment_rows, association_rows, health_rows = [], [], [], []
    regression_rows = _read(RESULT_ROOT / "stage12_regression.csv")
    if mode == "smoke":
        regression_rows = [r for r in regression_rows if r.get("mode") == "preflight"]
    intervention_files = {
        "intervention_history_off.csv": [], "intervention_operation_profile_off.csv": [],
        "intervention_operation_alignment_swap.csv": [], "intervention_relation_env_off.csv": [],
        "intervention_node_conditioning_off.csv": [], "intervention_global_query.csv": [],
        "intervention_token_drop.csv": [],
    }
    p0_output = []
    for dataset, seed, variant in selected:
        path = _checkpoint(mode, dataset, variant, seed)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("selection") != "best_val_accuracy" or any(k.startswith("test_") for k in checkpoint.get("metrics", {})):
            raise RuntimeError(f"Checkpoint is not validation-only: {path}")
        cfg, data = _load_data(dataset, seed)
        info = checkpoint["data_info"]
        model = _model_for_s3(dataset, seed, variant, info, device)
        model.load_state_dict(checkpoint["model_state"])
        head = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
        head.load_state_dict(checkpoint["head_state"])
        model.eval(); head.eval()
        x, edges = data.x.to(device), data.edge_index.to(device)
        val_idx = data.val_idx.to(device)
        full, val_logits, val_metrics = _evaluate(model, head, data, x, edges, val_idx)
        if variant != "terminal":
            alpha = full["readout_attention_heads"]
            if not torch.isfinite(alpha).all() or not torch.allclose(alpha.sum(-1), torch.ones_like(alpha.sum(-1)), atol=1e-6, rtol=1e-6):
                raise RuntimeError(f"Invalid Stage-3 readout attention: {dataset}/{variant}/seed{seed}")
            if not torch.isfinite(full["history_branch"]).all() or not torch.isfinite(full["history_tokens"]).all():
                raise RuntimeError(f"Non-finite Stage-3 history state: {dataset}/{variant}/seed{seed}")
        if variant == "terminal":
            dst = full["canonical_edge_index"][1]
            for modality in ("text", "visual"):
                modulation = full["modulation"][f"{modality}_step0"]
                modulation1 = full["modulation"][f"{modality}_step1"]
                full[f"operation_profile_{modality}_step0"] = model._profile(modulation, dst, x.size(0))
                full[f"operation_profile_{modality}_step1"] = model._profile(modulation1, dst, x.size(0))
                full[f"relation_environment_{modality}"] = model._profile(full[f"base_relation_{modality}"], dst, x.size(0))
        if abs(val_metrics["val_acc"] - float(checkpoint["metrics"]["val_acc"])) > 1e-6:
            raise RuntimeError(f"Validation accuracy did not reproduce: {path}")
        if abs(val_metrics["val_macro_f1"] - float(checkpoint["metrics"]["val_macro_f1"])) > 1e-6:
            raise RuntimeError(f"Validation Macro-F1 did not reproduce: {path}")
        v3 = _v3_model(dataset, seed, info, device)
        shared = {key: value for key, value in checkpoint["model_state"].items() if key in v3.state_dict()}
        missing, unexpected = v3.load_state_dict(shared, strict=False)
        if unexpected or set(missing) != {key for key in v3.state_dict() if key not in shared}:
            raise RuntimeError(f"Could not map S3 shared state into v3 for regression: {path}")
        v3.eval()
        old_values = v3.analyze(x, edges)
        checks = _compare_stage12(old_values, full, dataset, seed, variant, "trained")
        regression_rows.extend(checks)
        if any(not row["passed"] for row in checks):
            raise RuntimeError(f"Stage-I/II regression failed for trained {dataset}/{variant}/seed{seed}")
        health_rows.extend(_stage12_health(full, dataset, seed, variant))
        branch_rows.extend(_history_branch_rows(full, model, dataset, seed, variant))
        attention_rows.extend(_attention_rows(full, dataset, seed, variant))
        preference_rows.extend(_preference_rows(full, dataset, seed, variant))
        op_rows, env_rows = _profile_rows(full, dataset, seed, variant)
        operation_rows.extend(op_rows); environment_rows.extend(env_rows)
        association_rows.extend(_association_rows(full, dataset, seed, variant))

        if mode == "full" and variant == "relation_operation_history":
            p0_path = P0_ROOT / "p02" / dataset / f"seed{seed}" / "conditional_feature" / "edge_diagnostics.pt"
            if not p0_path.exists():
                raise FileNotFoundError(f"Required post-hoc P0 artifact is missing: {p0_path}")
            p0 = torch.load(p0_path, map_location="cpu", weights_only=False)
            p0_output.extend(_p0_rows(dataset, seed, p0, full))
            # Keep only node-level baseline readout tensors across interventions;
            # relation-edge diagnostics are no longer needed after P0/regression.
            baseline_readout = {key: full[key] for key in (
                "terminal_z", "history_branch", "history_tokens", "history_representation",
                "readout_query", "readout_attention",
            )}
            full = baseline_readout
            del old_values
            if device.type == "cuda":
                torch.cuda.empty_cache()
            interventions = [
                ("intervention_history_off.csv", "history_branch_off", {"history_off": True}),
                ("intervention_operation_profile_off.csv", "operation_profile_off", {"operation_profile_off": True}),
                ("intervention_operation_alignment_swap.csv", "operation_history_alignment_swap", {"operation_alignment_swap": True}),
                ("intervention_relation_env_off.csv", "relation_environment_off", {"relation_env_off": True}),
                ("intervention_node_conditioning_off.csv", "node_intrinsic_query_off", {"node_conditioning_off": True}),
                ("intervention_global_query.csv", "global_query_only", {"global_query": True}),
            ]
            for filename, label, kwargs in interventions:
                intervention_files[filename].append(_intervention_rows(
                    model, head, data, x, edges, val_idx, baseline_readout, val_logits, val_metrics,
                    dataset, seed, label, kwargs,
                ))
            for token_id, label in enumerate(TOKEN_LABELS):
                intervention_files["intervention_token_drop.csv"].append(_intervention_rows(
                    model, head, data, x, edges, val_idx, baseline_readout, val_logits, val_metrics,
                    dataset, seed, f"drop_{label}", {"drop_token": token_id},
                ))
        print(f"ANALYZED {mode} {dataset}/{variant}/seed{seed} val={val_metrics['val_acc']:.4f}", flush=True)
        del model, v3, head, data, full, x, edges
        if device.type == "cuda":
            torch.cuda.empty_cache()

    prefix = "smoke_" if mode == "smoke" else ""
    _write(RESULT_ROOT / f"{prefix}history_branch_diagnostics.csv", branch_rows)
    _write(RESULT_ROOT / f"{prefix}history_attention.csv", attention_rows)
    _write(RESULT_ROOT / f"{prefix}history_preference_heterogeneity.csv", preference_rows)
    _write(RESULT_ROOT / f"{prefix}operation_profile_diagnostics.csv", operation_rows)
    _write(RESULT_ROOT / f"{prefix}relation_environment_diagnostics.csv", environment_rows)
    _write(RESULT_ROOT / f"{prefix}history_operation_association.csv", association_rows)
    _write(RESULT_ROOT / f"{prefix}stage12_health.csv", health_rows)
    _write(RESULT_ROOT / f"{prefix}stage12_regression.csv", regression_rows)
    if mode == "full":
        for name, rows in intervention_files.items():
            flat = [row for group in rows for row in (group if isinstance(group, list) else [group])]
            _write(RESULT_ROOT / name, flat)
        _write(RESULT_ROOT / "p0_history_readout.csv", p0_output)
    print(f"WROTE {mode}: {len(branch_rows)} branch, {len(attention_rows)} attention, "
          f"{len(health_rows)} Stage-I/II health rows", flush=True)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze M0-S3 ROHC pilot without test evaluation.")
    parser.add_argument("--mode", choices=("regression", "smoke", "full"), default="full")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if args.mode == "regression":
        _preflight(device)
    else:
        _run_analysis(args.mode, device)


if __name__ == "__main__":
    main()
