from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_mag_data  # noqa: E402
from src.models.interaction_core_v3 import Model as V3Model  # noqa: E402
from src.models.interaction_prior_retaining import Model, READOUT_VARIANTS  # noqa: E402

OUTPUT_ROOT = ROOT / "outputs" / "m0" / "prior_retention"
V3_ROOT = ROOT / "outputs" / "m0" / "conditioner_v3"
RESULT_ROOT = ROOT / "results" / "m0" / "prior_retention"
DATASETS = ("Movies", "Grocery", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = READOUT_VARIANTS
REGRESSION_ATOL = 1e-5
REGRESSION_RTOL = 1e-5
REGRESSION_KEYS = (
    "H0_text", "H0_visual", "H1_text", "H1_visual", "H2_text", "H2_visual",
    "relation_text", "relation_visual", "relation_memory",
    "base_relation_text", "base_relation_visual",
    "modulation_text_step0", "modulation_text_step1",
    "modulation_visual_step0", "modulation_visual_step1",
    "execution_code_text_step0", "execution_code_text_step1",
    "execution_code_visual_step0", "execution_code_visual_step1",
    "delta_execution_code_text_step0", "delta_execution_code_text_step1",
    "delta_execution_code_visual_step0", "delta_execution_code_visual_step1",
    "delta_message_text_step0", "delta_message_text_step1",
    "delta_message_visual_step0", "delta_message_visual_step1",
)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    if not fields:
        fields = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _compose(dataset: str, seed: int, variant: str, model_name="interaction_prior_retaining"):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        overrides = [
            f"dataset={dataset}", "task=nc", f"model={model_name}", f"seed={seed}",
            "num_runs=1", "task.evaluate_test=false",
        ]
        if model_name == "interaction_prior_retaining":
            overrides.append(f"model.readout_variant={variant}")
        else:
            overrides.append("model.variant=context_bilinear_absolute")
        return compose(config_name="config", overrides=overrides)


def _checkpoint(mode: str, dataset: str, variant: str, seed: int) -> Path:
    if mode == "v3":
        return V3_ROOT / "checkpoints" / dataset / "context_bilinear_absolute" / f"seed{seed}.pt"
    root = OUTPUT_ROOT / ("smoke" if mode == "smoke" else "full")
    return root / "checkpoints" / dataset / variant / f"seed{seed}.pt"


def _info(data):
    return {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.size(1)),
        "visual_dim": int(data.x_i.size(1)),
    }


def _stats(tensor) -> dict:
    value = torch.as_tensor(tensor).detach().float().reshape(-1)
    value = value[torch.isfinite(value)]
    if value.numel() == 0:
        return {"mean": float("nan"), "median": float("nan"), "p10": float("nan"),
                "p90": float("nan"), "p95": float("nan"), "n": 0}
    q = torch.quantile(value, torch.tensor([0.1, 0.5, 0.9, 0.95], device=value.device))
    return {"mean": float(value.mean().item()), "median": float(q[1].item()),
            "p10": float(q[0].item()), "p90": float(q[2].item()),
            "p95": float(q[3].item()), "n": int(value.numel())}


def _metric(logits, labels, num_classes: int) -> dict:
    pred = logits.argmax(-1)
    labels_cpu = labels.detach().cpu().reshape(-1)
    pred_cpu = pred.detach().cpu().reshape(-1)
    return {
        "val_acc": float((pred_cpu == labels_cpu).float().mean().item()),
        "val_macro_f1": float(f1_score(
            labels_cpu.numpy(), pred_cpu.numpy(), labels=list(range(int(num_classes))),
            average="macro", zero_division=0,
        )),
        "pred": pred_cpu,
    }


def _regression_rows(expected: dict, observed: dict, dataset: str, seed: int,
                     variant: str, phase: str) -> list[dict]:
    rows = []
    for key in REGRESSION_KEYS:
        left = expected[key].detach().float().cpu()
        right = observed[key].detach().float().cpu()
        if left.shape != right.shape:
            rows.append({
                "phase": phase, "dataset": dataset, "seed": seed, "variant": variant,
                "tensor": key, "shape_match": False, "max_abs_error": float("inf"),
                "passed": False, "atol": REGRESSION_ATOL, "rtol": REGRESSION_RTOL,
            })
            continue
        diff = (left - right).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        rows.append({
            "phase": phase, "dataset": dataset, "seed": seed, "variant": variant,
            "tensor": key, "shape_match": True, "max_abs_error": max_abs,
            "passed": bool(torch.allclose(left, right, atol=REGRESSION_ATOL, rtol=REGRESSION_RTOL)),
            "atol": REGRESSION_ATOL, "rtol": REGRESSION_RTOL,
        })
    return rows


def _terminal_snapshot(model: Model, full: dict) -> dict:
    with torch.no_grad():
        z_term = model.fusion(torch.cat((full["H2_text"], full["H2_visual"]), dim=-1))
    snapshot = {key: full[key].detach().cpu() for key in REGRESSION_KEYS}
    snapshot["terminal_fused_z"] = z_term.detach().cpu()
    return snapshot


def _compare_to_v3(checkpoint, data, x, edges, snapshot, dataset, seed, variant, phase, device):
    v3_cfg = _compose(dataset, seed, variant, model_name="interaction_core_v3")
    v3_model = V3Model(v3_cfg, _info(data)).to(device)
    source_state = checkpoint["model_state"]
    target_state = v3_model.state_dict()
    shared = {k: v for k, v in source_state.items()
              if k in target_state and target_state[k].shape == v.shape}
    if len(shared) < 80:
        raise RuntimeError(f"Only {len(shared)} shared v3 weights mapped for {dataset}/{seed}/{variant}")
    missing, unexpected = v3_model.load_state_dict(shared, strict=False)
    if any(not key.startswith(("trace_epoch", "gradient_rms_trace", "gradient_numel_trace",
                               "parameter_norm_trace", "operator_up_norm_trace",
                               "conditioner_output_norm_trace")) for key in missing):
        raise RuntimeError(f"Unexpected missing shared v3 weights: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected v3 keys during shared mapping: {unexpected}")
    v3_model.eval()
    with torch.no_grad():
        expected = v3_model.analyze(x, edges)
        adapted = {key: expected[key] for key in REGRESSION_KEYS if key in expected}
        adapted["H2_text"] = expected["final_text"]
        adapted["H2_visual"] = expected["final_visual"]
        adapted["terminal_fused_z"] = expected["fused_z"]
    rows = _regression_rows(adapted, snapshot, dataset, seed, variant, phase)
    del v3_model, expected
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def _stage_health(full: dict, dataset: str, seed: int, variant: str) -> list[dict]:
    rows = []
    for modality in ("text", "visual"):
        relation = full[f"relation_{modality}"].float()
        rows.append({"dataset": dataset, "seed": seed, "variant": variant, "stage": "stage1",
                     "measure": f"relation_feature_variance_{modality}",
                     "value": float(relation.var(0, unbiased=False).mean().item())})
        reverse = []
        edges = full["canonical_edge_index"].detach().cpu()
        index = {(int(s), int(d)): i for i, (s, d) in enumerate(edges.t().tolist())}
        for i, (src, dst) in enumerate(edges.t().tolist()):
            j = index.get((dst, src))
            if j is not None and src < dst:
                reverse.append((i, j))
        if reverse:
            left = torch.tensor([pair[0] for pair in reverse], device=relation.device)
            right = torch.tensor([pair[1] for pair in reverse], device=relation.device)
            direction = 1.0 - F.cosine_similarity(relation[left], relation[right], dim=-1, eps=1e-8)
        else:
            direction = relation.new_empty((0,))
        rows.append({"dataset": dataset, "seed": seed, "variant": variant, "stage": "stage1",
                     "measure": f"directed_reverse_1_minus_cosine_{modality}",
                     "value": _stats(direction)["mean"]})
        for step in (0, 1):
            for name in ("operator_deviation_ratio",):
                measure = full[f"{name}_{modality}_step{step}"]
                rows.append({"dataset": dataset, "seed": seed, "variant": variant, "stage": "stage2",
                             "measure": f"{name}_{modality}_step{step}",
                             "value": _stats(measure)["mean"]})
            for name in ("modulation", "execution_code"):
                tensor = full[f"{name}_{modality}_step{step}"].float()
                rows.append({"dataset": dataset, "seed": seed, "variant": variant, "stage": "stage2",
                             "measure": f"{name}_feature_variance_{modality}_step{step}",
                             "value": float(tensor.var(0, unbiased=False).mean().item()) if tensor.numel() else 0.0})
    modality_cos = 1.0 - F.cosine_similarity(
        full["relation_text"].float(), full["relation_visual"].float(), dim=-1, eps=1e-8
    )
    rows.append({"dataset": dataset, "seed": seed, "variant": variant, "stage": "stage1",
                 "measure": "modality_relation_1_minus_cosine_mean", "value": _stats(modality_cos)["mean"]})
    return rows


def _update_rows(full: dict, dataset: str, seed: int) -> list[dict]:
    rows = []
    for modality in ("text", "visual"):
        u0 = full[f"U_{modality}_step0"].float()
        u1 = full[f"U_{modality}_step1"].float()
        h0 = full[f"H0_{modality}"].float()
        for step, update in ((0, u0), (1, u1)):
            rows.append({
                "dataset": dataset, "seed": seed, "modality": modality, "step": step,
                "update_norm_mean": _stats(update.norm(dim=-1))["mean"],
                "update_feature_variance": float(update.var(0, unbiased=False).mean().item()),
                "update_node_variance": float(update.norm(dim=-1).var(unbiased=False).item()),
            })
        cosine = F.cosine_similarity(u0, u1, dim=-1, eps=1e-8)
        ratio = (u0 + u1).norm(dim=-1) / (h0.norm(dim=-1) + 1e-12)
        rows.append({
            "dataset": dataset, "seed": seed, "modality": modality, "step": "pair",
            "cos_U0_U1_mean": _stats(cosine)["mean"],
            "context_update_over_prior_mean": _stats(ratio)["mean"],
            "update_sum_norm_mean": _stats((u0 + u1).norm(dim=-1))["mean"],
        })
    return rows


def _prior_rows(full: dict, dataset: str, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    injection, retention, gates = [], [], []
    eps = 1e-12
    for modality in ("text", "visual"):
        h0 = full[f"H0_{modality}"].float()
        h2 = full[f"H2_{modality}"].float()
        final = full[f"final_{modality}"].float()
        context = full[f"C_update_{modality}"].float()
        enrich = full[f"E_{modality}"].float()
        gate = full[f"gate_{modality}"].float()
        residual = full[f"R_{modality}"].float()
        rho = residual.norm(dim=-1) / (h0.norm(dim=-1) + eps)
        align = F.cosine_similarity(h0, residual, dim=-1, eps=1e-8)
        final_drift = 1.0 - F.cosine_similarity(h0, final, dim=-1, eps=1e-8)
        terminal_drift = 1.0 - F.cosine_similarity(h0, h2, dim=-1, eps=1e-8)
        for label, values in (("rho_context", rho), ("prior_norm", h0.norm(dim=-1)),
                              ("context_norm", context.norm(dim=-1)), ("enrichment_norm", enrich.norm(dim=-1)),
                              ("injected_residual_norm", residual.norm(dim=-1)), ("cos_prior_context", align),
                              ("final_prior_drift", final_drift)):
            injection.append({"dataset": dataset, "seed": seed, "modality": modality,
                              "measure": label, **_stats(values)})
        gain = terminal_drift - final_drift
        retention.append({
            "dataset": dataset, "seed": seed, "modality": modality,
            "terminal_drift_mean": _stats(terminal_drift)["mean"],
            "terminal_drift_median": _stats(terminal_drift)["median"],
            "terminal_drift_p90": _stats(terminal_drift)["p90"],
            "prior_drift_mean": _stats(final_drift)["mean"],
            "prior_drift_median": _stats(final_drift)["median"],
            "prior_drift_p90": _stats(final_drift)["p90"],
            "retention_gain_mean": _stats(gain)["mean"],
            "retention_gain_median": _stats(gain)["median"],
            "retention_gain_p90": _stats(gain)["p90"],
            "fraction_prior_closer_than_terminal": float((final_drift < terminal_drift).float().mean().item()),
        })
        feature_var = gate.var(dim=-1, unbiased=False)
        node_var = gate.var(dim=0, unbiased=False)
        gates.append({
            "dataset": dataset, "seed": seed, "modality": modality,
            "mean": float(gate.mean().item()), "std": float(gate.std(unbiased=False).item()),
            "median": _stats(gate)["median"], "p10": _stats(gate)["p10"], "p90": _stats(gate)["p90"],
            "feature_variance_mean": float(feature_var.mean().item()),
            "node_variance_mean": float(node_var.mean().item()),
            "fraction_below_0_05": float((gate < 0.05).float().mean().item()),
            "fraction_above_0_95": float((gate > 0.95).float().mean().item()),
            "alignment_positive_gt_0_3": float((align > 0.3).float().mean().item()),
            "alignment_negative_lt_minus_0_3": float((align < -0.3).float().mean().item()),
            "alignment_near_orthogonal": float((align.abs() <= 0.3).float().mean().item()),
        })
    return injection, retention, gates


def _intervention_metrics(head, changed: dict, x_cpu, val_idx_cpu, labels_cpu, num_classes,
                          baseline_logits_cpu, baseline_pred_cpu, baseline_final_cpu, baseline_r_cpu,
                          baseline_gate_cpu, dataset, seed, name, note="same-checkpoint validation intervention"):
    with torch.no_grad():
        z = changed["fused_z"]
        logits = head(z[val_idx_cpu.to(z.device)]).detach().cpu()
        metrics = _metric(logits, labels_cpu, num_classes)
        base_metrics = _metric(baseline_logits_cpu, labels_cpu, num_classes)
        val_l2 = (logits - baseline_logits_cpu).norm(dim=-1)
        flip = (metrics["pred"] != baseline_pred_cpu).float().mean()
        rep = torch.cat((changed["final_text"], changed["final_visual"]), dim=-1).detach().cpu()
        rep_l2 = (rep - baseline_final_cpu).norm(dim=-1)
        residual = torch.cat((changed["R_text"], changed["R_visual"]), dim=-1).detach().cpu()
        residual_l2 = (residual - baseline_r_cpu).norm(dim=-1)
        gates = torch.cat((changed["gate_text"], changed["gate_visual"]), dim=-1).detach().cpu()
        gate_l1 = (gates - baseline_gate_cpu).abs().mean()
        gate_l2 = (gates - baseline_gate_cpu).norm(dim=-1).mean()
    return {
        "dataset": dataset, "seed": seed, "variant": "prior_update_gate", "intervention": name,
        "interpretation": note,
        "val_acc_full": base_metrics["val_acc"], "val_acc_intervened": metrics["val_acc"],
        "val_acc_change_pp": 100.0 * (metrics["val_acc"] - base_metrics["val_acc"]),
        "val_macro_f1_full": base_metrics["val_macro_f1"], "val_macro_f1_intervened": metrics["val_macro_f1"],
        "val_macro_f1_change_pp": 100.0 * (metrics["val_macro_f1"] - base_metrics["val_macro_f1"]),
        "logit_l2_mean": float(val_l2.mean().item()), "prediction_flip_rate": float(flip.item()),
        "representation_l2_mean": float(rep_l2.mean().item()),
        "context_residual_l2_mean": float(residual_l2.mean().item()),
        "gate_l1_mean": float(gate_l1.item()), "gate_l2_mean": float(gate_l2.item()),
    }


def _full_interventions(model, head, x, edges, val_idx, labels, num_classes, baseline, dataset, seed):
    rows = []
    baseline_logits = baseline["baseline_logits_cpu"]
    baseline_pred = baseline["baseline_pred_cpu"]
    final_cpu = baseline["final_cpu"]
    residual_cpu = baseline["residual_cpu"]
    gate_cpu = baseline["gate_cpu"]
    specifications = (
        ("context_off", {"context_off": True}, "primary graph-context-off intervention"),
        ("context_source_substitution_delta", {"context_source_override": "delta"}, "post-hoc update-to-delta source substitution; not causal proof"),
        ("gate_one_unconditional_injection", {"force_gate_one": True}, "same-checkpoint feature gate forced to one"),
        ("prior_bypass_off", {"prior_bypass_off": True}, "strong/OOD prior-bypass intervention; not a causal estimate"),
        ("operator_off", {"operator_enabled": False}, "Stage-II operator-off regression intervention"),
        ("within_target_relation_shuffle", {"relation_shuffle": True}, "within-target cyclic relation-memory shuffle"),
    )
    for name, kwargs, note in specifications:
        changed = model.analyze(x, edges, **kwargs)
        rows.append(_intervention_metrics(
            head, changed, x, val_idx, labels, num_classes, baseline_logits, baseline_pred,
            final_cpu, residual_cpu, gate_cpu, dataset, seed, name, note,
        ))
        del changed
        if x.is_cuda:
            torch.cuda.empty_cache()
    return rows


def _run_checkpoint(mode, dataset, seed, variant, device, regression_rows,
                    stage_rows, residual_rows, injection_rows, retention_rows, gate_rows,
                    intervention_rows):
    path = _checkpoint(mode, dataset, variant, seed)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("selection") != "best_val_accuracy":
        raise RuntimeError(f"Checkpoint selection is not best-val-accuracy: {path}")
    if checkpoint.get("protocol_version") != "unified_full_graph_nc_v1":
        raise RuntimeError(f"Wrong NC protocol: {path}")
    if any(key.startswith("test_") for key in checkpoint.get("metrics", {})):
        raise RuntimeError(f"Checkpoint includes test metrics: {path}")
    cfg = _compose(dataset, seed, variant)
    data = load_mag_data(cfg, "nc", seed)
    model = Model(cfg, _info(data)).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    head = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    head.load_state_dict(checkpoint["head_state"], strict=True)
    model.eval()
    head.eval()
    x = data.x.to(device)
    edges = data.edge_index.to(device)
    val_idx = data.val_idx.detach().cpu()
    labels = data.y.index_select(0, val_idx).detach().cpu()

    with torch.no_grad():
        full = model.analyze(x, edges)
        val_logits = head(full["fused_z"][val_idx.to(device)]).detach().cpu()
        measured = _metric(val_logits, labels, int(data.num_classes))
        if abs(measured["val_acc"] - float(checkpoint["metrics"]["val_acc"])) > 1e-6:
            raise RuntimeError(f"Validation accuracy does not reproduce for {path}")
        if abs(measured["val_macro_f1"] - float(checkpoint["metrics"]["val_macro_f1"])) > 1e-6:
            raise RuntimeError(f"Validation Macro-F1 does not reproduce for {path}")
        stage_rows.extend(_stage_health(full, dataset, seed, variant))
        if variant == "prior_update_gate":
            residual_rows.extend(_update_rows(full, dataset, seed))
            inj, ret, gate = _prior_rows(full, dataset, seed)
            injection_rows.extend(inj)
            retention_rows.extend(ret)
            gate_rows.extend(gate)
        snapshot = _terminal_snapshot(model, full)
        if variant == "prior_update_gate":
            val_pred = measured["pred"]
            baseline = {
                "baseline_logits_cpu": val_logits,
                "baseline_pred_cpu": val_pred,
                "final_cpu": torch.cat((full["final_text"], full["final_visual"]), dim=-1).detach().cpu(),
                "residual_cpu": torch.cat((full["R_text"], full["R_visual"]), dim=-1).detach().cpu(),
                "gate_cpu": torch.cat((full["gate_text"], full["gate_visual"]), dim=-1).detach().cpu(),
            }
        else:
            baseline = None
    regression_rows.extend(_compare_to_v3(
        checkpoint, data, x, edges, snapshot, dataset, seed, variant, "trained_" + variant, device
    ))
    del full, snapshot
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if baseline is not None:
        intervention_rows.extend(_full_interventions(
            model, head, x, edges, val_idx, labels, int(data.num_classes), baseline, dataset, seed
        ))
    del model, head, data, x, edges
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "dataset": dataset, "seed": seed, "variant": variant,
        "val_acc_reproduced": measured["val_acc"],
        "val_macro_f1_reproduced": measured["val_macro_f1"],
    }


def _preflight(device) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            checkpoint = torch.load(_checkpoint("v3", dataset, "context_bilinear_absolute", seed),
                                    map_location="cpu", weights_only=False)
            cfg = _compose(dataset, seed, "terminal")
            data = load_mag_data(cfg, "nc", seed)
            info = _info(data)
            prior_model = Model(cfg, info).to(device)
            v3_cfg = _compose(dataset, seed, "terminal", model_name="interaction_core_v3")
            v3_model = V3Model(v3_cfg, info).to(device)
            shared = {k: v for k, v in checkpoint["model_state"].items()
                      if k in prior_model.state_dict() and prior_model.state_dict()[k].shape == v.shape}
            if len(shared) < 80:
                raise RuntimeError(f"Frozen v3 checkpoint did not map to PRCI terminal: {dataset}/{seed}")
            prior_model.load_state_dict(shared, strict=False)
            v3_model.load_state_dict(checkpoint["model_state"], strict=True)
            prior_model.eval()
            v3_model.eval()
            x, edges = data.x.to(device), data.edge_index.to(device)
            with torch.no_grad():
                observed = prior_model.analyze(x, edges)
                expected = v3_model.analyze(x, edges)
                snap = _terminal_snapshot(prior_model, observed)
                adapted = {key: expected[key] for key in REGRESSION_KEYS if key in expected}
                adapted["H2_text"] = expected["final_text"]
                adapted["H2_visual"] = expected["final_visual"]
                adapted["terminal_fused_z"] = expected["fused_z"]
                current = _regression_rows(adapted, snap, dataset, seed, "terminal", "frozen_v3_preflight")
            rows.extend(current)
            if any(not row["passed"] for row in current):
                raise RuntimeError(f"Frozen v3 exact Stage-I/II gate failed for {dataset}/seed{seed}")
            print(f"REGRESSION PASS frozen v3 {dataset}/seed{seed}", flush=True)
            del prior_model, v3_model, observed, expected, data, x, edges
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Audit the prior-retaining readout on validation-only checkpoints.")
    parser.add_argument("--mode", choices=("preflight", "smoke", "full"), default="full")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    prefix = "smoke_" if args.mode == "smoke" else ""
    regression_rows = []
    if args.mode in ("preflight", "smoke", "full"):
        regression_rows.extend(_preflight(device))
    if args.mode == "preflight":
        _write(RESULT_ROOT / "stage12_preflight_regression.csv", regression_rows)
        print(f"PREFLIGHT PASS: {len(regression_rows)} tensor comparisons", flush=True)
        return

    selected = [("Movies", 42, v) for v in VARIANTS] if args.mode == "smoke" else [
        (d, s, v) for d in DATASETS for s in SEEDS for v in VARIANTS
    ]
    stage_rows, residual_rows, injection_rows, retention_rows, gate_rows = [], [], [], [], []
    intervention_rows, validation_rows = [], []
    for dataset, seed, variant in selected:
        validation_rows.append(_run_checkpoint(
            args.mode, dataset, seed, variant, device, regression_rows,
            stage_rows, residual_rows, injection_rows, retention_rows, gate_rows,
            intervention_rows,
        ))
        print(f"ANALYZED {args.mode} {dataset}/{variant}/seed{seed}", flush=True)
    _write(RESULT_ROOT / f"{prefix}stage12_regression.csv", regression_rows)
    _write(RESULT_ROOT / f"{prefix}stage12_health.csv", stage_rows)
    _write(RESULT_ROOT / f"{prefix}validation_reproduction.csv", validation_rows)
    _write(RESULT_ROOT / f"{prefix}update_residual_diagnostics.csv", residual_rows)
    _write(RESULT_ROOT / f"{prefix}prior_injection_diagnostics.csv", injection_rows)
    _write(RESULT_ROOT / f"{prefix}prior_retention_diagnostics.csv", retention_rows)
    _write(RESULT_ROOT / f"{prefix}gate_diagnostics.csv", gate_rows)
    if args.mode == "full":
        by_name = {
            "intervention_context_off.csv": "context_off",
            "intervention_context_source_swap.csv": "context_source_substitution_delta",
            "intervention_gate_one.csv": "gate_one_unconditional_injection",
            "intervention_prior_off.csv": "prior_bypass_off",
            "intervention_operator_off.csv": "operator_off",
            "intervention_relation_shuffle.csv": "within_target_relation_shuffle",
        }
        for filename, name in by_name.items():
            _write(RESULT_ROOT / filename, [r for r in intervention_rows if r["intervention"] == name])
    print(f"WROTE {args.mode}: {len(regression_rows)} regression rows, "
          f"{len(stage_rows)} Stage-I/II health rows, {len(intervention_rows)} intervention rows", flush=True)


if __name__ == "__main__":
    main()
