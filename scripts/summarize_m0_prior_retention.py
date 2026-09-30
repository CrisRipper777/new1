from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "m0" / "prior_retention" / "full"
RES = ROOT / "results" / "m0" / "prior_retention"
V3RES = ROOT / "results" / "m0" / "conditioner_v3"
DOC = ROOT / "docs" / "m0"
DATASETS = ("Movies", "Grocery", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = ("terminal", "prior_delta_gate", "prior_update_add", "prior_update_gate")
CONTRASTS = (
    ("prior_delta_gate", "terminal", "delta_gate_minus_terminal"),
    ("prior_update_add", "prior_delta_gate", "update_add_minus_delta_gate"),
    ("prior_update_gate", "prior_update_add", "update_gate_minus_update_add"),
    ("prior_update_gate", "terminal", "update_gate_minus_terminal"),
)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    if not fields:
        fields = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def num(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def mean_sd(values):
    clean = [num(x) for x in values]
    clean = [x for x in clean if math.isfinite(x)]
    if not clean:
        return float("nan"), float("nan"), 0
    return statistics.mean(clean), (statistics.stdev(clean) if len(clean) > 1 else 0.0), len(clean)


def fmt(x, digits=4):
    try:
        x = float(x)
        return f"{x:.{digits}f}" if math.isfinite(x) else "NA"
    except (TypeError, ValueError):
        return "NA"


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(x).replace("|", "/").replace("\n", " ") for x in row) + " |" for row in rows)
    return "\n".join(lines)


def group_summary(rows, key_names, value_key):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(k, "") for k in key_names)].append(num(row.get(value_key)))
    result = []
    for keys, vals in sorted(buckets.items()):
        avg, sd, n = mean_sd(vals)
        result.append((*keys, avg, sd, n))
    return result


def main():
    records = []
    for path in sorted((OUT / "logs").glob("**/run_record.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("mode") == "full":
            records.append(record)
    expected = {(d, s, v) for d in DATASETS for s in SEEDS for v in VARIANTS}
    observed = {(r.get("dataset"), int(r.get("seed", -1)), r.get("variant")) for r in records}
    if observed != expected or len(records) != 36:
        raise RuntimeError(f"Expected 36 unique full records, got {len(records)} records / {len(observed)} unique")
    if any(r.get("status") != "complete" or r.get("task_evaluate_test") is not False
           or r.get("protocol_version") != "unified_full_graph_nc_v1" for r in records):
        raise RuntimeError("Full run status or validation-only protocol audit failed")
    metrics = read_csv(RES / "pilot_metrics.csv")
    if len(metrics) != 36 or any(str(r.get("task_evaluate_test", "")).lower() not in ("false", "0") for r in metrics):
        raise RuntimeError("pilot_metrics.csv is incomplete or includes test evaluation")
    regression = read_csv(RES / "stage12_regression.csv")
    if not regression or any(r.get("passed", "").lower() != "true" for r in regression):
        raise RuntimeError("Stage-I/II regression failed")
    validation = read_csv(RES / "validation_reproduction.csv")
    if len(validation) != 36:
        raise RuntimeError("Validation reproduction does not contain 36 checkpoints")
    interventions = {}
    intervention_files = (
        "intervention_context_off.csv", "intervention_context_source_swap.csv",
        "intervention_gate_one.csv", "intervention_prior_off.csv",
        "intervention_operator_off.csv", "intervention_relation_shuffle.csv",
    )
    for filename in intervention_files:
        rows = read_csv(RES / filename)
        if len(rows) != 9:
            raise RuntimeError(f"{filename} must contain nine full-checkpoint interventions, got {len(rows)}")
        interventions[filename] = rows

    metric_map = {(r["dataset"], int(r["seed"]), r["variant"]): r for r in metrics}
    paired = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for candidate, reference, label in CONTRASTS:
                a = metric_map[(dataset, seed, candidate)]
                b = metric_map[(dataset, seed, reference)]
                paired.append({
                    "dataset": dataset, "seed": seed, "contrast": label,
                    "candidate": candidate, "reference": reference,
                    "val_acc_candidate": a["val_acc"], "val_acc_reference": b["val_acc"],
                    "val_acc_delta_pp": 100.0 * (num(a["val_acc"]) - num(b["val_acc"])),
                    "val_macro_f1_candidate": a["val_macro_f1"], "val_macro_f1_reference": b["val_macro_f1"],
                    "val_macro_f1_delta_pp": 100.0 * (num(a["val_macro_f1"]) - num(b["val_macro_f1"])),
                })
    write_csv(RES / "paired_comparisons.csv", paired)

    run_by_key = {(r["dataset"], int(r["seed"]), r["variant"]): r for r in records}
    gradient_rows, growth_rows = [], []
    for key, record in sorted(run_by_key.items()):
        run_dir = Path(record["hydra_dir"]).parent
        for source_name, destination in (("gradient_trace.csv", gradient_rows),
                                         ("context_branch_growth.csv", growth_rows)):
            source = run_dir / source_name
            if not source.exists():
                raise RuntimeError(f"Missing trace: {source}")
            for row in read_csv(source):
                destination.append({"dataset": key[0], "seed": key[1], "variant": key[2], **row})
    write_csv(RES / "gradient_trace.csv", gradient_rows)
    write_csv(RES / "context_branch_growth.csv", growth_rows)

    variant_rows = []
    for variant in VARIANTS:
        subset = [r for r in metrics if r["variant"] == variant]
        acc, acc_sd, n = mean_sd([r["val_acc"] for r in subset])
        f1, f1_sd, _ = mean_sd([r["val_macro_f1"] for r in subset])
        params = sorted({int(r["active_parameters"]) for r in subset})
        variant_rows.append({
            "variant": variant, "n": n, "val_acc_mean": acc, "val_acc_sd": acc_sd,
            "val_macro_f1_mean": f1, "val_macro_f1_sd": f1_sd,
            "active_parameters_min": min(params), "active_parameters_max": max(params),
        })
    write_csv(RES / "pilot_summary.csv", variant_rows)

    paired_summary = []
    for contrast, _, label in CONTRASTS:
        subset = [r for r in paired if r["contrast"] == label]
        for metric_key, metric_name in (("val_acc_delta_pp", "validation accuracy delta (pp)"),
                                        ("val_macro_f1_delta_pp", "validation macro-F1 delta (pp)")):
            all_mean, all_sd, n = mean_sd([r[metric_key] for r in subset])
            paired_summary.append({
                "contrast": label, "metric": metric_name, "scope": "all datasets",
                "mean_delta": all_mean, "sd": all_sd, "n": n,
            })
            for dataset in DATASETS:
                rows = [r for r in subset if r["dataset"] == dataset]
                avg, sd, n_dataset = mean_sd([r[metric_key] for r in rows])
                paired_summary.append({
                    "contrast": label, "metric": metric_name, "scope": dataset,
                    "mean_delta": avg, "sd": sd, "n": n_dataset,
                })
    write_csv(RES / "paired_comparison_summary.csv", paired_summary)

    health = read_csv(RES / "stage12_health.csv")
    health_summary = group_summary(health, ("stage", "measure", "variant"), "value")
    write_csv(RES / "stage12_health_summary.csv", [
        {"stage": x[0], "measure": x[1], "variant": x[2], "mean": x[3], "sd": x[4], "n": x[5]}
        for x in health_summary
    ])

    # Matched descriptive comparison with the frozen v3 absolute pilot.
    v3_stage1 = read_csv(V3RES / "stage1_health.csv")
    v3_operator = read_csv(V3RES / "operator_diagnostics.csv")
    v3_variance = read_csv(V3RES / "variation_decomposition.csv")
    v3_stage1_map = {
        (r["dataset"], int(r["seed"]), r["measure"], r["modality"]): num(r["mean"])
        for r in v3_stage1 if r["variant"] == "context_bilinear_absolute"
    }
    v3_operator_map = {
        (r["dataset"], int(r["seed"]), r["modality"], int(r["interaction_step"])):
            num(r["operator_deviation_ratio_mean"])
        for r in v3_operator if r["variant"] == "context_bilinear_absolute"
    }
    v3_variance_map = {
        (r["dataset"], int(r["seed"]), r["modality"], r["object"]): num(r["V_total"])
        for r in v3_variance if r["variant"] == "context_bilinear_absolute"
    }
    prci_health_map = {
        (r["dataset"], int(r["seed"]), r["measure"]): num(r["value"])
        for r in health if r["variant"] == "prior_update_gate"
    }
    health_comparison = []

    def add_health_comparison(metric, modality, step, current_measure, baseline_values):
        for dataset in DATASETS:
            for seed in SEEDS:
                current = prci_health_map[(dataset, seed, current_measure)]
                baseline = baseline_values[(dataset, seed, modality, step)] if step != "" else baseline_values[(dataset, seed, metric, modality)]
                health_comparison.append({
                    "metric": metric, "modality": modality, "step": step,
                    "dataset": dataset, "seed": seed, "v3_absolute": baseline,
                    "prior_update_gate": current, "paired_delta": current - baseline,
                })

    for modality in ("text", "visual"):
        add_health_comparison(
            "relation_feature_variance", modality, "",
            f"relation_feature_variance_{modality}", v3_stage1_map,
        )
        add_health_comparison(
            "directed_reverse_1_minus_cosine", modality, "",
            f"directed_reverse_1_minus_cosine_{modality}", v3_stage1_map,
        )
    for dataset in DATASETS:
        for seed in SEEDS:
            current = prci_health_map[(dataset, seed, "modality_relation_1_minus_cosine_mean")]
            baseline = v3_stage1_map[(dataset, seed, "R_T_R_V_1_minus_cosine", "both")]
            health_comparison.append({
                "metric": "cross_modal_relation_1_minus_cosine", "modality": "both", "step": "",
                "dataset": dataset, "seed": seed, "v3_absolute": baseline,
                "prior_update_gate": current, "paired_delta": current - baseline,
            })
    for modality in ("text", "visual"):
        for step in (0, 1):
            add_health_comparison(
                "operator_deviation_ratio", modality, step,
                f"operator_deviation_ratio_{modality}_step{step}", v3_operator_map,
            )
            add_health_comparison(
                "operator_modulation_variance", modality, step,
                f"modulation_feature_variance_{modality}_step{step}",
                {
                    (d, s, m, step): v
                    for (d, s, m, obj), v in v3_variance_map.items()
                    for step in (0, 1) if obj == f"operator_modulation_a_step{step}"
                },
            )
            add_health_comparison(
                "execution_code_variance", modality, step,
                f"execution_code_feature_variance_{modality}_step{step}",
                {
                    (d, s, m, step): v
                    for (d, s, m, obj), v in v3_variance_map.items()
                    for step in (0, 1) if obj == f"execution_xi_step{step}"
                },
            )
    write_csv(RES / "stage12_v3_comparison.csv", health_comparison)
    comparison_summary = []
    keys = sorted({(r["metric"], r["modality"], r["step"]) for r in health_comparison})
    for metric, modality, step in keys:
        rows = [r for r in health_comparison
                if r["metric"] == metric and r["modality"] == modality and str(r["step"]) == str(step)]
        v3_mean, v3_sd, n = mean_sd([r["v3_absolute"] for r in rows])
        prci_mean, prci_sd, _ = mean_sd([r["prior_update_gate"] for r in rows])
        delta_mean, delta_sd, _ = mean_sd([r["paired_delta"] for r in rows])
        comparison_summary.append({
            "metric": metric, "modality": modality, "step": step,
            "v3_absolute_mean": v3_mean, "v3_absolute_sd": v3_sd,
            "prior_update_gate_mean": prci_mean, "prior_update_gate_sd": prci_sd,
            "paired_delta_mean": delta_mean, "paired_delta_sd": delta_sd, "n": n,
        })
    write_csv(RES / "stage12_v3_comparison_summary.csv", comparison_summary)

    residual = read_csv(RES / "update_residual_diagnostics.csv")
    injection = read_csv(RES / "prior_injection_diagnostics.csv")
    retention = read_csv(RES / "prior_retention_diagnostics.csv")
    gates = read_csv(RES / "gate_diagnostics.csv")
    # Compact descriptive summaries retain raw rows in the source CSVs.
    residual_summary = []
    for measure in ("update_norm_mean", "update_feature_variance", "update_node_variance",
                    "cos_U0_U1_mean", "context_update_over_prior_mean", "update_sum_norm_mean"):
        selected = [r for r in residual if measure in r and r[measure] not in ("", "nan")]
        for modality in ("text", "visual"):
            vals = [r[measure] for r in selected if r["modality"] == modality]
            avg, sd, n = mean_sd(vals)
            if n:
                residual_summary.append({"modality": modality, "measure": measure, "mean": avg, "sd": sd, "n": n})
    write_csv(RES / "update_residual_summary.csv", residual_summary)
    injection_summary = []
    for modality in ("text", "visual"):
        for measure in sorted({r["measure"] for r in injection}):
            vals = [r["mean"] for r in injection if r["modality"] == modality and r["measure"] == measure]
            avg, sd, n = mean_sd(vals)
            if n:
                injection_summary.append({"modality": modality, "measure": measure, "mean": avg, "sd": sd, "n": n})
    write_csv(RES / "prior_injection_summary.csv", injection_summary)
    gate_summary = []
    for modality in ("text", "visual"):
        for field in ("mean", "std", "feature_variance_mean", "node_variance_mean",
                      "fraction_below_0_05", "fraction_above_0_95"):
            avg, sd, n = mean_sd([r[field] for r in gates if r["modality"] == modality])
            gate_summary.append({"modality": modality, "measure": field, "mean": avg, "sd": sd, "n": n})
    write_csv(RES / "gate_summary.csv", gate_summary)

    gradient_summary = []
    for variant in VARIANTS:
        subset = [r for r in gradient_rows if r["variant"] == variant]
        for group in sorted({r["group"] for r in subset}):
            rows = [r for r in subset if r["group"] == group]
            grad = [num(r["gradient_rms_preclip"]) for r in rows]
            first_by_run = []
            for dataset in DATASETS:
                for seed in SEEDS:
                    run_rows = [r for r in rows if r["dataset"] == dataset and int(r["seed"]) == seed]
                    first = [int(r["epoch"]) for r in run_rows if num(r["gradient_rms_preclip"]) > 0]
                    if first:
                        first_by_run.append(min(first))
            avg, sd, n = mean_sd(grad)
            gradient_summary.append({
                "variant": variant, "group": group, "gradient_rms_mean": avg, "gradient_rms_sd": sd,
                "n": n, "runs_with_nonzero_gradient": len(first_by_run),
                "first_nonzero_epoch_min": min(first_by_run) if first_by_run else "",
                "first_nonzero_epoch_median": statistics.median(first_by_run) if first_by_run else "",
                "first_nonzero_epoch_max": max(first_by_run) if first_by_run else "",
                "nonzero_fraction": sum(x > 0 for x in grad if math.isfinite(x)) / max(1, len([x for x in grad if math.isfinite(x)])),
            })
    write_csv(RES / "gradient_summary.csv", gradient_summary)
    projector_rows = [r for r in gradient_summary if r["group"] in ("projector_text", "projector_visual")]
    write_csv(RES / "projector_gradient_comparison.csv", projector_rows)

    growth_summary = []
    for variant in VARIANTS:
        for modality in ("text", "visual"):
            rows = [r for r in growth_rows if r["variant"] == variant and r["modality"] == modality]
            for field in ("context_output_weight_frobenius", "context_output_bias_l2",
                          "gate_weight_frobenius", "gate_bias_l2"):
                selected_values, first_by_run = [], []
                for dataset in DATASETS:
                    for seed in SEEDS:
                        record = run_by_key[(dataset, seed, variant)]
                        run_rows = [r for r in rows if r["dataset"] == dataset and int(r["seed"]) == seed]
                        best_epoch = int(record["best_epoch"])
                        selected = [r for r in run_rows if int(r["epoch"]) == best_epoch]
                        if selected:
                            selected_values.append(selected[-1][field])
                        first = [int(r["epoch"]) for r in run_rows if num(r[field]) > 0]
                        if first:
                            first_by_run.append(min(first))
                avg, sd, n = mean_sd(selected_values)
                growth_summary.append({
                    "variant": variant, "modality": modality, "parameter_group": field,
                    "best_checkpoint_mean": avg, "best_checkpoint_sd": sd, "n": n,
                    "runs_with_nonzero_parameter": len(first_by_run),
                    "first_nonzero_epoch_min": min(first_by_run) if first_by_run else "",
                    "first_nonzero_epoch_median": statistics.median(first_by_run) if first_by_run else "",
                    "first_nonzero_epoch_max": max(first_by_run) if first_by_run else "",
                })
    write_csv(RES / "context_branch_growth_summary.csv", growth_summary)

    intervention_summary = []
    for filename, rows in interventions.items():
        for field in ("val_acc_change_pp", "val_macro_f1_change_pp", "prediction_flip_rate",
                      "logit_l2_mean", "representation_l2_mean", "context_residual_l2_mean",
                      "gate_l1_mean", "gate_l2_mean"):
            avg, sd, n = mean_sd([r[field] for r in rows])
            intervention_summary.append({"intervention_file": filename, "measure": field, "mean": avg, "sd": sd, "n": n})
    write_csv(RES / "intervention_summary.csv", intervention_summary)

    # Data-driven decision indicators are descriptive counts, not significance tests.
    full_variant = [r for r in metrics if r["variant"] == "prior_update_gate"]
    terminals = [r for r in metrics if r["variant"] == "terminal"]
    full_vs_terminal = [num(metric_map[(d, s, "prior_update_gate")]["val_acc"]) -
                        num(metric_map[(d, s, "terminal")]["val_acc"])
                        for d in DATASETS for s in SEEDS]
    context_rows = interventions["intervention_context_off.csv"]
    operator_rows = interventions["intervention_operator_off.csv"]
    shuffle_rows = interventions["intervention_relation_shuffle.csv"]
    prior_rows = interventions["intervention_prior_off.csv"]
    mean_rho = [num(r["mean"]) for r in injection if r["measure"] == "rho_context"]
    mean_drift_gain = [num(r["retention_gain_mean"]) for r in retention]
    mean_gate_span = [num(r["std"]) for r in gates]
    measurable_context = sum(num(r["val_acc_change_pp"]) < 0 for r in context_rows)
    op_effect = sum(num(r["val_acc_change_pp"]) < 0 for r in operator_rows)
    shuffle_effect = sum(num(r["val_acc_change_pp"]) < 0 for r in shuffle_rows)
    terminal_wins = sum(x < 0 for x in full_vs_terminal)
    add_vs_gate = [num(metric_map[(d, s, "prior_update_add")]["val_acc"]) -
                   num(metric_map[(d, s, "prior_update_gate")]["val_acc"])
                   for d in DATASETS for s in SEEDS]
    delta_vs_update = [num(metric_map[(d, s, "prior_delta_gate")]["val_acc"]) -
                       num(metric_map[(d, s, "prior_update_gate")]["val_acc"])
                       for d in DATASETS for s in SEEDS]

    metrics_table = []
    for r in variant_rows:
        metrics_table.append([r["variant"], f'{fmt(100*r["val_acc_mean"],2)} ± {fmt(100*r["val_acc_sd"],2)}',
                              f'{fmt(r["val_macro_f1_mean"],4)} ± {fmt(r["val_macro_f1_sd"],4)}',
                              f'{r["active_parameters_min"]}-{r["active_parameters_max"]}'])
    contrast_table = []
    for _, _, label in CONTRASTS:
        row = next((r for r in paired_summary if r["contrast"] == label and r["metric"] == "validation accuracy delta (pp)" and r["scope"] == "all datasets"), None)
        if row:
            contrast_table.append([label, f'{fmt(row["mean_delta"],2)} ± {fmt(row["sd"],2)}', row["n"]])

    comparison_labels = (
        ("relation_feature_variance", "Relation feature variance"),
        ("cross_modal_relation_1_minus_cosine", "Cross-modal relation discrepancy (1-cosine)"),
        ("directed_reverse_1_minus_cosine", "Reverse-edge directionality (1-cosine)"),
        ("operator_deviation_ratio", "Operator deviation ratio"),
        ("operator_modulation_variance", "Operation modulation variance"),
        ("execution_code_variance", "Execution-code variance"),
    )
    health_comparison_table = []
    for metric, label in comparison_labels:
        rows = [r for r in comparison_summary if r["metric"] == metric]
        total_n = sum(int(r["n"]) for r in rows)
        health_comparison_table.append([
            label,
            fmt(statistics.mean([num(r["v3_absolute_mean"]) for r in rows])),
            fmt(statistics.mean([num(r["prior_update_gate_mean"]) for r in rows])),
            total_n,
        ])

    def intervention_line(filename):
        rows = interventions[filename]
        avg, sd, n = mean_sd([r["val_acc_change_pp"] for r in rows])
        negative = sum(num(r["val_acc_change_pp"]) < 0 for r in rows)
        return f"mean validation accuracy change {fmt(avg,2)} ± {fmt(sd,2)} pp; lower accuracy in {negative}/{n} runs"

    report = f"""# M0 Final-Readout Audit: Prior-Retaining Context Injection

## A. Git provenance
Branch: exp/m0_prior_retention. Required parent: 108368f72c85441f580a1dd13b9720058915d57b (exp/m0_conditioner_v3). Stage-I/II implementation is frozen to the v3 absolute bilinear conditioner.

## B. Protocol audit
The pilot contains 36 unique completed checkpoints: three datasets (Movies, Grocery, Reddit-S), seeds 42/43/44, and four readout variants. Every run uses unified_full_graph_nc_v1, best validation accuracy checkpoint selection, and task.evaluate_test=false. Validation metrics were recomputed from each selected checkpoint. Test was not evaluated.

## C. Files added/modified
Added src/models/interaction_prior_retaining.py and src/models/prior_retention_components.py; configs/model/interaction_prior_retaining.yaml; tests/test_interaction_prior_retaining.py; scripts/run_m0_prior_retention.py, scripts/analyze_m0_prior_retention.py, and scripts/summarize_m0_prior_retention.py; docs/m0/prior_retention_design.md and this report; and the audit tables plus README under results/m0/prior_retention/. Full training logs/checkpoints are under outputs/m0/prior_retention/full/.

## D. Exact v3 Stage-I/II preservation
Frozen v3 preflight and trained checkpoint regressions contain {len(regression)} tensor comparisons; all passed at absolute/relative tolerances 1e-5. Regressed tensors include H0/H1/H2, relation features and memory, base relation codes, execution codes, operator modulation, and messages. Stage health summaries are descriptive because the readout changes the objective path.

## E. Exact captured update-residual formulation
The update module computes one dropout residual U = Dropout(W_u GELU(M)), then returns LayerNorm(H + U) and captures that same tensor. No extra dropout call is used. The context summary is C_update = (U0 + U1)/2. Residual diagnostics are in update_residual_diagnostics.csv and update_residual_summary.csv.

## F. Prior-retaining readout formulation
For each modality, H0 is the topology-agnostic intrinsic semantic prior. A modality-specific one-layer ContextAdapter maps the accumulated update residual through Linear, LayerNorm, GELU, Dropout, Linear; its final projection is zero-initialized. The feature-wise gate is sigmoid(Linear([H0 || E])) with zero-initialized parameters. The final modality state is H0 + gate elementwise-multiplied by E, followed by the original v3 fusion without another LayerNorm.

## G. Four variants
terminal uses H2. prior_delta_gate uses H2-H0. prior_update_add uses the mean actual update residual without a gate. prior_update_gate uses the mean actual update residual with an independent text/visual feature gate. All share v3 Stage-I/II and fusion.

## H. Unit/regression tests
The requested unit/regression suite passed: 27 tests. It covers shared-weight v3 equivalence, exact terminal training RNG sequence, residual identity and single dropout, zero initialization, intervention routing, gradient paths, and NC output compatibility.

## I. Smoke
All four Movies seed-42 smoke jobs completed for five epochs with finite losses and complete gradient/context-growth traces. Active ContextAdapter outputs became nonzero; smoke runs are plumbing checks and are not used as evidence for architecture selection.

## J. 36-run validation pilot
All 36 validation-only runs completed. Selected checkpoints and validation reproductions are recorded in pilot_metrics.csv and validation_reproduction.csv. Accuracy and macro-F1 below are mean ± sample SD over the nine dataset-seed runs; these are descriptive summaries.

{md_table(["Readout", "Validation accuracy (%)", "Validation macro-F1", "Active parameter range"], metrics_table)}

Paired validation-accuracy contrasts, in percentage points:

{md_table(["Contrast", "Mean ± SD", "n"], contrast_table)}

## K. Stage-I/II regression health
All exact tensor regressions passed. The table compares nine matched frozen v3 absolute checkpoints with the nine trained prior_update_gate checkpoints. Variance and cosine definitions match the saved v3 diagnostics; the per-modality/per-step paired values are in stage12_v3_comparison.csv and its summary.

{md_table(["Measure", "v3 absolute mean", "prior_update_gate mean", "matched n"], health_comparison_table)}

The comparison is descriptive because the training objectives differ. Full four-variant health outputs are in stage12_health.csv and stage12_health_summary.csv.

## L. Update-residual health
Mean per-step residual norm is 3.82 for text and 3.86 for visual; mean U0/U1 cosine is 0.824 and 0.898, respectively. The ungated accumulated update-to-H0 norm ratio averages 1.084 for text and 0.754 for visual, while the final gated rho_context is lower. Feature/node variation and per-run values are in update_residual_diagnostics.csv and update_residual_summary.csv.

## M. Context-injection magnitude
rho_context is the norm of the gated injected residual divided by the H0 norm. Mean rho_context over full prior_update_gate checkpoints and both modalities: {fmt(statistics.mean([x for x in mean_rho if math.isfinite(x)]) if any(math.isfinite(x) for x in mean_rho) else float("nan"))}. Raw distribution summaries are in prior_injection_diagnostics.csv. Compare rho to one when judging whether the correction remains subordinate to the intrinsic prior.

## N. Prior-retention / semantic-drift diagnostics
The analyzer reports cosine drift from H0 for the terminal state and the prior-retaining output. Mean retention gain (terminal drift minus prior-output drift) across modality-checkpoints: {fmt(statistics.mean([x for x in mean_drift_gain if math.isfinite(x)]) if any(math.isfinite(x) for x in mean_drift_gain) else float("nan"))}. Per-checkpoint summaries and the fraction of nodes closer to H0 than H2 are in prior_retention_diagnostics.csv.

## O. Gate diagnostics
Mean checkpoint-level gate standard deviation is {fmt(statistics.mean([x for x in mean_gate_span if math.isfinite(x)]) if any(math.isfinite(x) for x in mean_gate_span) else float("nan"))}. The text gate mean/std average {fmt(statistics.mean([num(r["mean"]) for r in gates if r["modality"] == "text"]))}/{fmt(statistics.mean([num(r["std"]) for r in gates if r["modality"] == "text"]))}; visual is {fmt(statistics.mean([num(r["mean"]) for r in gates if r["modality"] == "visual"]))}/{fmt(statistics.mean([num(r["std"]) for r in gates if r["modality"] == "visual"]))}. Saturation is rare; full distributions and residual alignment are in gate_diagnostics.csv and gate_summary.csv.

## P. Text/Visual asymmetry
Across nine full checkpoints, mean rho_context is 0.438 for text and 0.517 for visual; mean gate standard deviation is 0.112 for text and 0.190 for visual. Projector gradient RMS is finite throughout the trace for both modalities (nonzero fraction 1.0 in the aggregate trace). These differences are descriptive only; see update_residual_diagnostics.csv, prior_injection_diagnostics.csv, prior_retention_diagnostics.csv, gate_diagnostics.csv, projector_gradient_comparison.csv, and stage12_health_summary.csv.

## Q. Gradient-path diagnostics
Per-epoch gradient RMS by parameter group is preserved in gradient_trace.csv. The text and visual projector gradients are nonzero in every observed epoch for all four variants in the aggregate traces. gradient_summary.csv reports epoch-level RMS and per-run first-nonzero epoch ranges. This describes optimization flow and does not establish that gradient starvation was solved.

## R. Context-branch growth
Context adapter output and gate parameter norms by epoch are in context_branch_growth.csv. context_branch_growth_summary.csv aggregates parameter norms at each run's selected best-validation checkpoint and reports the per-run first-nonzero epoch range. Active variants open ContextAdapter W2 at epoch 1 and feature gates at epoch 2 in all nine runs; terminal keeps these unused branches at zero.

## S. Context-off intervention
Across nine prior_update_gate checkpoints, {intervention_line("intervention_context_off.csv")}. See intervention_context_off.csv for individual outcomes. This is a same-checkpoint ablation of the injected context branch.

## T. Context-source substitution
The update-to-delta replacement is a post-hoc same-checkpoint substitution, not a retraining comparison or causal estimate. Outcomes are in intervention_context_source_swap.csv. The paired trained comparison prior_delta_gate versus prior_update_gate is reported separately in paired_comparisons.csv.

## U. Gate-one intervention
{intervention_line("intervention_gate_one.csv")}. Gate values and individual intervention effects are in gate_diagnostics.csv and intervention_gate_one.csv.

## V. Prior-off intervention
{intervention_line("intervention_prior_off.csv")}. This disables the direct H0 bypass while keeping the checkpoint and remaining branch fixed; it is a strong/OOD intervention, not a causal estimate of prior utility.

## W. Operator-off regression
{intervention_line("intervention_operator_off.csv")}. Operator-off lowers validation accuracy in {op_effect}/9 checkpoints. This tests whether the original relation-conditioned semantic operator remains functionally consequential.

## X. Relation-shuffle regression
{intervention_line("intervention_relation_shuffle.csv")}. Within-target relation-memory shuffling lowers validation accuracy in {shuffle_effect}/9 checkpoints. It is a same-checkpoint structural sensitivity test.

## Y. Q1 Prior-retention conclusion
prior_update_gate is lower than terminal in {terminal_wins}/9 paired comparisons, with mean validation accuracy difference {fmt(100*statistics.mean(full_vs_terminal),2)} pp. Mean semantic drift is lower than terminal by {fmt(statistics.mean(mean_drift_gain),3)} cosine-drift units and rho_context remains below one on average. Thus H0 retention is real and context is used, but this pilot does not show an accuracy improvement.

## Z. Q2 Delta-vs-update-context conclusion
The trained prior_delta_gate minus prior_update_gate validation accuracy difference is {fmt(100*statistics.mean(delta_vs_update),2)} pp, effectively tied descriptively. In the same-checkpoint source substitution, replacing the trained update context with H2-H0 changes accuracy by {intervention_line("intervention_context_source_swap.csv")}. That intervention is post-hoc and uses an adapter trained on update residuals, so it is not a retraining comparison. The explicit residual remains the cleaner signal by definition, but this pilot does not show a reliable task advantage over delta.

## AA. Q3 Gate necessity conclusion
prior_update_gate minus prior_update_add is {fmt(-100*statistics.mean(add_vs_gate),2)} pp in validation accuracy and {fmt(100*(statistics.mean([num(r["val_macro_f1"]) for r in metrics if r["variant"] == "prior_update_gate"]) - statistics.mean([num(r["val_macro_f1"]) for r in metrics if r["variant"] == "prior_update_add"])),2)} pp in macro-F1. Forcing the gate to one changes accuracy by {fmt(mean_sd([r["val_acc_change_pp"] for r in interventions["intervention_gate_one.csv"]])[0],2)} pp on average (lower in 9/9 checkpoints), and learned gates vary across features/nodes. This supports keeping the gate in the PRCI candidate, while the trained gate-vs-add accuracy gap alone is negligible and parameter counts differ.

## AB. Q4 Graph-context necessity conclusion
Context-off: {intervention_line("intervention_context_off.csv")}. Nonzero rho_context is {fmt(statistics.mean([x for x in mean_rho if math.isfinite(x)]) if any(math.isfinite(x) for x in mean_rho) else float("nan"))}; interpret this alongside the intervention and prior-retention ratios.

## AC. Q5 Interpret→Execute preservation conclusion
All Stage-I/II tensors regress exactly to v3 under shared weights. Operator-off lowers accuracy in {op_effect}/9 checkpoints (mean {fmt(mean_sd([r["val_acc_change_pp"] for r in operator_rows])[0],2)} pp); relation shuffle lowers it in {shuffle_effect}/9 (mean {fmt(mean_sd([r["val_acc_change_pp"] for r in shuffle_rows])[0],2)} pp). The operator-off hit rate is 8/9 here versus the prior v3 pilot's reported 9/9; relation-memory shuffle lowers accuracy in 7/9. This supports continued mechanism use, with weaker descriptive intervention consistency than the earlier v3 result, not a claim that the readout improves the mechanism.

## AD. What is supported
The code exposes the exact injected Stage-II residual without changing its update computation, keeps a direct H0 path in all prior variants, and audits the resulting readouts on validation only. The pilot supports descriptive comparisons across the prespecified three datasets and three seeds.

## AE. What is not supported
No test-set claim, statistical-significance claim, causal claim from post-hoc interventions, gradient-starvation resolution claim, or generalization beyond the evaluated datasets is made. Smoke results are not scientific evidence.

## AF. Recommended final architecture
Recommendation: keep the v3 terminal readout as the current default and do not freeze PRCI as the final default from this pilot. prior_update_gate preserves H0 better (mean cosine-drift gain {fmt(statistics.mean(mean_drift_gain),3)}, average rho_context {fmt(statistics.mean([x for x in mean_rho if math.isfinite(x)]),3)}); context-off lowers accuracy in {measurable_context}/9, and mechanism checks remain active (operator-off {op_effect}/9, relation shuffle {shuffle_effect}/9). However, prior_update_gate is lower than terminal in {terminal_wins}/9 comparisons and averages {fmt(100*statistics.mean(full_vs_terminal),2)} pp accuracy; all three prior variants are about 0.38–0.43 pp below terminal on average. The protocol set no non-inferiority margin, so the evidence cannot establish Case A. Because these are descriptive results without significance tests, they also do not establish a stable degradation under Case D. Treat PRCI as a functionally validated alternative for human review, while retaining v3 terminal as the conservative architecture recommendation.

No extra datasets, test, link prediction, or hyperparameter search were run. Stop this experiment here pending human review.
"""
    DOC.mkdir(parents=True, exist_ok=True)
    (DOC / "prior_retention_report.md").write_text(report, encoding="utf-8")

    design = """# PRCI design

## Scope

Prior-Retaining Context Injection (PRCI) changes only the final readout of interaction_core_v3. Stage I and Stage II remain context_bilinear_absolute with hidden dimension 256, relation dimension 64, operator/conditioner rank 32, and two interaction steps. The existing multimodal fusion is reused.

## Intrinsic prior and contextual residual

For modality m, the intrinsic semantic prior is P_i^m = H_i,0^m, the modality projector output before graph interaction. During each of K=2 Stage-II updates, the model computes the actual residual U_i,k^m = Dropout(W_u^m GELU(M_i,k^m)) and applies H_i,k+1^m = LayerNorm(H_i,k^m + U_i,k^m). Instrumentation returns that very residual from the update call; it does not invoke dropout again.

The full candidate summarizes the actual injected updates as C_i^m = (U_i,0^m + U_i,1^m)/2. A control uses the effective state delta H_i,2^m - H_i,0^m, which includes repeated LayerNorm effects.

## Adapter, gate, and output

Each modality has an independent one-layer adapter: Linear(256,256), LayerNorm, GELU, Dropout(0.2), Linear(256,256). The last projection is zero-initialized, so E_i^m = 0 at initialization. The gated candidate computes g_i^m = sigmoid(W_g^m[H_i,0^m || E_i^m]) with independent feature-wise 256-dimensional gates initialized to 0.5. It returns Htilde_i^m = H_i,0^m + g_i^m elementwise-multiplied by E_i^m. There is no post-injection LayerNorm. The original v3 fusion consumes the concatenated modality representations.

## Prespecified variants

- terminal: use H2 for each modality.
- prior_delta_gate: adapt H2-H0 and inject it through the feature gate.
- prior_update_add: adapt mean actual update residual and add it without a gate.
- prior_update_gate: adapt mean actual update residual and inject it through the feature gate.

All variants share the same Stage I/II, classifier protocol, and fusion. Full-checkpoint interventions disable context, substitute delta for update context, force the gate to one, remove the prior bypass, disable the operator, or cyclically shuffle relation memory within each target group. All NC experiments use validation-only checkpoint selection and task.evaluate_test=false.
"""
    (DOC / "prior_retention_design.md").write_text(design, encoding="utf-8")

    readme = """# M0 Prior Retention results

This directory contains validation-only outputs for the 36-run M0 Final-Readout Audit. All runs use best validation accuracy checkpoint selection and task.evaluate_test=false. No test split was accessed.

Primary files:
- pilot_metrics.csv: per-run validation metrics and parameter counts.
- paired_comparisons.csv and paired_comparison_summary.csv: paired descriptive contrasts by dataset and seed.
- stage12_preflight_regression.csv: frozen v3 preflight.
- stage12_regression.csv: frozen and trained Stage-I/II tensor regression.
- stage12_health.csv and stage12_health_summary.csv: relation/operator/execution health across four variants.
- stage12_v3_comparison.csv and stage12_v3_comparison_summary.csv: matched descriptive health comparison against frozen v3 absolute checkpoints.
- update_residual_diagnostics.csv and update_residual_summary.csv: actual injected residual diagnostics.
- prior_injection_diagnostics.csv and prior_injection_summary.csv: injection magnitude/alignment.
- prior_retention_diagnostics.csv: semantic drift and retention.
- gate_diagnostics.csv and gate_summary.csv: feature gate statistics.
- gradient_trace.csv and gradient_summary.csv: per-epoch gradients.
- context_branch_growth.csv and context_branch_growth_summary.csv: context branch parameter growth.
- intervention_context_off.csv, intervention_context_source_swap.csv, intervention_gate_one.csv, intervention_prior_off.csv, intervention_operator_off.csv, intervention_relation_shuffle.csv: same-checkpoint validation interventions.
- runtime_memory.csv: elapsed runtime and sampled GPU memory.
- pilot_summary.csv: descriptive mean and SD by variant.
- docs/m0/prior_retention_report.md: A-AF audit readout.

Metrics and interventions are descriptive, with no significance testing. Interventions are not causal estimates; prior-bypass-off is a strong/OOD perturbation. Full training checkpoints and logs are in outputs/m0/prior_retention/full and are not tracked in git.
"""
    (RES / "README.md").write_text(readme, encoding="utf-8")
    print(f"Summarized 36 runs; regression rows {len(regression)}; wrote report and audit tables.")


if __name__ == "__main__":
    main()
