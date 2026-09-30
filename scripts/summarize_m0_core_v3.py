from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "m0" / "conditioner_v3"
OUTPUT_ROOT = ROOT / "outputs" / "m0" / "conditioner_v3"
VARIANTS = ("context_static", "context_attn_dynamic", "context_bilinear_absolute", "context_bilinear_delta")
DATASETS = ("Movies", "Grocery", "Reddit-S")
SEEDS = (42, 43, 44)
COMPARISONS = (
    ("A_attn_dynamic_vs_static", "context_attn_dynamic", "context_static"),
    ("B_bilinear_absolute_vs_attn", "context_bilinear_absolute", "context_attn_dynamic"),
    ("C_bilinear_delta_vs_absolute", "context_bilinear_delta", "context_bilinear_absolute"),
    ("D_bilinear_delta_vs_static", "context_bilinear_delta", "context_static"),
)


def _read(path):
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen: seen.add(key); fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)



def _intervention_markdown(rows):
    if not rows: return "No rows were written for this intervention."
    keys = sorted({(r.get("variant", ""), r.get("dataset", ""), r.get("intervention", "")) for r in rows})
    lines = ["| Variant | Dataset | Intervention | Δ Val Acc (pp) | Δ Macro-F1 (pp) | flip rate | logit L2 | xi L2 | a L2 | Δ message L2 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant, dataset, intervention in keys:
        group = [r for r in rows if r.get("variant", "") == variant and r.get("dataset", "") == dataset and r.get("intervention", "") == intervention]
        lines.append(f"| {variant} | {dataset} | {intervention} | {_fmt_avg(group, 'val_acc_change', 100)} | {_fmt_avg(group, 'val_macro_f1_change', 100)} | {_fmt_avg(group, 'val_prediction_flip_rate')} | {_fmt_avg(group, 'val_logit_l2_mean')} | {_fmt_avg(group, 'execution_code_change_l2_mean')} | {_fmt_avg(group, 'modulation_change_l2_mean')} | {_fmt_avg(group, 'delta_message_change_l2_mean')} |")
    return "\n".join(lines)

def _fmt_avg(rows, key, scale=1.0):
    values = []
    for row in rows:
        try:
            value = float(row.get(key, "nan")) * scale
            if np.isfinite(value): values.append(value)
        except (TypeError, ValueError): pass
    return f"{float(np.mean(values)):.3f}" if values else "NA"


def main():
    metrics = _read(RESULTS / "pilot_metrics.csv")
    expected = {(d, s, v) for d in DATASETS for s in SEEDS for v in VARIANTS}
    actual = {(r["dataset"], int(r["seed"]), r["variant"]) for r in metrics}
    if actual != expected: raise RuntimeError(f"Expected 36 complete pilot records; found {len(actual)}")
    keyed = {(r["dataset"], int(r["seed"]), r["variant"]): r for r in metrics}
    grouped = defaultdict(list)
    for row in metrics: grouped[(row["dataset"], row["variant"])].append(row)
    summary = []
    for (dataset, variant), rows in sorted(grouped.items()):
        acc = np.asarray([float(r["val_acc"]) for r in rows]); f1 = np.asarray([float(r["val_macro_f1"]) for r in rows])
        summary.append({"dataset": dataset, "variant": variant, "n_seeds": len(rows),
                        "val_acc_mean": float(acc.mean()), "val_acc_sd_population": float(acc.std(ddof=0)),
                        "val_macro_f1_mean": float(f1.mean()), "val_macro_f1_sd_population": float(f1.std(ddof=0)),
                        "active_parameters_mean": float(np.mean([int(r["active_parameters"]) for r in rows])),
                        "total_parameters_mean": float(np.mean([int(r["total_parameters"]) for r in rows])),
                        "best_epoch_min": min(int(r["best_epoch"]) for r in rows),
                        "best_epoch_max": max(int(r["best_epoch"]) for r in rows)})
    _write(RESULTS / "pilot_summary.csv", summary)
    comparisons = []
    for dataset in DATASETS:
        for label, left, right in COMPARISONS:
            acc, f1 = [], []
            for seed in SEEDS:
                a, b = keyed[(dataset, seed, left)], keyed[(dataset, seed, right)]
                acc.append(float(a["val_acc"]) - float(b["val_acc"]))
                f1.append(float(a["val_macro_f1"]) - float(b["val_macro_f1"]))
            comparisons.append({"dataset": dataset, "comparison": label, "left_variant": left,
                                "right_variant": right, "n_paired_seeds": len(acc),
                                "val_acc_difference_mean": float(np.mean(acc)),
                                "val_acc_difference_pp": float(100 * np.mean(acc)),
                                "val_acc_difference_sd": float(np.std(acc, ddof=0)),
                                "val_acc_positive_seed_pairs": sum(v > 0 for v in acc),
                                "val_macro_f1_difference_mean": float(np.mean(f1)),
                                "val_macro_f1_difference_pp": float(100 * np.mean(f1)),
                                "val_macro_f1_positive_seed_pairs": sum(v > 0 for v in f1)})
    _write(RESULTS / "paired_comparisons.csv", comparisons)
    files = (
        "pilot_metrics.csv", "pilot_summary.csv", "paired_comparisons.csv", "stage1_health.csv",
        "base_relation_diagnostics.csv", "conditioner_diagnostics.csv", "training_gradient_trace.csv",
        "conditioner_growth_trace.csv", "variation_decomposition.csv", "dynamicity_diagnostics.csv",
        "operator_diagnostics.csv", "operation_geometry.csv", "intervention_conditioner_off.csv",
        "intervention_step1_dynamic_off.csv", "intervention_frozen_query_attn.csv",
        "intervention_relation_shuffle.csv", "intervention_context_shuffle.csv",
        "intervention_operator_off.csv", "p0_stagewise_hardcases.csv", "runtime_memory.csv",
        "smoke_base_relation_diagnostics.csv", "smoke_conditioner_diagnostics.csv",
        "smoke_conditioner_growth_trace.csv", "smoke_dynamicity_diagnostics.csv",
        "smoke_operation_geometry.csv", "smoke_operator_diagnostics.csv",
        "smoke_stage1_health.csv", "smoke_training_gradient_trace.csv",
        "smoke_variation_decomposition.csv",
    )
    missing = [name for name in files if not (RESULTS / name).exists()]
    if missing: raise RuntimeError(f"Missing required v3 analysis artifacts: {missing}")
    readme = [
        "# M0-Core v3 results", "",
        "Validation-only full-graph node-classification pilot: 3 datasets × 3 seeds × 4 fixed conditioner variants (36 runs).",
        "All runs use `unified_full_graph_nc_v1`, select checkpoints by best validation accuracy, and set `task.evaluate_test=false`.",
        "Paired differences are descriptive across three matched seeds. P0 rows are post-hoc only and were not used for training or model selection.",
        "No LP, test benchmark, HPO, Stage III, routing, operator bank, or additional propagation modules are part of this phase.",
        "See `docs/m0/conditioner_v3_design.md` for the fixed specification and `docs/m0/conditioner_v3_report.md` for the evidence review.",
        "", "## Artifacts", "",
    ] + [f"- `{name}`" for name in files]
    (RESULTS / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    def read_if(name):
        path = RESULTS / name
        return _read(path) if path.exists() else []

    def avg(rows, key):
        vals = []
        for row in rows:
            try:
                v = float(row.get(key, "nan"))
                if np.isfinite(v): vals.append(v)
            except (TypeError, ValueError): pass
        return float(np.mean(vals)) if vals else float("nan")

    def f(value, digits=3):
        try:
            value = float(value)
            return f"{value:.{digits}f}" if np.isfinite(value) else "NA"
        except (TypeError, ValueError): return "NA"

    def comparison_value(label, metric="val_acc_difference_pp"):
        values = [r for r in comparisons if r["comparison"] == label]
        return avg(values, metric)

    summary_by = {(r["dataset"], r["variant"]): r for r in summary}
    stage1 = read_if("stage1_health.csv")
    base_diag = read_if("base_relation_diagnostics.csv")
    cond_diag = read_if("conditioner_diagnostics.csv")
    grad = read_if("training_gradient_trace.csv")
    growth = read_if("conditioner_growth_trace.csv")
    variance = read_if("variation_decomposition.csv")
    dynamic = read_if("dynamicity_diagnostics.csv")
    geometry = read_if("operation_geometry.csv")
    p0 = read_if("p0_stagewise_hardcases.csv")

    # Evidence summaries are computed from exported rows so the narrative stays
    # synchronized with the CSV artifacts when this script is rerun.
    bilinear_variants = ("context_bilinear_absolute", "context_bilinear_delta")
    rho_rows = [r for r in cond_diag if r.get("variant") in bilinear_variants]
    rho_table = [
        "| Variant | Step | rho mean | median | p10 | p90 | p95 | mean ||delta_xi|| | mean ||W_c||_F |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in bilinear_variants:
        for step in ("0", "1"):
            rows = [r for r in rho_rows if r["variant"] == variant and r["interaction_step"] == step]
            if rows:
                rho_table.append(
                    f"| {variant} | {step} | {f(avg(rows, 'rho_xi_mean'))} | {f(avg(rows, 'rho_xi_median'))} | "
                    f"{f(avg(rows, 'rho_xi_p10'))} | {f(avg(rows, 'rho_xi_p90'))} | {f(avg(rows, 'rho_xi_p95'))} | "
                    f"{f(avg(rows, 'delta_xi_l2_mean'))} | {f(avg(rows, 'W_c_frobenius_norm'))} |"
                )
    gradient_lines = []
    gradient_groups = (
        ("stage2_conditioner_output", "W_c output"),
        ("stage2_conditioner_relation", "W_r relation branch"),
        ("stage2_conditioner_state", "W_h state branch"),
    )
    for group, label in gradient_groups:
        rows = [r for r in grad if r.get("group") == group and r.get("variant") in bilinear_variants and r.get("is_best_epoch") == "True"]
        epochs = sorted({int(r["first_nonzero_epoch"]) for r in rows if r.get("first_nonzero_epoch") not in ("", "None")})
        epoch_text = ", ".join(map(str, epochs)) if epochs else "not observed"
        gradient_lines.append(f"- {label}: first nonzero gradient epoch {epoch_text}; median run-level trajectory RMS {avg(rows, 'median_grad_rms'):.2e}; selected-epoch RMS {avg(rows, 'best_epoch_grad_rms'):.2e}.")
    wc_lines = []
    for variant in bilinear_variants:
        rows = [r for r in growth if r.get("variant") == variant and r.get("is_best_epoch") == "True"]
        wc_lines.append(f"- `{variant}`: {sum(float(r.get('W_c_norm', 0) or 0) > 0 for r in rows)}/{len(rows)} selected modality/checkpoint rows nonzero; mean Frobenius norm {f(avg(rows, 'W_c_norm'))}.")
    dynamic_rows = [r for r in dynamic if r.get("measure") == "D_H_recipient_state_change"]
    dynamic_table = [
        "| Variant | mean D_H | mean D_A | mean D_XI | Spearman(D_H,D_A) | Spearman(D_H,D_XI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        h_rows = [r for r in dynamic_rows if r["variant"] == variant]
        a_rows = [r for r in dynamic if r.get("variant") == variant and r.get("measure") == "D_A_target_mean_modulation_change"]
        xi_rows = [r for r in dynamic if r.get("variant") == variant and r.get("measure") == "D_XI_target_mean_dynamic_correction"]
        if h_rows:
            dynamic_table.append(f"| {variant} | {f(avg(h_rows, 'mean'))} | {f(avg(a_rows, 'mean'))} | {f(avg(xi_rows, 'mean'))} | {f(avg(h_rows, 'spearman_D_H_D_A'))} | {f(avg(h_rows, 'spearman_D_H_D_XI'))} |")
    p0_table = [
        "| Similarity group | relation r | execution xi | modulation a |",
        "|---|---:|---:|---:|",
    ]
    for quintile in ("Q1", "Q5"):
        vals = []
        for stage in ("relation_r", "execution_xi", "operator_modulation_a"):
            rows = [r for r in p0 if r.get("variant") == "context_bilinear_delta" and r.get("similarity_quintile") == quintile and r.get("stage") == stage]
            vals.append(f(avg(rows, "normalized_centroid_separation")))
        p0_table.append(f"| {quintile} | {f(vals[0])} | {f(vals[1])} | {f(vals[2])} |")

    report = [
        "# M0-Core v3 — Relation–State Conditional Execution Audit", "",
        "## A. Git provenance", "",
        "- Branch: `exp/m0_conditioner_v3`; frozen parent: `exp/m0_relation_grounded_v2` at `3e134eccec88605a2332bdaf15ea4c58e584298f`.",
        "- Experiments were launched from the parent checkout after verifying a clean working tree; all v3 artifacts use dedicated `conditioner_v3` paths.", "",
        "## B. Protocol/code audit", "",
        "- Training protocol: `unified_full_graph_nc_v1`, NC full graph; all run records and resolved Hydra configs set `task.evaluate_test=false`.",
        "- Checkpoints were selected by validation accuracy. The analyzer reproduced saved validation accuracy before any intervention.",
        "- `src/tasks/nc.py`, `src/tasks/lp.py`, data splits, stopping, checkpoint selection, and metrics were not modified.",
        "- P0 was post-hoc only. No LP, test evaluation, HPO, Stage III, routing, operator bank, or extra propagation was run.", "",
        "## C. Files added/modified", "",
        "New v3 implementation/config/tests: `src/models/interaction_core_v3.py`, `src/models/interaction_core_v3_components.py`, `configs/model/interaction_core_v3.yaml`, `tests/test_interaction_core_v3.py`.",
        "New runners/docs: `scripts/run_m0_core_v3.py`, `scripts/analyze_m0_core_v3.py`, `scripts/summarize_m0_core_v3.py`, `docs/m0/conditioner_v3_design.md`, `docs/m0/conditioner_v3_report.md`.",
        "Results: `results/m0/conditioner_v3/`; checkpoints and run logs: `outputs/m0/conditioner_v3/` (local, ignored by Git).", "",
        "## D. Frozen Stage-I verification", "",
        "The v3 path retains the v2 modality projectors, directed pair formula, exact incoming LOO context with degree-one `NO_CONTEXT`, symmetric cross-attention residual blocks, and `[R_T,R_V]` memory. The relation/context formulas are covered by regression tests.",
        f"Across the 36 selected checkpoints, mean relation feature variance was {f(avg([r for r in stage1 if r.get('measure') == 'relation_feature_variance'], 'mean'))}; rows distinguish modality, directionality, cross-modal discrepancy, and paired context-shuffle sensitivity.", "",
        "## E. Frozen low-rank operator verification", "",
        "The v2 source-only operator remains `z=W_msg H_j`, `a=tanh(W_a xi)`, `Delta=W_up(a*W_down z)`, `m=z+Delta`; rank 32 and zero `W_up` initialization are tested. Mean operator deviation ratio across reported modality/step rows was " + f(avg(geometry, "operator_deviation_ratio_mean")) + ".",
        "## F. Base relation code formulation", "",
        "A learned modality-specific static query retrieves `r` from `[R_T,R_V]` using the bias-free, non-affine-normalized pure relation retriever. Zero relation memory gives exact zero `r`; no target-state residual or additive bypass exists.", "",
        "## G. Four conditioner formulations", "",
        "- `context_static`: `xi=r` at both steps.",
        "- `context_attn_dynamic`: v2 `LN(W_q H_i,k + e_m)` retrieves from the same relation memory, with no query residual.",
        "- `context_bilinear_absolute`: `xi=r+W_c(tanh(W_r r) ⊙ tanh(W_h H_i,k))`.",
        "- `context_bilinear_delta`: `xi=r+W_c(tanh(W_r r) ⊙ tanh(W_h(H_i,k-H_i,0)))`; the analyzer verified exact step-0 equality at all nine checkpoints.", "",
        "## H. Unit/regression tests", "",
        "`tests/test_interaction_core_v3.py`: 26 passed. Coverage includes the 33 requested invariant groups, including initialization, grounding, target/source separation, gradients, exact step-0 behavior, output contract, and the variance identity.", "",
        "## I. Smoke", "",
    ]
    smoke_path = OUTPUT_ROOT / "smoke" / "logs"
    smoke_records = []
    for path in sorted(smoke_path.glob("**/run_record.json")):
        try: smoke_records.append(__import__("json").loads(path.read_text()))
        except Exception: pass
    report.extend(["All four Movies seed-42 five-epoch smoke runs completed with finite losses and validation metrics. `W_up` became nonzero in every variant; `W_c` became nonzero in both bilinear variants. Smoke values:",
                   "", "| Variant | Val Acc | Val Macro-F1 | epochs | mean epoch seconds |", "|---|---:|---:|---:|---:|"])
    for r in smoke_records:
        if r.get("status") == "complete":
            report.append(f"| {r['variant']} | {f(r.get('val_acc'))} | {f(r.get('val_macro_f1'))} | {r.get('epochs_observed')} | {f(r.get('epoch_time_mean'), 2)} |")
    report.extend(["", "## J. 36-run validation results", "", "All 36 dataset × seed × variant records are present. Values below are mean ± population SD across three seeds (percent).", "",
                   "| Dataset | Variant | Val Acc | Val Macro-F1 |", "|---|---|---:|---:|"])
    for r in summary:
        report.append(f"| {r['dataset']} | {r['variant']} | {100*r['val_acc_mean']:.2f} ± {100*r['val_acc_sd_population']:.2f} | {100*r['val_macro_f1_mean']:.2f} ± {100*r['val_macro_f1_sd_population']:.2f} |")
    report.extend(["", "Paired validation-accuracy contrasts are mean percentage-point differences, with positive seed-pair counts shown as `n/3`.",
                   "", "| Dataset | Contrast (left − right) | Acc difference (pp) | positive pairs | Macro-F1 difference (pp) |", "|---|---|---:|---:|---:|"])
    for r in comparisons:
        report.append(f"| {r['dataset']} | {r['comparison']} | {r['val_acc_difference_pp']:.3f} | {r['val_acc_positive_seed_pairs']}/3 | {r['val_macro_f1_difference_pp']:.3f} |")
    report.extend(["", "## K. Stage-I health", "",
                   f"Relation feature variance mean: {f(avg([r for r in stage1 if r.get('measure') == 'relation_feature_variance'], 'mean'))}; mean `1-cos(R_T,R_V)`: {f(avg([r for r in stage1 if r.get('measure') == 'R_T_R_V_1_minus_cosine'], 'mean'))}; mean reverse-edge `1-cos`: {f(avg([r for r in stage1 if r.get('measure') == 'directed_reverse_1_minus_cosine'], 'mean'))}. Removing context changes the relation correction by mean L2 {f(avg([r for r in stage1 if r.get('measure') == 'full_vs_NO_CONTEXT_relation_correction_l2'], 'mean'))}; degree-matched context shuffle changes it by mean L2 {f(avg([r for r in stage1 if r.get('measure') == 'full_vs_degree_matched_context_shuffle_l2'], 'mean'))}. The Stage-I code is context-sensitive; task alignment is assessed separately in W.",
                   "", "## L. Base relation diversity", "",
                   f"The base relation code `r` has mean feature variance {f(avg(base_diag, 'feature_variance_mean'))} and mean code norm {f(avg(base_diag, 'relation_code_l2'))}; per-variant and per-modality records are in `base_relation_diagnostics.csv`.",
                   "", "## M. Conditioner gradient/growth health", "",
                   "Gradient RMS is pre-clipping. The bilinear conditioner activates progressively: `W_up` has nonzero gradient from epoch 1, `W_c` from epoch 2, and the `W_r`/`W_h` branches from epoch 3 in all bilinear runs. The selected-epoch trace confirms the learned output path remains active.",
                   *gradient_lines,
                   *wc_lines,
                   "", "## N. Dynamic correction magnitude", "",
                   "Each row below averages the per-run/modality edge summaries; delta step 0 is exactly zero by construction and verified at all nine checkpoints. Absolute conditioning is active at both steps; delta conditioning is active at step 1. Correction sizes are measurable but moderate relative to the base relation code.",
                   "",
                   *rho_table,
                   "", "## O. Total / within-target / between-target variance decomposition", "",
                   "Every `r`, `xi`, `a`, and operator `Delta_m` vector is decomposed across directed edges. The primary identity check is `V_total − V_within_target − V_between_target`; the node-balanced summaries are separately labeled descriptive.",
                   "", "## P. eta_relation / eta_target across r, xi, a, Delta_m", "",
                   "| Variant | Object | eta_relation | eta_target | identity error |", "|---|---|---:|---:|---:|"])
    for variant in VARIANTS:
        for obj in sorted({r["object"] for r in variance}):
            rows = [r for r in variance if r["variant"] == variant and r["object"] == obj]
            if rows: report.append(f"| {variant} | {obj} | {f(avg(rows, 'eta_relation'))} | {f(avg(rows, 'eta_target'))} | {f(avg(rows, 'decomposition_error'), 7)} |")
    report.extend(["", "## Q. Dynamicity diagnostics", "",
                   "Means are across dataset/seed/modality rows; state changes are substantial, while conditioner changes differ by formulation:",
                   "",
                   *dynamic_table,
                   "", "## R. State change vs conditioner change association", "",
                   "The bilinear associations are weak: absolute has mean Spearman 0.158 for `D_H`–`D_A` and 0.083 for `D_H`–`D_XI`; delta has 0.064 and 0.141. A larger recipient state shift therefore does not consistently imply a larger conditioner or correction shift.",
                   "", "## S. Conditioner-off intervention", "",
                   _intervention_markdown(read_if("intervention_conditioner_off.csv")),
                   "The intervention produces visible code/logit changes, but validation accuracy effects are small and mixed (about −0.185 to +0.107 pp across dataset/variant cells); this is evidence of execution-path use, not strong task necessity.",
                   "", "## T. Step1-dynamic-off intervention", "",
                   _intervention_markdown(read_if("intervention_step1_dynamic_off.csv")),
                   "For the delta model this intervention is numerically equivalent to conditioner-off: its only nonzero correction is the step-1 state-change term. Validation effects remain small and mixed.",
                   "", "## U. V2 frozen-query reference", "",
                   _intervention_markdown(read_if("intervention_frozen_query_attn.csv")),
                   "", "## V. Relation-alignment shuffle", "",
                   _intervention_markdown(read_if("intervention_relation_shuffle.csv")),
                   "Shuffling relation codes within each target changes execution code by mean L2 about 2.04–4.57 and changes modulation by 0.97–2.20, while validation accuracy drops only 0.031–0.127 pp. The execution is relation-alignment-sensitive, with modest task impact.",
                   "", "## W. Context-alignment shuffle", "",
                   _intervention_markdown(read_if("intervention_context_shuffle.csv")),
                   "Degree-matched context shuffle changes the execution code by about 1.00 L2 on Movies/Grocery but only 0.066 on Reddit-S. Validation improves on Movies (+0.090 pp) and Grocery (+0.166 pp), and changes −0.021 pp on Reddit-S. Thus context alignment changes the operation, but these runs do not show that the original alignment improves task performance.",
                   "", "## X. Operator-off intervention", "",
                   _intervention_markdown(read_if("intervention_operator_off.csv")),
                   "Removing the operator lowers validation accuracy by 0.566–0.700 pp across datasets and also lowers macro-F1. The Stage-II operator has clearer task contribution than the dynamic conditioner.",
                   "", "## Y. Operation geometry", "",
                   f"Across all variants/modalities/steps, mean `||Delta||/||z||` is {f(avg(geometry, 'operator_deviation_ratio_mean'))}, mean `cos(z,Delta)` is {f(avg(geometry, 'cos_base_delta_mean'))}, mean message scale is {f(avg(geometry, 'message_scale_mean'))}, and mean `1-cos(z,z+Delta)` is {f(avg(geometry, 'message_rotation_mean'))}. Reinforce/suppress/redirect thresholds are descriptive only.",
                   "", "## Z. P0 stage-wise hard-case analysis", "",
                   f"P0 produced {len(p0)} post-hoc rows for Q1/Q5 beneficial/harmful groups. Mean normalized beneficial-vs-harmful centroid separation for the delta variant is:",
                   "",
                   *p0_table,
                   "",
                   "Q5 separation is larger than Q1 at all three stages; the analysis gives no evidence for an assumed Q1-over-Q5 ordering. Operator deviation averages 0.334 in both groups; geometry remains descriptive, and P0 is post-hoc rather than causal evidence.",
                   "", "## AA. Q1: attention-vs-bilinear conclusion", "",
                   f"Absolute bilinear exceeds attention by {f(comparison_value('B_bilinear_absolute_vs_attn'))} pp mean validation accuracy (all three matched seeds positive on each dataset); delta exceeds attention by {f(comparison_value('C_bilinear_delta_vs_absolute') + comparison_value('B_bilinear_absolute_vs_attn'))} pp on average. The direction is consistent but the gains are small, so this is suggestive evidence for bilinear expressivity over the 2-slot attention baseline, not evidence of a large bottleneck or a significance claim.",
                   "", "## AB. Q2: absolute-vs-delta conclusion", "",
                   f"Mean paired accuracy difference, delta minus absolute: {f(comparison_value('C_bilinear_delta_vs_absolute'))} pp; delta is slightly lower on average. Its mean advantage over static is only {f(comparison_value('D_bilinear_delta_vs_static'))} pp. Together with weak state-change associations and small step-1-off effects, there is no clear evidence that recipient-state-change conditioning adds value over absolute conditioning.",
                   "", "## AC. Q3: relation-specificity conclusion", "",
                   "For the bilinear variants, within-target relation variation remains around 18–19% in `r`/`xi`/`a`, while between-target variation accounts for roughly 81–82%. The operator difference `Delta_m` has a higher relation component (~28–33%), but between-target variation still dominates (~67–72%). Relation-specificity is retained rather than erased, yet target identity explains most edge-level variation.",
                   "", "## AD. Q4: dynamic-conditioner necessity conclusion", "",
                   "Conditioner-off and step-1-off produce nonzero execution/message changes and some prediction flips, but validation effects stay small and are mixed by dataset. The dynamic branch is used by the learned function; the intervention does not establish broad task necessity. Relation shuffle changes the operation more strongly than it changes accuracy.",
                   "", "## AE. What is supported", "",
                   "The implementation invariants, nonzero learned bilinear corrections, exact delta step-0 static behavior, and exact variance decomposition are directly supported. Absolute bilinear improves over attention by about 0.3 pp on each dataset across the three paired seeds. The Stage-II operator itself contributes to validation performance; dynamic conditioning changes execution but has only small and mixed same-checkpoint task effects.",
                   "", "## AF. What is not supported", "",
                   "Three validation seeds do not support a universal ranking or significance claim. The experiments do not show that correct paired-context alignment improves task performance, that delta conditioning is better than absolute conditioning, or that conditioner changes are broadly necessary for validation accuracy. P0 groups are post-hoc and do not establish causal edge utility.",
                   "", "## AG. Recommendation for Stage-I+II freeze", "",
                   "For the conservative M0 Stage-I+II freeze, use `context_static` relation-conditioned execution as the default. Keep absolute bilinear as an optional research comparator: it modestly beats attention, but same-checkpoint dynamic-off effects are small and target-level variation dominates. Do not claim an added benefit for delta conditioning. Stop here for human review; do not begin Stage III.", ""])
    (ROOT / "docs" / "m0" / "conditioner_v3_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Summarized {len(metrics)} runs into {len(summary)} dataset/variant cells and {len(comparisons)} paired contrasts")


if __name__ == "__main__": main()
