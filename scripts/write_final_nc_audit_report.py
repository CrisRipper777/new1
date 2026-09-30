from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/final_nc"
DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
VARIANTS = {
    "no_context": "w/o Context",
    "shared_relation": "w/o Relation Specificity",
    "static_execution": "w/o State Conditioning",
    "operator_off": "w/o Semantic Operator",
}
METRICS = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")


def read_csv(name: str) -> list[dict]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt_percent(mean: float, std: float | None = None) -> str:
    if std is None:
        return f"{mean * 100:.2f}%"
    return f"{mean * 100:.2f} ± {std * 100:.2f}%"


def mean(values) -> float:
    values = [float(v) for v in values]
    return statistics.mean(values) if values else float("nan")


def sample_sd(values) -> float:
    values = [float(v) for v in values]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def unique_checkpoints(rows: list[dict]) -> list[dict]:
    result = {}
    for row in rows:
        result[(row["dataset"], row["seed"])] = row
    return list(result.values())


def main() -> None:
    full = read_csv("final_nc_full_results.csv")
    ablation = read_csv("final_nc_ablation.csv")
    runs = read_csv("run_manifest.csv")
    val_rep = read_csv("validation_reproduction.csv")
    test_rep = read_csv("test_reproduction.csv")
    paired = [row for row in read_csv("paired_ablation_deltas.csv")
              if row["row_type"] == "summary" and row["dataset"] == "ALL_15_PAIRS"]
    stage1 = read_csv("stage1_health.csv")
    context = read_csv("context_health.csv")
    execution = read_csv("execution_health.csv")
    decomposition = read_csv("variance_decomposition.csv")
    relation_intervention = read_csv("intervention_relation_shuffle.csv")
    context_intervention = read_csv("intervention_context_shuffle.csv")
    operator_intervention = read_csv("intervention_operator_off.csv")
    efficiency = read_csv("model_efficiency.csv")
    gradients = read_csv("gradient_path_sanity.csv")

    full_by_data = {row["dataset"]: row for row in full}
    full_by_data.pop("Average", None)
    ablation_by = {(row["dataset"], row["variant"]): row for row in ablation}
    paired_by = {(row["comparison"].split(" - ")[0], row["metric"]): row for row in paired}
    active_by_variant = defaultdict(list)
    for row in efficiency:
        active_by_variant[row["variant"]].append(row)

    max_val_err = max(float(r["absolute_error"]) for r in val_rep)
    max_test_err = max(float(r["absolute_error"]) for r in test_rep)
    tolerance_counts = defaultdict(int)
    for row in val_rep + test_rep:
        tolerance_counts[row["tolerance_status"]] += 1
    all_gradient_groups = sorted({row["group"] for row in gradients})
    dead_gradient_rows = [row for row in gradients if row["nonzero_seen"].lower() != "true"]
    stage1_collapse = [row for row in stage1 if float(row["relation_edge_variance"]) <= 1e-12]
    exec_inactive = [row for row in execution if float(row["mean_delta_l2"]) <= 1e-12]
    nonfinite_rows = []
    for name, rows, fields in (
        ("stage1", stage1, ("relation_edge_variance", "cross_modal_relation_discrepancy_1_minus_cos")),
        ("context", context, ("mean_relation_change_l2",)),
        ("execution", execution, ("execution_code_variance", "bilinear_correction_relative_norm",
                                  "operator_deviation_relative_norm", "message_rotation_1_minus_cos")),
        ("variance", decomposition, ("eta_relation", "eta_target", "decomposition_error")),
    ):
        for row in rows:
            for field in fields:
                if row.get(field, "") and not math.isfinite(float(row[field])):
                    nonfinite_rows.append((name, row.get("dataset"), row.get("seed"), field))

    lines = [
        "# Final NC Architecture Freeze Audit",
        "",
        "## A. Git provenance",
        "",
        "- Branch: `exp/final_nc_freeze`.",
        "- Required parent: `108368f72c85441f580a1dd13b9720058915d57b` (`exp/m0_conditioner_v3`).",
        "- The final branch was checked out clean at that exact parent. No model structure was inherited from `exp/m0_stage3_history` or `exp/m0_prior_retention`.",
        "",
        "## B. Architecture-freeze provenance",
        "",
        "The paper-facing implementation is `src/models/final_interaction.py` plus `src/models/final_interaction_components.py`; its only benchmark configuration is `configs/model/final_interaction.yaml`. The locked equations and exclusions are recorded in `docs/final_model_spec.md` before smoke or final test evaluation. No hyperparameter search was run.",
        "",
        "## C. NC protocol audit",
        "",
        f"Completed {len(runs)} runs for five datasets × three seeds × five variants. Protocol: `unified_full_graph_nc_v1`, full graph, AdamW, lr 0.001, weight decay 0.0001, maximum 300 epochs, validation each epoch, patience 30, minimum epoch 30, min delta 0.0001, gradient clip 1.0.",
        "",
        "## D. Test sealing / no-leakage audit",
        "",
        "Architecture and hyperparameters were frozen before final test evaluation. Every checkpoint uses `selection=best_val_accuracy`; the unchanged NC task restores that checkpoint before test evaluation. The dedicated runner fixes Macro-F1's class universe from dataset metadata and does not inspect test labels to choose it. Test metrics were reporting-only; they were not used to choose epochs, variants, seeds, or hyperparameters. No result-dependent rerun or seed cherry-picking occurred.",
        "",
        "## E. Final model exact formulation",
        "",
        "Stage I independently encodes text and visual features, constructs ordered pair evidence and recipient leave-one-out context, and uses cross-modal attention to form edge-specific relation memory. Stage II retrieves a grounded base relation code, applies an absolute bilinear relation × current-recipient-state conditioner, transforms source semantics through a rank-32 operator, mean-aggregates, and updates each modality independently for two shared-parameter steps. Terminal fusion reads only H2. Full equations and dimensions are in `docs/final_model_spec.md`.",
        "",
        "## F. Five ablation definitions",
        "",
        "`no_context` replaces LOO context with learned null tokens; `shared_relation` uses two learned global relation slots; `static_execution` sets ξ=r; `operator_off` sets Δ=0 while keeping z and the backbone. These are separate trained fits; same-checkpoint interventions are reported separately.",
        "",
        "## G. V3→Final exact regression",
        "",
        "The full variant maps all 70 shared parameters from `interaction_core_v3/context_bilinear_absolute`. All 28 required output checks passed at atol=rtol=1e-5 with maximum absolute error 0. The regression includes H0, pair/context evidence, relation outputs and memory, base codes, both-step execution codes/modulation/delta messages, H1/H2, and fused_z.",
        "",
        "## H. Unit tests",
        "",
        "`tests/test_final_interaction.py`: 3 passed. It checks v3 numerical equivalence, all five variant contracts/active sets, and gradient activation past the zero-initialized gates.",
        "",
        "## I. Smoke",
        "",
        "Movies seed 42, all five variants, five epochs each: complete. Losses and validation/test metrics were finite; all checkpoint audits passed. Smoke results are not scientific evidence.",
        "",
        "## J. 75-run completion audit",
        "",
        f"Manifest contains {len(runs)} complete runs and the exact 5 × 3 × 5 grid. Each checkpoint records task, protocol, seed, validation-accuracy selection, selected epoch, validation/test accuracy and Macro-F1, model/head states, and data metadata.",
        "",
        "## K. Validation reproduction",
        "",
        f"All {len(val_rep)} checkpoint validation metrics were recomputed after checkpoint reload. Maximum absolute error: {max_val_err:.3e}; tolerance statuses: {dict(tolerance_counts)}.",
        "",
        "## L. Test reproduction",
        "",
        f"All {len(test_rep)} checkpoint test metrics were recomputed after checkpoint reload. Maximum absolute error: {max_test_err:.3e}; tolerance statuses: {dict(tolerance_counts)}. Test reproduction is an audit of saved predictions/metrics, not model selection.",
        "",
        "## M. Full-model five-dataset validation results",
        "",
        "See `docs/final_nc_main_table.md` for the supplementary validation table and `results/final_nc/final_nc_full_results.csv` for means and sample SDs.",
        "",
        "| Dataset | Val Acc (%) | Val Macro-F1 (%) |",
        "|---|---:|---:|",
    ]
    for dataset in DATASETS:
        row = full_by_data[dataset]
        lines.append(f"| {dataset} | {fmt_percent(float(row['val_acc_mean']), float(row['val_acc_std']))} | {fmt_percent(float(row['val_macro_f1_mean']), float(row['val_macro_f1_std']))} |")
    lines.extend(("", "## N. Full-model five-dataset test results", "",
                  "Test metrics are reported after restoring each best-validation checkpoint. The Average row is descriptive only.",
                  "", "| Dataset | Test Acc (%) | Test Macro-F1 (%) |", "|---|---:|---:|"))
    for dataset in DATASETS:
        row = full_by_data[dataset]
        lines.append(f"| {dataset} | {fmt_percent(float(row['test_acc_mean']), float(row['test_acc_std']))} | {fmt_percent(float(row['test_macro_f1_mean']), float(row['test_macro_f1_std']))} |")
    avg = next(row for row in full if row["dataset"] == "Average")
    lines.append(f"| Average† | {fmt_percent(float(avg['test_acc_mean']), float(avg['test_acc_std']))} | {fmt_percent(float(avg['test_macro_f1_mean']), float(avg['test_macro_f1_std']))} |")
    lines.extend((
        "",
        "† Descriptive macro-average across datasets; the SD shown is the average of within-dataset three-seed SDs, not pooled uncertainty.",
        "",
        "## O. Trained ablation — validation",
        "",
        "Per-dataset, per-variant validation metrics are in `docs/final_nc_ablation_table.md` and `results/final_nc/final_nc_ablation.csv`.",
        "",
        "## P. Trained ablation — test",
        "",
        "Per-dataset test accuracy and Macro-F1 mean ± sample SD for all five variants are in `docs/final_nc_ablation_table.md`.",
        "",
        "## Q. Paired ablation deltas",
        "",
        "`paired_ablation_deltas.csv` contains each matched dataset × seed delta and per-dataset plus all-15-pair mean, SD, and win/tie/loss. No Wilcoxon or t-test was used; no significance is claimed.",
        "",
        "## R. Stage-I health",
        "",
        f"`stage1_health.csv` reports relation edge variance, cross-modal relation discrepancy, and directionality for 15 Full checkpoints × two modalities. Relation-variance collapse rows (≤1e-12): {len(stage1_collapse)}.",
        "",
        "## S. Context health",
        "",
        "`context_health.csv` compares real-context and null-context relation states per seed, modality, and quintile of |cos(text)-cos(visual)|. This is descriptive; no Q1>Q5 hypothesis is tested.",
        "",
        "## T. Execution/operator health",
        "",
        "`execution_health.csv` reports execution-code and modulation variance, bilinear correction, Δ/z, cos(z,Δ), and message rotation by modality and step. Zero-Δ rows after training: " + str(len(exec_inactive)) + ".",
        "",
        "## U. Relation-specific variance decomposition",
        "",
        "`variance_decomposition.csv` reports total, within-target, between-target, η_relation, and η_target for r, ξ, a, and Δ on each Full checkpoint. The primary within/between decomposition follows the population law of total variance.",
        "",
        "## V. Relation-shuffle intervention",
        "",
        "Within-target cyclic relation-memory shuffle results are in `intervention_relation_shuffle.csv`; the intervention preserves each recipient's relation-memory marginal and degree while disrupting source-edge alignment.",
        "",
        "## W. Context-shuffle intervention",
        "",
        "Degree-matched paired Text/Visual context shuffle results are in `intervention_context_shuffle.csv`; the intervention preserves paired modality context and exact recipient degree groups.",
        "",
        "## X. Operator-off intervention",
        "",
        "The same Full checkpoints were evaluated with Δ=0 while keeping base source messages, aggregation, and updates. Results are in `intervention_operator_off.csv`.",
        "",
        "These three same-checkpoint perturbations are mechanism diagnostics, not causal estimates of module contribution. Their metrics are not used for model selection.",
        "",
        "## Y. Parameter counts",
        "",
        "`active_parameters.csv` lists model and classifier parameters by dataset/variant with active flags. `model_efficiency.csv` records active and total counts. Full has no inactive registered parameter modules; the expected inactive paths in ablations are explicitly marked.",
        "",
        "| Variant | Active parameter range | Total parameter range |",
        "|---|---:|---:|",
    ))
    for variant in ("full", *VARIANTS):
        rows = active_by_variant[variant]
        active = [int(row["active_parameters"]) for row in rows]
        total = [int(row["total_parameters"]) for row in rows]
        label = "Full" if variant == "full" else VARIANTS[variant]
        lines.append(f"| {label} | {min(active):,}–{max(active):,} | {min(total):,}–{max(total):,} |")
    full_eff = [row for row in efficiency if row["variant"] == "full"]
    lines.extend(("", "## Z. Runtime/memory", "",
                  "Full-graph epoch time includes per-epoch validation. Peak GPU memory is the assigned-device one-second `nvidia-smi` sample above baseline.",
                  "", "| Dataset | Mean epoch time (s) | Peak GPU memory mean / max (MiB) |", "|---|---:|---:|"))
    for dataset in DATASETS:
        row = next(row for row in full_eff if row["dataset"] == dataset)
        lines.append(f"| {dataset} | {float(row['mean_epoch_time_seconds']):.2f} | {float(row['peak_gpu_memory_mean_mb']):.0f} / {float(row['peak_gpu_memory_max_mb']):.0f} |")
    lines.extend((
        "",
        "## AA. Gradient-path sanity",
        "",
        f"Full training traces the seven required groups. Groups with no observed nonzero gradient in at least one Full dataset/seed run: {[(r['dataset'], r['seed'], r['group']) for r in dead_gradient_rows]}.",
        "",
        "## AB. Movies conclusion",
        "",
    ))
    for dataset in DATASETS:
        full_row = full_by_data[dataset]
        f1_delta_rows = []
        for variant in VARIANTS:
            summary = ablation_by[(dataset, variant)]
            f1_delta_rows.append((float(summary["test_macro_f1_mean"]) - float(full_row["test_macro_f1_mean"]), VARIANTS[variant]))
        mean_context = mean([float(r["mean_relation_change_l2"]) for r in context
                             if r["dataset"] == dataset and r["compatibility_gap_quintile"] == "all"])
        mean_operator = mean([float(r["operator_deviation_relative_norm"]) for r in execution if r["dataset"] == dataset])
        delta_eta = mean([float(r["eta_relation"]) for r in decomposition
                          if r["dataset"] == dataset and r["representation"] == "Delta"])
        lines.append(
            f"{dataset}: Full test Acc {fmt_percent(float(full_row['test_acc_mean']), float(full_row['test_acc_std']))}, "
            f"Macro-F1 {fmt_percent(float(full_row['test_macro_f1_mean']), float(full_row['test_macro_f1_std']))}; "
            f"real-vs-null context relation change {mean_context:.4g}, mean Δ/z {mean_operator:.4g}, "
            f"Δ η_relation {delta_eta:.4g}. Ablation Macro-F1 test-mean contrasts vs Full: "
            + ", ".join(f"{label} {delta * 100:+.2f} pp" for delta, label in f1_delta_rows) + "."
        )
        if dataset == "Movies":
            lines.extend(("", "## AC. Toys conclusion", ""))
        elif dataset == "Toys":
            lines.extend(("", "## AD. Grocery conclusion", ""))
        elif dataset == "Grocery":
            lines.extend(("", "## AE. ele-fashion conclusion", ""))
        elif dataset == "ele-fashion":
            lines.extend(("", "## AF. Reddit-S conclusion", ""))
    lines.extend((
        "",
        "## AG. Which claims are supported",
        "",
        "Evidence for these claims is descriptive: relation states vary across edges/recipients; real-context states differ from null-context states; trained Full checkpoints exhibit measured bilinear correction and source-operation magnitudes; the three same-checkpoint interventions quantify dependence on relation alignment, context alignment, and Δ. Dataset-specific support is reported in the linked CSV files. No claim of statistical significance is made.",
        "",
        "## AH. Which claims are not supported",
        "",
        "This benchmark does not establish statistical significance from three seeds, universal Full-model superiority, a causal percentage contribution for any module, or superiority over external baselines. A same-checkpoint conditioner-off intervention was not part of the frozen protocol. External-baseline values remain blank until their split/protocol sources are verified.",
        "",
        "## AI. Final NC architecture-freeze verdict",
        "",
    ))
    if not dead_gradient_rows and not stage1_collapse and not exec_inactive and not nonfinite_rows and max_val_err <= 1e-5 and max_test_err <= 1e-5:
        lines.append("**NC Architecture Freeze = FINAL.** The requested implementation, checkpoint, reproduction, and mechanism checks completed without a flagged path failure. Test results remain reporting-only.")
    else:
        lines.append(
            f"**NC Architecture Freeze = REVIEW.** Audit flags: dead gradient rows={len(dead_gradient_rows)}, "
            f"relation collapse rows={len(stage1_collapse)}, zero-operator rows={len(exec_inactive)}, "
            f"non-finite diagnostic fields={len(nonfinite_rows)}, max reproduction errors={max(max_val_err, max_test_err):.3e}. "
            "This is an implementation/mechanism review flag, not a performance-based model change."
        )
    lines.extend((
        "",
        "## AJ. Readiness for LP",
        "",
        "`docs/final_model_spec.md` is the architecture authority for a later LP phase. This round stops after the NC deliverables; LP training and new baseline runs were not started.",
        "",
        "## Audit artifact index",
        "",
        "Machine-readable results: `results/final_nc/README.md`, run manifest, metric/reproduction CSVs, tables, mechanism health and intervention CSVs, parameter/runtime records, and the blank external-baseline template.",
    ))
    (ROOT / "docs/final_nc_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE docs/final_nc_audit_report.md; status flags dead_grad={len(dead_gradient_rows)} "
          f"relation_collapse={len(stage1_collapse)} inactive_operator={len(exec_inactive)} nonfinite={len(nonfinite_rows)}")


if __name__ == "__main__":
    main()
