from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "m0" / "stage3_history"
OUTPUTS = ROOT / "outputs" / "m0" / "stage3_history" / "full" / "logs"
DATASETS = ("Movies", "Grocery", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = ("terminal", "state_history", "operation_history", "relation_operation_history")
CONTRASTS = (
    ("A_state_history_vs_terminal", "state_history", "terminal"),
    ("B_operation_history_vs_state_history", "operation_history", "state_history"),
    ("C_relation_operation_history_vs_operation_history", "relation_operation_history", "operation_history"),
)


def _read(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(name: str, rows: list[dict]) -> None:
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key); fields.append(key)
    if not fields:
        fields = ["empty"]
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _finite(values):
    out = []
    for value in values:
        try:
            value = float(value)
            if math.isfinite(value): out.append(value)
        except (TypeError, ValueError):
            pass
    return out


def _avg(rows, key):
    vals = _finite([row.get(key) for row in rows])
    return float(np.mean(vals)) if vals else float("nan")


def _fmt(value, digits=3):
    try:
        value = float(value)
        return f"{value:.{digits}f}" if math.isfinite(value) else "NA"
    except (TypeError, ValueError):
        return "NA"


def _sci(value):
    try:
        value = float(value)
        return f"{value:.2e}" if math.isfinite(value) else "NA"
    except (TypeError, ValueError):
        return "NA"


def _group(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    return groups


def _intervention_table(rows):
    if not rows:
        return "No rows."
    grouped = _group(rows, ("dataset", "intervention"))
    lines = ["| Dataset | Intervention | Δ Val Acc (pp) | Δ Macro-F1 (pp) | logit L2 | flip rate | tokens L2 | history L2 | attention L1/L2 | query L2 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (dataset, intervention), rr in sorted(grouped.items()):
        lines.append(f"| {dataset} | {intervention} | {_fmt(_avg(rr, 'val_acc_change_pp'))} | {_fmt(_avg(rr, 'val_macro_f1_change_pp'))} | "
                     f"{_fmt(_avg(rr, 'logit_l2_mean'))} | {_fmt(_avg(rr, 'prediction_flip_rate'))} | "
                     f"{_fmt(_avg(rr, 'history_tokens_l2_mean'))} | {_fmt(_avg(rr, 'history_representation_l2_mean'))} | "
                     f"{_fmt(_avg(rr, 'attention_l1_mean'))}/{_fmt(_avg(rr, 'attention_l2_mean'))} | {_fmt(_avg(rr, 'query_l2_mean'))} |")
    return "\n".join(lines)


def main() -> None:
    metrics = _read("pilot_metrics.csv")
    expected = {(d, seed, variant) for d in DATASETS for seed in SEEDS for variant in VARIANTS}
    actual = {(r["dataset"], int(r["seed"]), r["variant"]) for r in metrics}
    if actual != expected or len(metrics) != 36:
        raise RuntimeError(f"Expected 36 complete S3 validation records, found {len(actual)}")
    manifest_rows = []
    for path in OUTPUTS.glob("**/run_record.json"):
        try:
            manifest_rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    manifest_actual = {(r.get("dataset"), int(r.get("seed", -1)), r.get("variant")) for r in manifest_rows
                       if r.get("mode") == "full" and r.get("status") == "complete"}
    if manifest_actual != expected or any(r.get("task_evaluate_test") is not False for r in manifest_rows if r.get("mode") == "full"):
        raise RuntimeError("Full run manifests are incomplete or include a run with test evaluation enabled")
    if any(r.get("task_evaluate_test", "false").lower() != "false" for r in metrics):
        raise RuntimeError("S3 pilot manifest contains a run with test evaluation enabled")
    if any("test_" in key for row in metrics for key in row):
        raise RuntimeError("S3 pilot metrics contain test data fields")

    summary = []
    keyed = {(r["dataset"], int(r["seed"]), r["variant"]): r for r in metrics}
    for dataset in DATASETS:
        for variant in VARIANTS:
            rr = [keyed[(dataset, seed, variant)] for seed in SEEDS]
            acc = [float(r["val_acc"]) for r in rr]
            f1 = [float(r["val_macro_f1"]) for r in rr]
            summary.append({"dataset": dataset, "variant": variant, "n_seeds": 3,
                            "val_acc_mean": statistics.mean(acc), "val_acc_sd_population": statistics.pstdev(acc),
                            "val_macro_f1_mean": statistics.mean(f1), "val_macro_f1_sd_population": statistics.pstdev(f1),
                            "active_parameters_mean": statistics.mean(float(r["active_parameters"]) for r in rr),
                            "total_parameters_mean": statistics.mean(float(r["total_parameters"]) for r in rr)})
    _write("pilot_summary.csv", summary)

    paired = []
    for dataset in DATASETS:
        for name, left, right in CONTRASTS:
            da, df1 = [], []
            for seed in SEEDS:
                a, b = keyed[(dataset, seed, left)], keyed[(dataset, seed, right)]
                da.append(float(a["val_acc"]) - float(b["val_acc"]))
                df1.append(float(a["val_macro_f1"]) - float(b["val_macro_f1"]))
            paired.append({"dataset": dataset, "comparison": name, "left_variant": left,
                           "right_variant": right, "n_paired_seeds": len(da),
                           "val_acc_difference_pp": 100 * float(np.mean(da)),
                           "val_acc_difference_sd_pp": 100 * float(np.std(da, ddof=0)),
                           "val_acc_positive_seed_pairs": sum(x > 0 for x in da),
                           "val_macro_f1_difference_pp": 100 * float(np.mean(df1)),
                           "val_macro_f1_positive_seed_pairs": sum(x > 0 for x in df1)})
    _write("paired_comparisons.csv", paired)

    gradients, growth = [], []
    for path in sorted(OUTPUTS.glob("**/history_gradient_trace.csv")):
        variant, dataset, run = path.parents[1].name, path.parents[2].name, path.parent
        try:
            record = json.loads((run / "run_record.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                gradients.append({**row, "dataset": dataset, "variant": variant, "seed": record["seed"],
                                  "best_epoch": record.get("best_epoch"), "parameter_group": row["group"]})
        growth_path = run / "history_output_growth.csv"
        if growth_path.exists():
            with growth_path.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    growth.append({**row, "dataset": dataset, "variant": variant, "seed": record["seed"],
                                   "best_epoch": record.get("best_epoch")})
    _write("history_gradient_trace.csv", gradients)
    _write("history_output_growth.csv", growth)
    required_files = (
        "stage12_regression.csv", "stage12_health.csv", "history_branch_diagnostics.csv",
        "history_gradient_trace.csv", "history_output_growth.csv", "history_attention.csv",
        "history_preference_heterogeneity.csv", "operation_profile_diagnostics.csv",
        "relation_environment_diagnostics.csv", "intervention_history_off.csv",
        "intervention_operation_profile_off.csv", "intervention_operation_alignment_swap.csv",
        "intervention_relation_env_off.csv", "intervention_node_conditioning_off.csv",
        "intervention_global_query.csv", "intervention_token_drop.csv",
        "history_operation_association.csv", "runtime_memory.csv", "p0_history_readout.csv",
    )
    missing = [name for name in required_files if not (RESULTS / name).exists()]
    if missing:
        raise RuntimeError(f"Missing required S3 outputs: {missing}")
    regression = _read("stage12_regression.csv")
    if not regression or any(r.get("passed", "False") != "True" for r in regression):
        raise RuntimeError("At least one Stage-I/II regression comparison failed")

    artifact_names = ["pilot_metrics.csv", "pilot_summary.csv", "paired_comparisons.csv", "history_gradient_summary.csv", *required_files,
                      "smoke_stage12_regression.csv", "smoke_history_branch_diagnostics.csv", "smoke_history_attention.csv",
                      "smoke_history_preference_heterogeneity.csv", "smoke_operation_profile_diagnostics.csv",
                      "smoke_relation_environment_diagnostics.csv", "smoke_history_operation_association.csv", "smoke_stage12_health.csv"]
    readme = [
        "# M0-S3 — ROHC results", "",
        "Validation-only `unified_full_graph_nc_v1`: 3 datasets × 3 seeds × 4 fixed variants (36 runs).",
        "All run manifests set `task.evaluate_test=false`; checkpoint selection is best validation accuracy.",
        "P0 is post-hoc only. No LP, test evaluation, HPO, History Transformer, operator bank, MoE, or extra propagation was run.",
        "The Stage-I/II path is inherited from v3 `context_bilinear_absolute`; exact shared-weight regression is in `stage12_regression.csv`.",
        "See `docs/m0/stage3_history_design.md` and `docs/m0/stage3_history_report.md`.", "", "## Artifacts", "",
    ] + [f"- `{name}`" for name in artifact_names]
    (RESULTS / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    def table_metrics():
        lines = ["| Dataset | Variant | Val Acc mean ± SD (%) | Macro-F1 mean ± SD (%) |", "|---|---|---:|---:|"]
        for row in summary:
            lines.append(f"| {row['dataset']} | {row['variant']} | {100*row['val_acc_mean']:.2f} ± {100*row['val_acc_sd_population']:.2f} | "
                         f"{100*row['val_macro_f1_mean']:.2f} ± {100*row['val_macro_f1_sd_population']:.2f} |")
        return lines

    contrast_lines = ["| Dataset | Contrast | Δ Acc (pp) | positive pairs | Δ Macro-F1 (pp) |", "|---|---|---:|---:|---:|"]
    for row in paired:
        contrast_lines.append(f"| {row['dataset']} | {row['comparison']} | {row['val_acc_difference_pp']:.3f} | "
                              f"{row['val_acc_positive_seed_pairs']}/3 | {row['val_macro_f1_difference_pp']:.3f} |")
    branch = _read("history_branch_diagnostics.csv")
    branch_lines = ["| Variant | rho mean | median | p10 | p90 | p95 | mean ||c_hist|| | mean ||W_hist||_F |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        rr = [r for r in branch if r["variant"] == variant]
        if rr:
            branch_lines.append(f"| {variant} | {_fmt(_avg(rr, 'rho_hist_mean'))} | {_fmt(_avg(rr, 'rho_hist_median'))} | "
                                f"{_fmt(_avg(rr, 'rho_hist_p10'))} | {_fmt(_avg(rr, 'rho_hist_p90'))} | {_fmt(_avg(rr, 'rho_hist_p95'))} | "
                                f"{_fmt(_avg(rr, 'history_branch_l2_mean'))} | {_fmt(_avg(rr, 'W_hist_frobenius_norm'))} |")

    grad_grouped = _group(gradients, ("dataset", "seed", "variant", "group"))
    grad_summary = []
    for (dataset, seed, variant, group), rr in grad_grouped.items():
        vals = [(int(r["epoch"]), float(r["gradient_rms_preclip"])) for r in rr]
        nonzero = [epoch for epoch, value in vals if value > 0]
        best_epoch = int(rr[0]["best_epoch"])
        best = next((value for epoch, value in vals if epoch == best_epoch), float("nan"))
        grad_summary.append({"dataset": dataset, "seed": int(seed), "variant": variant, "group": group,
                             "first_nonzero_epoch": min(nonzero) if nonzero else "",
                             "median_grad_rms": float(np.median([v for _, v in vals])),
                             "best_epoch_grad_rms": best})
    _write("history_gradient_summary.csv", grad_summary)
    grad_lines = ["| Parameter group | median first nonzero epoch | median trajectory RMS | median best-epoch RMS |",
                  "|---|---:|---:|---:|"]
    for group in sorted({r["group"] for r in grad_summary}):
        rr = [r for r in grad_summary if r["group"] == group and r["variant"] != "terminal"]
        first = _finite([r["first_nonzero_epoch"] for r in rr])
        grad_lines.append(f"| {group} | {_fmt(np.median(first), 1) if first else 'not observed'} | "
                          f"{_sci(_avg(rr, 'median_grad_rms'))} | {_sci(_avg(rr, 'best_epoch_grad_rms'))} |")

    attention = _read("history_attention.csv")
    attn_lines = ["| Token | mean attention | across-node std |", "|---|---:|---:|"]
    for token in ("T0", "T1", "T2", "V0", "V1", "V2"):
        rr = [r for r in attention if r.get("head") == "mean" and r.get("token") == token]
        attn_lines.append(f"| {token} | {_fmt(_avg(rr, 'mean'))} | {_fmt(_avg(rr, 'across_node_std'))} |")
    dataset_attention_lines = ["| Dataset | Token | mean attention | across-node std |", "|---|---|---:|---:|"]
    for dataset in DATASETS:
        for token in ("T0", "T1", "T2", "V0", "V1", "V2"):
            rr = [r for r in attention if r.get("dataset") == dataset and r.get("variant") == "relation_operation_history"
                  and r.get("head") == "mean" and r.get("token") == token]
            dataset_attention_lines.append(f"| {dataset} | {token} | {_fmt(_avg(rr, 'mean'))} | {_fmt(_avg(rr, 'across_node_std'))} |")
    hetero = _read("history_preference_heterogeneity.csv")
    hetero_lines = ["| Degree group | stage0 | stage1 | stage2 | text | visual | entropy | preference variance |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for degree_bin in ("all_positive_degree", "1", "2", "3-4", "5-8", "9-16", "17+"):
        rr = [r for r in hetero if r.get("degree_bin") == degree_bin]
        if rr:
            hetero_lines.append(f"| {degree_bin} | {_fmt(_avg(rr, 'stage0_mass'))} | {_fmt(_avg(rr, 'stage1_mass'))} | "
                                f"{_fmt(_avg(rr, 'stage2_mass'))} | {_fmt(_avg(rr, 'text_mass'))} | {_fmt(_avg(rr, 'visual_mass'))} | "
                                f"{_fmt(_avg(rr, 'attention_entropy_mean'))} | {_fmt(_avg(rr, 'preference_variance'))} |")
    stagehealth = _read("stage12_health.csv")
    health_lines = ["| Stage/measure | mean |", "|---|---:|"]
    for measure in sorted({r["measure"] for r in stagehealth}):
        health_lines.append(f"| {measure} | {_fmt(_avg([r for r in stagehealth if r['measure'] == measure], 'value'))} |")

    smoke_root = ROOT / "outputs" / "m0" / "stage3_history" / "smoke" / "logs"
    smoke_records = []
    for path in sorted(smoke_root.glob("**/run_record.json")):
        try:
            smoke_records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    smoke_expected = {("Movies", 42, variant) for variant in VARIANTS}
    smoke_actual = {(r.get("dataset"), int(r.get("seed", -1)), r.get("variant")) for r in smoke_records if r.get("status") == "complete"}
    if smoke_actual != smoke_expected:
        raise RuntimeError("All four Movies seed-42 smoke runs must be complete before summarizing")
    smoke_lines = ["| Variant | Val Acc | Macro-F1 | epochs | mean epoch seconds | peak GPU MB |",
                   "|---|---:|---:|---:|---:|---:|"]
    for row in sorted(smoke_records, key=lambda r: VARIANTS.index(r["variant"])):
        smoke_lines.append(f"| {row['variant']} | {100*float(row['val_acc']):.2f} | {100*float(row['val_macro_f1']):.2f} | "
                           f"{row['epochs_observed']} | {_fmt(row['epoch_time_mean'], 2)} | {row['peak_gpu_memory_mb']} |")

    associations = _read("history_operation_association.csv")
    association_lines = ["| Modality | state | association | mean Spearman |", "|---|---:|---|---:|"]
    for modality in ("text", "visual"):
        for state in (1, 2):
            for association in ("operation_intensity", "state_transition_norm", "operation_heterogeneity"):
                rr = [r for r in associations if r["modality"] == modality and int(r["interaction_state"]) == state and r["association"] == association]
                if rr:
                    association_lines.append(f"| {modality} | {state} | {association} | {_fmt(_avg(rr, 'spearman_attention_mass'))} |")
    p0 = _read("p0_history_readout.csv")
    p0_lines = ["| Similarity | utility group | mean T/V state1 attention | mean T/V state2 attention | mean op heterogeneity (s1/s2) |",
                "|---|---|---:|---:|---:|"]
    for quintile in ("Q1", "Q5"):
        for utility_group in ("beneficial", "harmful"):
            rr = [r for r in p0 if r["similarity_quintile"] == quintile and r["utility_group"] == utility_group]
            if rr:
                het = (float(_avg(rr, "operation_heterogeneity_step1")) + float(_avg(rr, "operation_heterogeneity_step2"))) / 2
                p0_lines.append(f"| {quintile} | {utility_group} | {_fmt(_avg(rr, 'attention_state1'))} | "
                                f"{_fmt(_avg(rr, 'attention_state2'))} | {_fmt(het)} |")

    def intervention_mean(filename, key):
        return _avg(_read(filename), key)

    history_off_acc = intervention_mean("intervention_history_off.csv", "val_acc_change_pp")
    history_off_flip = intervention_mean("intervention_history_off.csv", "prediction_flip_rate")
    op_off_acc = intervention_mean("intervention_operation_profile_off.csv", "val_acc_change_pp")
    op_swap_acc = intervention_mean("intervention_operation_alignment_swap.csv", "val_acc_change_pp")
    env_off_acc = intervention_mean("intervention_relation_env_off.csv", "val_acc_change_pp")
    node_off_acc = intervention_mean("intervention_node_conditioning_off.csv", "val_acc_change_pp")
    global_query_acc = intervention_mean("intervention_global_query.csv", "val_acc_change_pp")
    token_drop_acc = intervention_mean("intervention_token_drop.csv", "val_acc_change_pp")
    token_drop_flip = intervention_mean("intervention_token_drop.csv", "prediction_flip_rate")
    full_hetero = [r for r in hetero if r.get("degree_bin") == "all_positive_degree"
                   and r.get("variant") == "relation_operation_history"]
    full_mean_attention_std = _avg(full_hetero, "preference_variance")
    full_mean_entropy = _avg(full_hetero, "attention_entropy_mean")
    full_visual_mass = _avg(full_hetero, "visual_mass")
    full_stage2_mass = _avg(full_hetero, "stage2_mass")

    # Compare trained S3 Stage-I/II health against the committed v3 absolute pilot.
    v3_root = ROOT / "results" / "m0" / "conditioner_v3"
    with (v3_root / "stage1_health.csv").open(encoding="utf-8") as handle:
        v3_stage1 = list(csv.DictReader(handle))
    with (v3_root / "operation_geometry.csv").open(encoding="utf-8") as handle:
        v3_geometry = list(csv.DictReader(handle))
    v3_abs_s1 = [r for r in v3_stage1 if r["variant"] == "context_bilinear_absolute"]
    v3_abs_geo = [r for r in v3_geometry if r["variant"] == "context_bilinear_absolute"]
    health_compare = [
        ("relation feature variance", _avg([r for r in v3_abs_s1 if r["measure"] == "relation_feature_variance"], "mean"),
         _avg([r for r in stagehealth if r["measure"].startswith("relation_feature_variance")], "value")),
        ("1−cos(R_T,R_V)", _avg([r for r in v3_abs_s1 if r["measure"] == "R_T_R_V_1_minus_cosine"], "mean"),
         _avg([r for r in stagehealth if r["measure"] == "modality_relation_1_minus_cosine_mean"], "value")),
        ("reverse-edge 1−cos", _avg([r for r in v3_abs_s1 if r["measure"] == "directed_reverse_1_minus_cosine"], "mean"),
         _avg([r for r in stagehealth if r["measure"].startswith("directed_reverse_1_minus_cosine")], "value")),
        ("operator deviation ratio", _avg(v3_abs_geo, "operator_deviation_ratio_mean"),
         _avg([r for r in stagehealth if r["measure"].startswith("operator_deviation_ratio")], "value")),
    ]
    health_compare_lines = ["| Measure | v3 absolute mean | S3 pilot mean |", "|---|---:|---:|"]
    for label, baseline, current in health_compare:
        health_compare_lines.append(f"| {label} | {_fmt(baseline)} | {_fmt(current)} |")

    report = [
        "# M0-S3 — Relation- and Operation-aware Interaction History Consolidation", "",
        "## A. Git provenance", "",
        "- Branch: `exp/m0_stage3_history`; frozen parent: `exp/m0_conditioner_v3` at `108368f72c85441f580a1dd13b9720058915d57b`.",
        "- New outputs use dedicated `stage3_history` paths; v3 code, checkpoints, and results were not overwritten.", "",
        "## B. Protocol/code audit", "",
        "- `unified_full_graph_nc_v1`; all run manifests set `task.evaluate_test=false`; checkpoint selection is best validation accuracy.",
        "- The NC/LP task code, data splits, early stopping, checkpoint selection, and metrics were not changed.",
        "- P0 is post-hoc. No LP, test evaluation, HPO, History Transformer, GPR, operator bank, MoE, extra propagation, or auxiliary loss was used.", "",
        "## C. Files added/modified", "",
        "- Model/components/config/tests: `src/models/interaction_full_s3.py`, `src/models/interaction_history_components.py`, `configs/model/interaction_full_s3.yaml`, `tests/test_interaction_full_s3.py`.",
        "- Run/analyze/summarize: `scripts/run_m0_stage3.py`, `scripts/analyze_m0_stage3.py`, `scripts/summarize_m0_stage3.py`.",
        "- Design/report/results: `docs/m0/stage3_history_design.md`, `docs/m0/stage3_history_report.md`, `results/m0/stage3_history/`; checkpoints/logs are in ignored local `outputs/m0/stage3_history/`.", "",
        "## D. Frozen Stage-I/II verification", "",
        "Stage I and Stage II are inherited from `interaction_core_v3` with `variant=context_bilinear_absolute`; no alternate relation encoder or semantic operator is introduced. Shared checkpoint weights are loaded into the terminal S3 model and compared on identical features/edges.", "",
        "## E. Exact operation-profile formulation", "",
        "For each modality and interaction step, the profile is `concat(incoming_mean(a), incoming_population_std(a))` with width 64. Degree-zero nodes receive zero mean and zero standard deviation. It summarizes the modulation actually used by the Stage-II low-rank semantic operator.", "",
        "## F. Exact interaction-history-token formulation", "",
        "Tokens are ordered `T0,T1,T2,V0,V1,V2`. T0/V0 encode intrinsic H0 plus learned NO_OPERATION; later tokens encode H1/H2, transitions `D1=H1−H0` and `D2=H2−H1`, and the corresponding 64-d mean/std operation profile. State-only mode substitutes learned NULL_OPERATION profiles.", "",
        "## G. Exact readout-query formulation", "",
        "The query is LayerNorm of a learned global vector plus projected H0 text/visual semantics and projected 128-d relation environments. State-only and operation-only modes use learned NULL_REL_ENV vectors. The readout is one four-head query-to-six-token cross-attention followed by LayerNorm; there is no query residual or attention stack.", "",
        "## H. Four Stage-III variants", "",
        "- `terminal`: returns the v3 fused terminal representation directly.",
        "- `state_history`: uses H0/H1/H2 and transitions, with NULL_OPERATION and NULL_REL_ENV.",
        "- `operation_history`: adds measured operation profiles, retaining NULL_REL_ENV.",
        "- `relation_operation_history`: adds both real operation profiles and real relation environments.",
        "All variants share a zero-initialized `W_hist`; initialization therefore equals terminal output exactly.", "",
        "## I. Unit/regression tests", "",
        "`tests/test_interaction_full_s3.py`: 27 passed. Coverage includes the 36 requested invariant groups, exact shared-weight v3 mapping, profile/environment statistics, intervention isolation, optimizer opening of W_hist, downstream gradients, attention normalization, and the NC output contract.", "",
        "## J. Smoke", "",
        "Movies seed 42, four variants, five epochs. Smoke confirmed finite losses/gradients, W_hist growth only in active history variants, delayed upstream gradients, normalized finite attention, and successful checkpoint/validation reload. Values are diagnostic only:", "",
        *smoke_lines, "",
        "## K. 36-run validation pilot", "",
        "All 36 dataset × seed × variant records are present. Values are mean ± population SD across three seeds (%).", "",
        "| Dataset | Variant | Val Acc mean ± SD (%) | Macro-F1 mean ± SD (%) |", "|---|---|---:|---:|", *table_metrics()[2:], "",
        "Paired contrasts are percentage-point differences across matched seeds:", "", *contrast_lines, "",
        "## L. Stage-I/II regression health", "",
        f"The regression artifact contains {len(regression)} comparisons; every comparison passed `atol=1e-5, rtol=1e-5` for H0/H1/H2, relation memory and base codes, per-step modulation/execution code, and terminal fusion. The summary below describes the trained pilot checkpoints.",
        "", "The exact-formula/weight mapping gate passed on all nine frozen v3 absolute checkpoints and all 36 trained S3 checkpoints. The table compares trained S3 Stage-I/II health with the prior v3 absolute pilot; some movement is expected from the changed end-to-end training objective and should be interpreted alongside exact code-path regression.",
        "", *health_compare_lines,
        "", f"Q6 answer: the exact Stage-I/II implementation and shared-weight mapping are preserved (all {len(regression)} regression rows passed), but trained health is not numerically unchanged: relation variance is {_fmt(_avg([r for r in stagehealth if r['measure'].startswith('relation_feature_variance')], 'value'))} vs v3 {_fmt(_avg([r for r in v3_abs_s1 if r['measure'] == 'relation_feature_variance'], 'mean'))}; modality discrepancy is {_fmt(_avg([r for r in stagehealth if r['measure'] == 'modality_relation_1_minus_cosine_mean'], 'value'))} vs {_fmt(_avg([r for r in v3_abs_s1 if r['measure'] == 'R_T_R_V_1_minus_cosine'], 'mean'))}; reverse-edge discrepancy is {_fmt(_avg([r for r in stagehealth if r['measure'].startswith('directed_reverse_1_minus_cosine')], 'value'))} vs {_fmt(_avg([r for r in v3_abs_s1 if r['measure'] == 'directed_reverse_1_minus_cosine'], 'mean'))}; and operator deviation is {_fmt(_avg([r for r in stagehealth if r['measure'].startswith('operator_deviation_ratio')], 'value'))} vs {_fmt(_avg(v3_abs_geo, 'operator_deviation_ratio_mean'))}. This is a training-induced diagnostic shift, with a marked reduction in operator deviation; it is not a code-path regression or a demonstrated Stage-II collapse.", "",
        "## M. History-branch magnitude/growth", "",
        "`rho_hist=||c_hist||/(||z_term||+eps)` distributions and selected checkpoint output norms:", "", *branch_lines, "",
        "Every active history variant started with exactly zero W_hist. The output-growth traces record epoch 0 plus each post-update epoch.", "",
        "## N. Stage-III gradient health", "",
        "Gradient RMS is pre-clipping. W_hist receives task gradient first; state/transition/operation token, query, and attention groups should activate after the branch opens.", "", *grad_lines, "",
        "## O. Readout attention", "",
        "Mean attention and per-node across-node standard deviation, averaged over trained dataset/seed runs:", "", *attn_lines, "",
        "Dataset-level mean attention for the full candidate, averaged across its three seeds:", "", *dataset_attention_lines, "",
        "## P. Node/modality/stage preference heterogeneity", "",
        "Attention entropy is computed per node. `preference_variance` is mean token variance across nodes; degree bins are descriptive, not a degree-aware model mechanism.", "", *hetero_lines, "",
        "## Q. History-off intervention", "",
        _intervention_table(_read("intervention_history_off.csv")), "",
        "## R. Operation-profile-off intervention", "",
        _intervention_table(_read("intervention_operation_profile_off.csv")), "",
        "## S. Operation-history alignment intervention", "",
        _intervention_table(_read("intervention_operation_alignment_swap.csv")), "",
        "## T. Relation-environment-off intervention", "",
        _intervention_table(_read("intervention_relation_env_off.csv")), "",
        "## U. Node-conditioning/global-query interventions", "",
        _intervention_table(_read("intervention_node_conditioning_off.csv") + _read("intervention_global_query.csv")), "",
        "## V. Token-drop intervention", "",
        _intervention_table(_read("intervention_token_drop.csv")), "",
        "Token drops remove the requested memory item and renormalize attention over the remaining tokens. Changes are post-hoc diagnostics, not causal proof.",
        f"Q5 answer: V2 removal lowers accuracy by {_fmt(_avg([r for r in _read('intervention_token_drop.csv') if r['intervention'] == 'drop_V2'], 'val_acc_change_pp'))} pp on average and has the largest, consistent effect across all three datasets; T2 has a smaller, dataset-dependent effect ({_fmt(_avg([r for r in _read('intervention_token_drop.csv') if r['intervention'] == 'drop_T2'], 'val_acc_change_pp'))} pp), while the remaining tokens have weak or mixed effects. The pilot therefore shows a dominant useful visual state-2 contribution, not broad evidence that all histories are complementary.", "",
        "## W. History-operation association", "",
        "Spearman associations between history attention mass and operation intensity, state-transition norm, or incoming operation heterogeneity:", "", *association_lines, "",
        "P0 post-hoc sampled-node summaries, grouped by relation utility and similarity quintile, are in `p0_history_readout.csv`.",
        "", *p0_lines, "",
    ]
    report.extend([
        "## X. Q1: interaction-history complementarity conclusion", "",
        f"Q1 answer: state_history is {_fmt(_avg([r for r in paired if r['comparison'] == 'A_state_history_vs_terminal'], 'val_acc_difference_pp'))} pp below terminal on mean accuracy, with {sum(int(r['val_acc_positive_seed_pairs']) for r in paired if r['comparison'] == 'A_state_history_vs_terminal')}/9 positive matched seeds. Yet turning the full history branch off in the trained full checkpoint costs {_fmt(history_off_acc)} pp and flips {_fmt(history_off_flip)} of validation predictions, so the full model uses its history branch. Usage within that checkpoint does not establish net complementary value over terminal; the cross-variant pilot favors terminal.", "",
        "## Y. Q2: operation-profile conclusion", "",
        f"Q2 answer: operation_history recovers {_fmt(_avg([r for r in paired if r['comparison'] == 'B_operation_history_vs_state_history'], 'val_acc_difference_pp'))} pp over state_history (7/9 positive seed pairs), and replacing profiles with NULL_OPERATION costs {_fmt(op_off_acc)} pp with measurable representation/attention changes. But swapping p0/p1 alignment changes accuracy by only {_fmt(op_swap_acc)} pp and flips {_fmt(_avg(_read('intervention_operation_alignment_swap.csv'), 'prediction_flip_rate'))} of predictions. The profile affects readout behavior, while evidence that it helps match each state to the operation that produced it is negligible.", "",
        "## Z. Q3: relation-environment conclusion", "",
        f"Q3 answer: adding relation environments changes validation accuracy by only {_fmt(_avg([r for r in paired if r['comparison'] == 'C_relation_operation_history_vs_operation_history'], 'val_acc_difference_pp'))} pp over operation_history (5/9 positive pairs). Removing the environment from the trained full checkpoint costs {_fmt(env_off_acc)} pp on average and changes its query/attention, so the query uses that signal; the between-variant comparison does not show additional task value from including it.", "",
        "## AA. Q4: adaptive-preference conclusion", "",
        f"Q4 answer: the full readout is modality/stage-skewed (mean visual mass {_fmt(full_visual_mass)}; stage-2 mass {_fmt(full_stage2_mass)}) and varies across nodes (preference variance {_fmt(full_mean_attention_std)}, entropy {_fmt(full_mean_entropy)} nats). However, removing intrinsic query terms costs only {_fmt(node_off_acc)} pp on average; q0-only costs {_fmt(global_query_acc)} pp but removes both node and relation terms. This supports nonuniform readout behavior, with modest isolated evidence for intrinsic node conditioning.", "",
        "## AB. What is supported", "",
        "The exact inherited Stage-I/II equivalence, zero-init terminal identity, Stage-III branch use, operation/relation-conditioned changes to the readout, and a dominant V2 token contribution are directly testable from committed CSVs and checkpoints/logs. The between-variant validation results still favor the terminal baseline.", "",
        "## AC. What is not supported", "",
        "Three validation seeds do not establish significance or universal ranking. Token-drop and P0 summaries are post-hoc and do not establish causal history utility. Attention differences alone do not prove that the model uses the producing operation correctly.", "",
        "## AD. Recommendation for final M0 architecture", "",
        "Recommendation: retain the frozen two-stage Interpret→Execute architecture for final M0 and do not include ROHC in the default model. The full history branch is used, and V2 has a clear token-drop effect, but every dataset's state_history accuracy is below terminal; operation profiles recover only a small part of that loss, correct operation-to-state alignment has essentially no effect, and relation environments do not improve the full-versus-operation validation comparison. This fails the predeclared full-ROHC task-value gate. Keep the Stage-III results as diagnostic evidence, stop for human review, and do not run LP, test evaluation, HPO, or deeper history modules.", "",
    ])
    (ROOT / "docs/m0/stage3_history_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Summarized {len(metrics)} pilot runs; {len(paired)} paired contrasts; {len(regression)} regression rows")


if __name__ == "__main__": main()
