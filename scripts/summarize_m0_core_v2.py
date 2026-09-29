from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "m0" / "relation_grounded_v2"
VARIANTS = ("global_grounded_dynamic", "pair_grounded_dynamic",
            "context_grounded_static", "context_grounded_dynamic")
COMPARISONS = (
    ("Q1_pair_vs_global", "pair_grounded_dynamic", "global_grounded_dynamic"),
    ("Q2_context_dynamic_vs_pair", "context_grounded_dynamic", "pair_grounded_dynamic"),
    ("Q3_dynamic_vs_static", "context_grounded_dynamic", "context_grounded_static"),
)


def _read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    metrics = _read(RESULTS / "pilot_metrics.csv")
    expected = {(d, s, v) for d in ("Movies", "Grocery", "Reddit-S")
                for s in (42, 43, 44) for v in VARIANTS}
    actual = {(r["dataset"], int(r["seed"]), r["variant"]) for r in metrics}
    if actual != expected:
        raise RuntimeError(f"Expected all 36 validation-only run records; found {len(actual)}")
    by_key = {(r["dataset"], int(r["seed"]), r["variant"]): r for r in metrics}
    groups = defaultdict(list)
    for row in metrics:
        groups[(row["dataset"], row["variant"])].append(row)
    summary = []
    for (dataset, variant), rows in sorted(groups.items()):
        acc = np.asarray([float(r["val_acc"]) for r in rows])
        f1 = np.asarray([float(r["val_macro_f1"]) for r in rows])
        summary.append({
            "dataset": dataset, "variant": variant, "n_seeds": len(rows),
            "val_acc_mean": float(acc.mean()), "val_acc_sd_population": float(acc.std(ddof=0)),
            "val_macro_f1_mean": float(f1.mean()), "val_macro_f1_sd_population": float(f1.std(ddof=0)),
            "active_parameters_mean": float(np.mean([int(r["active_parameters"]) for r in rows])),
            "total_parameters_mean": float(np.mean([int(r["total_parameters"]) for r in rows])),
            "best_epoch_min": min(int(r["best_epoch"]) for r in rows),
            "best_epoch_max": max(int(r["best_epoch"]) for r in rows),
        })
    _write(RESULTS / "pilot_summary.csv", summary)

    comparisons = []
    for dataset in ("Movies", "Grocery", "Reddit-S"):
        for label, left, right in COMPARISONS:
            acc, f1 = [], []
            for seed in (42, 43, 44):
                a, b = by_key[(dataset, seed, left)], by_key[(dataset, seed, right)]
                acc.append(float(a["val_acc"]) - float(b["val_acc"]))
                f1.append(float(a["val_macro_f1"]) - float(b["val_macro_f1"]))
            comparisons.append({
                "dataset": dataset, "comparison": label, "left_variant": left,
                "right_variant": right, "n_paired_seeds": len(acc),
                "val_acc_difference_mean": float(np.mean(acc)),
                "val_acc_difference_pp": float(100 * np.mean(acc)),
                "val_acc_difference_sd": float(np.std(acc, ddof=0)),
                "val_acc_positive_seed_pairs": sum(x > 0 for x in acc),
                "val_macro_f1_difference_mean": float(np.mean(f1)),
                "val_macro_f1_difference_pp": float(100 * np.mean(f1)),
                "val_macro_f1_positive_seed_pairs": sum(x > 0 for x in f1),
            })
    _write(RESULTS / "paired_comparisons.csv", comparisons)
    files = (
        "pilot_metrics.csv", "pilot_summary.csv", "paired_comparisons.csv", "stage1_health.csv",
        "stage1_attention.csv", "training_gradient_trace.csv", "operator_growth_trace.csv",
        "execution_attention.csv", "operator_diagnostics.csv", "operation_geometry.csv",
        "dynamicity_diagnostics.csv", "intervention_relation_shuffle.csv",
        "intervention_context_shuffle.csv", "intervention_frozen_query.csv",
        "intervention_operator_off.csv", "p0_operation_hardcases.csv", "runtime_memory.csv",
    )
    readme = [
        "# M0-Core v2 results", "",
        "Validation-only full-graph NC experiment: 3 datasets × 3 seeds × 4 fixed variants (36 runs).",
        "All checkpoints are selected by validation accuracy and all runs set `task.evaluate_test=false`.",
        "Paired contrasts are descriptive across seeds; P0 operation summaries are post-hoc only and were not used in training or selection.",
        "See `docs/m0/relation_grounded_v2_report.md` for the A–AA evidence review and limitations.", "", "## Artifacts", "",
    ] + [f"- `{name}`" for name in files]
    (RESULTS / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"Summarized {len(metrics)} runs into {len(summary)} dataset/variant cells and {len(comparisons)} paired contrasts")


if __name__ == "__main__":
    main()
