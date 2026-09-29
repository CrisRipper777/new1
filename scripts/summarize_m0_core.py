from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "m0" / "interpret_execute"


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    metrics = _read(RESULTS / "pilot_metrics.csv")
    if len(metrics) != 36:
        raise RuntimeError(f"Expected 36 pilot runs; found {len(metrics)}")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_key = {}
    for row in metrics:
        groups[(row["dataset"], row["variant"])].append(row)
        by_key[(row["dataset"], int(row["seed"]), row["variant"])] = row
    summary = []
    for (dataset, variant), rows in sorted(groups.items()):
        acc = np.asarray([float(row["val_acc"]) for row in rows])
        f1 = np.asarray([float(row["val_macro_f1"]) for row in rows])
        active = np.asarray([int(row["active_parameters"]) for row in rows])
        total = np.asarray([int(row["total_parameters"]) for row in rows])
        summary.append({
            "dataset": dataset, "variant": variant, "n_seeds": len(rows),
            "val_acc_mean": float(acc.mean()), "val_acc_std_population": float(acc.std(ddof=0)),
            "val_macro_f1_mean": float(f1.mean()), "val_macro_f1_std_population": float(f1.std(ddof=0)),
            "active_parameters_mean": float(active.mean()), "total_parameters_mean": float(total.mean()),
            "best_epoch_min": min(int(row["best_epoch"]) for row in rows),
            "best_epoch_max": max(int(row["best_epoch"]) for row in rows),
        })
    _write(RESULTS / "pilot_summary.csv", summary)

    comparisons = [
        ("pair_dynamic_vs_global_dynamic", "pair_dynamic", "global_dynamic"),
        ("context_dynamic_vs_pair_dynamic", "context_dynamic", "pair_dynamic"),
        ("context_dynamic_vs_context_static", "context_dynamic", "context_static"),
    ]
    contrast_rows = []
    for dataset in ("Movies", "Grocery", "Reddit-S"):
        for label, left, right in comparisons:
            diffs_acc, diffs_f1 = [], []
            for seed in (42, 43, 44):
                a = by_key[(dataset, seed, left)]
                b = by_key[(dataset, seed, right)]
                diffs_acc.append(float(a["val_acc"]) - float(b["val_acc"]))
                diffs_f1.append(float(a["val_macro_f1"]) - float(b["val_macro_f1"]))
            contrast_rows.append({
                "dataset": dataset, "comparison": label, "left_variant": left,
                "right_variant": right, "n_paired_seeds": len(diffs_acc),
                "val_acc_difference_mean": float(np.mean(diffs_acc)),
                "val_acc_difference_pp": float(100.0 * np.mean(diffs_acc)),
                "val_acc_difference_sd": float(np.std(diffs_acc, ddof=0)),
                "val_acc_positive_seed_pairs": sum(value > 0 for value in diffs_acc),
                "val_macro_f1_difference_mean": float(np.mean(diffs_f1)),
                "val_macro_f1_difference_pp": float(100.0 * np.mean(diffs_f1)),
                "val_macro_f1_positive_seed_pairs": sum(value > 0 for value in diffs_f1),
            })
    _write(RESULTS / "paired_comparisons.csv", contrast_rows)

    artifacts = [
        "pilot_metrics.csv", "pilot_summary.csv", "paired_comparisons.csv",
        "operator_diagnostics.csv", "modulation_diagnostics.csv", "dynamicity_diagnostics.csv",
        "intervention_diagnostics.csv", "p0_operation_hardcases.csv", "runtime_memory.csv",
    ]
    text = [
        "# M0-Core pilot artifacts", "",
        "Validation-only `unified_full_graph_nc_v1` pilot: Movies, Grocery, Reddit-S;",
        "seeds 42/43/44; four fixed variants; 36 completed runs. Every run set",
        "`task.evaluate_test=false`; best checkpoints were selected by validation accuracy.",
        "No P0 artifact entered training or checkpoint selection. P0 operation rows are",
        "post-hoc descriptions only. See `docs/m0/interpret_execute_report.md` for the",
        "scientific interpretation and limits.", "", "## Files", "",
    ]
    text.extend(f"- `{name}`" for name in artifacts)
    (RESULTS / "README.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"Summarized {len(metrics)} pilot rows into {len(summary)} dataset/variant cells and {len(contrast_rows)} paired comparisons")


if __name__ == "__main__":
    main()
