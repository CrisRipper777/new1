from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POINT_METRICS = (
    "low_relation_count",
    "high_relation_count",
    "low_mean_ce",
    "high_mean_ce",
    "delta_mean_ce",
    "low_beneficial_rate_ce",
    "high_beneficial_rate_ce",
    "delta_beneficial_rate_ce",
    "low_harmful_rate_ce",
    "high_harmful_rate_ce",
    "delta_harmful_rate_ce",
    "low_mean_margin",
    "high_mean_margin",
    "delta_mean_margin",
    "low_beneficial_rate_margin",
    "high_beneficial_rate_margin",
    "delta_beneficial_rate_margin",
    "low_harmful_rate_margin",
    "high_harmful_rate_margin",
    "delta_harmful_rate_margin",
)
DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
MODALITIES = ("Text", "Visual")
ANALYSES = ("recipient_redundancy", "cross_modal_disagreement")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _cross_seed_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["modality"], row["analysis"], row["similarity_bin"])].append(row)
    result: list[dict[str, Any]] = []
    for key in sorted(grouped):
        members = grouped[key]
        out: dict[str, Any] = {
            "dataset": key[0],
            "modality": key[1],
            "analysis": key[2],
            "similarity_bin": key[3],
            "seed_count": len({int(row["seed"]) for row in members}),
            "aggregation": "mean and population SD over model seeds; not a pooled bootstrap CI",
        }
        for metric in POINT_METRICS:
            values = np.asarray([float(row[metric]) for row in members], dtype=np.float64)
            finite = values[np.isfinite(values)]
            out[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            out[f"{metric}_seed_sd"] = float(np.std(finite, ddof=0)) if finite.size else float("nan")
        result.append(out)
    return result


def _make_plot(dataset: str, rows: list[dict[str, str]], plot_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True, constrained_layout=True)
    bins = ("Q1", "Q2", "Q3", "Q4", "Q5")
    styles = {
        "recipient_redundancy": ("#087e68", "Recipient context redundancy (primary)", "o", "-") ,
        "cross_modal_disagreement": ("#d97706", "Cross-modal disagreement (secondary)", "s", "--"),
    }
    for ax, modality in zip(axes, MODALITIES, strict=True):
        for analysis in ANALYSES:
            color, label, marker, linestyle = styles[analysis]
            means: list[float] = []
            seed_sds: list[float] = []
            for q in bins:
                values = [
                    float(row["delta_mean_ce"])
                    for row in rows
                    if row["modality"] == modality and row["analysis"] == analysis and row["similarity_bin"] == q
                ]
                means.append(float(np.mean(values)) if values else float("nan"))
                seed_sds.append(float(np.std(values, ddof=0)) if values else 0.0)
            ax.errorbar(
                np.arange(1, 6), means, yerr=seed_sds, color=color, label=label,
                marker=marker, linestyle=linestyle, linewidth=1.8, capsize=3,
            )
        ax.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.7)
        ax.set_title(modality)
        ax.set_xticks(np.arange(1, 6), bins)
        ax.set_xlabel("Probe-similarity quintile")
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Low − high mean CE utility")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(f"{dataset}: context-conditioned utility (mean ± seed SD)")
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"p02_{dataset}_conditional_effects.png", dpi=220)
    fig.savefig(plot_dir / f"p02_{dataset}_conditional_effects.pdf")
    plt.close(fig)


def _write_readme(output_root: Path, cross_rows: list[dict[str, Any]]) -> None:
    readme = """# P0.2 Problem Validation

This directory contains an offline analysis of the fixed P0.1 validation-relation population. No model was trained, no relation was re-sampled, and no test labels or metrics were read.

## Analysis

- Input artifacts: the 15 `edge_analysis.pt`, `semantic_embeddings.pt`, and `splits/<dataset>.pt` files under `/hdd1/DataInHere/YHF/mag_model/outputs/problem_validation/`.
- Each split cache's stored target and directed relation arrays were checked for exact sequence equality against `edge_analysis.pt`.
- The current `new1` physical graph was checked against the frozen target degrees and directed relation endpoints.
- Recipient redundancy uses `cos(H_j^m, (d_i N_i^m - H_j^m)/(d_i-1))`; `d_i=1` rows are marked undefined and excluded only from this primary analysis. Coverage is in `p02_context_coverage.csv`.
- For each dataset, seed, and modality, P0.1+ stable-mergesort balanced quintiles of probe similarity are reused. Within each quintile, stable-mergesort balanced halves define low/high descriptor groups.
- Primary descriptor: recipient context redundancy. Secondary descriptor: absolute difference between text and visual probe similarities.
- CE utility is primary; margin utility is a robustness view. Differences are low descriptor minus high descriptor, with no assumed direction.
- Confidence intervals use 1,000 target-node bootstrap replicates (seed 42 by default); all sampled relations for a selected target are resampled together. Per-seed intervals are reported in `p02_per_seed.csv`.

## Files

- `p02_per_seed.csv`: Q1–Q5 and relation-count-weighted overall point estimates with per-seed node-bootstrap 95% CIs.
- `p02_cross_dataset.csv`: per-dataset means and population SDs over the three model seeds; these are not pooled inferential intervals.
- `p02_context_coverage.csv`: degree-one count/proportion and retained context-analysis coverage.
- `p02_counterexamples.csv`: descriptive redundancy and disagreement distributions in the predeclared Q1/Q5 sign subsets.
- `plots/`: per-dataset CE effect plots; error bars are across-seed SD.

## Reproduction

From the `new1` repository, run `PYTHONPATH=. python scripts/run_problem_validation_p02.py --input-root /hdd1/DataInHere/YHF/mag_model/outputs/problem_validation --split-root /hdd1/DataInHere/YHF/mag_model/outputs/problem_validation/splits`.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def summarize(output_root: Path) -> None:
    per_seed_path = output_root / "p02_per_seed.csv"
    rows = _read_csv(per_seed_path)
    cross_rows = _cross_seed_summary(rows)
    _write_csv(output_root / "p02_cross_dataset.csv", cross_rows)
    plot_dir = output_root / "plots"
    for dataset in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        if dataset_rows:
            _make_plot(dataset, dataset_rows, plot_dir)
    _write_readme(output_root, cross_rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Summarize offline P0.2 context results.")
    parser.add_argument("--output-root", type=Path, default=Path("results/problem_validation/p02"))
    args = parser.parse_args()
    summarize(args.output_root)


if __name__ == "__main__":
    main()

