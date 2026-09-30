from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.models.final_interaction import Model


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = ("full", "no_context", "shared_relation", "static_execution", "operator_off")
METRICS = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
DISPLAY = {
    "full": "Full", "no_context": "w/o Context",
    "shared_relation": "w/o Relation Specificity",
    "static_execution": "w/o State Conditioning",
    "operator_off": "w/o Semantic Operator",
}
RESULTS = ROOT / "results/final_nc"


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def _sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _fmt(mean: float, sd: float) -> str:
    return f"{mean * 100:.2f} ± {sd * 100:.2f}"


def _summaries(records: list[dict]):
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        groups[(row["dataset"], row["variant"])].append(row)
    full_rows = []
    ablation_rows = []
    by_dataset_variant = {}
    for dataset in DATASETS:
        for variant in VARIANTS:
            values = groups[(dataset, variant)]
            if len(values) != 3:
                raise ValueError(f"expected 3 seeds for {dataset}/{variant}; found {len(values)}")
            summary = {"dataset": dataset, "variant": variant, "n_seeds": len(values)}
            for metric in METRICS:
                numbers = [float(row[metric]) for row in values]
                summary[f"{metric}_mean"] = _mean(numbers)
                summary[f"{metric}_std"] = _sd(numbers)
            by_dataset_variant[(dataset, variant)] = summary
            ablation_rows.append(summary)
            if variant == "full":
                full_rows.append(summary)
    average = {"dataset": "Average", "variant": "full", "n_seeds": 15,
               "average_type": "descriptive macro-average across five datasets"}
    for metric in METRICS:
        average[f"{metric}_mean"] = _mean([row[f"{metric}_mean"] for row in full_rows])
        average[f"{metric}_std"] = _mean([row[f"{metric}_std"] for row in full_rows])
    full_rows.append(average)
    fields = ["dataset", "variant", "n_seeds", "average_type"] + [
        name for metric in METRICS for name in (f"{metric}_mean", f"{metric}_std")
    ]
    _write(RESULTS / "final_nc_full_results.csv", full_rows, fields)
    _write(RESULTS / "final_nc_ablation.csv", ablation_rows, fields)
    return full_rows, ablation_rows, by_dataset_variant


def _paired(records: list[dict]):
    lookup = {(r["dataset"], int(r["seed"]), r["variant"]): r for r in records}
    variants = ("no_context", "shared_relation", "static_execution", "operator_off")
    pair_rows, summary_rows = [], []
    for variant in variants:
        comparison = f"{variant} - full"
        for metric in METRICS:
            grouped: dict[str, list[float]] = defaultdict(list)
            for dataset in DATASETS:
                for seed in SEEDS:
                    delta = (float(lookup[(dataset, seed, variant)][metric])
                             - float(lookup[(dataset, seed, "full")][metric]))
                    grouped[dataset].append(delta)
                    grouped["ALL_15_PAIRS"].append(delta)
                    pair_rows.append({
                        "row_type": "paired_seed", "dataset": dataset, "seed": seed,
                        "comparison": comparison, "metric": metric, "delta": delta,
                        "win_tie_loss": "win" if delta > 1e-12 else ("loss" if delta < -1e-12 else "tie"),
                    })
            for dataset in (*DATASETS, "ALL_15_PAIRS"):
                values = grouped[dataset]
                summary_rows.append({
                    "row_type": "summary", "dataset": dataset, "seed": "",
                    "comparison": comparison, "metric": metric,
                    "mean": _mean(values), "sd": _sd(values),
                    "wins": sum(value > 1e-12 for value in values),
                    "ties": sum(abs(value) <= 1e-12 for value in values),
                    "losses": sum(value < -1e-12 for value in values),
                    "n_pairs": len(values),
                })
    fields = ["row_type", "dataset", "seed", "comparison", "metric", "delta",
              "win_tie_loss", "mean", "sd", "wins", "ties", "losses", "n_pairs"]
    _write(RESULTS / "paired_ablation_deltas.csv", pair_rows + summary_rows, fields)
    return pair_rows, summary_rows


def _parameter_rows(records: list[dict]):
    by_dataset = {dataset: next(r for r in records if r["dataset"] == dataset and int(r["seed"]) == 42)
                  for dataset in DATASETS}
    rows = []
    for dataset in DATASETS:
        checkpoint = torch.load(by_dataset[dataset]["checkpoint"], map_location="cpu", weights_only=False)
        data_info = checkpoint["data_info"]
        for variant in VARIANTS:
            cfg = OmegaConf.load(ROOT / "configs/model/final_interaction.yaml")
            cfg.variant = variant
            model = Model(OmegaConf.create({"model": cfg}), data_info)
            active = model.active_parameter_names()
            for name, parameter in model.named_parameters():
                rows.append({
                    "dataset": dataset, "variant": variant, "module": "model",
                    "parameter_name": name, "numel": parameter.numel(),
                    "active": name in active,
                })
            head = nn.Linear(model.out_dim, int(data_info["num_classes"]))
            for name, parameter in head.named_parameters():
                rows.append({
                    "dataset": dataset, "variant": variant, "module": "classifier",
                    "parameter_name": name, "numel": parameter.numel(), "active": True,
                })
    _write(RESULTS / "active_parameters.csv", rows,
           ["dataset", "variant", "module", "parameter_name", "numel", "active"])
    return rows


def _efficiency(records: list[dict]):
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        groups[(row["dataset"], row["variant"])].append(row)
    rows = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            values = groups[(dataset, variant)]
            rows.append({
                "dataset": dataset, "variant": variant, "n_seeds": len(values),
                "active_parameters": int(values[0]["active_parameters"]),
                "total_parameters": int(values[0]["total_parameters"]),
                "mean_epoch_time_seconds": _mean([float(row["epoch_time_mean"]) for row in values]),
                "peak_gpu_memory_mean_mb": _mean([float(row["peak_gpu_memory_mb"]) for row in values]),
                "peak_gpu_memory_max_mb": max(float(row["peak_gpu_memory_mb"]) for row in values),
                "runtime_note": "run wall time divided by observed epochs; includes startup and per-epoch validation",
            })
    _write(RESULTS / "model_efficiency.csv", rows,
           ["dataset", "variant", "n_seeds", "active_parameters", "total_parameters",
            "mean_epoch_time_seconds", "peak_gpu_memory_mean_mb", "peak_gpu_memory_max_mb", "runtime_note"])


def _gradient_sanity(records: list[dict]):
    rows = []
    for run in records:
        if run["variant"] != "full":
            continue
        trace_path = Path(run.get("gradient_trace") or (Path(run["hydra_dir"]).parent / "gradient_trace.csv"))
        with trace_path.open(encoding="utf-8", newline="") as handle:
            traces = list(csv.DictReader(handle))
        for group in ("projector", "stage1_pair", "stage1_context", "stage2_conditioner",
                      "stage2_operator", "state_update", "fusion"):
            selected = [row for row in traces if row["group"] == group]
            nonzero = [int(row["epoch"]) for row in selected if float(row["gradient_rms"]) > 0.0]
            at_best = next((float(row["gradient_rms"]) for row in selected
                            if int(row["epoch"]) == int(run["best_epoch"])), 0.0)
            rows.append({
                "dataset": run["dataset"], "seed": run["seed"], "group": group,
                "first_nonzero_epoch": min(nonzero) if nonzero else "",
                "best_checkpoint_epoch": run["best_epoch"],
                "gradient_rms_at_best_epoch": at_best,
                "nonzero_seen": bool(nonzero),
            })
    _write(RESULTS / "gradient_path_sanity.csv", rows,
           ["dataset", "seed", "group", "first_nonzero_epoch", "best_checkpoint_epoch",
            "gradient_rms_at_best_epoch", "nonzero_seen"])


def _markdown(full_rows, ablation_rows, paired_summaries):
    full = {row["dataset"]: row for row in full_rows}
    main = [
        "# Final NC Main Results",
        "",
        "Full model, validation-selected checkpoint, test evaluated after restoration. Values are mean ± sample SD across seeds 42, 43, and 44.",
        "",
        "| Dataset | Test Acc (%) | Test Macro-F1 (%) |",
        "|---|---:|---:|",
    ]
    for dataset in DATASETS:
        row = full[dataset]
        main.append(f"| {dataset} | {_fmt(row['test_acc_mean'], row['test_acc_std'])} | {_fmt(row['test_macro_f1_mean'], row['test_macro_f1_std'])} |")
    avg = full["Average"]
    main.append(f"| Average† | {_fmt(avg['test_acc_mean'], avg['test_acc_std'])} | {_fmt(avg['test_macro_f1_mean'], avg['test_macro_f1_std'])} |")
    main.extend((
        "",
        "† Descriptive macro-average of the five dataset means; the ± term is the descriptive average of the five within-dataset seed SDs. It is not a pooled uncertainty estimate or a statistical test.",
        "",
        "## Validation supplementary table",
        "",
        "| Dataset | Val Acc (%) | Val Macro-F1 (%) |",
        "|---|---:|---:|",
    ))
    for dataset in DATASETS:
        row = full[dataset]
        main.append(f"| {dataset} | {_fmt(row['val_acc_mean'], row['val_acc_std'])} | {_fmt(row['val_macro_f1_mean'], row['val_macro_f1_std'])} |")
    main.append(f"| Average† | {_fmt(avg['val_acc_mean'], avg['val_acc_std'])} | {_fmt(avg['val_macro_f1_mean'], avg['val_macro_f1_std'])} |")
    (ROOT / "docs/final_nc_main_table.md").write_text("\n".join(main) + "\n", encoding="utf-8")

    lookup = {(row["dataset"], row["variant"]): row for row in ablation_rows}
    ablation = [
        "# Final NC Ablation Results",
        "",
        "Values are mean ± sample SD across three matched seeds. The trained variants are separate fits; their differences are not same-checkpoint intervention effects.",
        "",
        "## Test metrics",
        "",
        "| Dataset | Variant | Test Acc (%) | Test Macro-F1 (%) |",
        "|---|---|---:|---:|",
    ]
    for dataset in DATASETS:
        for variant in VARIANTS:
            row = lookup[(dataset, variant)]
            ablation.append(f"| {dataset} | {DISPLAY[variant]} | {_fmt(row['test_acc_mean'], row['test_acc_std'])} | {_fmt(row['test_macro_f1_mean'], row['test_macro_f1_std'])} |")
    ablation.extend((
        "",
        "## Validation supplementary metrics",
        "",
        "| Dataset | Variant | Val Acc (%) | Val Macro-F1 (%) |",
        "|---|---|---:|---:|",
    ))
    for dataset in DATASETS:
        for variant in VARIANTS:
            row = lookup[(dataset, variant)]
            ablation.append(f"| {dataset} | {DISPLAY[variant]} | {_fmt(row['val_acc_mean'], row['val_acc_std'])} | {_fmt(row['val_macro_f1_mean'], row['val_macro_f1_std'])} |")
    ablation.append("")
    (ROOT / "docs/final_nc_ablation_table.md").write_text("\n".join(ablation), encoding="utf-8")

    _write(RESULTS / "external_baseline_template.csv", [], [
        "dataset", "model", "test_acc_mean", "test_acc_std", "test_macro_f1_mean",
        "test_macro_f1_std", "protocol_source", "result_source", "verified_same_split", "notes",
    ])
    template = (
        "# Final NC Results README\n\n"
        "This directory contains the locked five-dataset, three-seed, five-variant NC benchmark and final full-model mechanism audits. "
        "The final grid has 75 independent training runs. Full-graph training uses `unified_full_graph_nc_v1`; checkpoint selection is validation accuracy only. "
        "Each test metric was computed after restoring the selected checkpoint. No test result was used to select an epoch, variant, seed, or hyperparameter.\n\n"
        "`final_nc_metrics.csv` contains all run metrics; `validation_reproduction.csv` and `test_reproduction.csv` independently reload every checkpoint. "
        "The full-model diagnostics cover all 15 Full checkpoints. Same-checkpoint interventions are diagnostic perturbations and are not causal contribution estimates. "
        "The external-baseline template is intentionally blank pending source-verified results on the same split and protocol.\n"
    )
    (RESULTS / "README.md").write_text(template, encoding="utf-8")


def main() -> None:
    records = _read(RESULTS / "run_manifest.csv")
    if len(records) != 75:
        raise ValueError(f"expected 75 manifest rows, found {len(records)}")
    if {(row["dataset"], int(row["seed"]), row["variant"]) for row in records} != {
        (dataset, seed, variant) for dataset in DATASETS for seed in SEEDS for variant in VARIANTS
    }:
        raise ValueError("run manifest does not match the frozen 5 x 3 x 5 grid")
    full, ablation, _ = _summaries(records)
    pair_rows, paired_summaries = _paired(records)
    parameter_rows = _parameter_rows(records)
    _efficiency(records)
    _gradient_sanity(records)
    _markdown(full, ablation, paired_summaries)
    print(f"SUMMARIZED runs={len(records)} paired_deltas={len(pair_rows)} active_parameter_rows={len(parameter_rows)}")


if __name__ == "__main__":
    main()
