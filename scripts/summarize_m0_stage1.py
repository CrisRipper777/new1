from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "m0" / "stage1_relation"


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the fixed M0-S1 pilot grid.")
    parser.add_argument("--input", type=Path, default=RESULTS / "pilot_metrics.csv")
    parser.add_argument("--runtime", type=Path, default=RESULTS / "runtime_memory.csv")
    parser.add_argument("--output", type=Path, default=RESULTS / "pilot_summary.csv")
    args = parser.parse_args()
    metrics = _read(args.input)
    runtime = _read(args.runtime)
    runtime_by_key = {(r["dataset"], r["seed"], r["variant"]): r for r in runtime}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in metrics:
        groups[(row["dataset"], row["variant"])].append(row)
    rows = []
    for (dataset, variant), group in sorted(groups.items()):
        vals = [float(r["val_acc"]) for r in group]
        f1 = [float(r["val_macro_f1"]) for r in group]
        seconds = []
        memory = []
        for item in group:
            runtime_row = runtime_by_key.get((dataset, item["seed"], variant), {})
            if runtime_row.get("epoch_time_mean") not in (None, ""):
                seconds.append(float(runtime_row["epoch_time_mean"]))
            if runtime_row.get("peak_gpu_memory_mb") not in (None, ""):
                memory.append(float(runtime_row["peak_gpu_memory_mb"]))
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "n_seeds": len(group),
                "val_acc_mean": float(np.mean(vals)),
                "val_acc_std_population": float(np.std(vals, ddof=0)),
                "val_macro_f1_mean": float(np.mean(f1)),
                "val_macro_f1_std_population": float(np.std(f1, ddof=0)),
                "epoch_time_mean_seconds": float(np.mean(seconds)) if seconds else float("nan"),
                "peak_gpu_memory_mean_mb": float(np.mean(memory)) if memory else float("nan"),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} dataset-variant summaries to {args.output}")


if __name__ == "__main__":
    main()
