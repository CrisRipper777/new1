from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "m0" / "stage1_relation"
RESULT_ROOT = ROOT / "results" / "m0" / "stage1_relation"
VARIANTS = ("generic", "pair", "pair_compat", "pair_compat_context")
DATASETS = ("Movies", "Grocery", "Reddit-S")
SEEDS = (42, 43, 44)


def _gpu_memory(gpu_id: int) -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id", str(gpu_id), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return int(result.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def _timestamp_epoch_seconds(log_path: Path) -> tuple[float, int]:
    epoch_rows: list[tuple[dt.datetime, int]] = []
    pattern = re.compile(r"^\[(.*?)\].*?Epoch\s+(\d+)")
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.search(line)
            if match:
                try:
                    epoch_rows.append((dt.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S"), int(match.group(2))))
                except ValueError:
                    continue
    count = len(epoch_rows)
    if count >= 2:
        elapsed = (epoch_rows[-1][0] - epoch_rows[0][0]).total_seconds()
        return max(elapsed / (count - 1), 0.0), count
    return 0.0, count


def _job_paths(mode: str, dataset: str, variant: str, seed: int):
    if mode == "smoke":
        checkpoint = OUTPUT_ROOT / "smoke" / "checkpoints" / dataset / variant / f"seed{seed}.pt"
        run_dir = OUTPUT_ROOT / "smoke" / "logs" / dataset / variant / f"seed{seed}"
    else:
        checkpoint = OUTPUT_ROOT / dataset / variant / f"seed{seed}.pt"
        run_dir = OUTPUT_ROOT / "logs" / dataset / variant / f"seed{seed}"
    return checkpoint, run_dir


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_records(mode: str) -> list[dict]:
    base = OUTPUT_ROOT / "smoke" / "logs" if mode == "smoke" else OUTPUT_ROOT / "logs"
    records = []
    if base.exists():
        for path in sorted(base.glob("**/run_record.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("status") == "complete":
                    records.append(record)
            except (OSError, json.JSONDecodeError):
                continue
    return records


def _export_pilot_csvs() -> None:
    records = _read_records("pilot")
    metrics_fields = ["dataset", "seed", "variant", "best_epoch", "val_acc", "val_macro_f1", "num_parameters"]
    runtime_fields = ["dataset", "seed", "variant", "epoch_time_mean", "peak_gpu_memory_mb"]
    metrics_rows = [{key: row.get(key) for key in metrics_fields} for row in records]
    runtime_rows = [{key: row.get(key) for key in runtime_fields} for row in records]
    _write_csv(RESULT_ROOT / "pilot_metrics.csv", metrics_rows, metrics_fields)
    _write_csv(RESULT_ROOT / "runtime_memory.csv", runtime_rows, runtime_fields)


def _run_one(mode: str, dataset: str, variant: str, seed: int, gpu_id: int, force: bool) -> dict:
    checkpoint, run_dir = _job_paths(mode, dataset, variant, seed)
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    record_path = run_dir / "run_record.json"
    stdout_path = run_dir / "run.stdout.log"
    hydra_dir = run_dir / "hydra"
    if not force and record_path.exists() and checkpoint.exists():
        old = json.loads(record_path.read_text(encoding="utf-8"))
        if old.get("status") == "complete":
            print(f"SKIP {mode} {dataset}/{variant}/seed{seed}", flush=True)
            return old

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    hydra_dir.mkdir(parents=True, exist_ok=True)
    epochs = 5 if mode == "smoke" else 300
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        "model=interaction_m0",
        f"model.stage1_variant={variant}",
        f"seed={seed}",
        "num_runs=1",
        f"device=cuda:{gpu_id}",
        f"task.epochs={epochs}",
        "task.evaluate_test=false",
        f"task.save_ckpt_path={checkpoint.resolve()}",
        f"hydra.run.dir={hydra_dir.resolve()}",
    ]
    if mode == "smoke":
        # Engineering smoke runs complete all five epochs irrespective of val accuracy.
        overrides.extend(("task.patience=10", "task.early_stop_min_epoch=10"))
    command = [sys.executable, "scripts/run_m0_single.py", *overrides]
    baseline_memory = _gpu_memory(gpu_id)
    peak_memory = baseline_memory
    start = time.monotonic()
    print(f"START {mode} {dataset}/{variant}/seed{seed} on cuda:{gpu_id}", flush=True)
    with stdout_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")},
        )
        while process.poll() is None:
            peak_memory = max(peak_memory, _gpu_memory(gpu_id))
            time.sleep(1.0)
        code = process.returncode
    peak_memory = max(peak_memory, _gpu_memory(gpu_id))
    wall_seconds = time.monotonic() - start
    log_path = hydra_dir / "main.log"
    epoch_time, epoch_count = _timestamp_epoch_seconds(log_path)
    if epoch_time <= 0 and epoch_count:
        epoch_time = wall_seconds / epoch_count

    record = {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "gpu": gpu_id,
        "mode": mode,
        "status": "failed",
        "return_code": code,
        "task_evaluate_test": False,
        "epochs_observed": epoch_count,
        "wall_seconds": wall_seconds,
        "epoch_time_mean": epoch_time,
        "peak_gpu_memory_mb": max(0, peak_memory - baseline_memory),
        "gpu_memory_sampling": "assigned-device nvidia-smi 1s polling, baseline-subtracted",
    }
    if code == 0 and log_path.exists() and checkpoint.exists():
        result_path = hydra_dir / "results.json"
        raw_results = json.loads(result_path.read_text(encoding="utf-8"))
        val_acc = float(raw_results["val_acc"]["mean"])
        val_macro_f1 = float(raw_results["val_macro_f1"]["mean"])
        if not (val_acc == val_acc and val_macro_f1 == val_macro_f1):
            raise RuntimeError(f"non-finite validation metrics in {result_path}")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"\[Run 1\] Best Val Acc .*?epoch=(\d+)", log_text)
        if not match:
            raise RuntimeError(f"could not recover validation-selected epoch from {log_path}")
        import torch

        checkpoint_values = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if checkpoint_values.get("selection") != "best_val_accuracy":
            raise RuntimeError("M0-S1 checkpoint was not selected by validation accuracy")
        if any(key.startswith("test_") for key in checkpoint_values.get("metrics", {})):
            raise RuntimeError("test metrics found in an M0-S1 checkpoint")
        parameter_count = sum(value.numel() for value in checkpoint_values["model_state"].values())
        parameter_count += sum(value.numel() for value in checkpoint_values["head_state"].values())
        record.update(
            {
                "status": "complete",
                "best_epoch": int(match.group(1)),
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
                "num_parameters": int(parameter_count),
                "checkpoint": str(checkpoint.resolve()),
                "hydra_output_dir": str(hydra_dir.resolve()),
            }
        )
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if record["status"] != "complete":
        print(f"FAIL {mode} {dataset}/{variant}/seed{seed}; see {stdout_path}", flush=True)
        return record
    print(
        f"DONE {mode} {dataset}/{variant}/seed{seed} epochs={epoch_count} "
        f"val_acc={record['val_acc']:.4f} epoch_s={epoch_time:.2f} "
        f"peak_gpu_delta_mb={record['peak_gpu_memory_mb']}",
        flush=True,
    )
    return record


def _run_gpu_queue(mode: str, gpu_id: int, jobs: list[tuple[str, str, int]], force: bool) -> list[dict]:
    """Keep no more than one training process active on each physical GPU."""
    return [_run_one(mode, dataset, variant, seed, gpu_id, force) for dataset, variant, seed in jobs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the authorized M0-S1 NC experiment grid.")
    parser.add_argument("--mode", choices=("smoke", "pilot"), default="pilot")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.mode == "smoke":
        datasets, seeds = ["Movies"], [42]
        variants = args.variants
    else:
        datasets, seeds, variants = args.datasets, args.seeds, args.variants
    if args.workers < 1 or args.workers > 2:
        parser.error("workers must be 1 or 2 (one independent run per available RTX 3090)")
    jobs = [(ds, variant, seed) for ds in datasets for seed in seeds for variant in variants]
    failed = []
    gpu_queues = {gpu_id: jobs[gpu_id::args.workers] for gpu_id in range(args.workers)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_gpu_queue, args.mode, gpu_id, queue, args.force): gpu_id
            for gpu_id, queue in gpu_queues.items()
            if queue
        }
        for future in concurrent.futures.as_completed(futures):
            for result in future.result():
                if result.get("status") != "complete":
                    failed.append((result.get("dataset"), result.get("variant"), result.get("seed")))
    if args.mode == "pilot":
        _export_pilot_csvs()
    if failed:
        raise SystemExit(f"M0-S1 jobs failed: {failed}")


if __name__ == "__main__":
    main()
