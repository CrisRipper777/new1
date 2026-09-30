from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT_ROOT = ROOT / "outputs" / "m0" / "stage3_history"
RESULT_ROOT = ROOT / "results" / "m0" / "stage3_history"
DATASETS = ("Movies", "Grocery", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = ("terminal", "state_history", "operation_history", "relation_operation_history")
STAGE3_GROUPS = 7


def _gpu_memory(gpu_id: int) -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id", str(gpu_id), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        return int(result.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def _run_regression_gate(device: str = "cuda:0") -> None:
    subprocess.run(
        [sys.executable, "scripts/analyze_m0_stage3.py", "--mode", "regression", "--device", device],
        cwd=ROOT, check=True,
    )
    path = RESULT_ROOT / "stage12_regression.csv"
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("passed") != "True" for row in rows):
        raise RuntimeError("Strict Stage-I/II terminal regression gate did not pass; refusing formal runs")


def _run_dir(mode: str, dataset: str, variant: str, seed: int) -> Path:
    base = OUTPUT_ROOT / ("smoke" if mode == "smoke" else "full")
    return base / "logs" / dataset / variant / f"seed{seed}"


def _checkpoint(mode: str, dataset: str, variant: str, seed: int) -> Path:
    base = OUTPUT_ROOT / ("smoke" if mode == "smoke" else "full")
    return base / "checkpoints" / dataset / variant / f"seed{seed}.pt"


def _records(mode: str) -> list[dict]:
    root = OUTPUT_ROOT / ("smoke" if mode == "smoke" else "full") / "logs"
    rows = []
    if root.exists():
        for path in sorted(root.glob("**/run_record.json")):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("mode") == mode:
                    rows.append(row)
            except (OSError, json.JSONDecodeError):
                continue
    return rows


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _run_one(mode: str, dataset: str, variant: str, seed: int, gpu_id: int, force: bool) -> dict:
    import torch
    from omegaconf import OmegaConf
    from src.models.interaction_full_s3 import Model

    checkpoint = _checkpoint(mode, dataset, variant, seed)
    run_dir = _run_dir(mode, dataset, variant, seed)
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    if force and checkpoint.exists():
        checkpoint.unlink()
    record_path = run_dir / "run_record.json"
    if not force and record_path.exists() and checkpoint.exists():
        prior = json.loads(record_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete":
            print(f"SKIP {mode} {dataset}/{variant}/seed{seed}", flush=True)
            return prior
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    hydra_dir = run_dir / "hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    gradient_path = (run_dir / "history_gradient_trace.csv").resolve()
    growth_path = (run_dir / "history_output_growth.csv").resolve()
    for path in (gradient_path, growth_path):
        if path.exists():
            path.unlink()
    # Record exact zero initialization before the first optimizer update.
    growth_path.write_text("epoch,W_hist_frobenius_norm\n0,0.0\n", encoding="utf-8")
    epochs = 5 if mode == "smoke" else 300
    overrides = [
        "dataset=" + dataset, "task=nc", "model=interaction_full_s3",
        "model.variant=context_bilinear_absolute", f"model.stage3_variant={variant}",
        f"model.stage3_gradient_trace_path={gradient_path}", f"model.stage3_growth_trace_path={growth_path}",
        f"seed={seed}", "num_runs=1", f"device=cuda:{gpu_id}", f"task.epochs={epochs}",
        "task.evaluate_test=false", f"task.save_ckpt_path={checkpoint.resolve()}",
        f"hydra.run.dir={hydra_dir.resolve()}",
    ]
    if mode == "smoke":
        overrides.extend(("task.patience=10", "task.early_stop_min_epoch=10"))
    command = [sys.executable, "scripts/run_m0_single.py", *overrides]
    baseline = _gpu_memory(gpu_id)
    peak = baseline
    start = time.monotonic()
    stdout_path = run_dir / "run.stdout.log"
    print(f"START {mode} {dataset}/{variant}/seed{seed} cuda:{gpu_id}", flush=True)
    with stdout_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, env=os.environ.copy())
        while process.poll() is None:
            peak = max(peak, _gpu_memory(gpu_id))
            time.sleep(1.0)
        return_code = process.returncode
    peak = max(peak, _gpu_memory(gpu_id))
    wall_seconds = time.monotonic() - start
    log_path = hydra_dir / "main.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    loss_values = re.findall(r"Train Loss ([^ |]+)", log_text)
    finite_losses = all(math.isfinite(float(v)) for v in loss_values)
    epoch_count = len(set(re.findall(r"Epoch\s+(\d+)", log_text)))
    record = {
        "dataset": dataset, "seed": seed, "variant": variant, "gpu": gpu_id,
        "mode": mode, "status": "failed", "return_code": return_code,
        "task_evaluate_test": False, "protocol_version": "unified_full_graph_nc_v1",
        "epochs_observed": epoch_count, "wall_seconds": wall_seconds,
        "epoch_time_mean": wall_seconds / max(epoch_count, 1),
        "peak_gpu_memory_mb": max(0, peak - baseline),
        "gpu_memory_sampling": "assigned-device nvidia-smi 1s polling, baseline-subtracted",
        "losses_finite": finite_losses,
    }
    if return_code == 0 and log_path.exists() and checkpoint.exists():
        raw = json.loads((hydra_dir / "results.json").read_text(encoding="utf-8"))
        val_acc, val_f1 = float(raw["val_acc"]["mean"]), float(raw["val_macro_f1"]["mean"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("selection") != "best_val_accuracy" or any(k.startswith("test_") for k in payload.get("metrics", {})):
            raise RuntimeError(f"Checkpoint protocol audit failed: {checkpoint}")
        if payload.get("protocol_version") != "unified_full_graph_nc_v1":
            raise RuntimeError(f"Unexpected NC protocol in {checkpoint}")
        if not (math.isfinite(val_acc) and math.isfinite(val_f1) and finite_losses):
            raise RuntimeError(f"Non-finite training or validation value: {dataset}/{variant}/{seed}")
        match = re.search(r"\[Run 1\] Best Val Acc .*?epoch=(\d+)", log_text)
        if not match:
            raise RuntimeError(f"Could not recover selected epoch from {log_path}")
        with gradient_path.open(encoding="utf-8") as f:
            gradient_rows = list(csv.DictReader(f))
        with growth_path.open(encoding="utf-8") as f:
            growth_rows = list(csv.DictReader(f))
        if len(gradient_rows) != epoch_count * STAGE3_GROUPS or len(growth_rows) != epoch_count + 1:
            raise RuntimeError(f"Per-epoch Stage-3 traces incomplete: {dataset}/{variant}/{seed}")
        for row in gradient_rows:
            if not math.isfinite(float(row["gradient_rms_preclip"])) or not math.isfinite(float(row["parameter_l2_norm"])):
                raise RuntimeError(f"Non-finite Stage-3 gradient/parameter trace: {dataset}/{variant}/{seed}")
        if variant != "terminal":
            first_epoch = {}
            for row in gradient_rows:
                if float(row["gradient_rms_preclip"]) > 0:
                    first_epoch[row["group"]] = min(first_epoch.get(row["group"], int(row["epoch"])), int(row["epoch"]))
            for group in ("stage3_history_token_state", "stage3_history_token_transition",
                          "stage3_history_token_operation", "stage3_query_intrinsic",
                          "stage3_query_relation_env", "stage3_readout_attention"):
                if group not in first_epoch or first_epoch[group] < 2:
                    raise RuntimeError(f"Upstream Stage-3 gradient did not appear after W_hist opened ({group}): {dataset}/{variant}/{seed}")
        growth_values = [float(row["W_hist_frobenius_norm"]) for row in growth_rows]
        if growth_values[0] != 0.0 or any(not math.isfinite(v) for v in growth_values):
            raise RuntimeError(f"Invalid W_hist initialization/growth trace: {dataset}/{variant}/{seed}")
        if variant == "terminal":
            if any(abs(v) > 1e-12 for v in growth_values):
                raise RuntimeError("Terminal variant unexpectedly opened W_hist")
        elif max(growth_values[1:]) <= 0:
            raise RuntimeError(f"W_hist did not open from zero: {dataset}/{variant}/{seed}")
        from omegaconf import OmegaConf as OC
        cfg_model = OC.load(ROOT / "configs/model/interaction_full_s3.yaml")
        cfg_model.stage3_variant = variant
        count_model = Model(OC.create({"model": cfg_model}), payload["data_info"])
        active = sum(p.numel() for name, p in count_model.named_parameters() if name in count_model.active_parameter_names())
        total = sum(p.numel() for p in count_model.parameters())
        record.update({
            "status": "complete", "best_epoch": int(match.group(1)),
            "val_acc": val_acc, "val_macro_f1": val_f1,
            "active_parameters": active + int(payload["data_info"]["num_classes"]) * count_model.out_dim + int(payload["data_info"]["num_classes"]),
            "total_parameters": total + int(payload["data_info"]["num_classes"]) * count_model.out_dim + int(payload["data_info"]["num_classes"]),
            "checkpoint": str(checkpoint.resolve()), "hydra_dir": str(hydra_dir.resolve()),
            "history_output_opened": variant != "terminal" and max(growth_values[1:]) > 0,
        })
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if record["status"] != "complete":
        print(f"FAIL {mode} {dataset}/{variant}/seed{seed}; inspect {stdout_path}", flush=True)
    else:
        print(f"DONE {mode} {dataset}/{variant}/seed{seed} epoch={record['best_epoch']} "
              f"val_acc={record['val_acc']:.4f} epoch_s={record['epoch_time_mean']:.2f}", flush=True)
    return record


def _run_grid(mode: str, workers: int, force: bool) -> None:
    if workers not in (1, 2):
        raise ValueError("workers must be 1 or 2 (one run per RTX 3090)")
    if mode == "smoke":
        datasets, seeds = ("Movies",), (42,)
    else:
        datasets, seeds = DATASETS, SEEDS
    jobs = [(dataset, variant, seed) for dataset in datasets for seed in seeds for variant in VARIANTS]
    queues = {gpu: jobs[gpu::workers] for gpu in range(workers)}
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(lambda q=q, g=g: [_run_one(mode, d, v, s, g, force) for d, v, s in q])
                   for g, q in queues.items() if q]
        for future in concurrent.futures.as_completed(futures):
            failures.extend((r.get("dataset"), r.get("variant"), r.get("seed"))
                            for r in future.result() if r.get("status") != "complete")
    if failures:
        raise RuntimeError(f"M0-S3 {mode} run failures: {failures}")
    if mode == "full":
        rows = [r for r in _records("full") if r.get("status") == "complete"]
        expected = {(d, s, v) for d in DATASETS for s in SEEDS for v in VARIANTS}
        actual = {(r["dataset"], int(r["seed"]), r["variant"]) for r in rows}
        if actual != expected:
            raise RuntimeError(f"Expected 36 complete S3 pilot records, found {len(actual)}")
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        _write_csv(RESULT_ROOT / "pilot_metrics.csv", rows, [
            "dataset", "seed", "variant", "best_epoch", "val_acc", "val_macro_f1",
            "active_parameters", "total_parameters", "history_output_opened",
        ])
        _write_csv(RESULT_ROOT / "runtime_memory.csv", rows, [
            "dataset", "seed", "variant", "wall_seconds", "epochs_observed", "epoch_time_mean",
            "peak_gpu_memory_mb", "gpu_memory_sampling",
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M0-S3 ROHC validation-only NC experiments.")
    parser.add_argument("--mode", choices=("all", "smoke", "full"), default="all")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    _run_regression_gate()
    if args.mode in ("all", "smoke"):
        _run_grid("smoke", args.workers, args.force)
        subprocess.run([sys.executable, "scripts/analyze_m0_stage3.py", "--mode", "smoke"], cwd=ROOT, check=True)
    if args.mode in ("all", "full"):
        smoke = {(r.get("dataset"), int(r.get("seed", -1)), r.get("variant"))
                 for r in _records("smoke") if r.get("status") == "complete"}
        required = {("Movies", 42, variant) for variant in VARIANTS}
        if not required.issubset(smoke):
            raise RuntimeError("Complete all four Movies seed-42 smoke runs before the 36-run pilot")
        _run_grid("full", args.workers, args.force)
        subprocess.run([sys.executable, "scripts/analyze_m0_stage3.py", "--mode", "full"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "scripts/summarize_m0_stage3.py"], cwd=ROOT, check=True)
    print(f"M0-S3 {args.mode.upper()} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
