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
OUTPUT_ROOT = ROOT / "outputs" / "final_nc"
RESULT_ROOT = ROOT / "results" / "final_nc"
DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = ("full", "no_context", "shared_relation", "static_execution", "operator_off")
PROTOCOL = "unified_full_graph_nc_v1"


def _gpu_memory(gpu_id: int) -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id", str(gpu_id), "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        return int(result.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def _paths(mode: str, dataset: str, variant: str, seed: int):
    base = OUTPUT_ROOT / ("smoke" if mode == "smoke" else "")
    checkpoint = base / "checkpoints" / dataset / variant / f"seed{seed}.pt"
    run_dir = base / "logs" / dataset / variant / f"seed{seed}"
    return checkpoint, run_dir


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _records(mode: str) -> list[dict]:
    base = OUTPUT_ROOT / ("smoke" if mode == "smoke" else "") / "logs"
    rows = []
    if base.exists():
        for path in sorted(base.glob("**/run_record.json")):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("mode") == mode:
                    rows.append(row)
            except (OSError, json.JSONDecodeError):
                continue
    return rows


def _parameter_audit(variant: str, checkpoint_payload: dict) -> tuple[int, int]:
    import torch.nn as nn
    from omegaconf import OmegaConf
    from src.models.final_interaction import Model

    model_cfg = OmegaConf.load(ROOT / "configs/model/final_interaction.yaml")
    model_cfg.variant = variant
    model = Model(OmegaConf.create({"model": model_cfg}), checkpoint_payload["data_info"])
    active_names = model.active_parameter_names()
    active_count = sum(p.numel() for name, p in model.named_parameters() if name in active_names)
    total_count = sum(p.numel() for p in model.parameters())
    head = nn.Linear(model.out_dim, int(checkpoint_payload["data_info"]["num_classes"]))
    head_count = sum(p.numel() for p in head.parameters())
    return active_count + head_count, total_count + head_count


def _run_one(mode: str, dataset: str, variant: str, seed: int, gpu_id: int, force: bool):
    import torch

    checkpoint, run_dir = _paths(mode, dataset, variant, seed)
    record_path = run_dir / "run_record.json"
    if not force and record_path.exists() and checkpoint.exists():
        try:
            prior = json.loads(record_path.read_text(encoding="utf-8"))
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if prior.get("status") == "complete" and payload.get("selection") == "best_val_accuracy":
                print(f"SKIP {mode} {dataset}/{variant}/seed{seed}", flush=True)
                return prior
        except Exception:
            pass
    if run_dir.exists():
        shutil.rmtree(run_dir)
    if checkpoint.exists():
        checkpoint.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    hydra_dir = run_dir / "hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "run.stdout.log"
    epochs = 5 if mode == "smoke" else 300
    gradient_path = (run_dir / "gradient_trace.csv").resolve()
    overrides = [
        f"dataset={dataset}", "task=nc", "model=final_interaction",
        f"model.variant={variant}", f"model.gradient_trace_path={gradient_path}",
        f"seed={seed}", "num_runs=1", f"device=cuda:{gpu_id}",
        f"task.epochs={epochs}", "task.training_mode=full_graph",
        "task.optimizer=adamw", "task.lr=0.001", "task.weight_decay=0.0001",
        "task.eval_every=1", "task.patience=30", "task.early_stop_min_epoch=30",
        "task.early_stop_min_delta=0.0001", "task.grad_clip=1.0",
        "task.evaluate_test=true", f"task.save_ckpt_path={checkpoint.resolve()}",
        f"hydra.run.dir={hydra_dir.resolve()}",
    ]
    command = [sys.executable, "scripts/run_final_nc_single.py", *overrides]
    baseline = _gpu_memory(gpu_id)
    peak = baseline
    start_time = time.monotonic()
    print(f"START {mode} {dataset}/{variant}/seed{seed} cuda:{gpu_id}", flush=True)
    with stdout_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, env=os.environ.copy()
        )
        while process.poll() is None:
            peak = max(peak, _gpu_memory(gpu_id))
            time.sleep(1.0)
        return_code = process.returncode
    peak = max(peak, _gpu_memory(gpu_id))
    wall_seconds = time.monotonic() - start_time
    log_path = hydra_dir / "main.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    epoch_lines = re.findall(r"Epoch\s+(\d+).*?Train Loss ([^ |]+)", log_text)
    finite_losses = all(math.isfinite(float(loss)) for _, loss in epoch_lines)
    epoch_count = len({int(epoch) for epoch, _ in epoch_lines})
    record = {
        "dataset": dataset, "seed": seed, "variant": variant, "gpu": gpu_id,
        "mode": mode, "status": "failed", "return_code": return_code,
        "task_evaluate_test": True, "protocol_version": PROTOCOL,
        "epochs_observed": epoch_count, "wall_seconds": wall_seconds,
        "epoch_time_mean": wall_seconds / max(epoch_count, 1),
        "peak_gpu_memory_mb": max(0, peak - baseline),
        "gpu_memory_sampling": "assigned-device nvidia-smi 1s polling, baseline-subtracted",
        "losses_finite": finite_losses,
        "checkpoint": str(checkpoint.resolve()), "hydra_dir": str(hydra_dir.resolve()),
        "gradient_trace": str(gradient_path),
    }
    if return_code == 0 and log_path.exists() and checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        expected_metrics = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
        metrics = payload.get("metrics", {})
        if payload.get("task") != "nc" or payload.get("protocol_version") != PROTOCOL:
            raise RuntimeError(f"Unexpected checkpoint protocol: {checkpoint}")
        if payload.get("selection") != "best_val_accuracy" or payload.get("epoch") is None:
            raise RuntimeError(f"Checkpoint selection audit failed: {checkpoint}")
        if any(key not in metrics or not math.isfinite(float(metrics[key])) for key in expected_metrics):
            raise RuntimeError(f"Checkpoint metrics are incomplete/non-finite: {checkpoint}")
        epoch_count_ok = (epoch_count == epochs) if mode == "smoke" else (30 <= epoch_count <= epochs)
        if not finite_losses or not epoch_count_ok:
            raise RuntimeError(f"Training loss/epoch audit failed: {dataset}/{variant}/{seed}")
        match = re.search(r"\[Run 1\] Best Val Acc .*?epoch=(\d+)", log_text)
        if not match or int(match.group(1)) != int(payload["epoch"]):
            raise RuntimeError(f"Selected epoch is missing or inconsistent: {checkpoint}")
        active, total = _parameter_audit(variant, payload)
        trace_rows = []
        if gradient_path.exists():
            with gradient_path.open(encoding="utf-8") as handle:
                trace_rows = list(csv.DictReader(handle))
        if len(trace_rows) != epoch_count * 7:
            raise RuntimeError(f"Gradient-health trace incomplete: {gradient_path}")
        record.update({
            "status": "complete", "best_epoch": int(payload["epoch"]),
            **{key: float(metrics[key]) for key in expected_metrics},
            "active_parameters": active, "total_parameters": total,
            "gradient_rows": len(trace_rows),
        })
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if record["status"] != "complete":
        print(f"FAIL {mode} {dataset}/{variant}/seed{seed}; inspect {stdout_path}", flush=True)
    else:
        print(f"DONE {mode} {dataset}/{variant}/seed{seed} epoch={record['best_epoch']} "
              f"val_acc={record['val_acc']:.4f} wall_s={wall_seconds:.1f}", flush=True)
    return record


def _queue(mode, gpu_id, jobs, force):
    return [_run_one(mode, dataset, variant, seed, gpu_id, force)
            for dataset, variant, seed in jobs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen five-dataset final NC benchmark.")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers not in (1, 2):
        parser.error("workers must be 1 or 2 (one full-graph experiment per GPU)")
    import torch
    if args.workers > torch.cuda.device_count():
        parser.error(f"requested {args.workers} workers but only {torch.cuda.device_count()} CUDA devices are visible")
    if args.mode == "smoke":
        datasets, seeds = ("Movies",), (42,)
    else:
        datasets, seeds = DATASETS, SEEDS
        smoke = {(r.get("dataset"), r.get("seed"), r.get("variant"))
                 for r in _records("smoke") if r.get("status") == "complete"}
        if not {("Movies", 42, variant) for variant in VARIANTS}.issubset(smoke):
            parser.error("complete all five Movies seed-42 smoke runs before starting the full grid")
    jobs = [(dataset, variant, seed) for dataset in datasets for seed in seeds for variant in VARIANTS]
    queues = {gpu: jobs[gpu::args.workers] for gpu in range(args.workers)}
    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_queue, args.mode, gpu, queue, args.force)
                   for gpu, queue in queues.items() if queue]
        for future in concurrent.futures.as_completed(futures):
            failed.extend((row.get("dataset"), row.get("variant"), row.get("seed"))
                          for row in future.result() if row.get("status") != "complete")
    if failed:
        raise SystemExit(f"Final NC runs failed: {failed}")
    records = [row for row in _records(args.mode) if row.get("status") == "complete"]
    fields = ["dataset", "seed", "variant", "mode", "status", "gpu", "protocol_version",
              "task_evaluate_test", "best_epoch", "val_acc", "val_macro_f1", "test_acc",
              "test_macro_f1", "active_parameters", "total_parameters", "epochs_observed",
              "wall_seconds", "epoch_time_mean", "peak_gpu_memory_mb", "checkpoint", "hydra_dir"]
    manifest_name = "smoke_manifest.csv" if args.mode == "smoke" else "run_manifest.csv"
    _write_csv(RESULT_ROOT / manifest_name, records, fields)
    expected = 5 if args.mode == "smoke" else 75
    if len(records) != expected:
        raise SystemExit(f"Expected {expected} complete {args.mode} runs, found {len(records)}")
    print(f"{args.mode.upper()} COMPLETE: {len(records)} runs", flush=True)


if __name__ == "__main__":
    main()
