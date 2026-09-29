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

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT_ROOT = ROOT / "outputs" / "m0" / "interpret_execute"
RESULT_ROOT = ROOT / "results" / "m0" / "interpret_execute"
VARIANTS = ("global_dynamic", "pair_dynamic", "context_static", "context_dynamic")
DATASETS = ("Movies", "Grocery", "Reddit-S")
SEEDS = (42, 43, 44)


def _gpu_memory(gpu_id: int) -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id", str(gpu_id), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        return int(result.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def _paths(mode: str, dataset: str, variant: str, seed: int):
    if mode == "smoke":
        base = OUTPUT_ROOT / "smoke"
        checkpoint = base / "checkpoints" / dataset / variant / f"seed{seed}.pt"
        run_dir = base / "logs" / dataset / variant / f"seed{seed}"
    else:
        checkpoint = OUTPUT_ROOT / "checkpoints" / dataset / variant / f"seed{seed}.pt"
        run_dir = OUTPUT_ROOT / "logs" / dataset / variant / f"seed{seed}"
    return checkpoint, run_dir


def _parameter_counts(checkpoint: dict, variant: str) -> tuple[int, int]:
    import torch

    from src.models.interaction_core import Model

    model_cfg = OmegaConf.load(ROOT / "configs" / "model" / "interaction_core.yaml")
    model_cfg.variant = variant
    cfg = OmegaConf.create({"model": model_cfg})
    model = Model(cfg, checkpoint["data_info"])
    active_names = model.active_parameter_names()
    active = sum(p.numel() for name, p in model.named_parameters() if name in active_names)
    classifier = int(checkpoint["data_info"]["num_classes"]) * model.out_dim + int(checkpoint["data_info"]["num_classes"])
    total = sum(p.numel() for p in model.parameters()) + classifier
    return active + classifier, total


def _read_records(mode: str) -> list[dict]:
    base = OUTPUT_ROOT / "smoke" / "logs" if mode == "smoke" else OUTPUT_ROOT / "logs"
    records = []
    if base.exists():
        for path in sorted(base.glob("**/run_record.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    return records


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _export_metrics() -> None:
    records = [r for r in _read_records("pilot") if r.get("status") == "complete"]
    fields = [
        "dataset", "seed", "variant", "best_epoch", "val_acc", "val_macro_f1",
        "active_parameters", "total_parameters",
    ]
    _write_csv(RESULT_ROOT / "pilot_metrics.csv", records, fields)
    runtime_fields = [
        "dataset", "seed", "variant", "wall_seconds", "epochs_observed",
        "epoch_time_mean", "peak_gpu_memory_mb", "gpu_memory_sampling",
    ]
    _write_csv(
        RESULT_ROOT / "runtime_memory.csv",
        [{key: row.get(key) for key in runtime_fields} for row in records],
        runtime_fields,
    )


def _run_one(mode: str, dataset: str, variant: str, seed: int, gpu_id: int, force: bool) -> dict:
    import torch

    checkpoint, run_dir = _paths(mode, dataset, variant, seed)
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    record_path = run_dir / "run_record.json"
    stdout_path = run_dir / "run.stdout.log"
    hydra_dir = run_dir / "hydra"
    if not force and record_path.exists() and checkpoint.exists():
        prior = json.loads(record_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete":
            print(f"SKIP {mode} {dataset}/{variant}/seed{seed}", flush=True)
            return prior

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    hydra_dir.mkdir(parents=True, exist_ok=True)
    epochs = 5 if mode == "smoke" else 300
    overrides = [
        f"dataset={dataset}", "task=nc", "model=interaction_core",
        f"model.variant={variant}", f"seed={seed}", "num_runs=1",
        f"device=cuda:{gpu_id}", f"task.epochs={epochs}",
        "task.evaluate_test=false", f"task.save_ckpt_path={checkpoint.resolve()}",
        f"hydra.run.dir={hydra_dir.resolve()}",
    ]
    if mode == "smoke":
        overrides.extend(("task.patience=10", "task.early_stop_min_epoch=10"))
    command = [sys.executable, "scripts/run_m0_single.py", *overrides]
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
    epoch_lines = []
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    for match in re.finditer(r"Epoch\s+(\d+).*?Train Loss ([^ |]+)", log_text):
        epoch_lines.append((int(match.group(1)), match.group(2)))
    finite_losses = all(math.isfinite(float(loss)) for _, loss in epoch_lines)
    epoch_count = len({epoch for epoch, _ in epoch_lines})
    record = {
        "dataset": dataset, "seed": seed, "variant": variant, "gpu": gpu_id,
        "mode": mode, "status": "failed", "return_code": return_code,
        "task_evaluate_test": False, "epochs_observed": epoch_count,
        "wall_seconds": wall_seconds,
        "epoch_time_mean": wall_seconds / max(epoch_count, 1),
        "peak_gpu_memory_mb": max(0, peak - baseline),
        "gpu_memory_sampling": "assigned-device nvidia-smi 1s polling, baseline-subtracted",
        "losses_finite": finite_losses,
    }
    if return_code == 0 and log_path.exists() and checkpoint.exists():
        result_path = hydra_dir / "results.json"
        raw = json.loads(result_path.read_text(encoding="utf-8"))
        val_acc = float(raw["val_acc"]["mean"])
        val_f1 = float(raw["val_macro_f1"]["mean"])
        if not (math.isfinite(val_acc) and math.isfinite(val_f1) and finite_losses):
            raise RuntimeError(f"Non-finite loss or validation metrics for {dataset}/{variant}/{seed}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("selection") != "best_val_accuracy" or any(
            key.startswith("test_") for key in payload.get("metrics", {})
        ):
            raise RuntimeError(f"Checkpoint selection/test-evaluation audit failed: {checkpoint}")
        up_text = payload["model_state"]["text_operator.up.weight"]
        up_visual = payload["model_state"]["visual_operator.up.weight"]
        if mode == "smoke" and (not torch.count_nonzero(up_text) or not torch.count_nonzero(up_visual)):
            raise RuntimeError(f"Zero-initialized operator did not become active: {checkpoint}")
        active, total = _parameter_counts(payload, variant)
        match = re.search(r"\[Run 1\] Best Val Acc .*?epoch=(\d+)", log_text)
        if not match:
            raise RuntimeError(f"Could not recover selected epoch from {log_path}")
        record.update({
            "status": "complete", "best_epoch": int(match.group(1)),
            "val_acc": val_acc, "val_macro_f1": val_f1,
            "active_parameters": active, "total_parameters": total,
            "operator_up_text_nonzero": bool(torch.count_nonzero(up_text)),
            "operator_up_visual_nonzero": bool(torch.count_nonzero(up_visual)),
            "checkpoint": str(checkpoint.resolve()),
            "hydra_dir": str(hydra_dir.resolve()),
        })
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if record["status"] != "complete":
        print(f"FAIL {mode} {dataset}/{variant}/seed{seed}; inspect {stdout_path}", flush=True)
    else:
        print(
            f"DONE {mode} {dataset}/{variant}/seed{seed} epoch={record['best_epoch']} "
            f"val_acc={record['val_acc']:.4f} epoch_s={record['epoch_time_mean']:.2f} "
            f"gpu_delta_mb={record['peak_gpu_memory_mb']}", flush=True,
        )
    return record


def _run_gpu_queue(mode: str, gpu_id: int, jobs: list[tuple[str, str, int]], force: bool):
    return [_run_one(mode, dataset, variant, seed, gpu_id, force) for dataset, variant, seed in jobs]


def _run_analysis(mode: str) -> None:
    subprocess.run(
        [sys.executable, "scripts/analyze_m0_core.py", "--mode", mode, "--device", "cuda:0"],
        cwd=ROOT, check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed M0-Core NC experiment grid.")
    parser.add_argument("--mode", choices=("smoke", "pilot"), default="pilot")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers not in (1, 2):
        parser.error("workers must be 1 or 2 (one run per available RTX 3090)")
    if set(args.variants) - set(VARIANTS):
        parser.error(f"variants must be selected from {VARIANTS}")
    if args.mode == "smoke":
        datasets, seeds, variants = ["Movies"], [42], args.variants
    else:
        datasets, seeds, variants = args.datasets, args.seeds, args.variants
        expected = {(d, s, v) for d in DATASETS for s in SEEDS for v in VARIANTS}
        requested = {(d, s, v) for d in datasets for s in seeds for v in variants}
        if requested != expected:
            parser.error("pilot is fixed to Movies/Grocery/Reddit-S x seeds 42/43/44 x four variants")
        smoke = [r for r in _read_records("smoke") if r.get("status") == "complete"]
        if not all(any((r.get("dataset"), r.get("seed"), r.get("variant")) == ("Movies", 42, v) for r in smoke) for v in VARIANTS):
            parser.error("complete all four Movies seed 42 smoke runs before starting the pilot")

    jobs = [(d, v, s) for d in datasets for s in seeds for v in variants]
    queues = {gpu: jobs[gpu::args.workers] for gpu in range(args.workers)}
    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_run_gpu_queue, args.mode, gpu, queue, args.force)
            for gpu, queue in queues.items() if queue
        ]
        for future in concurrent.futures.as_completed(futures):
            failed.extend(
                (r.get("dataset"), r.get("variant"), r.get("seed"))
                for r in future.result() if r.get("status") != "complete"
            )
    if args.mode == "pilot":
        _export_metrics()
    if failed:
        raise SystemExit(f"M0-Core runs failed: {failed}")

    _run_analysis(args.mode)
    if args.mode == "smoke":
        diag = list(csv.DictReader((RESULT_ROOT / "smoke_operator_diagnostics.csv").open(encoding="utf-8")))
        if not diag or any(float(row["delta_message_norm_mean"]) <= 0 for row in diag):
            raise SystemExit("Smoke operator diagnostics did not show a nonzero trained conditional branch")
        print("SMOKE CHECKS PASSED: finite losses/metrics, nonzero trained operator, and finite diagnostics", flush=True)
    else:
        subprocess.run([sys.executable, "scripts/summarize_m0_core.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
