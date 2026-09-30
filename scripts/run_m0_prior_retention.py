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
OUTPUT_ROOT = ROOT / "outputs" / "m0" / "prior_retention"
RESULT_ROOT = ROOT / "results" / "m0" / "prior_retention"
VARIANTS = ("terminal", "prior_delta_gate", "prior_update_add", "prior_update_gate")
DATASETS = ("Movies", "Grocery", "Reddit-S")
SEEDS = (42, 43, 44)

from src.models.interaction_prior_retaining import TRACE_GROUPS  # noqa: E402


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
    base = OUTPUT_ROOT / "smoke" if mode == "smoke" else OUTPUT_ROOT / "full"
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
    base = OUTPUT_ROOT / ("smoke" if mode == "smoke" else "full") / "logs"
    rows = []
    if base.exists():
        for path in sorted(base.glob("**/run_record.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("mode") == mode:
                    rows.append(record)
            except (OSError, json.JSONDecodeError):
                continue
    return rows


def _run_one(mode: str, dataset: str, variant: str, seed: int, gpu_id: int, force: bool):
    import torch

    checkpoint, run_dir = _paths(mode, dataset, variant, seed)
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
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
    stdout_path = run_dir / "run.stdout.log"
    epochs = 5 if mode == "smoke" else 300
    gradient_path = (run_dir / "gradient_trace.csv").resolve()
    growth_path = (run_dir / "context_branch_growth.csv").resolve()
    for trace_path in (gradient_path, growth_path):
        if trace_path.exists():
            trace_path.unlink()
    overrides = [
        f"dataset={dataset}", "task=nc", "model=interaction_prior_retaining",
        f"model.readout_variant={variant}",
        f"model.prior_gradient_trace_path={gradient_path}",
        f"model.context_branch_growth_path={growth_path}",
        f"seed={seed}", "num_runs=1", f"device=cuda:{gpu_id}",
        f"task.epochs={epochs}", "task.evaluate_test=false",
        f"task.save_ckpt_path={checkpoint.resolve()}",
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
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, env=os.environ.copy())
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
        "task_evaluate_test": False, "protocol_version": "unified_full_graph_nc_v1",
        "epochs_observed": epoch_count, "wall_seconds": wall_seconds,
        "epoch_time_mean": wall_seconds / max(epoch_count, 1),
        "peak_gpu_memory_mb": max(0, peak - baseline),
        "gpu_memory_sampling": "assigned-device nvidia-smi 1s polling, baseline-subtracted",
        "losses_finite": finite_losses,
    }
    if return_code == 0 and log_path.exists() and checkpoint.exists():
        raw = json.loads((hydra_dir / "results.json").read_text(encoding="utf-8"))
        val_acc = float(raw["val_acc"]["mean"])
        val_f1 = float(raw["val_macro_f1"]["mean"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("selection") != "best_val_accuracy" or any(k.startswith("test_") for k in payload.get("metrics", {})):
            raise RuntimeError(f"Checkpoint protocol audit failed: {checkpoint}")
        if payload.get("protocol_version") != "unified_full_graph_nc_v1":
            raise RuntimeError(f"Unexpected NC protocol: {checkpoint}")
        if not (math.isfinite(val_acc) and math.isfinite(val_f1) and finite_losses):
            raise RuntimeError(f"Non-finite training/validation value: {dataset}/{variant}/{seed}")
        state = payload["model_state"]
        active_w2 = (
            bool(torch.count_nonzero(state["context_adapter_text.output.weight"])),
            bool(torch.count_nonzero(state["context_adapter_visual.output.weight"])),
        )
        if mode == "smoke" and variant != "terminal" and not all(active_w2):
            raise RuntimeError(f"ContextAdapter W2 did not open from zero: {dataset}/{variant}/{seed}")
        if mode == "smoke":
            growth_rows = list(csv.DictReader(growth_path.open(encoding="utf-8")))
            if len(growth_rows) != (epoch_count + 1) * 2:
                raise RuntimeError(f"Context branch growth trace incomplete: {dataset}/{variant}/{seed}")
            gradient_rows = list(csv.DictReader(gradient_path.open(encoding="utf-8")))
            if len(gradient_rows) != epoch_count * len(TRACE_GROUPS):
                raise RuntimeError(f"Gradient trace incomplete: {dataset}/{variant}/{seed}")
            if variant != "terminal":
                nonzero_stage2 = [
                    int(r["epoch"]) for r in gradient_rows
                    if r["group"] in ("stage2_operator_up", "state_update")
                    and float(r["gradient_rms_preclip"]) > 0
                ]
                if not nonzero_stage2 or min(nonzero_stage2) < 2:
                    raise RuntimeError("Stage-I/II interaction gradient did not return after the zero-init branch opened")
        match = re.search(r"\[Run 1\] Best Val Acc .*?epoch=(\d+)", log_text)
        if not match:
            raise RuntimeError(f"Could not recover selected epoch from {log_path}")
        gradient_rows = list(csv.DictReader(gradient_path.open(encoding="utf-8")))
        growth_rows = list(csv.DictReader(growth_path.open(encoding="utf-8")))
        if len(gradient_rows) != epoch_count * len(TRACE_GROUPS):
            raise RuntimeError(f"Per-epoch gradient trace incomplete: {dataset}/{variant}/{seed}")
        if len(growth_rows) != (epoch_count + 1) * 2:
            raise RuntimeError(f"Context growth trace incomplete: {dataset}/{variant}/{seed}")
        from omegaconf import OmegaConf
        from src.models.interaction_prior_retaining import Model
        model_cfg = OmegaConf.load(ROOT / "configs" / "model" / "interaction_prior_retaining.yaml")
        model_cfg.readout_variant = variant
        count_model = Model(OmegaConf.create({"model": model_cfg}), payload["data_info"])
        active_names = count_model.active_parameter_names()
        active = sum(p.numel() for n, p in count_model.named_parameters() if n in active_names)
        total = sum(p.numel() for p in count_model.parameters())
        classifier = int(payload["data_info"]["num_classes"]) * count_model.out_dim + int(payload["data_info"]["num_classes"])
        record.update({
            "status": "complete", "best_epoch": int(match.group(1)),
            "val_acc": val_acc, "val_macro_f1": val_f1,
            "active_parameters": active + classifier, "total_parameters": total + classifier,
            "context_w2_text_nonzero": active_w2[0], "context_w2_visual_nonzero": active_w2[1],
            "checkpoint": str(checkpoint.resolve()), "hydra_dir": str(hydra_dir.resolve()),
        })
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if record["status"] != "complete":
        print(f"FAIL {mode} {dataset}/{variant}/seed{seed}; inspect {stdout_path}", flush=True)
    else:
        print(f"DONE {mode} {dataset}/{variant}/seed{seed} epoch={record['best_epoch']} "
              f"val_acc={record['val_acc']:.4f} epoch_s={record['epoch_time_mean']:.2f}", flush=True)
    return record


def _queue(mode, gpu_id, jobs, force):
    return [_run_one(mode, d, v, s, gpu_id, force) for d, v, s in jobs]


def _export_metrics():
    rows = [r for r in _records("full") if r.get("status") == "complete"]
    fields = ["dataset", "seed", "variant", "best_epoch", "val_acc", "val_macro_f1",
              "active_parameters", "total_parameters", "task_evaluate_test"]
    _write_csv(RESULT_ROOT / "pilot_metrics.csv", rows, fields)
    runtime_fields = ["dataset", "seed", "variant", "wall_seconds", "epochs_observed",
                      "epoch_time_mean", "peak_gpu_memory_mb", "gpu_memory_sampling"]
    _write_csv(RESULT_ROOT / "runtime_memory.csv", [{k: r.get(k) for k in runtime_fields} for r in rows], runtime_fields)


def main():
    parser = argparse.ArgumentParser(description="Run M0 prior-retaining readout validation experiments.")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers not in (1, 2):
        parser.error("workers must be 1 or 2 (one run per RTX 3090)")
    if args.mode == "smoke":
        datasets, seeds = ("Movies",), (42,)
    else:
        datasets, seeds = DATASETS, SEEDS
        smoke = {(r.get("dataset"), r.get("seed"), r.get("variant"))
                 for r in _records("smoke") if r.get("status") == "complete"}
        if not {("Movies", 42, v) for v in VARIANTS}.issubset(smoke):
            parser.error("Complete all four Movies seed-42 smoke runs before full pilot")
    # Gate all formal work against the nine frozen v3 absolute checkpoints.
    subprocess.run([sys.executable, "scripts/analyze_m0_prior_retention.py",
                    "--mode", "preflight", "--device", "cuda:0"], cwd=ROOT, check=True)
    jobs = [(d, v, s) for d in datasets for s in seeds for v in VARIANTS]
    queues = {gpu: jobs[gpu::args.workers] for gpu in range(args.workers)}
    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_queue, args.mode, gpu, queue, args.force) for gpu, queue in queues.items() if queue]
        for future in concurrent.futures.as_completed(futures):
            failed.extend((r.get("dataset"), r.get("variant"), r.get("seed"))
                          for r in future.result() if r.get("status") != "complete")
    if failed:
        raise SystemExit(f"Prior retention runs failed: {failed}")
    if args.mode == "full":
        _export_metrics()
    subprocess.run([sys.executable, "scripts/analyze_m0_prior_retention.py",
                    "--mode", args.mode, "--device", "cuda:0"], cwd=ROOT, check=True)
    if args.mode == "full":
        subprocess.run([sys.executable, "scripts/summarize_m0_prior_retention.py"], cwd=ROOT, check=True)
    print(f"{args.mode.upper()} COMPLETE: {len(jobs)} runs", flush=True)


if __name__ == "__main__":
    main()
