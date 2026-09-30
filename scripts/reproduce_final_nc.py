from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.data import load_mag_data
from src.models.final_interaction import Model
from src.tasks.inference import infer_all_embeddings
from src.tasks.nc import _evaluate_split


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = ("full", "no_context", "shared_relation", "static_execution", "operator_off")
METRICS = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")


def _write(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(sys.argv[1] if len(sys.argv) > 1 else parser_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for checkpoint reproduction but is unavailable")
    torch.set_num_threads(4)
    metric_rows = []
    val_rows, test_rows = [], []
    failures = []
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        for dataset in DATASETS:
            for seed in SEEDS:
                cfg = compose(config_name="config", overrides=[
                    f"dataset={dataset}", "task=nc", "model=final_interaction",
                    f"seed={seed}", "num_runs=1", "task.epochs=1",
                    "task.evaluate_test=true", "task.inference_mode=full",
                ])
                OmegaConf.resolve(cfg)
                data = load_mag_data(cfg, "nc", seed)
                data_info = {
                    "input_dim": data.input_dim, "num_nodes": data.num_nodes,
                    "num_classes": data.num_classes,
                    "text_dim": int(data.x_t.shape[1]), "visual_dim": int(data.x_i.shape[1]),
                }
                labels = list(range(int(data.num_classes)))
                for variant in VARIANTS:
                    checkpoint_path = ROOT / "outputs/final_nc/checkpoints" / dataset / variant / f"seed{seed}.pt"
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    required = {"task", "protocol_version", "seed", "selection", "epoch",
                                "metrics", "model_state", "head_state", "data_info"}
                    if not required.issubset(checkpoint):
                        raise AssertionError(f"checkpoint fields missing {sorted(required - set(checkpoint))}: {checkpoint_path}")
                    if checkpoint.get("task") != "nc" or checkpoint.get("protocol_version") != "unified_full_graph_nc_v1":
                        raise AssertionError(f"unexpected task/protocol: {checkpoint_path}")
                    if int(checkpoint.get("seed", -1)) != seed:
                        raise AssertionError(f"checkpoint seed mismatch: {checkpoint_path}")
                    if checkpoint.get("selection") != "best_val_accuracy":
                        raise AssertionError(f"not validation-selected: {checkpoint_path}")
                    if not all(metric in checkpoint.get("metrics", {}) for metric in METRICS):
                        raise AssertionError(f"missing saved test/validation metrics: {checkpoint_path}")
                    model_cfg = OmegaConf.load(ROOT / "configs/model/final_interaction.yaml")
                    model_cfg.variant = variant
                    model = Model(OmegaConf.create({"model": model_cfg}), data_info).to(device)
                    model.load_state_dict(checkpoint["model_state"], strict=True)
                    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
                    classifier.load_state_dict(checkpoint["head_state"], strict=True)
                    z = infer_all_embeddings(model, data, device, True, 4096, "full")
                    reproduced = {}
                    for split_name, idx in (("val", data.val_idx), ("test", data.test_idx)):
                        values = _evaluate_split(
                            classifier, z, data.y, idx, device, 4096, labels,
                        )
                        reproduced[f"{split_name}_acc"] = values["acc"]
                        reproduced[f"{split_name}_macro_f1"] = values["macro_f1"]
                    row = {
                        "dataset": dataset, "seed": seed, "variant": variant,
                        "best_epoch": int(checkpoint["epoch"]),
                        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                        **{key: float(checkpoint["metrics"][key]) for key in METRICS},
                    }
                    metric_rows.append(row)
                    for key in METRICS:
                        error = abs(float(reproduced[key]) - float(checkpoint["metrics"][key]))
                        status = "pass_1e-6" if error <= 1e-6 else (
                            "pass_1e-5_recorded" if error <= 1e-5 else "fail"
                        )
                        audit = {
                            "dataset": dataset, "seed": seed, "variant": variant,
                            "metric": key, "checkpoint_metric": checkpoint["metrics"][key],
                            "reproduced_metric": reproduced[key], "absolute_error": error,
                            "tolerance_status": status, "best_epoch": int(checkpoint["epoch"]),
                        }
                        (val_rows if key.startswith("val_") else test_rows).append(audit)
                        if status == "fail":
                            failures.append((dataset, seed, variant, key, error))
                    del model, classifier, z, checkpoint
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                print(f"REPRODUCED {dataset} seed={seed}", flush=True)

    result_dir = ROOT / "results/final_nc"
    _write(result_dir / "final_nc_metrics.csv", metric_rows,
           ["dataset", "seed", "variant", "best_epoch", *METRICS, "checkpoint"])
    audit_fields = ["dataset", "seed", "variant", "metric", "checkpoint_metric",
                    "reproduced_metric", "absolute_error", "tolerance_status", "best_epoch"]
    _write(result_dir / "validation_reproduction.csv", val_rows, audit_fields)
    _write(result_dir / "test_reproduction.csv", test_rows, audit_fields)
    print(f"reproduction rows={len(metric_rows)}; failures={len(failures)}; "
          f"max_abs={max((float(r['absolute_error']) for r in val_rows + test_rows), default=0.0):.3e}")
    if failures:
        raise SystemExit(f"checkpoint reproduction mismatch: {failures[:10]}")


if __name__ == "__main__":
    main()
