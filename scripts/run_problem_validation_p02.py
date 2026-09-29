from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.problem_validation.common import (  # noqa: E402
    DATASETS,
    OUTPUT_ROOT,
    RUN_SEEDS,
    make_physical_graph,
    mean_neighbor_messages,
)
from src.data.graph_utils import ensure_edge_index, preprocess_edge_index  # noqa: E402
from src.data.loaders import PROJECT_ROOT, resolve_path  # noqa: E402
from src.analysis.problem_validation.p01plus_statistics import quintile_partition_indices  # noqa: E402
from src.analysis.problem_validation.p02_context import (  # noqa: E402
    CI_METRICS,
    conditional_node_bootstrap,
    conditional_point_estimates,
    percentile_ci,
    recipient_context_redundancy,
)


MODALITIES = ("text", "visual")
EDGE_FIELDS = (
    "target_node",
    "neighbor_node",
    "target_degree",
    "analysis_target_nodes",
    "probe_sim_text",
    "probe_sim_visual",
    "utility_ce_text",
    "utility_ce_visual",
    "utility_margin_text",
    "utility_margin_visual",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline P0.2 context-conditioned utility analysis.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(RUN_SEEDS))
    parser.add_argument("--input-root", type=Path, default=OUTPUT_ROOT,
                        help="Existing P0.1 artifacts root; no training or resampling is performed.")
    parser.add_argument("--split-root", type=Path, default=None,
                        help="Existing P0.1 fixed split caches (defaults to INPUT_ROOT/splits).")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results" / "problem_validation" / "p02")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def _array(payload: dict[str, Any], name: str, dtype: Any | None = None) -> np.ndarray:
    value = payload[name]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _load_existing_run(
    input_root: Path,
    split_root: Path,
    dataset: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    run_dir = input_root / dataset / f"seed{seed}"
    edge_path = run_dir / "edge_analysis.pt"
    embedding_path = run_dir / "semantic_embeddings.pt"
    split_path = split_root / f"{dataset}.pt"
    for path in (edge_path, embedding_path, split_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required existing P0.1 artifact is missing: {path}")

    # Read only the named P0.1 artifacts and required fields. In particular, this
    # does not open metrics, checkpoints, test metrics, or any test-label artifact.
    edge_payload = torch.load(edge_path, map_location="cpu", weights_only=True)
    embedding_payload = torch.load(embedding_path, map_location="cpu", weights_only=True)
    split_payload = torch.load(split_path, map_location="cpu", weights_only=True)
    if not isinstance(edge_payload, dict) or not isinstance(embedding_payload, dict) or not isinstance(split_payload, dict):
        raise TypeError(f"P0.1 run artifacts must be dictionaries: {run_dir}")
    missing = set(EDGE_FIELDS) - edge_payload.keys()
    if missing:
        raise KeyError(f"{edge_path} is missing fields: {sorted(missing)}")
    if edge_payload.get("dataset") != dataset or int(edge_payload.get("run_seed", -1)) != seed:
        raise ValueError(f"Artifact identity mismatch in {edge_path}")
    if int(embedding_payload.get("run_seed", -1)) != seed:
        raise ValueError(f"Embedding seed mismatch in {embedding_path}")
    if int(split_payload.get("data_split_seed", -1)) != 42:
        raise ValueError(f"Expected P0.1 data split seed 42 in {split_path}")

    # Exact sequence equality establishes reuse of the original target/edge sample.
    checks = (
        ("analysis_target_nodes", "analysis_target_nodes"),
        ("target_node", "sampled_edge_target_node"),
        ("neighbor_node", "sampled_edge_neighbor_node"),
    )
    for edge_key, split_key in checks:
        if split_key not in split_payload:
            raise KeyError(f"{split_path} is missing the frozen population field {split_key}")
        if not torch.equal(
            torch.as_tensor(edge_payload[edge_key]).cpu().long(),
            torch.as_tensor(split_payload[split_key]).cpu().long(),
        ):
            raise AssertionError(f"P0.1 edge population changed: {edge_path} vs {split_path} ({edge_key})")
    for key in EDGE_FIELDS:
        value = edge_payload[key]
        if torch.is_tensor(value) and value.ndim == 1 and key not in ("analysis_target_nodes",):
            if value.numel() != int(edge_payload["sampled_relation_count"]):
                raise ValueError(f"P0.1 relation field length mismatch: {edge_path}:{key}")
    for key in ("H_text", "H_visual"):
        if key not in embedding_payload or not torch.is_tensor(embedding_payload[key]) or embedding_payload[key].ndim != 2:
            raise KeyError(f"{embedding_path} must contain a 2-D {key} tensor")
    selected = {key: edge_payload[key] for key in EDGE_FIELDS}
    selected["sampled_relation_count"] = int(edge_payload["sampled_relation_count"])
    embeddings = {key: embedding_payload[key].float().contiguous() for key in ("H_text", "H_visual")}
    return selected, embeddings


def _load_physical_graph(dataset: str, expected_nodes: int) -> torch.Tensor:
    # Load only the graph relation data needed for the context statistic. Do not
    # invoke the NC loader or read the dataset label/split files.
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[f"dataset={dataset}", "task=nc", "seed=42"])
    source = str(cfg.dataset.source).lower()
    if source == "magb":
        import dgl

        graphs, _ = dgl.load_graphs(str(resolve_path(cfg.dataset.graph_path)))
        if not graphs:
            raise ValueError(f"{dataset}: DGL graph file is empty")
        graph = graphs[0]
        num_nodes = int(graph.num_nodes())
        src, dst = graph.edges()
        raw_edges = torch.stack((src.long(), dst.long()), dim=0)
    elif source == "mmgraph":
        raw = torch.load(resolve_path(cfg.dataset.edge_path), map_location="cpu", weights_only=False)
        raw_edges = ensure_edge_index(torch.as_tensor(raw, dtype=torch.long))
        num_nodes = int(expected_nodes)
        raw_edges = preprocess_edge_index(
            raw_edges,
            num_nodes,
            make_undirected=bool(cfg.dataset.get("make_undirected", True)),
            with_self_loops=bool(cfg.dataset.get("add_self_loops", False)),
        )
    else:
        raise ValueError(f"{dataset}: unsupported graph source {source}")
    if num_nodes != int(expected_nodes):
        raise ValueError(f"{dataset}: graph has {num_nodes} nodes but frozen embeddings have {expected_nodes}")
    return make_physical_graph(raw_edges, num_nodes)


def _verify_graph_population(
    dataset: str,
    num_nodes: int,
    physical_edge_index: torch.Tensor,
    payload: dict[str, Any],
) -> None:
    target = torch.as_tensor(payload["target_node"], dtype=torch.long)
    neighbor = torch.as_tensor(payload["neighbor_node"], dtype=torch.long)
    degree = torch.as_tensor(payload["target_degree"], dtype=torch.long)
    target_nodes = torch.as_tensor(payload["analysis_target_nodes"], dtype=torch.long)
    if target.numel() != payload["sampled_relation_count"]:
        raise ValueError(f"{dataset}: sampled_relation_count does not match edge rows")
    if target_nodes.numel() == 0 or torch.unique(target_nodes).numel() != target_nodes.numel():
        raise ValueError(f"{dataset}: analysis_target_nodes must be non-empty and unique")
    if not torch.equal(target_nodes, target_nodes.sort().values):
        raise ValueError(f"{dataset}: analysis_target_nodes must be sorted")
    if bool((target == neighbor).any()):
        raise ValueError(f"{dataset}: P0.1 relation sample contains a self-loop")
    if target.numel() == 0 or int(torch.maximum(target.max(), neighbor.max())) >= num_nodes:
        raise ValueError(f"{dataset}: P0.1 relation sample is empty or has an out-of-range node index")

    # The P0.1 degree is the full physical degree, not the sampled edge count.
    graph_src, graph_dst = physical_edge_index
    graph_degree = torch.bincount(graph_dst, minlength=num_nodes)
    if not torch.equal(graph_degree[target], degree):
        raise AssertionError(f"{dataset}: P0.1 target_degree does not match the physical graph")
    n = int(num_nodes)
    graph_keys = torch.sort(graph_src * n + graph_dst).values
    relation_keys = neighbor * n + target
    positions = torch.searchsorted(graph_keys, relation_keys)
    if bool((positions >= graph_keys.numel()).any()) or not torch.equal(graph_keys[positions], relation_keys):
        raise AssertionError(f"{dataset}: at least one frozen P0.1 relation is absent from the physical graph")
    target_set = torch.isin(target, target_nodes)
    if not bool(target_set.all()):
        raise AssertionError(f"{dataset}: P0.1 sampled relation target is not in analysis_target_nodes")


def _describe(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "median": float("nan"),
                "q25": float("nan"), "q75": float("nan")}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=0)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def _counterexample_rows(
    dataset: str,
    seed: int,
    modality: str,
    similarity: np.ndarray,
    ce_utility: np.ndarray,
    redundancy: np.ndarray,
    disagreement: np.ndarray,
    context_defined: np.ndarray,
) -> list[dict[str, Any]]:
    q1, _, _, _, q5 = quintile_partition_indices(similarity)
    masks = {
        "high_similarity_harmful": (q5, ce_utility < 0.0),
        "high_similarity_expected": (q5, ce_utility > 0.0),
        "low_similarity_beneficial": (q1, ce_utility > 0.0),
        "low_similarity_expected": (q1, ce_utility < 0.0),
    }
    rows: list[dict[str, Any]] = []
    for name, (quantile_indices, sign_mask) in masks.items():
        selected = np.zeros(similarity.size, dtype=bool)
        selected[quantile_indices] = True
        selected &= sign_mask
        row: dict[str, Any] = {
            "dataset": dataset,
            "seed": seed,
            "modality": modality.title(),
            "subset": name,
            "relation_count": int(selected.sum()),
        }
        for label, values, valid_mask in (
            ("redundancy", redundancy, context_defined),
            ("disagreement", disagreement, np.ones(similarity.size, dtype=bool)),
        ):
            stats = _describe(values[selected & valid_mask])
            for key, value in stats.items():
                row[f"{label}_{key}"] = value
        rows.append(row)
    return rows


def _attach_ci(
    rows: list[dict[str, Any]],
    bootstrap: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    for row in rows:
        key = str(row["similarity_bin"])
        for metric in CI_METRICS:
            low, high = percentile_ci(bootstrap[key][metric])
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    invalid_seeds = sorted(set(args.seeds) - set(RUN_SEEDS))
    if invalid_seeds:
        raise SystemExit(f"seeds must be selected from {RUN_SEEDS}; got {invalid_seeds}")
    if args.bootstrap_replicates < 1:
        raise SystemExit("--bootstrap-replicates must be positive")
    split_root = args.split_root or args.input_root / "splits"
    print(
        f"P0.2 offline analysis: datasets={args.datasets}; seeds={args.seeds}; "
        f"input={args.input_root}; splits={split_root}; output={args.output_root}; "
        f"node bootstrap={args.bootstrap_replicates}, seed={args.bootstrap_seed}",
        flush=True,
    )
    per_seed_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    counterexample_rows: list[dict[str, Any]] = []

    for dataset in args.datasets:
        print(f"\n=== {dataset}: loading physical graph without NC labels or splits ===", flush=True)
        first_seed = args.seeds[0]
        first_payload, first_embeddings = _load_existing_run(args.input_root, split_root, dataset, first_seed)
        num_nodes = int(first_embeddings["H_text"].size(0))
        physical = _load_physical_graph(dataset, num_nodes)
        for seed in args.seeds:
            print(f"--- {dataset} seed{seed}: reuse exact P0.1 relation rows ---", flush=True)
            if seed == first_seed:
                payload, embeddings = first_payload, first_embeddings
            else:
                payload, embeddings = _load_existing_run(args.input_root, split_root, dataset, seed)
            _verify_graph_population(dataset, num_nodes, physical, payload)
            if embeddings["H_text"].size(0) != num_nodes or embeddings["H_visual"].size(0) != num_nodes:
                raise ValueError(f"{dataset} seed{seed}: embedding node count disagrees with physical graph")
            for modality, key in (("text", "probe_sim_text"), ("visual", "probe_sim_visual")):
                if not torch.isfinite(torch.as_tensor(payload[key])).all():
                    raise FloatingPointError(f"{dataset} seed{seed}: non-finite {key}")

            target = _array(payload, "target_node", np.int64)
            neighbor = _array(payload, "neighbor_node", np.int64)
            degree = _array(payload, "target_degree", np.int64)
            target_population = _array(payload, "analysis_target_nodes", np.int64)
            full_mean_text, isolated_text = mean_neighbor_messages(embeddings["H_text"], physical)
            full_mean_visual, isolated_visual = mean_neighbor_messages(embeddings["H_visual"], physical)
            if not torch.equal(isolated_text, isolated_visual):
                raise AssertionError(f"{dataset}: text and visual physical graph degree masks differ")
            redundancy_text, defined_text = recipient_context_redundancy(
                embeddings["H_text"], full_mean_text, target, neighbor, degree
            )
            redundancy_visual, defined_visual = recipient_context_redundancy(
                embeddings["H_visual"], full_mean_visual, target, neighbor, degree
            )
            if not np.array_equal(defined_text, degree > 1) or not np.array_equal(defined_visual, degree > 1):
                raise AssertionError("context_defined must be exactly equivalent to target_degree > 1")
            if not np.isfinite(redundancy_text[defined_text]).all() or not np.isfinite(redundancy_visual[defined_visual]).all():
                raise FloatingPointError(f"{dataset} seed{seed}: non-finite defined recipient redundancy")
            sim_text = _array(payload, "probe_sim_text", np.float64)
            sim_visual = _array(payload, "probe_sim_visual", np.float64)
            disagreement = np.abs(sim_text - sim_visual)
            context_defined = degree > 1
            total_count = int(degree.size)
            context_count = int(context_defined.sum())
            degree1_count = total_count - context_count
            coverage_rows.append({
                "dataset": dataset,
                "seed": seed,
                "sampled_relation_count": total_count,
                "context_defined_relation_count": context_count,
                "retained_relation_coverage": context_count / total_count,
                "degree1_relation_count": degree1_count,
                "degree1_relation_proportion": degree1_count / total_count,
                "context_defined_rule": "target_degree > 1",
            })

            for modality, sim, redundancy, ce, margin in (
                ("text", sim_text, redundancy_text, "utility_ce_text", "utility_margin_text"),
                ("visual", sim_visual, redundancy_visual, "utility_ce_visual", "utility_margin_visual"),
            ):
                ce_values = _array(payload, ce, np.float64)
                margin_values = _array(payload, margin, np.float64)
                utility_arrays = {"ce": ce_values, "margin": margin_values}
                # Primary: only relations whose recipient has another physical neighbor.
                for analysis, mask, descriptor in (
                    ("recipient_redundancy", context_defined, redundancy),
                    # Secondary: disagreement is defined for every physical relation.
                    ("cross_modal_disagreement", np.ones(total_count, dtype=bool), disagreement),
                ):
                    point = conditional_point_estimates(sim[mask], descriptor[mask],
                                                        {k: v[mask] for k, v in utility_arrays.items()})
                    boot = conditional_node_bootstrap(
                        sim[mask], descriptor[mask], {k: v[mask] for k, v in utility_arrays.items()},
                        target[mask], target_population,
                        replicates=args.bootstrap_replicates,
                        seed=args.bootstrap_seed,
                    )
                    output_rows: list[dict[str, Any]] = []
                    for row in point:
                        row.update({
                            "dataset": dataset,
                            "seed": seed,
                            "modality": modality.title(),
                            "analysis": analysis,
                            "context_population": "degree_gt_1" if analysis == "recipient_redundancy" else "all_sampled_relations",
                            "sampled_relation_count": total_count,
                            "analysis_relation_count": int(mask.sum()),
                            "bootstrap_replicates": args.bootstrap_replicates,
                            "bootstrap_seed": args.bootstrap_seed,
                            "bootstrap_resampling_unit": "target_node; all sampled relations carried together",
                            "similarity_quantile_rule": "P0.1+ stable mergesort + balanced array_split quintiles",
                            "conditional_split_rule": "stable mergesort + balanced low/high halves within quintile",
                            "low_group": "low recipient redundancy" if analysis == "recipient_redundancy" else "low cross-modal disagreement",
                            "high_group": "high recipient redundancy" if analysis == "recipient_redundancy" else "high cross-modal disagreement",
                        })
                        output_rows.append(row)
                    _attach_ci(output_rows, boot)
                    per_seed_rows.extend(output_rows)

                counterexample_rows.extend(_counterexample_rows(
                    dataset, seed, modality, sim, ce_values, redundancy, disagreement, context_defined
                ))
            print(
                f"  relations={total_count}; context-defined={context_count} ({context_count / total_count:.3%}); "
                f"degree-1={degree1_count}",
                flush=True,
            )
            del payload, embeddings, redundancy_text, redundancy_visual, full_mean_text, full_mean_visual
        del first_payload, first_embeddings, physical

    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_root / "p02_per_seed.csv", per_seed_rows)
    _write_csv(args.output_root / "p02_context_coverage.csv", coverage_rows)
    _write_csv(args.output_root / "p02_counterexamples.csv", counterexample_rows)
    from scripts.summarize_problem_validation_p02 import summarize

    summarize(args.output_root)
    print(f"P0.2 outputs written under {args.output_root}", flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

