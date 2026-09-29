from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.analysis.problem_validation.p01plus_statistics import quintile_partition_indices


UTILITY_KINDS = ("ce", "margin")
CI_METRICS = (
    "delta_mean_ce",
    "delta_beneficial_rate_ce",
    "delta_harmful_rate_ce",
    "delta_mean_margin",
    "delta_beneficial_rate_margin",
    "delta_harmful_rate_margin",
)


def _numpy(value: Any, name: str, dtype: Any | None = None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=dtype)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def recipient_context_redundancy(
    embeddings: torch.Tensor,
    full_mean_neighbor: torch.Tensor,
    target_node: Any,
    neighbor_node: Any,
    target_degree: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cosine(source, recipient's other-neighbor mean) per P0.1 row.

    The exact full-mean subtraction avoids an edge-by-edge Python neighbor walk.
    Degree-one rows are retained with ``context_defined=False`` and NaN values.
    """
    target = torch.as_tensor(target_node, dtype=torch.long, device=embeddings.device)
    neighbor = torch.as_tensor(neighbor_node, dtype=torch.long, device=embeddings.device)
    degree = torch.as_tensor(target_degree, dtype=torch.long, device=embeddings.device)
    if target.ndim != 1 or neighbor.ndim != 1 or degree.ndim != 1:
        raise ValueError("target_node, neighbor_node, and target_degree must be one-dimensional")
    if not (target.numel() == neighbor.numel() == degree.numel()):
        raise ValueError("relation arrays must have the same length")
    if embeddings.ndim != 2 or full_mean_neighbor.shape != embeddings.shape:
        raise ValueError("embeddings and full_mean_neighbor must have matching [nodes, dim] shapes")
    if target.numel() and (int(target.max()) >= embeddings.size(0) or int(neighbor.max()) >= embeddings.size(0)):
        raise ValueError("relation node index is outside the embedding table")
    if bool((degree < 1).any()):
        raise ValueError("sampled relations must have physical target degree >= 1")

    defined = degree > 1
    result = torch.full((target.numel(),), float("nan"), dtype=torch.float64, device=embeddings.device)
    if bool(defined.any()):
        d = degree[defined].to(embeddings.dtype).unsqueeze(-1)
        other_mean = (
            d * full_mean_neighbor[target[defined]] - embeddings[neighbor[defined]]
        ) / (d - 1.0)
        values = F.cosine_similarity(embeddings[neighbor[defined]], other_mean, dim=-1, eps=1e-8)
        result[defined] = values.to(torch.float64)
    return result.cpu().numpy(), defined.cpu().numpy()


def _validate_arrays(
    similarity: Any,
    descriptor: Any,
    utilities: dict[str, Any],
    target_node: Any,
    analysis_target_nodes: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    sim = _numpy(similarity, "similarity", np.float64)
    desc = _numpy(descriptor, "descriptor", np.float64)
    nodes = _numpy(target_node, "target_node", np.int64)
    population = _numpy(analysis_target_nodes, "analysis_target_nodes", np.int64)
    if sim.size == 0 or not (sim.size == desc.size == nodes.size):
        raise ValueError("similarity, descriptor, and target_node must have the same non-zero length")
    if not np.isfinite(sim).all() or not np.isfinite(desc).all():
        raise ValueError("similarity and context descriptor must be finite in the analysis population")
    if population.size == 0 or len(np.unique(population)) != population.size:
        raise ValueError("analysis_target_nodes must be a non-empty unique target population")
    if np.any(population[1:] < population[:-1]):
        raise ValueError("analysis_target_nodes must be sorted")
    edge_index = np.searchsorted(population, nodes)
    if (edge_index >= population.size).any() or not np.array_equal(population[edge_index], nodes):
        raise ValueError("every sampled relation target must belong to analysis_target_nodes")
    if not utilities:
        raise ValueError("at least one utility array is required")
    util_arrays: dict[str, np.ndarray] = {}
    for name, values in utilities.items():
        util = _numpy(values, name, np.float64)
        if util.size != sim.size or not np.isfinite(util).all():
            raise ValueError(f"{name} must be finite and match the relation population")
        util_arrays[name] = util
    return sim, desc, util_arrays, nodes, population


def _metric_values(
    low_weights: np.ndarray,
    high_weights: np.ndarray,
    utilities: dict[str, np.ndarray],
) -> dict[str, float]:
    result: dict[str, float] = {
        "low_relation_count": float(np.sum(low_weights)),
        "high_relation_count": float(np.sum(high_weights)),
    }
    for kind, values in utilities.items():
        low_n, high_n = result["low_relation_count"], result["high_relation_count"]
        low_mean = float(np.dot(low_weights, values) / low_n) if low_n else float("nan")
        high_mean = float(np.dot(high_weights, values) / high_n) if high_n else float("nan")
        low_benefit = float(np.dot(low_weights, values > 0.0) / low_n) if low_n else float("nan")
        high_benefit = float(np.dot(high_weights, values > 0.0) / high_n) if high_n else float("nan")
        low_harm = float(np.dot(low_weights, values < 0.0) / low_n) if low_n else float("nan")
        high_harm = float(np.dot(high_weights, values < 0.0) / high_n) if high_n else float("nan")
        result.update({
            f"low_mean_{kind}": low_mean,
            f"high_mean_{kind}": high_mean,
            f"delta_mean_{kind}": low_mean - high_mean,
            f"low_beneficial_rate_{kind}": low_benefit,
            f"high_beneficial_rate_{kind}": high_benefit,
            f"delta_beneficial_rate_{kind}": low_benefit - high_benefit,
            f"low_harmful_rate_{kind}": low_harm,
            f"high_harmful_rate_{kind}": high_harm,
            f"delta_harmful_rate_{kind}": low_harm - high_harm,
        })
    return result


def _balanced_weight_segments(order: np.ndarray, weights: np.ndarray, segments: int) -> np.ndarray:
    """Partition integer/fractional row mass into balanced stable rank segments."""
    if segments < 1:
        raise ValueError("segments must be positive")
    mass = float(np.sum(weights, dtype=np.float64))
    if mass <= 0.0:
        return np.zeros((segments, weights.size), dtype=np.float64)
    total = int(round(mass))
    if not np.isclose(mass, total, atol=1e-7):
        raise ValueError(f"balanced split expects integer total mass; got {mass}")
    base, extra = divmod(total, segments)
    sizes = np.full(segments, base, dtype=np.int64)
    sizes[:extra] += 1
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    end = np.cumsum(sorted_weights)
    start = end - sorted_weights
    boundaries = np.r_[0, np.cumsum(sizes)]
    output = np.zeros((segments, weights.size), dtype=np.float64)
    for segment in range(segments):
        overlap = np.maximum(
            0.0,
            np.minimum(end, float(boundaries[segment + 1])) - np.maximum(start, float(boundaries[segment])),
        )
        output[segment, order] = overlap
    return output


def conditional_point_estimates(
    similarity: Any,
    descriptor: Any,
    utilities: dict[str, Any],
) -> list[dict[str, float | str]]:
    """Stable balanced similarity quintiles followed by balanced descriptor halves."""
    sim, desc = _numpy(similarity, "similarity", np.float64), _numpy(descriptor, "descriptor", np.float64)
    if sim.size == 0 or sim.size != desc.size or not np.isfinite(sim).all() or not np.isfinite(desc).all():
        raise ValueError("similarity and descriptor must be finite non-empty arrays of equal length")
    util_arrays = {name: _numpy(value, name, np.float64) for name, value in utilities.items()}
    if not util_arrays or any(v.size != sim.size or not np.isfinite(v).all() for v in util_arrays.values()):
        raise ValueError("utility arrays must be finite and match similarity")
    rows: list[dict[str, float | str]] = []
    for q, indices in enumerate(quintile_partition_indices(sim), start=1):
        if indices.size < 2:
            raise ValueError("at least 10 relations are required for five quintiles and balanced halves")
        descriptor_order = np.argsort(desc[indices], kind="mergesort")
        halves = np.array_split(indices[descriptor_order], 2)
        low_weights = np.zeros(sim.size, dtype=np.float64)
        high_weights = np.zeros(sim.size, dtype=np.float64)
        low_weights[halves[0]] = 1.0
        high_weights[halves[1]] = 1.0
        row = _metric_values(low_weights, high_weights, util_arrays)
        row["similarity_bin"] = f"Q{q}"
        row["similarity_bin_relation_count"] = float(indices.size)
        rows.append(row)
    overall = {"similarity_bin": "weighted_overall"}
    q_counts = np.asarray([float(row["similarity_bin_relation_count"]) for row in rows])
    overall["similarity_bin_relation_count"] = float(q_counts.sum())
    numeric_keys = [k for k in rows[0] if k not in ("similarity_bin", "similarity_bin_relation_count")]
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        if key in ("low_relation_count", "high_relation_count"):
            overall[key] = float(np.sum(values))
        else:
            overall[key] = float(np.average(values, weights=q_counts))
    rows.append(overall)
    return rows


def _batch_overlap(sorted_weights: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    end = np.cumsum(sorted_weights, axis=1, dtype=np.float64)
    start = end - sorted_weights
    return np.maximum(0.0, np.minimum(end, upper[:, None]) - np.maximum(start, lower[:, None]))


def _batch_metric_values(
    low_weights: np.ndarray,
    high_weights: np.ndarray,
    utilities: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    low_n = np.sum(low_weights, axis=1)
    high_n = np.sum(high_weights, axis=1)
    result: dict[str, np.ndarray] = {
        "low_relation_count": low_n,
        "high_relation_count": high_n,
    }
    for kind, values in utilities.items():
        low_mean = np.divide(low_weights @ values, low_n, out=np.full(low_n.shape, np.nan), where=low_n > 0)
        high_mean = np.divide(high_weights @ values, high_n, out=np.full(high_n.shape, np.nan), where=high_n > 0)
        low_benefit = np.divide(low_weights @ (values > 0.0), low_n, out=np.full(low_n.shape, np.nan), where=low_n > 0)
        high_benefit = np.divide(high_weights @ (values > 0.0), high_n, out=np.full(high_n.shape, np.nan), where=high_n > 0)
        low_harm = np.divide(low_weights @ (values < 0.0), low_n, out=np.full(low_n.shape, np.nan), where=low_n > 0)
        high_harm = np.divide(high_weights @ (values < 0.0), high_n, out=np.full(high_n.shape, np.nan), where=high_n > 0)
        result.update({
            f"low_mean_{kind}": low_mean,
            f"high_mean_{kind}": high_mean,
            f"delta_mean_{kind}": low_mean - high_mean,
            f"low_beneficial_rate_{kind}": low_benefit,
            f"high_beneficial_rate_{kind}": high_benefit,
            f"delta_beneficial_rate_{kind}": low_benefit - high_benefit,
            f"low_harmful_rate_{kind}": low_harm,
            f"high_harmful_rate_{kind}": high_harm,
            f"delta_harmful_rate_{kind}": low_harm - high_harm,
        })
    return result


def conditional_node_bootstrap(
    similarity: Any,
    descriptor: Any,
    utilities: dict[str, Any],
    target_node: Any,
    analysis_target_nodes: Any,
    *,
    replicates: int = 1000,
    seed: int = 42,
    batch_size: int = 64,
) -> dict[str, dict[str, np.ndarray]]:
    # Batched target-node resampling; all relations for a node move together.
    if replicates < 1 or batch_size < 1:
        raise ValueError("replicates and batch_size must be positive")
    sim, desc, util_arrays, nodes, population = _validate_arrays(
        similarity, descriptor, utilities, target_node, analysis_target_nodes
    )
    node_index = np.searchsorted(population, nodes)
    sim_order = np.argsort(sim, kind="mergesort")
    descriptor_order = np.argsort(desc, kind="mergesort")
    sorted_utilities = {name: values[descriptor_order] for name, values in util_arrays.items()}
    keys = [*(f"Q{q}" for q in range(1, 6)), "weighted_overall"]
    batches: dict[str, dict[str, list[np.ndarray]]] = {
        key: {metric: [] for metric in CI_METRICS} for key in keys
    }
    rng = np.random.default_rng(seed)
    n_targets = population.size
    remaining = replicates
    while remaining:
        batch = min(batch_size, remaining)
        draws = rng.integers(0, n_targets, size=(batch, n_targets))
        multiplicity = np.zeros((batch, n_targets), dtype=np.int32)
        np.add.at(multiplicity, (np.arange(batch)[:, None], draws), 1)
        edge_weights = multiplicity[:, node_index].astype(np.float64, copy=False)
        sim_sorted_weights = edge_weights[:, sim_order]
        totals = np.sum(edge_weights, axis=1)
        base = totals // 5
        extra = totals % 5
        q_size = np.empty((batch, 5), dtype=np.float64)
        q_metric_values: list[dict[str, np.ndarray]] = []
        for q in range(5):
            lower = q * base + np.minimum(q, extra)
            size = base + (q < extra)
            upper = lower + size
            q_size[:, q] = size
            q_sorted = _batch_overlap(sim_sorted_weights, lower, upper)
            q_weights = np.zeros_like(q_sorted)
            q_weights[:, sim_order] = q_sorted
            descriptor_sorted = q_weights[:, descriptor_order]
            low_size = size // 2 + size % 2
            low_sorted = _batch_overlap(descriptor_sorted, np.zeros(batch), low_size)
            high_sorted = _batch_overlap(descriptor_sorted, low_size, size)
            estimates = _batch_metric_values(low_sorted, high_sorted, sorted_utilities)
            q_metric_values.append(estimates)
            q_key = f"Q{q + 1}"
            for metric in CI_METRICS:
                batches[q_key][metric].append(estimates[metric])

        denominator = np.sum(q_size, axis=1)
        for metric in CI_METRICS:
            weighted = np.zeros(batch, dtype=np.float64)
            for q in range(5):
                weighted += q_size[:, q] * q_metric_values[q][metric]
            overall = np.divide(weighted, denominator, out=np.full(batch, np.nan), where=denominator > 0)
            batches["weighted_overall"][metric].append(overall)
        remaining -= batch
    return {
        key: {metric: np.concatenate(chunks) for metric, chunks in metrics.items()}
        for key, metrics in batches.items()
    }


def percentile_ci(samples: Any) -> tuple[float, float]:
    values = _numpy(samples, "bootstrap samples", np.float64)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)

