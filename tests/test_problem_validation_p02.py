from __future__ import annotations

import numpy as np
import torch

from src.analysis.problem_validation.p01plus_statistics import quintile_partition_indices
from src.analysis.problem_validation.p02_context import (
    conditional_node_bootstrap,
    conditional_point_estimates,
    recipient_context_redundancy,
)


def test_recipient_context_redundancy_uses_exact_other_neighbor_mean() -> None:
    embeddings = torch.tensor([
        [0.5, 0.5],  # recipient 0 has neighbors 1 and 2
        [1.0, 0.0],
        [0.0, 2.0],
        [1.0, 1.0],  # recipient 3 has only neighbor 0
    ])
    full_mean = torch.tensor([
        [0.5, 1.0],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
    ])
    redundancy, defined = recipient_context_redundancy(
        embeddings,
        full_mean,
        target_node=np.array([0, 3]),
        neighbor_node=np.array([1, 0]),
        target_degree=np.array([2, 1]),
    )
    # Excluding source 1 from recipient 0 leaves exactly H_2, orthogonal to H_1.
    assert defined.tolist() == [True, False]
    assert redundancy[0] == 0.0
    assert np.isnan(redundancy[1])


def test_similarity_bins_reuse_stable_balanced_p01plus_partition() -> None:
    similarity = np.repeat(np.arange(5, dtype=float), 4)
    descriptor = np.array([3, 1, 2, 0] * 5, dtype=float)
    utility = np.arange(similarity.size, dtype=float)
    rows = conditional_point_estimates(similarity, descriptor, {"ce": utility})
    groups = quintile_partition_indices(similarity)
    assert [len(group) for group in groups] == [4, 4, 4, 4, 4]
    assert [row["similarity_bin"] for row in rows] == ["Q1", "Q2", "Q3", "Q4", "Q5", "weighted_overall"]
    for row in rows[:5]:
        assert abs(row["low_relation_count"] - row["high_relation_count"]) <= 1


def test_conditional_node_bootstrap_is_seeded_and_returns_all_bins() -> None:
    target_node = np.repeat(np.arange(20), 2)
    similarity = np.tile(np.arange(10, dtype=float), 4)
    descriptor = np.linspace(-1.0, 1.0, 40)
    utility = np.where(np.arange(40) % 3, 0.5, -0.25)
    args = (
        similarity,
        descriptor,
        {"ce": utility},
        target_node,
        np.arange(20),
    )
    one = conditional_node_bootstrap(*args, replicates=12, seed=42, batch_size=4)
    two = conditional_node_bootstrap(*args, replicates=12, seed=42, batch_size=7)
    assert set(one) == {"Q1", "Q2", "Q3", "Q4", "Q5", "weighted_overall"}
    for key in one:
        for metric in one[key]:
            np.testing.assert_allclose(one[key][metric], two[key][metric], equal_nan=True)

