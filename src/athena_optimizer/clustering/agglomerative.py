from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

from athena_optimizer.config import ClusteringConfig
from athena_optimizer.models import Cluster, QueryRecord


def cluster_records(
    records: Sequence[QueryRecord],
    matrix,
    config: ClusteringConfig,
    *,
    metric: str = "cosine",
) -> list[Cluster]:
    if not records:
        return []
    if len(records) == 1:
        return [
            Cluster(cluster_id="cluster-0", members=[records[0]], representative_query=records[0])
        ]

    if metric == "precomputed":
        distances = matrix
    elif metric == "cosine":
        distances = cosine_distances(matrix)
    else:
        raise ValueError(f"Unsupported metric '{metric}'.")

    params: dict[str, Any] = {
        "n_clusters": None,
        "metric": "precomputed",
        "linkage": config.linkage,
        "distance_threshold": config.distance_threshold,
    }
    model = AgglomerativeClustering(**params)
    labels = model.fit_predict(distances)

    grouped: dict[int, list[QueryRecord]] = defaultdict(list)
    for record, label in zip(records, labels, strict=False):
        grouped[int(label)].append(record)

    clusters: list[Cluster] = []
    for label, members in grouped.items():
        representative = members[0] if members else None
        clusters.append(
            Cluster(
                cluster_id=f"cluster-{label}",
                members=members,
                representative_query=representative,
            )
        )
    return clusters
