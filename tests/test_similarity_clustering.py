from __future__ import annotations

import numpy as np

from athena_optimizer.clustering import cluster_records
from athena_optimizer.config import ClusteringConfig, SimilarityConfig
from athena_optimizer.models import QueryRecord
from athena_optimizer.similarity import build_tfidf_matrix


def test_build_tfidf_matrix_prefers_normalized_query() -> None:
    records = [
        QueryRecord(query_id="a", query_text="select bar", normalized_query="select foo"),
        QueryRecord(query_id="b", query_text="select baz"),
    ]

    _, vectorizers = build_tfidf_matrix(records, SimilarityConfig())
    vocab = set(vectorizers["lexical"].vocabulary_.keys())

    assert "foo" in vocab
    assert "baz" in vocab
    assert "bar" not in vocab


def test_cluster_records_groups_similar_queries() -> None:
    records = [
        QueryRecord(query_id="a", query_text="select 1"),
        QueryRecord(query_id="b", query_text="select 2"),
        QueryRecord(query_id="c", query_text="select 3"),
    ]
    matrix = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ]
    )

    clusters = cluster_records(
        records,
        matrix,
        ClusteringConfig(distance_threshold=0.05),
    )
    sizes = sorted(len(cluster.members) for cluster in clusters)

    assert sizes == [1, 2]
