from __future__ import annotations

from typing import Any, cast

from athena_optimizer.models import QueryRecord
from athena_optimizer.similarity import (
    build_bm25_distance_matrix,
    build_hybrid_matrix,
    extract_structural_tokens,
)
from athena_optimizer.config import SimilarityConfig


def test_extract_structural_tokens_basic() -> None:
    sql = (
        "SELECT a, count(*) FROM sales s "
        "JOIN dim d ON s.id = d.id "
        "WHERE s.date > '2024-01-01' "
        "GROUP BY a ORDER BY a LIMIT 10"
    )
    tokens = set(extract_structural_tokens(sql))

    assert "table:sales" in tokens
    assert "table:dim" in tokens
    assert "join:inner" in tokens
    assert "col:a" in tokens
    assert "col:id" in tokens
    assert "func:count" in tokens
    assert "has_where" in tokens
    assert "has_group_by" in tokens
    assert "has_order_by" in tokens
    assert "has_limit" in tokens


def test_build_hybrid_matrix_includes_structural_vocab() -> None:
    records = [QueryRecord(query_id="q1", query_text="SELECT a FROM sales")]
    matrix, vectorizers = build_hybrid_matrix(records, SimilarityConfig(), use_structural=True)

    typed_matrix = cast(Any, matrix)
    assert typed_matrix.shape[0] == 1
    assert "structural" in vectorizers
    assert "table:sales" in vectorizers["structural"].vocabulary_


def test_bm25_distance_matrix_similarity() -> None:
    records = [
        QueryRecord(query_id="a", query_text="select foo from bar"),
        QueryRecord(query_id="b", query_text="select foo from bar"),
        QueryRecord(query_id="c", query_text="select baz from qux"),
    ]
    distances = build_bm25_distance_matrix(records)

    assert distances.shape == (3, 3)
    assert distances[0, 1] <= distances[0, 2]
