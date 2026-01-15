from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from athena_optimizer.models import QueryRecord

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_SQL_STOPWORDS = {
    "select",
    "from",
    "where",
    "join",
    "inner",
    "left",
    "right",
    "full",
    "outer",
    "on",
    "group",
    "by",
    "order",
    "limit",
    "having",
    "as",
    "distinct",
    "and",
    "or",
    "not",
    "case",
    "when",
    "then",
    "else",
    "end",
    "with",
    "union",
    "all",
    "insert",
    "update",
    "delete",
    "into",
    "values",
    "create",
    "table",
    "view",
    "materialized",
    "over",
}


def build_bm25_distance_matrix(records: Sequence[QueryRecord]) -> np.ndarray:
    tokens = [_tokenize(_record_text(record)) for record in records]
    count = len(tokens)
    if count == 0:
        return np.zeros((0, 0))

    bm25 = BM25Okapi(tokens)
    scores = np.vstack([bm25.get_scores(query) for query in tokens])
    sim = (scores + scores.T) / 2

    row_min = scores.min(axis=1, keepdims=True)
    row_max = scores.max(axis=1, keepdims=True)
    row_range = row_max - row_min
    zero_rows = (row_range == 0).flatten()

    row_range = np.where(row_range == 0, 1.0, row_range)
    row_norm = (scores - row_min) / row_range
    if zero_rows.any():
        row_norm[zero_rows, :] = 1.0

    sim = (row_norm + row_norm.T) / 2
    distances = 1.0 - sim
    np.fill_diagonal(distances, 0.0)
    return distances


def _tokenize(text: str) -> list[str]:
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    return [token for token in tokens if token not in _SQL_STOPWORDS]


def _record_text(record: QueryRecord) -> str:
    return record.normalized_query or record.query_text
