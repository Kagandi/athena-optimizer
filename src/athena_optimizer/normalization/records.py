from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from athena_optimizer.config import NormalizationConfig
from athena_optimizer.models import QueryRecord
from athena_optimizer.normalization.sqlglot_normalizer import normalize_sql


def normalize_records(
    records: Iterable[QueryRecord],
    config: NormalizationConfig,
) -> tuple[list[QueryRecord], int]:
    normalized: list[QueryRecord] = []
    failures = 0

    for record in records:
        try:
            normalized_query = normalize_sql(
                record.query_text,
                dialect=config.dialect,
                anonymize_literals=config.anonymize_literals,
            )
        except ValueError:
            normalized_query = None
            failures += 1
        normalized.append(replace(record, normalized_query=normalized_query))

    return normalized, failures
