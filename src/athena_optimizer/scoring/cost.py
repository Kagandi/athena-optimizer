from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from athena_optimizer.config import ScoringConfig
from athena_optimizer.models import QueryRecord

_BYTES_PER_TB = 10**12


@dataclass(frozen=True)
class ScoredRecord:
    record: QueryRecord
    frequency: int
    cost_score: float
    estimated_cost_usd: float


def score_records(
    records: Iterable[QueryRecord],
    config: ScoringConfig,
) -> list[ScoredRecord]:
    record_list = list(records)
    frequency_map = _build_frequency_map(record_list)

    scored: list[ScoredRecord] = []
    for record in record_list:
        frequency = frequency_map[_frequency_key(record)]
        cost_score = _cost_score(record, frequency=frequency, config=config)
        estimated_cost_usd = _estimate_cost_usd(record, config=config)
        scored.append(
            ScoredRecord(
                record=record,
                frequency=frequency,
                cost_score=cost_score,
                estimated_cost_usd=estimated_cost_usd,
            )
        )
    return scored


def _build_frequency_map(records: list[QueryRecord]) -> dict[str, int]:
    frequency_map: dict[str, int] = {}
    for record in records:
        key = _frequency_key(record)
        frequency_map[key] = frequency_map.get(key, 0) + 1
    return frequency_map


def _frequency_key(record: QueryRecord) -> str:
    return record.normalized_query or record.query_text


def _cost_score(record: QueryRecord, *, frequency: int, config: ScoringConfig) -> float:
    bytes_score = _log1p(record.data_scanned_bytes)
    runtime_score = _log1p(record.runtime_ms)
    frequency_score = _log1p(frequency)

    return (
        config.bytes_weight * bytes_score
        + config.runtime_weight * runtime_score
        + config.frequency_weight * frequency_score
    )


def _estimate_cost_usd(record: QueryRecord, *, config: ScoringConfig) -> float:
    bytes_scanned = record.data_scanned_bytes or 0
    if bytes_scanned <= 0 or config.price_per_tb_usd <= 0:
        return 0.0
    tb_scanned = bytes_scanned / _BYTES_PER_TB
    return tb_scanned * config.price_per_tb_usd


def _log1p(value: int | None) -> float:
    if value is None:
        return 0.0
    if value < 0:
        value = 0
    return math.log1p(value)
