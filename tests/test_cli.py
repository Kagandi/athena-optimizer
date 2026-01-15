from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from athena_optimizer.cli import (
    _apply_thresholds,
    _auto_distance_threshold,
    _build_similarity_config,
    _bytes_to_gb,
    _cluster_to_dict,
    _format_scanned,
    _ms_to_seconds,
    _parse_time,
    _scored_to_dict,
    _shorten,
)
from athena_optimizer.models import Cluster, QueryRecord
from athena_optimizer.scoring import ScoredRecord


def _make_scored(
    *,
    query_id: str,
    bytes_scanned: int | None,
    runtime_ms: int | None,
) -> ScoredRecord:
    record = QueryRecord(
        query_id=query_id,
        query_text="SELECT 1",
        data_scanned_bytes=bytes_scanned,
        runtime_ms=runtime_ms,
    )
    return ScoredRecord(record=record, frequency=1, cost_score=1.0, estimated_cost_usd=0.0)


def test_apply_thresholds_or_semantics() -> None:
    scored = [
        _make_scored(query_id="a", bytes_scanned=2 * 1024**3, runtime_ms=500),
        _make_scored(query_id="b", bytes_scanned=100 * 1024**2, runtime_ms=5000),
        _make_scored(query_id="c", bytes_scanned=0, runtime_ms=0),
    ]

    filtered = _apply_thresholds(scored, min_gb=1.0, min_runtime_seconds=3.0)
    kept_ids = [item.record.query_id for item in filtered]

    assert kept_ids == ["a", "b"]


def test_apply_thresholds_none_active_returns_all() -> None:
    scored = [_make_scored(query_id="a", bytes_scanned=0, runtime_ms=0)]
    filtered = _apply_thresholds(scored, min_gb=0.0, min_runtime_seconds=0.0)
    assert filtered == scored


def test_unit_conversions() -> None:
    assert _bytes_to_gb(1024**3) == 1.0
    assert _bytes_to_gb(None) is None
    assert _format_scanned(999.9) == "999.900 GB"
    assert _format_scanned(1000.0) == "1.000 TB"
    assert _ms_to_seconds(2500) == 2.5
    assert _ms_to_seconds(None) is None


def test_scored_to_dict_units() -> None:
    record = QueryRecord(
        query_id="q1",
        query_text="SELECT 1",
        data_scanned_bytes=1024**3,
        bytes_output=2 * 1024**3,
        runtime_ms=2000,
        submission_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    scored = ScoredRecord(
        record=record,
        frequency=2,
        cost_score=3.5,
        estimated_cost_usd=0.01,
    )

    payload = _scored_to_dict(scored, metadata_as_json=False)

    assert payload["runtime_seconds"] == 2.0
    assert payload["data_scanned_gb"] == 1.0
    assert payload["data_scanned_display"] == "1.000 GB"
    assert payload["bytes_output_gb"] == 2.0
    assert payload["bytes_output_display"] == "2.000 GB"
    assert payload["frequency"] == 2
    assert payload["cost_score"] == 3.5
    assert payload["estimated_cost_usd"] == 0.01


def test_shorten_handles_none() -> None:
    assert _shorten(None, 10) == ""
    assert _shorten("short", 10) == "short"
    assert _shorten("0123456789AB", 10) == "0123456..."


def test_build_similarity_config_validates_ngram_range() -> None:
    config = _build_similarity_config(
        max_features=100,
        ngram_min=1,
        ngram_max=2,
        min_df=1,
        max_df=1.0,
        lexical_weight=0.8,
        structural_weight=0.2,
    )
    assert config.ngram_range == (1, 2)

    with pytest.raises(Exception):
        _build_similarity_config(
            max_features=100,
            ngram_min=2,
            ngram_max=1,
            min_df=1,
            max_df=1.0,
            lexical_weight=0.8,
            structural_weight=0.2,
        )


def test_cluster_to_dict_includes_scanned_gb() -> None:
    members = [
        QueryRecord(
            query_id="a",
            query_text="SELECT 1",
            data_scanned_bytes=1024**3,
            runtime_ms=1000,
        ),
        QueryRecord(
            query_id="b",
            query_text="SELECT 2",
            data_scanned_bytes=2 * 1024**3,
            runtime_ms=3000,
        ),
    ]
    cluster = Cluster(cluster_id="cluster-1", members=members, representative_query=members[0])

    payload = _cluster_to_dict(
        cluster,
        member_ids_as_string=False,
        price_per_tb_usd=1.0,
    )

    expected_cost = (3 * 1024**3) / 10**12
    assert payload["total_scanned_gb"] == 3.0
    assert payload["total_scanned_display"] == "3.000 GB"
    assert payload["total_cost_usd"] == pytest.approx(expected_cost)
    assert payload["median_runtime_seconds"] == pytest.approx(2.0)
    assert payload["median_runtime_display"] == "2.00 s"


def test_auto_distance_threshold_uses_quantile() -> None:
    matrix = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    threshold = _auto_distance_threshold(matrix, quantile=0.5, max_samples=10)

    assert 0.05 <= threshold <= 1.5


def test_parse_time_accepts_date_only() -> None:
    value = _parse_time("2024-01-01")
    assert value is not None
    assert value.isoformat().startswith("2024-01-01")
