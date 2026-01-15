from __future__ import annotations

from athena_optimizer.models import QueryRecord
from athena_optimizer.recommendations import recommend_for_scored
from athena_optimizer.scoring import ScoredRecord


def test_recommend_select_star() -> None:
    record = QueryRecord(
        query_id="q1",
        query_text="SELECT * FROM table",
        data_scanned_bytes=5 * 1024**3,
        runtime_ms=5000,
    )
    scored = [ScoredRecord(record=record, frequency=1, cost_score=1.0, estimated_cost_usd=0.1)]

    recommendations = recommend_for_scored(scored)

    types = {rec.recommendation.recommendation_type for rec in recommendations}
    assert "avoid_select_star" in types
    assert recommendations[0].query_text == "SELECT * FROM table"
    assert recommendations[0].data_scanned_display == "5.000 GB"


def test_recommend_missing_where() -> None:
    record = QueryRecord(
        query_id="q2",
        query_text="SELECT col FROM table",
        data_scanned_bytes=2 * 1024**3,
        runtime_ms=1000,
    )
    scored = [ScoredRecord(record=record, frequency=1, cost_score=1.0, estimated_cost_usd=0.1)]

    recommendations = recommend_for_scored(scored)

    types = {rec.recommendation.recommendation_type for rec in recommendations}
    assert "add_filters_or_partitions" in types


def test_recommend_high_frequency() -> None:
    record = QueryRecord(
        query_id="q3",
        query_text="SELECT col FROM table WHERE col = 1",
        data_scanned_bytes=2 * 1024**3,
        runtime_ms=1000,
    )
    scored = [ScoredRecord(record=record, frequency=5, cost_score=1.0, estimated_cost_usd=0.1)]

    recommendations = recommend_for_scored(scored)

    types = {rec.recommendation.recommendation_type for rec in recommendations}
    assert "consider_materialized_view" in types


def test_recommend_long_runtime() -> None:
    record = QueryRecord(
        query_id="q4",
        query_text="SELECT col FROM table WHERE col = 1",
        data_scanned_bytes=1 * 1024**3,
        runtime_ms=120_000,
    )
    scored = [ScoredRecord(record=record, frequency=1, cost_score=1.0, estimated_cost_usd=0.1)]

    recommendations = recommend_for_scored(scored)

    types = {rec.recommendation.recommendation_type for rec in recommendations}
    assert "consider_temp_table" in types
