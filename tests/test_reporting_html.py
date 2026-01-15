from __future__ import annotations

from athena_optimizer.models import Cluster, QueryRecommendation, QueryRecord, Recommendation
from athena_optimizer.reporting import (
    render_cluster_report_html,
    render_recommendation_report_html,
    render_scoring_report_html,
)
from athena_optimizer.scoring import ScoredRecord


def test_render_scoring_report_html_contains_table() -> None:
    record = QueryRecord(query_id="q1", query_text="SELECT 1", runtime_ms=100)
    scored = [ScoredRecord(record=record, frequency=1, cost_score=1.2, estimated_cost_usd=0.01)]

    report = render_scoring_report_html(scored, failures=0, filtered_out=0, top_n=10)

    assert "<table" in report
    assert "data-table" in report
    assert "q1" in report


def test_render_cluster_report_html_contains_query() -> None:
    members = [
        QueryRecord(
            query_id="q1",
            query_text="SELECT 1",
            data_scanned_bytes=1024**3,
            runtime_ms=1000,
        ),
        QueryRecord(
            query_id="q2",
            query_text="SELECT 2",
            data_scanned_bytes=512**3,
            runtime_ms=3000,
        ),
    ]
    cluster = Cluster(cluster_id="cluster-1", members=members, representative_query=members[0])

    report = render_cluster_report_html(
        [cluster],
        failures=0,
        filtered_out=0,
        top_n=5,
        price_per_tb_usd=1.0,
        extra_queries=1,
    )

    assert "cluster-1" in report
    assert "SELECT 1" in report
    assert "Additional queries" in report
    assert "2.00 s" in report


def test_render_recommendation_report_html_contains_query() -> None:
    record = QueryRecord(query_id="q1", query_text="SELECT * FROM t")
    recommendation = Recommendation(
        recommendation_type="avoid_select_star",
        description="Replace SELECT * with only required columns.",
    )
    rec = QueryRecommendation(
        query_id="q1",
        query_text=record.query_text,
        recommendation=recommendation,
        cost_score=1.0,
        estimated_cost_usd=0.01,
        frequency=1,
        data_scanned_display="5.000 GB",
        bytes_output_display="1.000 GB",
    )

    report = render_recommendation_report_html([rec], failures=0, top_n=5)

    assert "avoid_select_star" in report
    assert "SELECT * FROM t" in report
    assert "5.000 GB" in report
