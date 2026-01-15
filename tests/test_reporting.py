from __future__ import annotations

from athena_optimizer.models import Cluster, QueryRecord
from athena_optimizer.reporting import render_cluster_report, render_scoring_report
from athena_optimizer.scoring import ScoredRecord


def test_render_scoring_report_contains_query() -> None:
    record = QueryRecord(query_id="q1", query_text="SELECT 1", runtime_ms=100)
    scored = [ScoredRecord(record=record, frequency=1, cost_score=1.2, estimated_cost_usd=0.01)]

    report = render_scoring_report(scored, failures=0, filtered_out=0, top_n=10)

    assert "# Athena Optimizer Report" in report
    assert "Queries scored: 1" in report
    assert "q1" in report


def test_render_cluster_report_contains_cluster() -> None:
    members = [
        QueryRecord(query_id="q1", query_text="SELECT 1", data_scanned_bytes=1024**3),
        QueryRecord(query_id="q2", query_text="SELECT 2", data_scanned_bytes=1024**3),
    ]
    cluster = Cluster(cluster_id="cluster-1", members=members, representative_query=members[0])

    report = render_cluster_report(
        [cluster],
        failures=0,
        filtered_out=0,
        top_n=5,
        price_per_tb_usd=1.0,
    )

    assert "Clusters: 1" in report
    assert "cluster-1" in report
    assert "SELECT 1" in report
