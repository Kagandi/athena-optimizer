from __future__ import annotations

from datetime import datetime
import csv
import json
from pathlib import Path
import random
from statistics import median

import numpy as np
import typer
from sklearn.metrics.pairwise import cosine_distances

from athena_optimizer.config import (
    AwsConfig,
    ClusteringConfig,
    NormalizationConfig,
    ScoringConfig,
    SimilarityConfig,
)
from athena_optimizer.ingestion.athena_api import AthenaApiIngestor
from athena_optimizer.ingestion.s3 import S3QueryJsonIngestor
from athena_optimizer.normalization import normalize_records
from athena_optimizer.scoring import score_records
from athena_optimizer.similarity import build_bm25_distance_matrix, build_hybrid_matrix
from athena_optimizer.clustering import cluster_records
from athena_optimizer.recommendations import recommend_for_scored
from athena_optimizer.reporting import (
    render_cluster_report,
    render_cluster_report_html,
    render_recommendation_report_html,
    render_scoring_report,
    render_scoring_report_html,
)

_BYTES_PER_GB = 1024**3
_BYTES_PER_TB = 10**12
_MS_PER_SECOND = 1000

app = typer.Typer(help="Analyze Athena query history for cost and optimization.")


@app.command("ingest-athena")
def ingest_athena(
    workgroup: str | None = None,
    max_results: int | None = 1000,
    state: list[str] | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    profile: str | None = None,
    region: str | None = None,
) -> None:
    start_dt = _parse_time(start_time)
    end_dt = _parse_time(end_time)

    ingestor = AthenaApiIngestor(
        aws=AwsConfig(profile=profile, region=region),
        workgroup=workgroup,
        max_results=max_results,
        state_filter=set(state) if state else None,
        start_time=start_dt,
        end_time=end_dt,
    )
    records = ingestor.fetch()
    typer.echo(f"Fetched {len(records)} query executions.")


@app.command("ingest-s3")
def ingest_s3(
    bucket: str,
    prefix: str = "",
    max_records: int | None = None,
    profile: str | None = None,
    region: str | None = None,
) -> None:
    ingestor = S3QueryJsonIngestor(
        aws=AwsConfig(profile=profile, region=region),
        bucket=bucket,
        prefix=prefix,
        max_records=max_records,
    )
    records = ingestor.fetch()
    typer.echo(f"Fetched {len(records)} query records.")


@app.command("analyze-athena")
def analyze_athena(
    workgroup: str | None = None,
    max_results: int | None = 1000,
    state: list[str] | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    profile: str | None = None,
    region: str | None = None,
    dialect: str = "athena",
    anonymize_literals: bool = False,
    bytes_weight: float = 0.6,
    runtime_weight: float = 0.3,
    frequency_weight: float = 0.1,
    price_per_tb_usd: float = 5.0,
    top_n: int = 10,
    min_gb: float = 0.0,
    min_runtime_seconds: float = 0.0,
    output: str | None = None,
    output_format: str = "json",
    report_md: str | None = typer.Option(
        None,
        help="Write a Markdown report to this path.",
    ),
    report_html: str | None = typer.Option(
        None,
        help="Write an HTML report to this path.",
    ),
) -> None:
    start_dt = _parse_time(start_time)
    end_dt = _parse_time(end_time)

    ingestor = AthenaApiIngestor(
        aws=AwsConfig(profile=profile, region=region),
        workgroup=workgroup,
        max_results=max_results,
        state_filter=set(state) if state else None,
        start_time=start_dt,
        end_time=end_dt,
    )
    records = ingestor.fetch()
    normalized, failures = normalize_records(
        records,
        NormalizationConfig(dialect=dialect, anonymize_literals=anonymize_literals),
    )
    scored = score_records(
        normalized,
        ScoringConfig(
            bytes_weight=bytes_weight,
            runtime_weight=runtime_weight,
            frequency_weight=frequency_weight,
            price_per_tb_usd=price_per_tb_usd,
        ),
    )
    filtered = _apply_thresholds(
        scored,
        min_gb=min_gb,
        min_runtime_seconds=min_runtime_seconds,
    )
    filtered_out = len(scored) - len(filtered)
    _print_scoring_summary(filtered, failures, top_n, filtered_out)
    _write_scored_output(filtered, output, output_format)
    _write_markdown_report(
        render_scoring_report(
            filtered,
            failures=failures,
            filtered_out=filtered_out,
            top_n=top_n,
        ),
        report_md,
    )
    _write_html_report(
        render_scoring_report_html(
            filtered,
            failures=failures,
            filtered_out=filtered_out,
            top_n=top_n,
        ),
        report_html,
    )


@app.command("analyze-s3")
def analyze_s3(
    bucket: str,
    prefix: str = "",
    max_records: int | None = None,
    profile: str | None = None,
    region: str | None = None,
    dialect: str = "athena",
    anonymize_literals: bool = False,
    bytes_weight: float = 0.6,
    runtime_weight: float = 0.3,
    frequency_weight: float = 0.1,
    price_per_tb_usd: float = 5.0,
    top_n: int = 10,
    min_gb: float = 0.0,
    min_runtime_seconds: float = 0.0,
    output: str | None = None,
    output_format: str = "json",
    report_md: str | None = typer.Option(
        None,
        help="Write a Markdown report to this path.",
    ),
    report_html: str | None = typer.Option(
        None,
        help="Write an HTML report to this path.",
    ),
) -> None:
    ingestor = S3QueryJsonIngestor(
        aws=AwsConfig(profile=profile, region=region),
        bucket=bucket,
        prefix=prefix,
        max_records=max_records,
    )
    records = ingestor.fetch()
    normalized, failures = normalize_records(
        records,
        NormalizationConfig(dialect=dialect, anonymize_literals=anonymize_literals),
    )
    scored = score_records(
        normalized,
        ScoringConfig(
            bytes_weight=bytes_weight,
            runtime_weight=runtime_weight,
            frequency_weight=frequency_weight,
            price_per_tb_usd=price_per_tb_usd,
        ),
    )
    filtered = _apply_thresholds(
        scored,
        min_gb=min_gb,
        min_runtime_seconds=min_runtime_seconds,
    )
    filtered_out = len(scored) - len(filtered)
    _print_scoring_summary(filtered, failures, top_n, filtered_out)
    _write_scored_output(filtered, output, output_format)
    _write_markdown_report(
        render_scoring_report(
            filtered,
            failures=failures,
            filtered_out=filtered_out,
            top_n=top_n,
        ),
        report_md,
    )
    _write_html_report(
        render_scoring_report_html(
            filtered,
            failures=failures,
            filtered_out=filtered_out,
            top_n=top_n,
        ),
        report_html,
    )


@app.command("cluster-athena")
def cluster_athena(
    workgroup: str | None = None,
    max_results: int | None = 1000,
    state: list[str] | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    profile: str | None = None,
    region: str | None = None,
    dialect: str = "athena",
    anonymize_literals: bool = False,
    bytes_weight: float = 0.6,
    runtime_weight: float = 0.3,
    frequency_weight: float = 0.1,
    price_per_tb_usd: float = 5.0,
    min_gb: float = 0.0,
    min_runtime_seconds: float = 0.0,
    max_features: int | None = 5000,
    ngram_min: int = 1,
    ngram_max: int = 2,
    min_df: int = 1,
    max_df: float = 1.0,
    similarity_method: str = "tfidf",
    use_structural: bool = True,
    lexical_weight: float = 0.8,
    structural_weight: float = 0.2,
    distance_threshold: float = typer.Option(
        0.3,
        help="Cosine distance threshold for agglomerative clustering (0-2).",
    ),
    auto_threshold: bool = typer.Option(
        False,
        help="Auto-select the distance threshold from pairwise distances.",
    ),
    linkage: str = "average",
    top_n: int = 10,
    extra_queries: int = 5,
    output: str | None = None,
    output_format: str = "json",
    report_md: str | None = typer.Option(
        None,
        help="Write a Markdown report to this path.",
    ),
    report_html: str | None = typer.Option(
        None,
        help="Write an HTML report to this path.",
    ),
) -> None:
    start_dt = _parse_time(start_time)
    end_dt = _parse_time(end_time)

    ingestor = AthenaApiIngestor(
        aws=AwsConfig(profile=profile, region=region),
        workgroup=workgroup,
        max_results=max_results,
        state_filter=set(state) if state else None,
        start_time=start_dt,
        end_time=end_dt,
    )
    records = ingestor.fetch()
    normalized, failures = normalize_records(
        records,
        NormalizationConfig(dialect=dialect, anonymize_literals=anonymize_literals),
    )
    filtered, filtered_out = _filter_records(
        normalized,
        bytes_weight=bytes_weight,
        runtime_weight=runtime_weight,
        frequency_weight=frequency_weight,
        price_per_tb_usd=price_per_tb_usd,
        min_gb=min_gb,
        min_runtime_seconds=min_runtime_seconds,
    )
    if not filtered:
        _print_cluster_summary([], failures, filtered_out, top_n, price_per_tb_usd)
        _write_cluster_output([], output, output_format, price_per_tb_usd)
        _write_markdown_report(
            render_cluster_report(
                [],
                failures=failures,
                filtered_out=filtered_out,
                top_n=top_n,
                price_per_tb_usd=price_per_tb_usd,
            ),
            report_md,
        )
        _write_html_report(
            render_cluster_report_html(
                [],
                failures=failures,
                filtered_out=filtered_out,
                top_n=top_n,
                price_per_tb_usd=price_per_tb_usd,
                extra_queries=extra_queries,
            ),
            report_html,
        )
        return
    similarity_config = _build_similarity_config(
        max_features=max_features,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        min_df=min_df,
        max_df=max_df,
        lexical_weight=lexical_weight,
        structural_weight=structural_weight,
    )
    matrix, metric = _build_similarity_matrix(
        filtered,
        method=similarity_method,
        use_structural=use_structural,
        config=similarity_config,
    )
    if auto_threshold:
        distance_threshold = _auto_distance_threshold(matrix, metric=metric)
        typer.echo(f"Auto distance threshold: {distance_threshold:.3f}")
    clusters = cluster_records(
        filtered,
        matrix,
        ClusteringConfig(distance_threshold=distance_threshold, linkage=linkage),
        metric=metric,
    )
    _print_cluster_summary(clusters, failures, filtered_out, top_n, price_per_tb_usd)
    _write_cluster_output(clusters, output, output_format, price_per_tb_usd)
    _write_markdown_report(
        render_cluster_report(
            clusters,
            failures=failures,
            filtered_out=filtered_out,
            top_n=top_n,
            price_per_tb_usd=price_per_tb_usd,
        ),
        report_md,
    )
    _write_html_report(
        render_cluster_report_html(
            clusters,
            failures=failures,
            filtered_out=filtered_out,
            top_n=top_n,
            price_per_tb_usd=price_per_tb_usd,
            extra_queries=extra_queries,
        ),
        report_html,
    )


@app.command("cluster-s3")
def cluster_s3(
    bucket: str,
    prefix: str = "",
    max_records: int | None = None,
    profile: str | None = None,
    region: str | None = None,
    dialect: str = "athena",
    anonymize_literals: bool = False,
    bytes_weight: float = 0.6,
    runtime_weight: float = 0.3,
    frequency_weight: float = 0.1,
    price_per_tb_usd: float = 5.0,
    min_gb: float = 0.0,
    min_runtime_seconds: float = 0.0,
    max_features: int | None = 5000,
    ngram_min: int = 1,
    ngram_max: int = 2,
    min_df: int = 1,
    max_df: float = 1.0,
    similarity_method: str = "tfidf",
    use_structural: bool = True,
    lexical_weight: float = 0.8,
    structural_weight: float = 0.2,
    distance_threshold: float = typer.Option(
        0.3,
        help="Cosine distance threshold for agglomerative clustering (0-2).",
    ),
    auto_threshold: bool = typer.Option(
        False,
        help="Auto-select the distance threshold from pairwise distances.",
    ),
    linkage: str = "average",
    top_n: int = 10,
    extra_queries: int = 5,
    output: str | None = None,
    output_format: str = "json",
    report_md: str | None = typer.Option(
        None,
        help="Write a Markdown report to this path.",
    ),
    report_html: str | None = typer.Option(
        None,
        help="Write an HTML report to this path.",
    ),
) -> None:
    ingestor = S3QueryJsonIngestor(
        aws=AwsConfig(profile=profile, region=region),
        bucket=bucket,
        prefix=prefix,
        max_records=max_records,
    )
    records = ingestor.fetch()
    normalized, failures = normalize_records(
        records,
        NormalizationConfig(dialect=dialect, anonymize_literals=anonymize_literals),
    )
    filtered, filtered_out = _filter_records(
        normalized,
        bytes_weight=bytes_weight,
        runtime_weight=runtime_weight,
        frequency_weight=frequency_weight,
        price_per_tb_usd=price_per_tb_usd,
        min_gb=min_gb,
        min_runtime_seconds=min_runtime_seconds,
    )
    if not filtered:
        _print_cluster_summary([], failures, filtered_out, top_n, price_per_tb_usd)
        _write_cluster_output([], output, output_format, price_per_tb_usd)
        _write_markdown_report(
            render_cluster_report(
                [],
                failures=failures,
                filtered_out=filtered_out,
                top_n=top_n,
                price_per_tb_usd=price_per_tb_usd,
            ),
            report_md,
        )
        _write_html_report(
            render_cluster_report_html(
                [],
                failures=failures,
                filtered_out=filtered_out,
                top_n=top_n,
                price_per_tb_usd=price_per_tb_usd,
                extra_queries=extra_queries,
            ),
            report_html,
        )
        return
    similarity_config = _build_similarity_config(
        max_features=max_features,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        min_df=min_df,
        max_df=max_df,
        lexical_weight=lexical_weight,
        structural_weight=structural_weight,
    )
    matrix, metric = _build_similarity_matrix(
        filtered,
        method=similarity_method,
        use_structural=use_structural,
        config=similarity_config,
    )
    if auto_threshold:
        distance_threshold = _auto_distance_threshold(matrix, metric=metric)
        typer.echo(f"Auto distance threshold: {distance_threshold:.3f}")
    clusters = cluster_records(
        filtered,
        matrix,
        ClusteringConfig(distance_threshold=distance_threshold, linkage=linkage),
        metric=metric,
    )
    _print_cluster_summary(clusters, failures, filtered_out, top_n, price_per_tb_usd)
    _write_cluster_output(clusters, output, output_format, price_per_tb_usd)
    _write_markdown_report(
        render_cluster_report(
            clusters,
            failures=failures,
            filtered_out=filtered_out,
            top_n=top_n,
            price_per_tb_usd=price_per_tb_usd,
        ),
        report_md,
    )
    _write_html_report(
        render_cluster_report_html(
            clusters,
            failures=failures,
            filtered_out=filtered_out,
            top_n=top_n,
            price_per_tb_usd=price_per_tb_usd,
            extra_queries=extra_queries,
        ),
        report_html,
    )


@app.command("recommend-athena")
def recommend_athena(
    workgroup: str | None = None,
    max_results: int | None = 1000,
    state: list[str] | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    profile: str | None = None,
    region: str | None = None,
    dialect: str = "athena",
    anonymize_literals: bool = False,
    bytes_weight: float = 0.6,
    runtime_weight: float = 0.3,
    frequency_weight: float = 0.1,
    price_per_tb_usd: float = 5.0,
    min_gb: float = 0.0,
    min_runtime_seconds: float = 0.0,
    min_frequency: int = 1,
    top_n: int = 50,
    output: str | None = None,
    output_format: str = "json",
    report_md: str | None = typer.Option(
        None,
        help="Write a Markdown report to this path.",
    ),
    report_html: str | None = typer.Option(
        None,
        help="Write an HTML report to this path.",
    ),
) -> None:
    start_dt = _parse_time(start_time)
    end_dt = _parse_time(end_time)

    ingestor = AthenaApiIngestor(
        aws=AwsConfig(profile=profile, region=region),
        workgroup=workgroup,
        max_results=max_results,
        state_filter=set(state) if state else None,
        start_time=start_dt,
        end_time=end_dt,
    )
    records = ingestor.fetch()
    normalized, failures = normalize_records(
        records,
        NormalizationConfig(dialect=dialect, anonymize_literals=anonymize_literals),
    )
    scored = score_records(
        normalized,
        ScoringConfig(
            bytes_weight=bytes_weight,
            runtime_weight=runtime_weight,
            frequency_weight=frequency_weight,
            price_per_tb_usd=price_per_tb_usd,
        ),
    )
    filtered = _apply_thresholds(
        scored,
        min_gb=min_gb,
        min_runtime_seconds=min_runtime_seconds,
    )
    recommendations = recommend_for_scored(filtered, min_frequency=min_frequency)
    _print_recommendation_summary(recommendations, failures, top_n)
    _write_recommendations_output(recommendations, output, output_format)
    _write_markdown_report(
        render_scoring_report(
            [item for item in filtered],
            failures=failures,
            filtered_out=len(scored) - len(filtered),
            top_n=top_n,
        ),
        report_md,
    )
    _write_html_report(
        render_recommendation_report_html(
            recommendations,
            failures=failures,
            top_n=top_n,
        ),
        report_html,
    )


@app.command("recommend-s3")
def recommend_s3(
    bucket: str,
    prefix: str = "",
    max_records: int | None = None,
    profile: str | None = None,
    region: str | None = None,
    dialect: str = "athena",
    anonymize_literals: bool = False,
    bytes_weight: float = 0.6,
    runtime_weight: float = 0.3,
    frequency_weight: float = 0.1,
    price_per_tb_usd: float = 5.0,
    min_gb: float = 0.0,
    min_runtime_seconds: float = 0.0,
    min_frequency: int = 1,
    top_n: int = 50,
    output: str | None = None,
    output_format: str = "json",
    report_md: str | None = typer.Option(
        None,
        help="Write a Markdown report to this path.",
    ),
    report_html: str | None = typer.Option(
        None,
        help="Write an HTML report to this path.",
    ),
) -> None:
    ingestor = S3QueryJsonIngestor(
        aws=AwsConfig(profile=profile, region=region),
        bucket=bucket,
        prefix=prefix,
        max_records=max_records,
    )
    records = ingestor.fetch()
    normalized, failures = normalize_records(
        records,
        NormalizationConfig(dialect=dialect, anonymize_literals=anonymize_literals),
    )
    scored = score_records(
        normalized,
        ScoringConfig(
            bytes_weight=bytes_weight,
            runtime_weight=runtime_weight,
            frequency_weight=frequency_weight,
            price_per_tb_usd=price_per_tb_usd,
        ),
    )
    filtered = _apply_thresholds(
        scored,
        min_gb=min_gb,
        min_runtime_seconds=min_runtime_seconds,
    )
    recommendations = recommend_for_scored(filtered, min_frequency=min_frequency)
    _print_recommendation_summary(recommendations, failures, top_n)
    _write_recommendations_output(recommendations, output, output_format)
    _write_markdown_report(
        render_scoring_report(
            [item for item in filtered],
            failures=failures,
            filtered_out=len(scored) - len(filtered),
            top_n=top_n,
        ),
        report_md,
    )
    _write_html_report(
        render_recommendation_report_html(
            recommendations,
            failures=failures,
            top_n=top_n,
        ),
        report_html,
    )


def _filter_records(
    records,
    *,
    bytes_weight: float,
    runtime_weight: float,
    frequency_weight: float,
    price_per_tb_usd: float,
    min_gb: float,
    min_runtime_seconds: float,
):
    record_list = list(records)
    if min_gb <= 0 and min_runtime_seconds <= 0:
        return record_list, 0

    scored = score_records(
        record_list,
        ScoringConfig(
            bytes_weight=bytes_weight,
            runtime_weight=runtime_weight,
            frequency_weight=frequency_weight,
            price_per_tb_usd=price_per_tb_usd,
        ),
    )
    filtered = _apply_thresholds(
        scored,
        min_gb=min_gb,
        min_runtime_seconds=min_runtime_seconds,
    )
    return [item.record for item in filtered], len(scored) - len(filtered)


def _build_similarity_config(
    *,
    max_features: int | None,
    ngram_min: int,
    ngram_max: int,
    min_df: int,
    max_df: float,
    lexical_weight: float,
    structural_weight: float,
) -> SimilarityConfig:
    if ngram_min <= 0 or ngram_max <= 0:
        raise typer.BadParameter("ngram-min and ngram-max must be positive.")
    if ngram_min > ngram_max:
        raise typer.BadParameter("ngram-min cannot exceed ngram-max.")

    return SimilarityConfig(
        max_features=max_features,
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        max_df=max_df,
        lexical_weight=lexical_weight,
        structural_weight=structural_weight,
    )


def _build_similarity_matrix(
    records,
    *,
    method: str,
    use_structural: bool,
    config: SimilarityConfig,
):
    method_key = method.lower().strip()
    if method_key == "tfidf":
        matrix, _ = build_hybrid_matrix(records, config, use_structural=use_structural)
        return matrix, "cosine"
    if method_key == "bm25":
        distances = build_bm25_distance_matrix(records)
        return distances, "precomputed"

    raise typer.BadParameter("similarity-method must be 'tfidf' or 'bm25'.")


def _auto_distance_threshold(
    matrix,
    *,
    metric: str = "cosine",
    quantile: float = 0.25,
    max_samples: int = 10000,
    seed: int = 42,
) -> float:
    if metric == "cosine":
        distances = cosine_distances(matrix)
    elif metric == "precomputed":
        distances = np.asarray(matrix)
    else:
        raise typer.BadParameter("auto-threshold only supports cosine or precomputed distances.")

    if distances.shape[0] < 2:
        return 0.3
    upper = distances[np.triu_indices(distances.shape[0], k=1)]
    if upper.size == 0:
        return 0.3

    values = upper
    if upper.size > max_samples:
        rng = random.Random(seed)
        indices = rng.sample(range(upper.size), k=max_samples)
        values = upper[indices]

    threshold = float(np.quantile(values, quantile))
    return max(min(threshold, 1.5), 0.05)


def _apply_thresholds(scored, *, min_gb: float, min_runtime_seconds: float):
    records = list(scored)
    if min_gb <= 0 and min_runtime_seconds <= 0:
        return records

    min_bytes = min_gb * _BYTES_PER_GB if min_gb > 0 else None
    min_runtime_ms = min_runtime_seconds * _MS_PER_SECOND if min_runtime_seconds > 0 else None

    filtered = []
    for item in records:
        bytes_value = item.record.data_scanned_bytes or 0
        runtime_ms = item.record.runtime_ms or 0
        checks: list[bool] = []
        if min_bytes is not None:
            checks.append(bytes_value >= min_bytes)
        if min_runtime_ms is not None:
            checks.append(runtime_ms >= min_runtime_ms)
        if checks and not any(checks):
            continue
        filtered.append(item)
    return filtered


def _write_markdown_report(report: str, output: str | None) -> None:
    if not output:
        return
    path = Path(output)
    path.write_text(report, encoding="utf-8")
    typer.echo(f"Wrote report to {path}.")


def _write_html_report(report: str, output: str | None) -> None:
    if not output:
        return
    path = Path(output)
    path.write_text(report, encoding="utf-8")
    typer.echo(f"Wrote report to {path}.")


def _write_recommendations_output(
    recommendations,
    output: str | None,
    output_format: str,
) -> None:
    if not output:
        return

    fmt = output_format.lower().strip()
    if fmt not in {"json", "csv"}:
        raise typer.BadParameter("output-format must be 'json' or 'csv'.")

    path = Path(output)
    payload = [_recommendation_to_dict(item) for item in recommendations]
    if fmt == "json":
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        typer.echo(f"Wrote {len(payload)} recommendations to {path}.")
        return

    fieldnames = _recommendation_csv_fieldnames()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload)
    typer.echo(f"Wrote {len(payload)} recommendations to {path}.")


def _recommendation_csv_fieldnames() -> list[str]:
    return [
        "query_id",
        "query_text",
        "recommendation_type",
        "description",
        "rationale",
        "estimated_impact",
        "cost_score",
        "estimated_cost_usd",
        "frequency",
        "data_scanned_gb",
        "data_scanned_display",
        "bytes_output_gb",
        "bytes_output_display",
    ]


def _recommendation_to_dict(item) -> dict[str, object]:
    return {
        "query_id": item.query_id,
        "query_text": item.query_text,
        "recommendation_type": item.recommendation.recommendation_type,
        "description": item.recommendation.description,
        "rationale": item.recommendation.rationale,
        "estimated_impact": item.recommendation.estimated_impact,
        "cost_score": item.cost_score,
        "estimated_cost_usd": item.estimated_cost_usd,
        "frequency": item.frequency,
        "data_scanned_gb": item.data_scanned_gb,
        "data_scanned_display": item.data_scanned_display,
        "bytes_output_gb": item.bytes_output_gb,
        "bytes_output_display": item.bytes_output_display,
    }


def _write_scored_output(scored, output: str | None, output_format: str) -> None:
    if not output:
        return

    fmt = output_format.lower().strip()
    if fmt not in {"json", "csv"}:
        raise typer.BadParameter("output-format must be 'json' or 'csv'.")

    path = Path(output)
    if fmt == "json":
        payload = [_scored_to_dict(item, metadata_as_json=False) for item in scored]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        typer.echo(f"Wrote {len(payload)} records to {path}.")
        return

    rows = [_scored_to_dict(item, metadata_as_json=True) for item in scored]
    fieldnames = _csv_fieldnames()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    typer.echo(f"Wrote {len(rows)} records to {path}.")


def _write_cluster_output(
    clusters,
    output: str | None,
    output_format: str,
    price_per_tb_usd: float,
) -> None:
    if not output:
        return

    fmt = output_format.lower().strip()
    if fmt not in {"json", "csv"}:
        raise typer.BadParameter("output-format must be 'json' or 'csv'.")

    path = Path(output)
    if fmt == "json":
        payload = [
            _cluster_to_dict(
                cluster,
                member_ids_as_string=False,
                price_per_tb_usd=price_per_tb_usd,
            )
            for cluster in clusters
        ]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        typer.echo(f"Wrote {len(payload)} clusters to {path}.")
        return

    rows = [
        _cluster_to_dict(
            cluster,
            member_ids_as_string=True,
            price_per_tb_usd=price_per_tb_usd,
        )
        for cluster in clusters
    ]
    fieldnames = _cluster_csv_fieldnames()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    typer.echo(f"Wrote {len(rows)} clusters to {path}.")


def _cluster_csv_fieldnames() -> list[str]:
    return [
        "cluster_id",
        "size",
        "total_scanned_gb",
        "total_scanned_display",
        "total_cost_usd",
        "median_runtime_seconds",
        "median_runtime_display",
        "representative_query_id",
        "representative_query_text",
        "member_query_ids",
    ]


def _cluster_to_dict(
    cluster,
    *,
    member_ids_as_string: bool,
    price_per_tb_usd: float,
) -> dict[str, object]:
    representative = cluster.representative_query
    if representative is None and cluster.members:
        representative = cluster.members[0]
    member_ids = [member.query_id for member in cluster.members]
    member_ids_value: object = member_ids
    if member_ids_as_string:
        member_ids_value = ";".join(member_ids)

    total_bytes = sum(member.data_scanned_bytes or 0 for member in cluster.members)
    total_gb = _bytes_to_gb(total_bytes) or 0.0
    total_cost = _estimate_cost_usd_bytes(total_bytes, price_per_tb_usd)
    median_runtime_seconds = _median_runtime_seconds(
        [member.runtime_ms for member in cluster.members]
    )

    return {
        "cluster_id": cluster.cluster_id,
        "size": len(cluster.members),
        "total_scanned_gb": total_gb,
        "total_scanned_display": _format_scanned(total_gb),
        "total_cost_usd": total_cost,
        "median_runtime_seconds": median_runtime_seconds,
        "median_runtime_display": _format_runtime_seconds(median_runtime_seconds),
        "representative_query_id": representative.query_id if representative else None,
        "representative_query_text": representative.query_text if representative else None,
        "member_query_ids": member_ids_value,
    }


def _print_cluster_summary(
    clusters,
    failures: int,
    filtered_out: int,
    top_n: int,
    price_per_tb_usd: float,
) -> None:
    total_clusters = len(clusters)
    total_queries = sum(len(cluster.members) for cluster in clusters)
    total_bytes = sum(
        (member.data_scanned_bytes or 0) for cluster in clusters for member in cluster.members
    )
    total_gb = _bytes_to_gb(total_bytes) or 0.0
    total_cost = _estimate_cost_usd_bytes(total_bytes, price_per_tb_usd)
    typer.echo(
        "Clustered "
        f"{total_queries} queries into {total_clusters} clusters. "
        f"Total scanned: {_format_scanned(total_gb)} (${total_cost:.2f}). "
        f"Normalization failures: {failures}. Filtered out: {filtered_out}."
    )
    if total_clusters == 0:
        return

    top_n = max(top_n, 0)
    typer.echo(f"Top {top_n} Clusters:")
    ranked = sorted(
        clusters,
        key=lambda cluster: sum(member.data_scanned_bytes or 0 for member in cluster.members),
        reverse=True,
    )
    for index, cluster in enumerate(ranked[:top_n], start=1):
        size = len(cluster.members)
        representative = cluster.representative_query
        if representative is None and cluster.members:
            representative = cluster.members[0]
        rep_id = representative.query_id if representative else ""
        cluster_bytes = sum(member.data_scanned_bytes or 0 for member in cluster.members)
        cluster_gb = _bytes_to_gb(cluster_bytes) or 0.0
        cluster_cost = _estimate_cost_usd_bytes(cluster_bytes, price_per_tb_usd)
        median_runtime_seconds = _median_runtime_seconds(
            [member.runtime_ms for member in cluster.members]
        )
        snippet = _shorten(representative.query_text if representative else "", 120)
        typer.echo(
            f"{index}. cluster={cluster.cluster_id} size={size} scanned={_format_scanned(cluster_gb)} "
            f"median_runtime={_format_runtime_seconds(median_runtime_seconds)} "
            f"cost=${cluster_cost:.2f} rep_id={rep_id}"
        )
        if snippet:
            typer.echo(f"   {snippet}")


def _print_recommendation_summary(recommendations, failures: int, top_n: int) -> None:
    total = len(recommendations)
    typer.echo(f"Generated {total} recommendations. Normalization failures: {failures}.")
    if total == 0:
        return

    top_n = max(top_n, 0)
    ranked = sorted(
        recommendations,
        key=lambda item: (item.estimated_cost_usd or 0.0, item.cost_score or 0.0),
        reverse=True,
    )
    typer.echo(f"Top {top_n} Recommendations:")
    for index, item in enumerate(ranked[:top_n], start=1):
        rec = item.recommendation
        cost = item.estimated_cost_usd or 0.0
        typer.echo(f"{index}. {rec.recommendation_type} cost=${cost:.2f} id={item.query_id}")
        typer.echo(f"   {rec.description}")


def _bytes_to_gb(value: int | None) -> float | None:
    if value is None:
        return None
    return value / _BYTES_PER_GB


def _format_scanned(value_gb: float | None) -> str:
    if value_gb is None:
        return "0.000 GB"
    if value_gb >= 1000:
        return f"{value_gb / 1000:.3f} TB"
    return f"{value_gb:.3f} GB"


def _median_runtime_seconds(values_ms: list[int | None]) -> float | None:
    cleaned = [value for value in values_ms if value is not None]
    if not cleaned:
        return None
    seconds = [value / _MS_PER_SECOND for value in cleaned]
    return float(median(seconds))


def _format_runtime_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f} s"


def _estimate_cost_usd_bytes(bytes_scanned: int | None, price_per_tb_usd: float) -> float:
    if bytes_scanned is None or bytes_scanned <= 0 or price_per_tb_usd <= 0:
        return 0.0
    tb_scanned = bytes_scanned / _BYTES_PER_TB
    return tb_scanned * price_per_tb_usd


def _ms_to_seconds(value: int | None) -> float | None:
    if value is None:
        return None
    return value / _MS_PER_SECOND


def _csv_fieldnames() -> list[str]:
    return [
        "query_id",
        "query_text",
        "normalized_query",
        "workgroup",
        "user",
        "submission_time",
        "runtime_seconds",
        "data_scanned_gb",
        "data_scanned_display",
        "bytes_output_gb",
        "bytes_output_display",
        "state",
        "frequency",
        "cost_score",
        "estimated_cost_usd",
        "source",
        "metadata",
    ]


def _scored_to_dict(item, *, metadata_as_json: bool) -> dict[str, object]:
    record = item.record
    submission_time = record.submission_time.isoformat() if record.submission_time else None
    metadata_value: object = record.metadata
    if metadata_as_json:
        metadata_value = json.dumps(record.metadata, ensure_ascii=True, sort_keys=True)

    scanned_gb = _bytes_to_gb(record.data_scanned_bytes)
    output_gb = _bytes_to_gb(record.bytes_output)

    return {
        "query_id": record.query_id,
        "query_text": record.query_text,
        "normalized_query": record.normalized_query,
        "workgroup": record.workgroup,
        "user": record.user,
        "submission_time": submission_time,
        "runtime_seconds": _ms_to_seconds(record.runtime_ms),
        "data_scanned_gb": scanned_gb,
        "data_scanned_display": _format_scanned(scanned_gb),
        "bytes_output_gb": output_gb,
        "bytes_output_display": _format_scanned(output_gb),
        "state": record.state,
        "frequency": item.frequency,
        "cost_score": item.cost_score,
        "estimated_cost_usd": item.estimated_cost_usd,
        "source": record.source,
        "metadata": metadata_value,
    }


def _print_scoring_summary(scored, failures: int, top_n: int, filtered_out: int) -> None:
    total = len(scored) + filtered_out
    total_cost = sum(item.estimated_cost_usd for item in scored)
    typer.echo(
        f"Scored {total} queries. Normalization failures: {failures}. Filtered out: {filtered_out}. "
        f"Estimated cost: ${total_cost:.2f}."
    )
    if len(scored) == 0:
        return

    top_n = max(top_n, 0)
    ranked = sorted(scored, key=lambda item: item.cost_score, reverse=True)
    for index, item in enumerate(ranked[:top_n], start=1):
        record = item.record
        scanned_gb = _bytes_to_gb(record.data_scanned_bytes) or 0.0
        runtime_seconds = _ms_to_seconds(record.runtime_ms) or 0.0
        snippet = _shorten(record.query_text, 120)
        typer.echo(
            f"{index}. score={item.cost_score:.2f} scanned={_format_scanned(scanned_gb)} "
            f"runtime_s={runtime_seconds:.2f} cost=${item.estimated_cost_usd:.2f} "
            f"freq={item.frequency} id={record.query_id}"
        )
        if snippet:
            typer.echo(f"   {snippet}")


def _shorten(text: str | None, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None

    cleaned = value.strip()
    if cleaned.endswith("Z") and "T" not in cleaned:
        cleaned = cleaned[:-1]

    try:
        return datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError as exc:
        raise typer.BadParameter(
            "Use ISO-8601 date or datetime (e.g. 2024-01-01 or 2024-01-01T00:00:00Z)."
        ) from exc
