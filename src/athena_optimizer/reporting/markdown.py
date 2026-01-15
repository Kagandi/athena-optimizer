from __future__ import annotations

from collections.abc import Sequence
from statistics import median

from athena_optimizer.models import Cluster
from athena_optimizer.scoring import ScoredRecord

_BYTES_PER_GB = 1024**3
_BYTES_PER_TB = 10**12
_MS_PER_SECOND = 1000


def render_scoring_report(
    scored: Sequence[ScoredRecord],
    *,
    failures: int,
    filtered_out: int,
    top_n: int,
) -> str:
    total = len(scored) + filtered_out
    total_bytes = sum(item.record.data_scanned_bytes or 0 for item in scored)
    total_gb = total_bytes / _BYTES_PER_GB
    total_cost = sum(item.estimated_cost_usd for item in scored)

    lines = [
        "# Athena Optimizer Report",
        "",
        "## Summary",
        f"- Queries scored: {total}",
        f"- Normalization failures: {failures}",
        f"- Filtered out: {filtered_out}",
        f"- Total scanned: {_format_scanned(total_gb)}",
        f"- Estimated cost: ${total_cost:.2f}",
        "",
        "## Top Queries",
        "| rank | query_id | cost_score | cost_usd | scanned | runtime_s | frequency |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    top_n = max(top_n, 0)
    ranked = sorted(scored, key=lambda item: item.cost_score, reverse=True)
    for index, item in enumerate(ranked[:top_n], start=1):
        record = item.record
        scanned_gb = (record.data_scanned_bytes or 0) / _BYTES_PER_GB
        runtime_s = (record.runtime_ms or 0) / _MS_PER_SECOND
        lines.append(
            "| "
            f"{index} | {record.query_id} | {item.cost_score:.2f} | "
            f"{item.estimated_cost_usd:.2f} | {_format_scanned(scanned_gb)} | {runtime_s:.2f} | "
            f"{item.frequency} |"
        )

    return "\n".join(lines)


def render_cluster_report(
    clusters: Sequence[Cluster],
    *,
    failures: int,
    filtered_out: int,
    top_n: int,
    price_per_tb_usd: float,
) -> str:
    total_queries = sum(len(cluster.members) for cluster in clusters)
    total_bytes = sum(
        (member.data_scanned_bytes or 0) for cluster in clusters for member in cluster.members
    )
    total_gb = total_bytes / _BYTES_PER_GB
    total_cost = _estimate_cost_usd_bytes(total_bytes, price_per_tb_usd)

    lines = [
        "# Athena Optimizer Report",
        "",
        "## Cluster Summary",
        f"- Queries clustered: {total_queries}",
        f"- Clusters: {len(clusters)}",
        f"- Normalization failures: {failures}",
        f"- Filtered out: {filtered_out}",
        f"- Total scanned: {_format_scanned(total_gb)}",
        f"- Estimated cost: ${total_cost:.2f}",
        "",
        "## Top Clusters",
        "| rank | cluster_id | size | scanned | median_runtime_s | cost_usd | representative_id |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    top_n = max(top_n, 0)
    ranked = sorted(
        clusters,
        key=lambda cluster: sum(member.data_scanned_bytes or 0 for member in cluster.members),
        reverse=True,
    )
    top_clusters = ranked[:top_n]
    for index, cluster in enumerate(top_clusters, start=1):
        representative = cluster.representative_query
        if representative is None and cluster.members:
            representative = cluster.members[0]
        rep_id = representative.query_id if representative else ""
        cluster_bytes = sum(member.data_scanned_bytes or 0 for member in cluster.members)
        cluster_gb = cluster_bytes / _BYTES_PER_GB
        cluster_cost = _estimate_cost_usd_bytes(cluster_bytes, price_per_tb_usd)
        median_runtime = _median_runtime_seconds([member.runtime_ms for member in cluster.members])
        lines.append(
            "| "
            f"{index} | {cluster.cluster_id} | {len(cluster.members)} | "
            f"{_format_scanned(cluster_gb)} | {_format_runtime_seconds(median_runtime)} | "
            f"{cluster_cost:.2f} | {rep_id} |"
        )

    lines.append("")
    lines.append("## Representative Queries")
    for cluster in top_clusters:
        representative = cluster.representative_query
        if representative is None and cluster.members:
            representative = cluster.members[0]
        if representative is None:
            continue
        lines.append("")
        lines.append(f"### {cluster.cluster_id} ({representative.query_id})")
        lines.append("```sql")
        lines.append(representative.query_text)
        lines.append("```")

    return "\n".join(lines)


def _format_scanned(value_gb: float) -> str:
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
    return f"{value:.2f}"


def _median_runtime_seconds(values_ms: list[int | None]) -> float | None:
    cleaned = [value for value in values_ms if value is not None]
    if not cleaned:
        return None
    seconds = [value / _MS_PER_SECOND for value in cleaned]
    return float(median(seconds))


def _format_runtime_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _estimate_cost_usd_bytes(bytes_scanned: int, price_per_tb_usd: float) -> float:
    if bytes_scanned <= 0 or price_per_tb_usd <= 0:
        return 0.0
    tb_scanned = bytes_scanned / _BYTES_PER_TB
    return tb_scanned * price_per_tb_usd
