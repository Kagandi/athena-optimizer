from __future__ import annotations

from collections.abc import Sequence
import html
import re
from statistics import median

from athena_optimizer.models import Cluster, QueryRecommendation
from athena_optimizer.scoring import ScoredRecord

_BYTES_PER_GB = 1024**3
_BYTES_PER_TB = 10**12
_MS_PER_SECOND = 1000


def _anchor_id(value: str) -> str:
    cleaned = value.lower()
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", cleaned).strip("-")
    return f"cluster-{cleaned}" if cleaned else "cluster"


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
    return f"{value:.2f} s"


def render_scoring_report_html(
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

    ranked = sorted(scored, key=lambda item: item.cost_score, reverse=True)[: max(top_n, 0)]

    rows = []
    for index, item in enumerate(ranked, start=1):
        record = item.record
        scanned_gb = (record.data_scanned_bytes or 0) / _BYTES_PER_GB
        runtime_s = (record.runtime_ms or 0) / _MS_PER_SECOND
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{html.escape(record.query_id)}</td>"
            f"<td>{item.cost_score:.2f}</td>"
            f"<td>{item.estimated_cost_usd:.2f}</td>"
            f"<td>{_format_scanned(scanned_gb)}</td>"
            f"<td>{runtime_s:.2f}</td>"
            f"<td>{item.frequency}</td>"
            "</tr>"
        )

    body = f"""
    <header class="page-header">
      <div>
        <h1>Athena Optimizer Report</h1>
        <p class="subtitle">Scoring overview for expensive queries.</p>
      </div>
      <span class="badge">Scoring</span>
    </header>

    <section class="card">
      <div class="card-header">
        <h2>Summary</h2>
      </div>
      <div class="stat-grid">
        <div class="stat"><div class="stat-label">Queries scored</div><div class="stat-value">{total}</div></div>
        <div class="stat"><div class="stat-label">Normalization failures</div><div class="stat-value">{failures}</div></div>
        <div class="stat"><div class="stat-label">Filtered out</div><div class="stat-value">{filtered_out}</div></div>
        <div class="stat"><div class="stat-label">Total scanned</div><div class="stat-value">{_format_scanned(total_gb)}</div></div>
        <div class="stat"><div class="stat-label">Estimated cost</div><div class="stat-value">${total_cost:.2f}</div></div>
      </div>
    </section>

    <section class="card">
      <div class="card-header">
        <h2>Top Queries</h2>
        <span class="note">Top {len(ranked)}</span>
      </div>
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Query ID</th>
              <th>Cost Score</th>
              <th>Cost USD</th>
              <th>Scanned</th>
              <th>Runtime s</th>
              <th>Frequency</th>
            </tr>
          </thead>
          <tbody>
            {"".join(rows)}
          </tbody>
        </table>
      </div>
    </section>
    """
    return _html_page(body)


def render_cluster_report_html(
    clusters: Sequence[Cluster],
    *,
    failures: int,
    filtered_out: int,
    top_n: int,
    price_per_tb_usd: float,
    extra_queries: int = 5,
) -> str:
    total_queries = sum(len(cluster.members) for cluster in clusters)
    total_bytes = sum(
        (member.data_scanned_bytes or 0) for cluster in clusters for member in cluster.members
    )
    total_gb = total_bytes / _BYTES_PER_GB
    total_cost = _estimate_cost_usd_bytes(total_bytes, price_per_tb_usd)

    ranked = sorted(
        clusters,
        key=lambda cluster: sum(member.data_scanned_bytes or 0 for member in cluster.members),
        reverse=True,
    )[: max(top_n, 0)]

    extra_limit = max(extra_queries, 0)

    rows = []
    for index, cluster in enumerate(ranked, start=1):
        representative = cluster.representative_query
        if representative is None and cluster.members:
            representative = cluster.members[0]
        rep_id = representative.query_id if representative else ""
        cluster_bytes = sum(member.data_scanned_bytes or 0 for member in cluster.members)
        cluster_gb = cluster_bytes / _BYTES_PER_GB
        cluster_cost = _estimate_cost_usd_bytes(cluster_bytes, price_per_tb_usd)
        median_runtime = _median_runtime_seconds([member.runtime_ms for member in cluster.members])
        anchor_id = _anchor_id(cluster.cluster_id)
        rows.append(
            f'<tr class="clickable" data-href="#{anchor_id}" role="link" tabindex="0">'
            f"<td>{index}</td>"
            f'<td><a class="link" href="#{anchor_id}">{html.escape(cluster.cluster_id)}</a></td>'
            f"<td>{len(cluster.members)}</td>"
            f"<td>{_format_scanned(cluster_gb)}</td>"
            f"<td>{_format_runtime_seconds(median_runtime)}</td>"
            f"<td>{cluster_cost:.2f}</td>"
            f"<td>{html.escape(rep_id)}</td>"
            "</tr>"
        )

    query_blocks = []
    for cluster in ranked:
        representative = cluster.representative_query
        if representative is None and cluster.members:
            representative = cluster.members[0]
        if representative is None:
            continue
        anchor_id = _anchor_id(cluster.cluster_id)
        extra_block = ""
        if extra_limit > 0:
            extra_members = [member for member in cluster.members if member is not representative]
            if extra_members:
                shown = extra_members[:extra_limit]
                extra_rows = []
                for member in shown:
                    query_text = member.query_text or ""
                    extra_rows.append(
                        '<div class="query-sub">'
                        f"<h4>{html.escape(member.query_id)}</h4>"
                        f'<pre class="sql-block"><code>{html.escape(query_text)}</code></pre>'
                        "</div>"
                    )
                summary = f"Additional queries ({len(extra_members)} total, showing {len(shown)})"
                extra_block = (
                    '<details class="extra-queries">'
                    f"<summary>{summary}</summary>"
                    f"{''.join(extra_rows)}"
                    "</details>"
                )

        query_blocks.append(
            f'<div class="query-block" id="{anchor_id}">'
            f"<h3>{html.escape(cluster.cluster_id)} ({html.escape(representative.query_id)})</h3>"
            f'<pre class="sql-block"><code>{html.escape(representative.query_text)}</code></pre>'
            f"{extra_block}"
            "</div>"
        )

    body = f"""
    <header class=\"page-header\">
      <div>
        <h1>Athena Optimizer Report</h1>
        <p class=\"subtitle\">Clustered view of similar queries.</p>
      </div>
      <span class=\"badge\">Clusters</span>
    </header>

    <section class=\"card\">
      <div class=\"card-header\">
        <h2>Cluster Summary</h2>
      </div>
      <div class=\"stat-grid\">
        <div class=\"stat\"><div class=\"stat-label\">Queries clustered</div><div class=\"stat-value\">{total_queries}</div></div>
        <div class=\"stat\"><div class=\"stat-label\">Clusters</div><div class=\"stat-value\">{len(clusters)}</div></div>
        <div class=\"stat\"><div class=\"stat-label\">Normalization failures</div><div class=\"stat-value\">{failures}</div></div>
        <div class=\"stat\"><div class=\"stat-label\">Filtered out</div><div class=\"stat-value\">{filtered_out}</div></div>
        <div class="stat"><div class="stat-label">Total scanned</div><div class="stat-value">{_format_scanned(total_gb)}</div></div>
        <div class=\"stat\"><div class=\"stat-label\">Estimated cost</div><div class=\"stat-value\">${total_cost:.2f}</div></div>
      </div>
    </section>

    <section class=\"card\">
      <div class=\"card-header\">
        <h2>Top Clusters</h2>
        <span class=\"note\">Top {len(ranked)}</span>
      </div>
      <div class=\"table-wrap\">
        <table class=\"data-table\">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Cluster ID</th>
              <th>Size</th>
              <th>Scanned</th>
              <th>Median Runtime</th>
              <th>Cost USD</th>
              <th>Representative ID</th>
            </tr>
          </thead>
          <tbody>
            {"".join(rows)}
          </tbody>
        </table>
      </div>
    </section>

    <section class=\"card\">
      <div class=\"card-header\">
        <h2>Representative Queries</h2>
        <span class=\"note\">Top {len(ranked)}</span>
      </div>
      {"".join(query_blocks)}
    </section>
    """
    return _html_page(body)


def render_recommendation_report_html(
    recommendations: Sequence[QueryRecommendation],
    *,
    failures: int,
    top_n: int,
) -> str:
    ranked = sorted(
        recommendations,
        key=lambda item: (item.estimated_cost_usd or 0.0, item.cost_score or 0.0),
        reverse=True,
    )[: max(top_n, 0)]

    rows = []
    for index, item in enumerate(ranked, start=1):
        rec = item.recommendation
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{html.escape(item.query_id)}</td>"
            f"<td>{html.escape(rec.recommendation_type)}</td>"
            f"<td>{html.escape(rec.description)}</td>"
            f"<td>{html.escape(rec.rationale or '')}</td>"
            f"<td>{html.escape(rec.estimated_impact or '')}</td>"
            f"<td>{item.cost_score or 0.0:.2f}</td>"
            f"<td>{item.estimated_cost_usd or 0.0:.2f}</td>"
            f"<td>{item.frequency or 0}</td>"
            f"<td>{html.escape(item.data_scanned_display or '')}</td>"
            f"<td>{html.escape(item.bytes_output_display or '')}</td>"
            "</tr>"
        )

    query_blocks = []
    for item in ranked:
        query_blocks.append(
            '<div class="query-block">'
            f"<h3>{html.escape(item.query_id)} ({html.escape(item.recommendation.recommendation_type)})</h3>"
            f'<pre class="sql-block"><code>{html.escape(item.query_text)}</code></pre>'
            "</div>"
        )

    body = f"""
    <header class=\"page-header\">
      <div>
        <h1>Athena Optimizer Report</h1>
        <p class=\"subtitle\">Rule-based optimization recommendations.</p>
      </div>
      <span class=\"badge\">Recommendations</span>
    </header>

    <section class=\"card\">
      <div class=\"card-header\">
        <h2>Recommendation Summary</h2>
      </div>
      <div class=\"stat-grid\">
        <div class=\"stat\"><div class=\"stat-label\">Recommendations</div><div class=\"stat-value\">{len(recommendations)}</div></div>
        <div class=\"stat\"><div class=\"stat-label\">Normalization failures</div><div class=\"stat-value\">{failures}</div></div>
      </div>
    </section>

    <section class=\"card\">
      <div class=\"card-header\">
        <h2>Top Recommendations</h2>
        <span class=\"note\">Top {len(ranked)}</span>
      </div>
      <div class=\"table-wrap\">
        <table class=\"data-table\">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Query ID</th>
              <th>Type</th>
              <th>Description</th>
              <th>Rationale</th>
              <th>Impact</th>
              <th>Cost Score</th>
              <th>Cost USD</th>
              <th>Frequency</th>
              <th>Scanned</th>
              <th>Output</th>
            </tr>
          </thead>
          <tbody>
            {"".join(rows)}
          </tbody>
        </table>
      </div>
    </section>

    <section class=\"card\">
      <div class=\"card-header\">
        <h2>Query Text</h2>
        <span class=\"note\">Top {len(ranked)}</span>
      </div>
      {"".join(query_blocks)}
    </section>
    """
    return _html_page(body)


def _estimate_cost_usd_bytes(bytes_scanned: int, price_per_tb_usd: float) -> float:
    if bytes_scanned <= 0 or price_per_tb_usd <= 0:
        return 0.0
    tb_scanned = bytes_scanned / _BYTES_PER_TB
    return tb_scanned * price_per_tb_usd


def _html_page(body: str) -> str:
    template = """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Athena Optimizer Report</title>
    <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\" />
    <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin />
    <link href=\"https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,500;700&display=swap\" rel=\"stylesheet\" />
    <style>
      :root {
        --bg: #f7f4ec;
        --ink: #1c1b18;
        --muted: #5c5a52;
        --card: #ffffff;
        --border: #e7e0d3;
        --accent: #0f6a5a;
        --accent-2: #cc7a28;
        --code: #f4efe5;
        --shadow: 0 18px 45px rgba(18, 16, 12, 0.12);
      }

      * { box-sizing: border-box; }

      body {
        margin: 0;
        color: var(--ink);
        background:
          radial-gradient(800px 420px at 10% -10%, rgba(204, 122, 40, 0.18), rgba(204, 122, 40, 0)),
          radial-gradient(900px 520px at 95% -20%, rgba(15, 106, 90, 0.18), rgba(15, 106, 90, 0)),
          var(--bg);
        font-family: "Space Grotesk", "Work Sans", sans-serif;
      }

      .report {
        max-width: 1120px;
        margin: 32px auto 64px;
        padding: 0 24px 32px;
      }

      h1, h2, h3 {
        font-family: "Source Serif 4", "Times New Roman", serif;
        letter-spacing: -0.02em;
      }

      h1 { font-size: 36px; margin: 0; }
      h2 { font-size: 22px; margin: 0; }
      h3 { font-size: 16px; margin: 0 0 8px; }

      .subtitle {
        margin: 6px 0 0;
        color: var(--muted);
        font-size: 14px;
      }

      .page-header {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 24px;
      }

      .badge {
        background: var(--accent);
        color: #fff;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
      }

      .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: var(--shadow);
        animation: fadeUp 0.5s ease both;
      }

      .card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
      }

      .note {
        font-size: 11px;
        color: var(--muted);
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }

      .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 12px;
      }

      .stat {
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px;
        background: #fbf9f2;
      }

      .stat-label {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 11px;
        color: var(--muted);
        margin-bottom: 6px;
      }

      .stat-value {
        font-size: 22px;
        font-weight: 600;
      }

      .table-wrap { overflow-x: auto; }

      .data-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
      }

      .data-table th,
      .data-table td {
        padding: 10px 12px;
        border-bottom: 1px solid var(--border);
        text-align: left;
        font-size: 13px;
      }

      .data-table th {
        background: #f7f0e1;
        position: sticky;
        top: 0;
        z-index: 1;
      }

      .data-table tr:nth-child(even) td {
        background: #faf7ee;
      }

      .clickable { cursor: pointer; }
      .clickable:hover td { background: #f2e9d7; }
      .clickable:focus { outline: 2px solid var(--accent-2); outline-offset: 2px; }

      .link { color: var(--accent); text-decoration: none; font-weight: 600; }
      .link:hover { text-decoration: underline; }

      .query-block {
        border-top: 1px dashed var(--border);
        padding-top: 12px;
        margin-top: 12px;
        scroll-margin-top: 80px;
      }

      .query-block:target {
        background: #fff5e0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 10px 24px rgba(15, 106, 90, 0.15);
      }

      .sql-block {
        background: var(--code);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px;
        overflow-x: auto;
      }

      .extra-queries {
        margin-top: 12px;
        border: 1px dashed var(--border);
        border-radius: 12px;
        padding: 12px;
        background: #fbf7ef;
      }

      .extra-queries summary {
        cursor: pointer;
        font-weight: 600;
        color: var(--accent-2);
        margin-bottom: 12px;
      }

      .query-sub h4 {
        font-size: 14px;
        margin: 0 0 6px;
      }

      @keyframes fadeUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
      }

      @media (max-width: 720px) {
        .page-header { flex-direction: column; align-items: flex-start; }
        h1 { font-size: 28px; }
      }
    </style>
  </head>
  <body>
    <main class=\"report\">
      {body}
    </main>
    <script>
      document.querySelectorAll("tr[data-href]").forEach((row) => {
        row.addEventListener("click", (event) => {
          const target = event.target;
          if (target && target.closest("a")) {
            return;
          }
          const href = row.getAttribute("data-href");
          if (href) {
            window.location.hash = href.substring(1);
          }
        });

        row.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            const href = row.getAttribute("data-href");
            if (href) {
              window.location.hash = href.substring(1);
            }
          }
        });
      });
    </script>
  </body>
</html>
"""
    return template.replace("{body}", body)
