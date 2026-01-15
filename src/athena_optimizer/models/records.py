from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    query_text: str
    normalized_query: str | None = None
    workgroup: str | None = None
    user: str | None = None
    submission_time: datetime | None = None
    runtime_ms: int | None = None
    data_scanned_bytes: int | None = None
    bytes_output: int | None = None
    state: str | None = None
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryStats:
    runtime_ms: int | None = None
    data_scanned_bytes: int | None = None
    frequency: int = 1
    cost_score: float | None = None


@dataclass(frozen=True)
class Cluster:
    cluster_id: str
    members: list[QueryRecord]
    representative_query: QueryRecord | None = None


@dataclass(frozen=True)
class Recommendation:
    recommendation_type: str
    description: str
    rationale: str | None = None
    estimated_impact: str | None = None


@dataclass(frozen=True)
class QueryRecommendation:
    query_id: str
    query_text: str
    recommendation: Recommendation
    cost_score: float | None = None
    estimated_cost_usd: float | None = None
    frequency: int | None = None
    data_scanned_gb: float | None = None
    data_scanned_display: str | None = None
    bytes_output_gb: float | None = None
    bytes_output_display: str | None = None
