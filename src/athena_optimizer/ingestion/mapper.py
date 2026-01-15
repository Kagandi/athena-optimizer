from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from athena_optimizer.models import QueryRecord


def parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        cleaned = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(cleaned)
        except ValueError:
            return None
    return None


def query_execution_to_record(
    execution: Mapping[str, Any],
    *,
    source: str,
) -> QueryRecord:
    status = execution.get("Status", {})
    stats = execution.get("Statistics", {})
    context = execution.get("QueryExecutionContext", {})
    result_conf = execution.get("ResultConfiguration", {})

    submission_time = parse_datetime(status.get("SubmissionDateTime"))
    runtime_ms = stats.get("TotalExecutionTimeInMillis") or stats.get("EngineExecutionTimeInMillis")
    data_scanned_bytes = stats.get("DataScannedInBytes")

    metadata = {
        "database": context.get("Database"),
        "catalog": context.get("Catalog"),
        "output_location": result_conf.get("OutputLocation"),
        "statement_type": execution.get("StatementType"),
        "engine_version": execution.get("EngineVersion", {}).get("SelectedEngineVersion"),
    }

    return QueryRecord(
        query_id=execution.get("QueryExecutionId", ""),
        query_text=execution.get("Query", ""),
        workgroup=execution.get("WorkGroup"),
        submission_time=submission_time,
        runtime_ms=runtime_ms,
        data_scanned_bytes=data_scanned_bytes,
        state=status.get("State"),
        source=source,
        metadata=metadata,
    )
