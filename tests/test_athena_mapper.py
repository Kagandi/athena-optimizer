from __future__ import annotations

from datetime import datetime, timezone

from athena_optimizer.ingestion.mapper import parse_datetime, query_execution_to_record


def test_parse_datetime_iso_string() -> None:
    value = "2024-01-01T12:30:45Z"
    parsed = parse_datetime(value)
    assert parsed is not None
    assert parsed.isoformat() == "2024-01-01T12:30:45+00:00"


def test_parse_datetime_invalid_string() -> None:
    assert parse_datetime("not-a-date") is None


def test_parse_datetime_passthrough() -> None:
    value = datetime(2024, 2, 2, tzinfo=timezone.utc)
    assert parse_datetime(value) is value


def test_query_execution_to_record_maps_fields() -> None:
    submission = datetime(2024, 5, 1, tzinfo=timezone.utc)
    execution = {
        "QueryExecutionId": "exec-123",
        "Query": "SELECT 1",
        "WorkGroup": "primary",
        "Status": {"SubmissionDateTime": submission, "State": "SUCCEEDED"},
        "Statistics": {
            "TotalExecutionTimeInMillis": 2500,
            "DataScannedInBytes": 1024,
        },
        "QueryExecutionContext": {"Database": "analytics", "Catalog": "AwsDataCatalog"},
        "ResultConfiguration": {"OutputLocation": "s3://bucket/prefix"},
        "StatementType": "DML",
        "EngineVersion": {"SelectedEngineVersion": "Athena engine version 3"},
    }

    record = query_execution_to_record(execution, source="athena_api")

    assert record.query_id == "exec-123"
    assert record.query_text == "SELECT 1"
    assert record.workgroup == "primary"
    assert record.submission_time == submission
    assert record.runtime_ms == 2500
    assert record.data_scanned_bytes == 1024
    assert record.state == "SUCCEEDED"
    assert record.source == "athena_api"
    assert record.metadata["database"] == "analytics"
    assert record.metadata["catalog"] == "AwsDataCatalog"
    assert record.metadata["output_location"] == "s3://bucket/prefix"
    assert record.metadata["statement_type"] == "DML"
    assert record.metadata["engine_version"] == "Athena engine version 3"


def test_query_execution_to_record_engine_execution_time_fallback() -> None:
    execution = {
        "QueryExecutionId": "exec-456",
        "Query": "SELECT 1",
        "Status": {"State": "SUCCEEDED"},
        "Statistics": {"EngineExecutionTimeInMillis": 900},
    }

    record = query_execution_to_record(execution, source="athena_api")

    assert record.runtime_ms == 900
