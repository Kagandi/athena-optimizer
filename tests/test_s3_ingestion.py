from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from athena_optimizer.ingestion import s3 as s3_ingestion


def test_parse_records_empty_blob() -> None:
    assert s3_ingestion._parse_records(b"") == []


def test_parse_records_json_array_filters_non_dict() -> None:
    payload = [
        {"query_id": "q1", "query_text": "SELECT 1"},
        "skip",
        123,
    ]
    records = s3_ingestion._parse_records(json.dumps(payload).encode("utf-8"))
    assert len(records) == 1
    assert records[0].query_id == "q1"


def test_parse_records_jsonl_lines() -> None:
    lines = [
        json.dumps({"query_id": "q1", "query_text": "SELECT 1"}),
        "",
        json.dumps({"query_id": "q2", "query_text": "SELECT 2"}),
    ]
    blob = "\n".join(lines).encode("utf-8")
    records = s3_ingestion._parse_records(blob)
    assert [record.query_id for record in records] == ["q1", "q2"]


def test_record_from_payload_accepts_legacy_runtime() -> None:
    payload = {
        "query_id": "q1",
        "query_text": "SELECT 1",
        "run_time_ms": 1234,
    }
    record = s3_ingestion._record_from_payload(payload)
    assert record.runtime_ms == 1234


def test_record_from_payload_query_execution_shape() -> None:
    submission = datetime(2024, 1, 1, tzinfo=timezone.utc)
    payload = {
        "QueryExecutionId": "exec-1",
        "Query": "SELECT 1",
        "Status": {"SubmissionDateTime": submission, "State": "SUCCEEDED"},
        "Statistics": {"TotalExecutionTimeInMillis": 500, "DataScannedInBytes": 2048},
    }
    record = s3_ingestion._record_from_payload(payload)
    assert record.query_id == "exec-1"
    assert record.query_text == "SELECT 1"
    assert record.state == "SUCCEEDED"
    assert record.source == "s3"


def test_parse_records_invalid_json_raises() -> None:
    with pytest.raises(json.JSONDecodeError):
        s3_ingestion._parse_records(b"{not-json}")


def test_is_json_key() -> None:
    assert s3_ingestion._is_json_key("path/file.json")
    assert s3_ingestion._is_json_key("path/file.jsonl")
    assert not s3_ingestion._is_json_key("path/file.txt")
