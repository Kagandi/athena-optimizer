from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import boto3

from athena_optimizer.config import AwsConfig
from athena_optimizer.ingestion.mapper import parse_datetime, query_execution_to_record
from athena_optimizer.models import QueryRecord


@dataclass
class S3QueryJsonIngestor:
    aws: AwsConfig
    bucket: str
    prefix: str = ""
    max_records: int | None = None

    def fetch(self) -> list[QueryRecord]:
        client = _build_client(self.aws)
        paginator = client.get_paginator("list_objects_v2")

        records: list[QueryRecord] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for item in page.get("Contents", []):
                key = item.get("Key")
                if not key or not _is_json_key(key):
                    continue
                response = client.get_object(Bucket=self.bucket, Key=key)
                body = response["Body"].read()
                for record in _parse_records(body):
                    records.append(record)
                    if self.max_records is not None and len(records) >= self.max_records:
                        return records[: self.max_records]
        return records


def _build_client(aws: AwsConfig):
    session = boto3.Session(profile_name=aws.profile, region_name=aws.region)
    return session.client("s3")


def _is_json_key(key: str) -> bool:
    return key.endswith(".json") or key.endswith(".jsonl")


def _parse_records(blob: bytes) -> list[QueryRecord]:
    text = blob.decode("utf-8")
    text = text.strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        return [_record_from_payload(item) for item in data if isinstance(item, dict)]
    records: list[QueryRecord] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(_record_from_payload(payload))
    return records


def _record_from_payload(payload: dict[str, Any]) -> QueryRecord:
    if "QueryExecutionId" in payload:
        return query_execution_to_record(payload, source="s3")

    return QueryRecord(
        query_id=payload.get("query_id", ""),
        query_text=payload.get("query_text", ""),
        normalized_query=payload.get("normalized_query"),
        workgroup=payload.get("workgroup"),
        user=payload.get("user"),
        submission_time=parse_datetime(payload.get("submission_time")),
        runtime_ms=payload.get("run_time_ms") or payload.get("runtime_ms"),
        data_scanned_bytes=payload.get("data_scanned_bytes"),
        bytes_output=payload.get("bytes_output"),
        state=payload.get("state"),
        source=payload.get("source", "s3"),
        metadata=payload.get("metadata", {}),
    )
