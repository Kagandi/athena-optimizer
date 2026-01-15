from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import boto3

from athena_optimizer.config import AwsConfig
from athena_optimizer.ingestion.mapper import query_execution_to_record
from athena_optimizer.models import QueryRecord

_MAX_PAGE_SIZE = 50
_BATCH_SIZE = 50


@dataclass
class AthenaApiIngestor:
    aws: AwsConfig
    workgroup: str | None = None
    max_results: int | None = 1000
    state_filter: set[str] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    page_size: int = 50

    def fetch(self) -> list[QueryRecord]:
        client = _build_client(self.aws)
        query_ids = _list_query_execution_ids(
            client,
            workgroup=self.workgroup,
            max_results=self.max_results,
            page_size=self.page_size,
        )

        records: list[QueryRecord] = []
        for chunk in _chunked(query_ids, _BATCH_SIZE):
            response = client.batch_get_query_execution(QueryExecutionIds=chunk)
            for execution in response.get("QueryExecutions", []):
                record = query_execution_to_record(execution, source="athena_api")
                if not record.query_id:
                    continue
                if self.state_filter and record.state not in self.state_filter:
                    continue
                if (
                    self.start_time
                    and record.submission_time
                    and record.submission_time < self.start_time
                ):
                    continue
                if (
                    self.end_time
                    and record.submission_time
                    and record.submission_time > self.end_time
                ):
                    continue
                records.append(record)
        return records


def _build_client(aws: AwsConfig):
    session = boto3.Session(profile_name=aws.profile, region_name=aws.region)
    return session.client("athena")


def _list_query_execution_ids(
    client,
    *,
    workgroup: str | None,
    max_results: int | None,
    page_size: int,
) -> list[str]:
    query_ids: list[str] = []
    next_token: str | None = None
    page_size = min(page_size, _MAX_PAGE_SIZE)

    while True:
        params: dict[str, object] = {"MaxResults": page_size}
        if workgroup:
            params["WorkGroup"] = workgroup
        if next_token:
            params["NextToken"] = next_token

        response = client.list_query_executions(**params)
        query_ids.extend(response.get("QueryExecutionIds", []))

        if max_results is not None and len(query_ids) >= max_results:
            return query_ids[:max_results]

        next_token = response.get("NextToken")
        if not next_token:
            break

    return query_ids


def _chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]
