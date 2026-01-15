from __future__ import annotations

import boto3
from botocore.stub import Stubber

from athena_optimizer.ingestion.athena_api import _list_query_execution_ids


def _athena_client():
    return boto3.client(
        "athena",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


def test_list_query_execution_ids_paginates() -> None:
    client = _athena_client()
    stubber = Stubber(client)

    stubber.add_response(
        "list_query_executions",
        {"QueryExecutionIds": ["q1", "q2"], "NextToken": "token-1"},
        {"MaxResults": 50, "WorkGroup": "primary"},
    )
    stubber.add_response(
        "list_query_executions",
        {"QueryExecutionIds": ["q3"]},
        {"MaxResults": 50, "WorkGroup": "primary", "NextToken": "token-1"},
    )

    with stubber:
        result = _list_query_execution_ids(
            client,
            workgroup="primary",
            max_results=None,
            page_size=200,
        )

    assert result == ["q1", "q2", "q3"]


def test_list_query_execution_ids_honors_max_results() -> None:
    client = _athena_client()
    stubber = Stubber(client)

    stubber.add_response(
        "list_query_executions",
        {"QueryExecutionIds": ["q1", "q2"], "NextToken": "token-1"},
        {"MaxResults": 10},
    )
    stubber.add_response(
        "list_query_executions",
        {"QueryExecutionIds": ["q3", "q4"]},
        {"MaxResults": 10, "NextToken": "token-1"},
    )

    with stubber:
        result = _list_query_execution_ids(
            client,
            workgroup=None,
            max_results=3,
            page_size=10,
        )

    assert result == ["q1", "q2", "q3"]
