from __future__ import annotations

from collections.abc import Sequence
import logging

from sqlglot import exp, parse_one

from athena_optimizer.models import QueryRecord

for _logger_name in ("sqlglot", "sqlglot.parser"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)


def build_structural_corpus(records: Sequence[QueryRecord]) -> list[str]:
    corpus: list[str] = []
    for record in records:
        sql = record.normalized_query or record.query_text
        tokens = extract_structural_tokens(sql)
        corpus.append(" ".join(tokens))
    return corpus


def extract_structural_tokens(sql: str) -> list[str]:
    try:
        expression = parse_one(sql, read="athena", error_level="ignore")
    except Exception:
        return []

    tokens: set[str] = set()

    for table in expression.find_all(exp.Table):
        if table.name:
            tokens.add(f"table:{table.name.lower()}")

    for column in expression.find_all(exp.Column):
        if column.name:
            tokens.add(f"col:{column.name.lower()}")

    for join in expression.find_all(exp.Join):
        kind = join.args.get("kind")
        kind_value = (kind or "inner").lower()
        tokens.add(f"join:{kind_value}")

    for func in expression.find_all(exp.Func):
        func_name = func.sql_name().lower()
        tokens.add(f"func:{func_name}")

    if expression.args.get("where") is not None:
        tokens.add("has_where")
    if expression.args.get("group") is not None:
        tokens.add("has_group_by")
    if expression.args.get("order") is not None:
        tokens.add("has_order_by")
    if expression.args.get("limit") is not None:
        tokens.add("has_limit")
    if expression.args.get("having") is not None:
        tokens.add("has_having")

    return sorted(tokens)
