import logging

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

for _logger_name in ("sqlglot", "sqlglot.parser"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)


def normalize_sql(sql: str, *, dialect: str = "athena", anonymize_literals: bool = False) -> str:
    try:
        expression = parse_one(sql, read=dialect, error_level="ignore")
    except ParseError as exc:
        message = f"Failed to parse SQL for dialect '{dialect}'."
        raise ValueError(message) from exc

    if anonymize_literals:
        expression = expression.transform(_anonymize_literals)

    return expression.sql(dialect=dialect, pretty=False)


def _anonymize_literals(node: exp.Expression) -> exp.Expression:
    if isinstance(node, exp.Literal):
        return exp.Literal.string("?")
    return node
