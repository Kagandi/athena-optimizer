from __future__ import annotations

from collections.abc import Sequence

import logging

from sqlglot import exp, parse_one

for _logger_name in ("sqlglot", "sqlglot.parser"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

from athena_optimizer.models import QueryRecommendation, Recommendation
from athena_optimizer.scoring import ScoredRecord

_BYTES_PER_GB = 1024**3
_BYTES_PER_TB = 10**12
_MS_PER_SECOND = 1000


def _format_scanned(value_gb: float) -> str:
    if value_gb >= 1000:
        return f"{value_gb / 1000:.3f} TB"
    return f"{value_gb:.3f} GB"


class RecommendationRule:
    def evaluate(self, scored: ScoredRecord) -> Recommendation | None:
        raise NotImplementedError


class SelectStarRule(RecommendationRule):
    def evaluate(self, scored: ScoredRecord) -> Recommendation | None:
        expression = _parse_expression(scored)
        if expression is None:
            return None

        for projection in expression.find_all(exp.Star):
            if projection is not None:
                return Recommendation(
                    recommendation_type="avoid_select_star",
                    description="Replace SELECT * with only required columns.",
                    rationale="Selecting all columns increases scanned bytes and cost.",
                    estimated_impact="Lower scanned data and cost.",
                )
        return None


class MissingWhereRule(RecommendationRule):
    def __init__(self, *, min_scanned_gb: float = 1.0) -> None:
        self._min_scanned_gb = min_scanned_gb

    def evaluate(self, scored: ScoredRecord) -> Recommendation | None:
        expression = _parse_expression(scored)
        if expression is None:
            return None
        if expression.args.get("where") is not None:
            return None

        scanned_gb = (scored.record.data_scanned_bytes or 0) / _BYTES_PER_GB
        if scanned_gb < self._min_scanned_gb:
            return None

        return Recommendation(
            recommendation_type="add_filters_or_partitions",
            description="Add WHERE filters or partition predicates to reduce scan.",
            rationale=f"Query scanned {scanned_gb:.3f} GB without filters.",
            estimated_impact="Lower scanned data and cost.",
        )


class HighFrequencyRule(RecommendationRule):
    def __init__(self, *, min_frequency: int = 5, min_scanned_gb: float = 1.0) -> None:
        self._min_frequency = min_frequency
        self._min_scanned_gb = min_scanned_gb

    def evaluate(self, scored: ScoredRecord) -> Recommendation | None:
        scanned_gb = (scored.record.data_scanned_bytes or 0) / _BYTES_PER_GB
        if scored.frequency < self._min_frequency or scanned_gb < self._min_scanned_gb:
            return None

        return Recommendation(
            recommendation_type="consider_materialized_view",
            description="Consider a materialized view or CTAS for repeated queries.",
            rationale=(
                f"Query runs frequently (freq={scored.frequency}) and scans {scanned_gb:.3f} GB."
            ),
            estimated_impact="Reduce repeated scans and runtime.",
        )


class LongRuntimeRule(RecommendationRule):
    def __init__(self, *, min_runtime_seconds: float = 60.0) -> None:
        self._min_runtime_seconds = min_runtime_seconds

    def evaluate(self, scored: ScoredRecord) -> Recommendation | None:
        runtime_s = (scored.record.runtime_ms or 0) / _MS_PER_SECOND
        if runtime_s < self._min_runtime_seconds:
            return None

        return Recommendation(
            recommendation_type="consider_temp_table",
            description="Consider a temp table/CTAS to reduce repeated compute.",
            rationale=f"Query runtime is {runtime_s:.2f} seconds.",
            estimated_impact="Reduce runtime and repeated compute cost.",
        )


_DEFAULT_RULES = (
    SelectStarRule(),
    MissingWhereRule(),
    HighFrequencyRule(),
    LongRuntimeRule(),
)


def recommend_for_scored(
    scored: Sequence[ScoredRecord],
    *,
    min_frequency: int = 1,
) -> list[QueryRecommendation]:
    recommendations: list[QueryRecommendation] = []

    for item in scored:
        if item.frequency < min_frequency:
            continue
        for rule in _DEFAULT_RULES:
            rec = rule.evaluate(item)
            if rec is None:
                continue
            scanned_gb = (item.record.data_scanned_bytes or 0) / _BYTES_PER_GB
            output_gb = (item.record.bytes_output or 0) / _BYTES_PER_GB
            recommendations.append(
                QueryRecommendation(
                    query_id=item.record.query_id,
                    query_text=item.record.query_text,
                    recommendation=rec,
                    cost_score=item.cost_score,
                    estimated_cost_usd=item.estimated_cost_usd,
                    frequency=item.frequency,
                    data_scanned_gb=scanned_gb,
                    data_scanned_display=_format_scanned(scanned_gb),
                    bytes_output_gb=output_gb,
                    bytes_output_display=_format_scanned(output_gb),
                )
            )
    return recommendations


def _parse_expression(scored: ScoredRecord):
    sql = scored.record.normalized_query or scored.record.query_text
    try:
        return parse_one(sql, read="athena", error_level="ignore")
    except Exception:
        return None
