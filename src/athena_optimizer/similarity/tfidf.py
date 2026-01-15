from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from athena_optimizer.config import SimilarityConfig
from athena_optimizer.models import QueryRecord
from athena_optimizer.similarity.structural import build_structural_corpus


def build_tfidf_matrix(
    records: Sequence[QueryRecord],
    config: SimilarityConfig,
) -> tuple[object, dict[str, TfidfVectorizer]]:
    return build_hybrid_matrix(records, config, use_structural=False)


def build_hybrid_matrix(
    records: Sequence[QueryRecord],
    config: SimilarityConfig,
    *,
    use_structural: bool = True,
) -> tuple[object, dict[str, TfidfVectorizer]]:
    corpus = [_record_text(record) for record in records]
    lexical_vectorizer = TfidfVectorizer(
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
    )
    lexical_matrix: Any = lexical_vectorizer.fit_transform(corpus)

    if not use_structural or config.structural_weight <= 0:
        return lexical_matrix, {"lexical": lexical_vectorizer}

    structural_corpus = build_structural_corpus(records)
    try:
        structural_vectorizer = TfidfVectorizer(
            max_features=config.max_features,
            ngram_range=(1, 1),
            min_df=1,
            max_df=1.0,
            token_pattern=r"(?u)\b[\w:]+\b",
            lowercase=False,
        )
        structural_matrix: Any = structural_vectorizer.fit_transform(structural_corpus)
    except ValueError:
        return lexical_matrix, {"lexical": lexical_vectorizer}

    weight_total = config.lexical_weight + config.structural_weight
    if weight_total <= 0:
        return lexical_matrix, {"lexical": lexical_vectorizer}

    lexical_weight = config.lexical_weight / weight_total
    structural_weight = config.structural_weight / weight_total
    combined = hstack(
        [
            lexical_matrix.multiply(lexical_weight),
            structural_matrix.multiply(structural_weight),
        ]
    )
    return combined, {
        "lexical": lexical_vectorizer,
        "structural": structural_vectorizer,
    }


def _record_text(record: QueryRecord) -> str:
    if record.normalized_query:
        return record.normalized_query
    return record.query_text
