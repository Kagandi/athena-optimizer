from dataclasses import dataclass, field


@dataclass(frozen=True)
class AwsConfig:
    profile: str | None = None
    region: str | None = None


@dataclass(frozen=True)
class NormalizationConfig:
    dialect: str = "athena"
    anonymize_literals: bool = False


@dataclass(frozen=True)
class ScoringConfig:
    bytes_weight: float = 0.6
    runtime_weight: float = 0.3
    frequency_weight: float = 0.1
    price_per_tb_usd: float = 1.0


@dataclass(frozen=True)
class SimilarityConfig:
    lexical_weight: float = 0.8
    structural_weight: float = 0.2
    max_features: int | None = 5000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 1
    max_df: float = 1.0


@dataclass(frozen=True)
class ClusteringConfig:
    method: str = "agglomerative"
    distance_threshold: float = 0.3
    linkage: str = "average"


@dataclass(frozen=True)
class AppConfig:
    aws: AwsConfig = field(default_factory=AwsConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
