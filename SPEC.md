# Athena Query Optimizer - Project Specification

## 1. Overview
The Athena Query Optimizer (AQO) is a modular library that identifies expensive, repetitive Amazon Athena queries, clusters similar queries, and suggests optimizations. It uses statistical and NLP techniques (e.g., BM25, clustering, embeddings) to group similar queries and rank cost offenders by scanned data, runtime, and frequency. The system produces actionable recommendations such as query rewrites, temporary tables, partitioning, clustering, and materialized views.

## 2. Goals
- Detect expensive Athena queries by cost signals (scan bytes, runtime, concurrency impact).
- Identify repetitive or structurally similar queries using NLP + clustering.
- Recommend query optimization techniques with actionable guidance.
- Provide a modular architecture to swap scoring, vectorization, and clustering strategies.
- Support batch analysis and incremental updates over time.

## 3. Non-Goals
- Executing query rewrites automatically.
- Managing AWS account configuration or permissions.
- Replacing Athena usage reports or billing tools.
- Full SQL validation or correctness proofs for rewritten SQL.

## 4. Users and Use Cases
- Data engineers and analysts reviewing Athena cost and performance.
- FinOps teams tracking large query spend and proposing savings.
- Platform teams enforcing query patterns and optimization standards.

Example workflows:
1. Ingest Athena query history from Athena API and S3 query result metadata.
2. Identify top expensive queries by bytes scanned and runtime.
3. Cluster similar queries and surface canonical query patterns.
4. Recommend optimizations with reasoning and expected impact.

## 5. Functional Requirements
### 5.1 Query Ingestion
- Support ingestion from:
  - Athena Query History (via AWS SDK)
  - S3 query result metadata
  - CloudWatch logs (optional)
  - CSV/JSON exports
- Initial focus: Athena API and S3 sources.
- AWS access uses boto3 with the default credential chain.
- Default credentials are local AWS SSO profiles (AWS_PROFILE).
- Allow explicit profile and region overrides via config/CLI.
- Normalize query text and metadata:
  - query_id, workgroup, user, submission_time, runtime_ms, data_scanned_bytes, bytes_output, state

### 5.2 Query Normalization
- SQL normalization and canonicalization (case, whitespace, literals)
- SQL parser: sqlglot
- Extract structure and features:
  - tables, columns, predicates, joins, group by, order by
- Optional anonymization of literals for privacy

### 5.3 Cost and Importance Scoring
- Cost score based on:
  - data_scanned_bytes
  - runtime_ms
  - frequency of execution
  - concurrency impact
- Estimated cost in USD using Athena pricing ($5 per 1 TB scanned)
- Configurable scoring weights and pricing overrides
- Threshold filters (min scanned GB, min runtime seconds) before ranking
- Rank query groups by total cost and waste potential

### 5.4 Similarity and Clustering
- Similarity methods (pluggable):
  - BM25 (tokenized SQL)
  - TF-IDF + cosine similarity
  - Embeddings (optional)
- Feature sets for similarity (configurable):
  - Lexical tokens from normalized SQL (keywords, identifiers)
  - Structural signatures (tables, join types, predicate columns/operators, group/order by)
  - Projection and aggregation features (selected columns, functions, aggregations)
  - Literal placeholders and query length stats
- Default weighting: lexical-first, structural for refinement
- Clustering algorithms (pluggable):
  - Agglomerative (default, configurable distance_threshold)
  - DBSCAN
  - KMeans
- Output representative queries per cluster
- Optional auto-threshold selection based on distance distribution

### 5.5 Optimization Suggestions
- Rule-based recommendations:
  - Partitioning and predicate pushdown
  - Clustering/bucketing suggestions
  - Temp tables or CTAS
  - Materialized views
  - Rewrite suggestions (e.g., remove SELECT *, use pre-aggregations)
- Each suggestion should include rationale and potential impact

### 5.6 Output and Reporting
- Structured output (JSON, CSV)
- Human-readable report (Markdown/HTML)
- Optional CLI to run analysis
- CLI can write Markdown or HTML reports to file
- CLI output uses GB for scanned/output size and seconds for runtime

## 6. Non-Functional Requirements
- Modular architecture with clear interfaces for each stage
- Extensible with new algorithms and heuristics
- Testable components with deterministic behavior
- Configurable and environment-agnostic

## 7. Architecture (High-Level)
1. Ingestion Layer
2. Normalization Layer
3. Scoring Layer
4. Similarity + Clustering Layer
5. Recommendation Engine
6. Reporting/Export

## 8. Data Model (Initial)
- QueryRecord:
  - query_id, query_text, normalized_query, workgroup, user, submission_time
  - runtime_ms, data_scanned_bytes, bytes_output, state, source, metadata
- QueryStats:
  - runtime_ms, data_scanned_bytes, frequency, cost_score
- Cluster:
  - cluster_id, members, representative_query
- Recommendation:
  - recommendation_type, description, rationale, estimated_impact

## 9. Milestones
### Milestone 1: Project Skeleton (Done)
- Repository layout
- Initial data model definitions
- Basic CLI entry point
- Example sample data ingestion from JSON

### Milestone 2: Normalization + Scoring (Done)
- SQL normalization module
- Cost scoring and ranking
- Configurable weights and thresholds

### Milestone 3: Similarity + Clustering (In Progress)
- TF-IDF similarity (done)
- Structural feature extraction (done)
- Hybrid TF-IDF similarity (done)
- BM25 similarity (done)
- Agglomerative clustering (done)
- Cluster summaries and representative query extraction (in progress)

### Milestone 4: Recommendation Engine (In Progress)
- Rule-based recommendation system (done)
- Per-query recommendations (done)
- Per-cluster recommendations (planned)

### Milestone 5: Reporting + Output (In Progress)
- Markdown report generation (done)
- HTML report generation (done)
- JSON/CSV output (done)
- CLI integration (done)

### Milestone 6: Extensions (Planned)
- Embeddings integration
- Incremental updates and caching
- Performance optimization

## 10. Open Questions
- Minimum dataset size target for clustering performance?

## 11. Risks
- Noisy queries and lack of structure in raw SQL
- High variance in query templates across teams
- False positives in recommendations
- Performance constraints for large query histories
