# Athena Optimizer

Athena Optimizer analyzes Amazon Athena query history to identify expensive, repetitive queries and produce optimization signals.

## Requirements
- Python 3.13+
- uv
- AWS credentials (AWS SSO profile or default chain)

## Setup
```bash
uv sync --directory /Users/kagandima/Projects/athena-optimizer
```

If you use AWS SSO:
```bash
aws sso login --profile <profile>
```

## Run
The CLI entrypoint is installed by uv, so you can run:
```bash
uv run --directory /Users/kagandima/Projects/athena-optimizer -- athena-optimizer --help
```

## CLI Commands
- `ingest-athena`: Fetch query executions from the Athena API and print a count.
- `ingest-s3`: Load query records from S3 JSON/JSONL and print a count.
- `analyze-athena`: Ingest, normalize, score, filter, and export Athena API results.
- `analyze-s3`: Ingest, normalize, score, filter, and export S3 query records.
- `cluster-athena`: Ingest, normalize, cluster, and export Athena API results.
- `cluster-s3`: Ingest, normalize, cluster, and export S3 query records.
- `recommend-athena`: Generate rule-based recommendations from Athena API results.
- `recommend-s3`: Generate rule-based recommendations from S3 query records.

### Analyze Athena API
```bash
uv run --directory /Users/kagandima/Projects/athena-optimizer -- \
  athena-optimizer analyze-athena \
  --max-results 100 \
  --profile <profile> \
  --region <region>
```

`--start-time` and `--end-time` accept either `YYYY-MM-DD` or full ISO-8601 timestamps.

### Analyze from S3 JSON/JSONL
```bash
uv run --directory /Users/kagandima/Projects/athena-optimizer -- \
  athena-optimizer analyze-s3 \
  --bucket <bucket> \
  --prefix <prefix>
```

S3 JSON accepts `runtime_ms` or legacy `run_time_ms` fields.

### Output results
```bash
uv run --directory /Users/kagandima/Projects/athena-optimizer -- \
  athena-optimizer analyze-athena \
  --max-results 100 \
  --output /tmp/athena_scored.json \
  --output-format json
```

```bash
uv run --directory /Users/kagandima/Projects/athena-optimizer -- \
  athena-optimizer analyze-athena \
  --max-results 100 \
  --report-md /tmp/athena_report.md
```

```bash
uv run --directory /Users/kagandima/Projects/athena-optimizer -- \
  athena-optimizer analyze-athena \
  --max-results 100 \
  --report-html /tmp/athena_report.html
```

Cost estimate defaults to $1/TB (Athena $5 per 5TB). Override with `--price-per-tb-usd`.

### Filter thresholds
```bash
uv run --directory /Users/kagandima/Projects/athena-optimizer -- \
  athena-optimizer analyze-athena \
  --min-gb 1.0 \
  --min-runtime-seconds 1.0
```

Thresholds are applied with OR semantics; a query is kept if it meets any active threshold.

### Cluster similar queries
```bash
uv run --directory /Users/kagandima/Projects/athena-optimizer -- \
  athena-optimizer cluster-athena \
  --max-results 200 \
  --min-gb 1.0 \
  --min-runtime-seconds 1.0 \
  --distance-threshold 0.3 \
  --output /tmp/athena_clusters.json \
  --output-format json \
  --report-md /tmp/athena_clusters.md \
  --report-html /tmp/athena_clusters.html
```

```bash
uv run --directory /Users/kagandima/Projects/athena-optimizer -- \
  athena-optimizer cluster-s3 \
  --bucket <bucket> \
  --prefix <prefix> \
  --distance-threshold 0.3 \
  --output /tmp/athena_clusters.json \
  --output-format json \
  --report-md /tmp/athena_clusters.md
```

Similarity options:
- `--similarity-method tfidf|bm25` (default: tfidf)
- `--use-structural/--no-structural` to include SQL structural tokens
- `--lexical-weight` and `--structural-weight` to balance feature groups
- `--extra-queries` to include more cluster members in HTML reports

### Generate recommendations
```bash
uv run --directory /Users/kagandima/Projects/athena-optimizer -- \
  athena-optimizer recommend-athena \
  --max-results 200 \
  --min-gb 1.0 \
  --min-runtime-seconds 1.0 \
  --output /tmp/athena_recommendations.json \
  --output-format json
```

```bash
uv run --directory /Users/kagandima/Projects/athena-optimizer -- \
  athena-optimizer recommend-athena \
  --max-results 200 \
  --report-html /tmp/athena_recommendations.html
```

```bash
uv run --directory /Users/kagandima/Projects/athena-optimizer -- \
  athena-optimizer recommend-s3 \
  --bucket <bucket> \
  --prefix <prefix> \
  --output /tmp/athena_recommendations.json \
  --output-format json
```

Distance threshold notes:
- Smaller values produce tighter clusters (more clusters).
- Larger values merge more queries into fewer clusters.
- Typical range: 0.2 to 0.6 for cosine distance.
- Use `--auto-threshold` to pick a threshold from the distance distribution.

## Project Structure
- `src/athena_optimizer/`: core library
- `tests/`: test suite
- `notebooks/`: marimo notebooks

## Notes
- Athena ingestion uses boto3 and the default credential chain.
- Use `--profile` and `--region` to override defaults.
- CLI output uses GB for scanned/output size and seconds for runtime.
- Estimated cost uses decimal TB pricing (default $1/TB).
