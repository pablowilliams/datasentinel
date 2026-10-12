# Architecture

DataSentinel watches a daily batch of loan applications. It loads them, checks
them against declared rules, compares the population against a reference window,
scores an approval model on them, and records incidents that state what was
expected and what arrived.

## Flow

```
CSV or generated batch
        |
        v
  ingestion/loaders.py ......... writes a run record BEFORE attempting the load
        |                        idempotent on a payload hash
        v
  raw_loan_applications ........ the raw layer accepts what arrived
        |
        +--> quality/checks.py ..... nine deterministic checks, declared in YAML
        |                            "is this data valid?"
        |
        +--> drift/detector.py ..... numeric, categorical and missingness tests
        |                            "is it still the same world?"
        |
        +--> model/monitor.py ...... the approval classifier scored on the batch
        |                            "does the model still agree with reality?"
        v
  incidents/service.py ......... severity, category, and the evidence for the claim
        v
  incidents + quality_results + drift_results + model_metrics
```

Three layers ask three different questions and none subsumes another. Measured:
`income_shift` passes all seventeen deterministic checks while moving median
income 48% and halving the approval rate, so only drift sees it. `invalid_values`
damages the model without moving any distribution the drift tests examine, so
only the range check sees it. A platform with one layer has a blind spot shaped
like the other.

## Why PostgreSQL holds both the data and the metadata

An incident is only meaningful next to the run that produced it, and a run is
only meaningful next to the rows it loaded. Splitting them across two stores buys
nothing here and costs every join that matters.

The warehouse is PostgreSQL 17. There is no second analytical store: an earlier
dependency list included DuckDB, which no code ever used, and it has been
removed rather than left in to look modern.

## Modules

| Module | Lines | Does |
|---|---:|---|
| `generator.py` | 276 | Twelve scenarios: healthy, failure, benign |
| `schema.py` | 56 | One declaration of the columns and their bounds |
| `ingestion/loaders.py` | 154 | Run tracking, idempotency, raw load |
| `quality/checks.py` | 254 | Nine check types and their registry |
| `quality/runner.py` | 28 | YAML to check specs |
| `drift/distributions.py` | 130 | KS, PSI, chi-squared, two-proportion z |
| `drift/detector.py` | 167 | The two-gate rule and FDR control |
| `incidents/service.py` | 71 | Failures to incidents, with severity |
| `model/features.py` | 44 | Decision-time features, and the leakage guard |
| `model/baseline.py` | 121 | The approval classifier and its card |
| `model/monitor.py` | 122 | Scoring, metrics, persistence |
| `model/decay.py` | 283 | Does detected drift mean real damage? |
| `monitoring/pipeline.py` | 181 | One run, end to end |
| `monitoring/evaluation.py` | 186 | Detection scored across scenarios |
| `monitoring/sensitivity.py` | 126 | The smallest fault reliably caught |
| `api/export.py` | 113 | Snapshot for the static dashboard |
| `cli.py` | 155 | Commands |

## Check types

| Type | Incident category | Asks |
|---|---|---|
| `schema_required` | schema | Are the required columns present? |
| `schema_unexpected` | schema | Has anything new appeared? |
| `schema_type` | schema | Does the column parse as declared? |
| `null_rate` | completeness | Is missingness within tolerance? |
| `range` | validity | Are values inside their bounds? |
| `allowed_values` | category | Are the categories known? |
| `unique` | uniqueness | Is the key a key? |
| `freshness` | freshness | Is the newest record recent enough? |
| `volume` | volume | Did roughly the expected number of rows arrive? |

Every column-level check stays silent when its column is absent, so one schema
break produces one incident about one cause rather than a dozen about the same
one.

## Tables

`datasets`, `pipeline_runs`, `raw_loan_applications`, `quality_results`,
`drift_results`, `incidents`, `model_metrics`. The schema is in
`src/datasentinel/db/schema.sql` and every statement is idempotent, so `migrate`
is safe to rerun.

## What is not here

No dbt project, no FastAPI service, no Docker. The published dashboard is a
static page reading a committed snapshot, because GitHub Pages serves files
rather than processes. Orchestration is a Makefile and a CLI rather than Prefect.
See the README's "State of the work".
