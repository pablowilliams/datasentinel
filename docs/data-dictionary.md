# Data dictionary

## loan_applications, the business data

Declared once in `src/datasentinel/schema.py` and used by the generator, the
ingestion validator and the quality checks alike. Three copies of a schema is how
a schema drifts.

| Column | Type | Null | Bounds | Critical | Notes |
|---|---|---|---|---|---|
| `application_id` | string | no | | yes | The key. Duplicates are a critical incident |
| `application_date` | date | no | | yes | Drives the freshness check |
| `customer_age` | int | yes | 18 to 100 | | |
| `annual_income` | float | yes | 0 to 2,000,000 | yes | 1.2% null on a healthy day |
| `employment_status` | string | yes | five known categories | | A sixth is a category incident |
| `loan_amount` | float | no | 1 to 1,000,000 | yes | |
| `loan_term_months` | int | no | 12, 24, 36, 48, 60 | | |
| `credit_score` | float | yes | 300 to 850 | yes | 2.0% null on a healthy day |
| `debt_to_income_ratio` | float | yes | 0 to 1 | | |
| `previous_defaults` | int | yes | 0 to 20 | | |
| `approved` | bool | no | | yes | The underwriting decision, and the model's label |
| `interest_rate` | float | yes | 0 to 40 | | **Post-decision.** Never a model feature |

**Critical** means the column carries the business meaning. Missingness in a
critical column escalates its incident one severity step, because a 10% null rate
in `interest_rate` is untidy while the same rate in `credit_score` changes who
gets a loan.

Healthy missingness is deliberate and non-zero. A threshold calibrated against a
perfectly complete dataset would alert on the first ordinary day.

## Monitoring metadata

| Table | Holds | Written by |
|---|---|---|
| `datasets` | Registered datasets and their SLA | `ingestion/loaders.py` |
| `pipeline_runs` | One row per load attempt, including failures | `ingestion/loaders.py` |
| `raw_loan_applications` | The raw layer, as it arrived | `ingestion/loaders.py` |
| `quality_results` | Every check on every run, passed or not | `monitoring/pipeline.py` |
| `drift_results` | Every drift test, including the ones that found nothing | `monitoring/pipeline.py` |
| `incidents` | What a person should act on, with evidence | `monitoring/pipeline.py` |
| `model_metrics` | One row per metric per evaluation window | `model/monitor.py` |

Passing checks and negative drift results are stored, not just failures. A
monitoring platform that only records what went wrong cannot tell you whether it
was looking.

### Known gaps

`datasets.freshness_sla_minutes` is written and never read; the 24 hour SLA is
hardcoded in `config/checks.yml`. The plan's `quality_checks` table was never
created, because checks live in YAML instead (`docs/decisions/0006`).

## Scenarios

| Scenario | Kind | Should raise |
|---|---|---|
| `healthy` | healthy | nothing |
| `minor_noise` | benign | nothing |
| `weekend` | benign | nothing |
| `trivial_income_tick` | benign | nothing |
| `missing_income` | failure | completeness |
| `schema_break` | failure | schema |
| `population_shift` | failure | drift, category |
| `income_shift` | failure | drift |
| `stale` | failure | freshness |
| `duplicates` | failure | uniqueness |
| `volume_drop` | failure | volume |
| `invalid_values` | failure | validity |

The benign three are the reason precision means anything. Without them a detector
scores perfect recall by alerting on everything.
