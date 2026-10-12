# DataSentinel

**Data quality and ML monitoring for batch pipelines, characterised by the smallest
fault it can actually see.**

DataSentinel watches a daily batch of loan applications. It loads them, checks them
against declared rules, compares the population against a reference window, and
raises incidents that state what was expected, what arrived, and how much data is
affected.

Python 3.12 · PostgreSQL 17 · pandas · SciPy · scikit-learn · dbt · FastAPI · pytest

## The number that matters is not the recall

Across 720 scenario-days, twelve scenarios over sixty days:

| | |
|---|---|
| Failures detected, by the right category | **480 / 480** |
| False alarms on non-failure days | **3 / 240** (1.3%) |
| False alarms on healthy days | **0 / 60** |
| Precision / recall / F1 | 99.4% / 100% / 99.7% |

Those figures are close to uninformative on their own, and reporting them alone
would be the mistake this project is built to avoid. Recall of 100% against a
fault chosen to be large measures the choice of fault. The honest
characterisation is the smallest fault reliably caught, on at least 90% of days:

| Fault | Reliably caught from | Detection at half that severity |
|---|---|---|
| Duplicate records | **0.2%** of the batch | 100% |
| Invalid values | **0.2%** of rows | 95% at 0.2%, 76% at 0.1% |
| Missing income | **5%** null rate | 62% at 4%, 14% at 3% |
| Rows lost upstream | **30%** of the batch | 81% at 25%, 48% at 20% |
| Population shift, income only | **15%** uplift | 29% at 10%, 10% at 6% |

Read the last row as the real limitation. **A population whose income has moved by
10% is caught on fewer than a third of days.** Anyone relying on this to notice a
gradual shift would be relying on something it cannot do.

The missing-income floor of 5% is not a coincidence: it is the configured
threshold, and detection climbs from 14% to 100% across the three points either
side of it. That is what a correctly calibrated check looks like.

## Three things measurement changed

**The volume floor could not see a quarter of the data go missing.** The original
`min_ratio` of 0.70 caught a 30% row loss on 58% of days and a 25% loss on 23%.
Rather than guess a better number, the floor was swept against both detection and
false alarms over 240 quiet days:

| floor | 15% lost | 20% | 25% | 30% | 40% | false alarms |
|---|---|---|---|---|---|---|
| 0.70 | 0% | 5% | 23% | 58% | 100% | 0.0% |
| **0.80** | **30%** | **58%** | **85%** | **97%** | **100%** | **0.4%** |
| 0.85 | 58% | 83% | 97% | 100% | 100% | 3.8% |
| 0.90 | 83% | 97% | 100% | 100% | 100% | 12.9% |
| 0.95 | 95% | 100% | 100% | 100% | 100% | 27.9% |

0.80 reliably catches a 30% loss for one false alarm in 240 quiet days. Past 0.85
the false alarms arrive faster than the detections, and an alert people learn to
ignore protects nothing.

That curve was wrong the first time it was computed. The generator delivered an
identical row count every weekday, so every floor up to 0.95 reported zero false
alarms and the recommendation would have been 0.95. Real intake varies; the
generator now varies with it, standard deviation 8%, and the answer changed.

**Some faults are invisible to deterministic checks, and this proves it rather
than asserting it.** The `income_shift` scenario moves median income up 48% and
halves the approval rate, from 55.7% to 26.8%. Every range, null, uniqueness,
freshness and category rule passes. Seventeen checks, zero incidents. Only the
distribution comparison sees it. That is the entire argument for drift detection,
and it is a measurement in the test suite rather than a claim in a README.

**A multiple-testing correction that turned out to be redundant.** Ten columns are
tested for drift every run, so over a month the chance of some column looking
significant for no reason approaches one, and an uncorrected run did produce a
false alarm on a 470-row Saturday. Benjamini-Hochberg was added to control it,
then ablated: one false alarm over 360 days either way. The practical-magnitude
gate was already rejecting everything the correction would have caught. It is kept
because it is cheap and protects the system if the effect gates are ever loosened,
and it is documented as redundant at current thresholds rather than advertised as
an improvement.

## Two gates, both of which must open

Data quality asks whether the data is valid. Drift asks whether it still describes
the same world. A batch can pass every rule and be a different population.

Drift is raised only when a change is both unlikely to be chance and large enough
to matter. The second gate does the work, and the measured numbers say where. A
1.5% shift in median income against this reference window:

| rows | KS p-value | PSI | outcome |
|---|---|---|---|
| 1,200 | 0.59 | 0.006 | silent on both gates |
| 5,000 | 0.031 | 0.004 | significant, not material |
| 100,000 | 7.2e-10 | 0.001 | significant, not material |
| 500,000 | 3.7e-13 | 0.001 | significant, not material |

At a daily batch of 1,200 the significance gate suffices. From five thousand rows
up it stops protecting anything: the p-value falls through the floor while the
shift stays at a tenth of a PSI point, true and significant and of no consequence
to anybody.

## Benign scenarios are what make precision mean anything

Three of the twelve scenarios change without being wrong, and the correct
behaviour is silence:

- **minor_noise** · an ordinary day that differs from the mean
- **weekend** · Saturday, at 40% of weekday volume because it is Saturday
- **trivial_income_tick** · incomes up 1.5%, which no analyst would act on

Without them a detector scores perfect recall by alerting on everything, and the
precision figure that hides it is worth nothing.

The volume check is why the weekend passes. It compares against the same weekday
in the reference window rather than a flat average, so a quiet Saturday at 470
rows is silent while a lost partition at 480 rows is raised. Near-identical
counts, opposite verdicts. A pipeline that cries wolf every weekend gets muted,
and then it catches nothing at all.

The three false alarms are all that Saturday: in a 470-row batch the approval rate
carries a sampling error of about 2.3 points, and a 6.7-point deviation is a 2.9
sigma day. That is noise behaving as noise, not a bug, and it has not been tuned
away.

## What these numbers are, and are not

Injected faults on synthetic data, detected by rules the same author configured.
That is a real test of the wiring and a weak test of the world. The generator and
the detector were written to agree, so agreement is the floor of what they can
demonstrate, not evidence about real pipelines. The benign scenarios and the
false-alarm count are what keep it from being circular, and the sensitivity floors
are what make it informative. A reader should treat the floors as the finding and
the 99.7% F1 as scene-setting.

## Architecture

```
CSV or API  ->  ingestion  ->  PostgreSQL raw  ->  dbt staging and marts
                    |                                      |
                    |                      +---------------+---------------+
                    |                      v               v               v
                    |               quality checks    drift tests    model metrics
                    |                      +---------------+---------------+
                    |                                      v
                    +-------------> pipeline_runs      incidents  ->  API, dashboard
```

PostgreSQL holds the business data and the monitoring metadata side by side: an
incident is only meaningful next to the run that produced it. Loading is
idempotent on a payload hash, because a retried delivery is the most ordinary
cause of duplicate rows, and a failed load still leaves a run record, because
silence is the one outcome a monitoring platform must never produce about itself.

Checks and drift gates live in `config/*.yml`. Adding a rule never means editing
the pipeline.

## Running it

Needs PostgreSQL on localhost. `brew services start postgresql@17`, then
`createdb datasentinel`.

```bash
make install        # venv and dependencies
make init           # schema plus 90 days of healthy reference data
make demo           # a healthy run, then three different failures
make evaluate       # score detection across every scenario
make sensitivity    # find the smallest fault reliably caught
make test           # 91 tests
```

Individual runs:

```bash
.venv/bin/python -m datasentinel.cli scenarios
.venv/bin/python -m datasentinel.cli run --scenario income_shift --day 2026-10-12
```

Machine-readable results land in `results/evaluation.json` and
`results/sensitivity.json`.

## State of the work

Built and measured: the generator with twelve scenarios, ingestion with run
tracking and idempotency, nine check types driven by YAML, drift across numeric,
categorical and missingness dimensions with FDR control, the incident model with
severity escalation on business-critical columns, the evaluation harness, the
sensitivity sweep, and 91 tests including integration tests against a real
PostgreSQL.

Not built yet: the dbt staging and mart models are scaffolded but empty, so the
checks currently read the raw layer directly. The FastAPI service, the Streamlit
dashboard, the baseline approval classifier and its performance-decay monitoring,
and the Docker Compose setup are all still to come. Docker is not installed on the
machine this was developed on, so the compose file will be written untested and
labelled as such.

## Limitations worth stating before an interviewer finds them

- Synthetic data throughout. No claim here transfers to a real pipeline.
- The faults were injected by the author and the thresholds set by the author.
- A population shift under 15% in income is not reliably detected.
- The reference window is a fixed 90 days with no seasonality beyond day of week,
  so a genuine annual cycle would read as drift.
- Severity mapping is a judgement encoded as a lookup table, not a derived
  quantity. It is defensible, not objective.
- The health score the original plan called for is deliberately absent. A single
  number summarising dataset health invites exactly the overclaiming this project
  spends its effort avoiding.

## Licence

MIT
