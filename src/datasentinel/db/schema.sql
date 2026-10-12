-- DataSentinel storage. Business data and the monitoring metadata about it live
-- side by side: an incident is only meaningful next to the run that produced it.

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id              UUID PRIMARY KEY,
    name                    TEXT NOT NULL UNIQUE,
    owner                   TEXT,
    source_type             TEXT NOT NULL,
    expected_frequency      TEXT NOT NULL,
    freshness_sla_minutes   INTEGER NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id          UUID PRIMARY KEY,
    dataset_id      UUID NOT NULL REFERENCES datasets(dataset_id),
    pipeline_name   TEXT NOT NULL,
    scenario        TEXT,
    source_uri      TEXT,
    source_hash     TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    status          TEXT NOT NULL CHECK (status IN ('running','success','failed','skipped')),
    rows_processed  INTEGER,
    error_message   TEXT
);

-- Idempotency. The same file loaded twice must not duplicate rows, so a
-- successful load of a given payload into a given dataset can exist only once.
CREATE UNIQUE INDEX IF NOT EXISTS pipeline_runs_dedupe
    ON pipeline_runs (dataset_id, source_hash)
    WHERE status = 'success' AND source_hash IS NOT NULL;

CREATE TABLE IF NOT EXISTS raw_loan_applications (
    row_id                  BIGSERIAL PRIMARY KEY,
    run_id                  UUID NOT NULL REFERENCES pipeline_runs(run_id),
    application_id          TEXT,
    application_date        DATE,
    customer_age            INTEGER,
    annual_income           NUMERIC,
    employment_status       TEXT,
    loan_amount             NUMERIC,
    loan_term_months        INTEGER,
    credit_score            NUMERIC,
    debt_to_income_ratio    NUMERIC,
    previous_defaults       INTEGER,
    approved                BOOLEAN,
    interest_rate           NUMERIC,
    ingested_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS raw_loans_run ON raw_loan_applications (run_id);
CREATE INDEX IF NOT EXISTS raw_loans_date ON raw_loan_applications (application_date);

CREATE TABLE IF NOT EXISTS quality_results (
    result_id       UUID PRIMARY KEY,
    run_id          UUID NOT NULL REFERENCES pipeline_runs(run_id),
    check_name      TEXT NOT NULL,
    check_type      TEXT NOT NULL,
    column_name     TEXT,
    passed          BOOLEAN NOT NULL,
    severity        TEXT NOT NULL,
    observed        JSONB,
    expected        JSONB,
    message         TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS quality_results_run ON quality_results (run_id);

CREATE TABLE IF NOT EXISTS drift_results (
    drift_id            UUID PRIMARY KEY,
    run_id              UUID NOT NULL REFERENCES pipeline_runs(run_id),
    column_name         TEXT NOT NULL,
    test_name           TEXT NOT NULL,
    statistic           NUMERIC,
    p_value             NUMERIC,
    effect_size         NUMERIC,
    threshold           NUMERIC,
    drift_detected      BOOLEAN NOT NULL,
    reference_rows      INTEGER,
    current_rows        INTEGER,
    explanation         TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS drift_results_run ON drift_results (run_id);

CREATE TABLE IF NOT EXISTS incidents (
    incident_id     UUID PRIMARY KEY,
    dataset_id      UUID NOT NULL REFERENCES datasets(dataset_id),
    run_id          UUID NOT NULL REFERENCES pipeline_runs(run_id),
    category        TEXT NOT NULL,
    severity        TEXT NOT NULL CHECK (severity IN ('critical','high','medium','low')),
    status          TEXT NOT NULL CHECK (status IN ('open','resolved')),
    title           TEXT NOT NULL,
    description     TEXT NOT NULL,
    column_name     TEXT,
    evidence        JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at     TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS incidents_dataset ON incidents (dataset_id, status);
CREATE INDEX IF NOT EXISTS incidents_run ON incidents (run_id);

CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id           UUID PRIMARY KEY,
    model_name          TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    evaluation_window   DATE NOT NULL,
    metric_name         TEXT NOT NULL,
    metric_value        NUMERIC NOT NULL,
    sample_count        INTEGER NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS model_metrics_window ON model_metrics (model_name, evaluation_window);
