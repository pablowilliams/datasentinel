VENV := .venv/bin
export DATASENTINEL_DSN ?= postgresql+psycopg://localhost:5432/datasentinel

.PHONY: help install init demo evaluate sensitivity test lint typecheck all

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:  ## create the venv and install everything
	python3 -m venv .venv
	$(VENV)/pip install -q -r requirements.txt -r requirements-dev.txt -e .

init:  ## create the schema and seed a healthy reference window
	$(VENV)/python -m datasentinel.cli init

demo:  ## one healthy run, then each failure, then back to healthy
	$(VENV)/python -m datasentinel.cli run --scenario healthy --day 2026-10-10
	$(VENV)/python -m datasentinel.cli run --scenario missing_income --day 2026-10-11
	$(VENV)/python -m datasentinel.cli run --scenario income_shift --day 2026-10-12
	$(VENV)/python -m datasentinel.cli run --scenario volume_drop --day 2026-10-13

evaluate:  ## score detection across every scenario
	$(VENV)/python -m datasentinel.cli evaluate

sensitivity:  ## find the smallest fault reliably caught
	$(VENV)/python -m datasentinel.cli sensitivity

test:  ## run the suite
	$(VENV)/python -m pytest -q

lint:
	$(VENV)/ruff check src tests

typecheck:
	$(VENV)/mypy src

all: lint test evaluate sensitivity

train-model:  ## train the approval classifier on healthy history
	$(VENV)/python -m datasentinel.cli model train

monitor-model:  ## score recent batches and record model metrics
	$(VENV)/python -m datasentinel.cli model monitor

decay:  ## does detected drift correspond to model damage?
	$(VENV)/python -m datasentinel.cli model decay

site:  ## rebuild the published dashboard's data from a real pipeline run
	$(VENV)/python scripts/build_site.py

serve-site:  ## preview the dashboard locally
	$(VENV)/python -m http.server 8787 --directory site
