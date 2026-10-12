# 0006. Checks are declared in YAML, not stored in a table

**Status:** accepted

## Context

The original plan specified a `quality_checks` table holding check definitions,
with a JSONB configuration column.

## Decision

Checks live in `config/checks.yml` and are resolved through a registry in
`quality/checks.py`. The `quality_checks` table was never created.

## Why

Check definitions are code in every way that matters: they are reviewed, they are
versioned alongside the logic that interprets them, and a change to them should
show up in a diff. Putting them in a database makes them invisible to review and
puts the current behaviour of the pipeline outside the repository.

`quality_results` remains a table, because results are data.

## Cost

A user interface for editing checks would need a different arrangement. Nothing
here has a user interface, so the cost is not yet paid.
