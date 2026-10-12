# 0007. The published dashboard is static

**Status:** accepted

## Context

The plan called for a Streamlit dashboard. The project needed somewhere a reader
could actually look at it.

## Decision

Publish a static page to GitHub Pages, reading a snapshot exported from
PostgreSQL by `scripts/build_site.py` and committed under `site/data`.

## Why

GitHub Pages serves files, not processes. A Streamlit app querying PostgreSQL
would need a host for both, which is a deployment problem rather than a
monitoring one.

The snapshot is produced by running the real pipeline over a week of scenarios, so
every number on the page came out of `src/`. Nothing is hand-written.

## Cost, stated on the page itself

It is a record of one run, not a live console, and it is only as current as the
last commit of `site/data`. The page says so in its first paragraph rather than
implying a service that is not there.
