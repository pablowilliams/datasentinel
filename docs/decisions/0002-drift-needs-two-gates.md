# 0002. Drift is raised only when it is both unlikely and large

**Status:** accepted

## Context

On a batch of any size a statistical test will eventually find a difference. The
question a monitoring platform has to answer is not whether the data changed but
whether anyone should care.

## Decision

A drift finding is raised only when both gates open: the change is unlikely to be
chance (`alpha`), and it is large enough to matter (`min_effect`). Significance
comes from Kolmogorov-Smirnov, chi-squared or a two-proportion z-test; magnitude
comes from PSI, total variation distance or the absolute rate gap.

## Evidence

A 1.5% uniform shift in median income against the reference window:

| rows | KS p-value | PSI | outcome |
|---|---|---|---|
| 1,200 | 0.59 | 0.006 | silent on both gates |
| 5,000 | 0.031 | 0.004 | significant, not material |
| 100,000 | 7.2e-10 | 0.001 | significant, not material |
| 500,000 | 3.7e-13 | 0.001 | significant, not material |

At a daily batch of 1,200 the significance gate is enough on its own. From five
thousand rows upward it stops protecting anything: the p-value falls through the
floor while the shift stays at a tenth of a PSI point, true and significant and of
no consequence to anybody.

## Consequences

Findings that are significant but immaterial are recorded with that reason
attached rather than discarded, so the explanation says which gate stayed shut.
