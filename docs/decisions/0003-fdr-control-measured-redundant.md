# 0003. Benjamini-Hochberg is kept although it was measured to change nothing

**Status:** accepted

## Context

Ten columns are tested for drift on every run. At alpha 0.05 that is roughly a
one-in-two chance per run of some column looking significant for no reason, and
over a month of runs it approaches certainty. An uncorrected run did produce a
false alarm: a 470-row Saturday whose approval rate sat 6.7 points off a reference
built from 89,020 rows, at p = 0.0035.

## Decision

Control the false discovery rate across the columns tested in a run, using
Benjamini-Hochberg, then measure whether it helped.

## What the measurement said

Nothing. Over 90 days and four non-failure scenarios, 360 scenario-days:

| | false drift alarms |
|---|---|
| uncorrected | 1 |
| Benjamini-Hochberg | 1 |

The correction did not remove that Saturday, because p = 0.0035 clears the
rank-one threshold of 0.005. More generally the practical-magnitude gate from
decision 0002 was already rejecting everything the correction would have caught.

## Decision

Keep it. It costs nothing, it is the right thing to do for ten simultaneous
tests, and it protects the system if the effect gates are ever loosened. Document
it as redundant at current thresholds rather than advertising it as an
improvement, because claiming an unmeasured benefit is how a README stops being
trustworthy.
