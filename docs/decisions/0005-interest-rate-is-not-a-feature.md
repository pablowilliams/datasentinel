# 0005. interest_rate is excluded from the model

**Status:** accepted

## Context

`interest_rate` is in the warehouse, correlates strongly with the approval
decision, and would improve every offline metric.

It is priced after a loan is approved. At the moment the model would have to run,
the field does not exist.

## Decision

Exclude it, and enforce the exclusion in code rather than in a comment.
`model/features.py` raises if a post-decision column or the label reaches the
feature set, and two tests assert that it does.

## Why it is enforced rather than remembered

A model given this column scores beautifully and cannot be deployed, and the
failure appears only in production where the column is empty. It is the same shape
as training on a field whose value is determined by the outcome, and it is far
easier to make than to notice.

## Consequence

AUC is 0.772. That is not a weak score: the label is a Bernoulli draw from a known
logistic function, so the generator's own logit is the ceiling, and it scores
0.7727 over 30 held-out days against the classifier's 0.7724. A materially higher
number would mean a leak, and that comparison is a test rather than a remark.
