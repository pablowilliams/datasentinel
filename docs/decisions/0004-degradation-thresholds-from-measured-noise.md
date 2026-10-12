# 0004. Model damage is defined by the model's own quiet days

**Status:** accepted

## Context

The drift-to-damage analysis needs a definition of damage. The first one was a
0.02 drop in AUC or a 0.02 rise in calibration error, picked by eye.

## What went wrong

Calibration error on untroubled days has a standard deviation of 0.0093. The
threshold therefore sat 1.7 standard deviations into ordinary variation and fired
on noise. It reported ten silent model failures out of 140 observations, every one
of them a quiet Saturday or an unremarkable Tuesday.

## Decision

Damage is a metric outside the 99% band of untroubled days, computed from at least
20 scored healthy days and refused outright below that. The window spans whole
weeks, so the band absorbs the fact that a 470-row Saturday measures calibration
less precisely than a 1,200-row Wednesday.

## Result

Silent failures fell from 10 in 140 to 5 in 196. Three of the five are benign days
clearing a 99th percentile, which is what a 99% band does by construction. The
other two are income shifts of 5% and 10%, below the 15% detection floor the
sensitivity sweep found independently.

Two analyses built for different purposes agree about where this platform stops
seeing, which is the strongest internal evidence in the repository.

## The pattern

This is the same mistake as decision 0001: a threshold chosen by eye rather than
from the distribution it is meant to separate. It was made twice, in two different
modules, weeks apart. Both are recorded rather than quietly fixed.
