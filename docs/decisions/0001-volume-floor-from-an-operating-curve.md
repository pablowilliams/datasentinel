# 0001. The volume floor comes from a measured curve, not a round number

**Status:** accepted

## Context

The volume check compares a batch's row count against the same weekday in the
reference window. It needs a floor. The first value was 0.70, chosen because it
looked reasonable.

Measured, 0.70 could not see a quarter of the data go missing: a 30% row loss was
caught on 58% of days and a 25% loss on 23%.

## Decision

Sweep the floor against both detection and false alarms, and read the answer off
the curve. Over 60 days and 240 quiet days:

| floor | 15% lost | 20% | 25% | 30% | 40% | false alarms |
|---|---|---|---|---|---|---|
| 0.70 | 0% | 5% | 23% | 58% | 100% | 0.0% |
| **0.80** | **30%** | **58%** | **85%** | **97%** | **100%** | **0.4%** |
| 0.85 | 58% | 83% | 97% | 100% | 100% | 3.8% |
| 0.90 | 83% | 97% | 100% | 100% | 100% | 12.9% |
| 0.95 | 95% | 100% | 100% | 100% | 100% | 27.9% |

0.80. It catches a 30% loss reliably for one false alarm in 240 quiet days. Past
0.85 the false alarms arrive faster than the detections, and an alert people
learn to ignore protects nothing.

## The part worth remembering

That curve was wrong the first time it was computed. The generator delivered an
identical row count every weekday, so every floor up to 0.95 reported zero false
alarms and the recommendation would have been 0.95. Real intake varies. The
generator now varies with it, standard deviation 8%, and the answer changed.

A threshold swept against a constant measures nothing.
