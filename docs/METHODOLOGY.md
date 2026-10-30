# Methodology and scientific limitations

## Objective

Demonstrate a reproducible short-horizon event-detection workflow for thunderstorm-related events using timestamped atmospheric observations. The current repository is a software prototype; it does not claim operational forecast skill.

## Input variables

The pipeline uses temperature, relative humidity, surface pressure, wind speed, recent precipitation, CAPE and a lightning count. Real projects should document sensor provenance, units, spatial footprint, sampling cadence, missing-data policy and quality flags for every variable.

## Target definition

The required `storm_event` field is binary. It must represent a clearly defined event occurring within a stated future horizon after the observation timestamp. This demo generator creates synthetic labels only to exercise the code. For real evaluation, construct labels from independent verified event observations and avoid using future information in the predictors.

## Baselines and model

The baseline is a transparent four-condition rule based on CAPE, humidity, precipitation and lightning count. The ML model is a scikit-learn Pipeline containing median imputation, standardization and class-balanced Logistic Regression. A 25% chronological holdout is used to reduce leakage from random mixing of time-adjacent records.

## Metrics

Precision, recall, F1, false-alarm rate and missed-event rate are reported. Accuracy alone is insufficient for rare severe-weather events. Lead time requires an archived sequence of prediction issue times and independently observed event times; it is not estimated from a static labelled table in this version.

## Recommended real-data validation

1. Obtain permissioned, quality-controlled radar, satellite, lightning and surface-station observations.
2. Co-register sources in time and space; document interpolation and missingness.
3. Define event type, minimum intensity, spatial radius and forecast horizon before training.
4. Hold out later dates and, where possible, entire locations or storm systems.
5. Tune thresholds and calibrate probabilities on a validation set, not on the final test set.
6. Report confidence intervals, event prevalence, reliability diagrams and performance by region/season/lead time.
7. Have meteorological experts review false alarms, missed events and warning communication.

## Safety statement

This academic prototype is not a certified warning service. It must not be used for emergency, aviation, public-safety or operational decisions. Always follow official meteorological alerts.
