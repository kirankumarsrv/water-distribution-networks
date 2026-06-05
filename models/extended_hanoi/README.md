# Extended Hanoi models and metrics

This folder contains the trained models, metrics, and notes for the Extended Hanoi network (`EPANETINPUTFILESFOR7NEWORKS/2_Extended Hanoi.inp`). These files represent the current practical model artifacts for the Extended Hanoi graph only.

## What is included

- `baseline_pressure_model.json` — baseline pressure lookup model used by localisation feature extraction
- `leak_detection_model.pkl` — original leak detection classifier
- `leak_detection_model_cleaned.pkl` — leak detector trained on cleaned features
- `leak_detection_model_no_leak.pkl` — final detector after removing leak-derived features
- `leak_detection_model_no_leak_quick.pkl` — faster quick retrain fallback
- `stage1_isolation_forest.pkl` — stage 1 anomaly detector
- `stage2_random_forest.pkl` — stage 2 fault classifier
- `stage2_zone_classifier.pkl` — original localisation classifier
- `stage2_zone_classifier_cleaned.pkl` — localisation classifier trained on cleaned feature set
- `stage2_zone_classifier_no_leak.pkl` — final localisation model after removing leak-derived features
- `stage2_zone_classifier_no_leak_quick.pkl` — faster quick retrain fallback localisation model
- `leak_detection_metrics.json` / `localization_metrics.json` — original metrics
- `leak_detection_metrics_cleaned.json` / `localization_metrics_cleaned.json` — cleaned feature metrics
- `leak_detection_metrics_no_leak.json` / `localization_metrics_no_leak.json` — final no-leak metrics
- `leak_detection_metrics_no_leak_quick.json` / `localization_metrics_no_leak_quick.json` — quick fallback metrics
- `evaluation_summary.json` — consolidated evaluation report

## Practical model notes

- The `*_no_leak` files are the practical Extended Hanoi model candidates after removing leak-derived features from the dataset.
- These trained models and dataset features are specific to the Extended Hanoi network. For other EPANET networks, a new training pipeline should be executed and new models built.
- The Extended Hanoi localisation model currently reports zero importance for all `gradient_*` features, so those features are strong candidates for removal in the next feature reduction step.
- `leak_detection_metrics_no_leak.json` is the current benchmark for the detection model after cleaning.
- `localization_metrics_no_leak.json` is the current benchmark for the localization model after cleaning.
