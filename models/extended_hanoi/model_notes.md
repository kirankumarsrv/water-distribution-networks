# Extended Hanoi model notes

## Extended Hanoi specificity

- All models in this folder are trained on the Extended Hanoi graph only.
- The input network is `EPANETINPUTFILESFOR7NEWORKS/2_Extended Hanoi.inp`.
- The generated dataset, feature names, and trained weights are not guaranteed to transfer to another network without retraining.

## No-leak model summary

- `leak_detection_metrics_no_leak.json` contains the final detection model metrics after removing leak-derived features.
- `localization_metrics_no_leak.json` contains the final localisation model metrics after removing leak-derived features.

## Current findings

- The cleaned Extended Hanoi leak detector still reaches perfect accuracy on the current test split, but it is now based on realistic hydraulic and sensor-derived features rather than direct leak flow features.
- The localisation model now depends more heavily on residuals and global flow summaries.
- The `gradient_*` set of features is currently zero importance in the final localisation model. These features can be removed in the next feature-selection pass.

## Model variants

- `*_cleaned` artifacts: models trained after removing leak-derived features from the dataset.
- `*_no_leak` artifacts: final practical model artifacts after the full no-leak feature cleaning step.
- `*_no_leak_quick` artifacts: faster retraining fallback with fixed hyperparameters.
