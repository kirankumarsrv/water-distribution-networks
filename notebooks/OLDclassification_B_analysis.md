# Dataset B Analysis: `classification_B.pt`

## Overview

This file documents the analysis of the classification dataset saved in `DATASETS/classification_B.pt`.
The dataset is a tabular fault classification dataset for water distribution network scenarios.
The future modeling plan is limited to:
- Random Forest (RF)
- Isolation Forest (anomaly detection)

## Dataset Summary

- `X` shape: `2500 x 15`
- `y` shape: `2500`
- Unique class labels: `[0, 1, 2, 3]`
- Label distribution: `[636, 642, 608, 614]`

### Class counts

| Label | Count |
|------:|------:|
| 0 | 636 |
| 1 | 642 |
| 2 | 608 |
| 3 | 614 |

The dataset is well balanced across the four classes, which is ideal for a multiclass Random Forest classifier.

## Feature Names

The 15 classification features are:

1. `H_in_mean`
2. `H_in_std`
3. `H_in_min`
4. `H_in_max`
5. `H_out_mean`
6. `H_out_std`
7. `Hm_mean`
8. `Hm_std`
9. `Q_EPANET_mean`
10. `Q_EPANET_std`
11. `Q1_mean`
12. `Q1_std`
13. `Q2_mean`
14. `Q2_std`
15. `f_mean`

## Key Insights

### 1. Physically meaningful feature design

The dataset uses aggregated hydraulic statistics rather than raw sensor values. This is a strong design for tabular classification because it captures system behavior in a compact, discriminative form.

### 2. No direct leak target leakage

Importantly, `Q_leak` is not included in the feature set for classification. This means models must infer faults indirectly from pressure, flow, and friction behavior rather than relying on a direct leak signal.

### 3. Balanced classes

The class counts are close to equal, which supports stable RF training and reduces the need for heavy class rebalancing.

### 4. Good separation potential

The current features include pressure means, variances, min/max values, and friction averages. These are the types of statistics that can distinguish between:
- normal operation
- leakage
- burst
- blockage

### 5. Friction feature is informative

`f_mean` is especially relevant for blockage detection because blockages should raise effective friction. This is a good physical indicator to include.

## Example behavior observed

A few example rows show that:
- normal and leak scenarios have similar aggregate pressures and flows
- blockage scenarios can exhibit larger friction mean values
- Q1 and Q2 statistics can help distinguish flow imbalance effects caused by different faults

## Recommended modeling approach

### Random Forest

This dataset is well suited to a Random Forest baseline because:
- it is tabular and numeric
- feature dimensionality is moderate (15 features)
- class balance is strong
- there is physical meaning in the engineered features

### Isolation Forest

For anomaly detection, train an Isolation Forest using only the normal samples (`label == 0`) and then use the model to detect deviations.
This will help verify whether the normal hydraulic regime is separable from fault regimes.

## Next steps

1. Compute a feature correlation matrix to understand redundancy and target signal strength.
2. Train a Random Forest classifier and evaluate with a confusion matrix.
3. Train an Isolation Forest on normal samples and inspect anomaly scores for the four classes.
4. Compute feature importance from the Random Forest model.
5. Optionally add feature differences such as `H_in_mean - H_out_mean` or `Q1_mean - Q2_mean` if additional signal is needed.

## Notes

- The dataset is *not* a raw time-series dataset; it is a tabular aggregated statistics dataset.
- Future work should remain focused on RF and Isolation Forest per the current plan.
