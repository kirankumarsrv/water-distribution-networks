# Dataset C Analysis: `water_dataset.pt`

## Overview

This file documents the analysis of the localization graph dataset saved in `DATASETS/water_dataset.pt`.
The dataset is a graph-based localization dataset for water distribution network faults.
The future modeling plan is focused on:
- Random Forest for tabular classification (Dataset B)
- Isolation Forest for anomaly detection

## Dataset Summary

Based on the current dataset:

- Number of graph samples: `1250`
- Each graph has:
  - `34` nodes (one node per pipe)
  - `88` directed edges
  - `8` node features
- `y` is a single label per graph indicating the cracked/faulty pipe index
- There are `34` unique cracked-pipe labels, matching the number of pipes in the network

## Graph Dataset Structure

Each sample is a `torch_geometric.data.Data` object with:
- `x`: node feature matrix of shape `[num_pipes, 8]`
- `edge_index`: directed graph connectivity of shape `[2, num_edges]`
- `y`: scalar tensor containing the faulty pipe index

### Example values from one sample

- `x` first pipe features:
  - `[5.5389, 5.5389, 0.0, 98.5704, 0.01221, 5.5389, 100.0, 97.1408]`
- `y`: `[28]` meaning the cracked pipe is node index `28`
- `edge_index` connectivity begins with:
  - `(0, 1), (1, 0), (1, 2), (2, 1), ...`

## Node feature names

The 8 node features in `x` are:

1. `Q1`
2. `Q2`
3. `Q_leak`
4. `Hm`
5. `f`
6. `Q_EPANET`
7. `H_in`
8. `H_out`

## Interpretation

### 1. Graph structure

The graph encodes pipe adjacency through shared network nodes. Each pipe is represented as a graph node, and each edge connects two pipes that meet at a junction.

### 2. Localization objective

The label `y` is the index of the faulty pipe in the sample graph. This makes the dataset suitable for:
- node-level localization models
- graph-based learning for fault localization

### 3. Feature information

`Q_leak` is available here as a node feature, which is appropriate for a localization graph where the target is the faulty pipe position.
The model may use pipe-level hydraulics and leak behavior together with graph connectivity to localize faults.

### 4. Dataset consistency

The dataset appears regular:
- all graphs have 34 nodes
- all graphs have 88 edges
This consistency simplifies batching and model design.

## Suggested usage and future direction

### Localization modeling

This dataset is best suited for graph-based localization approaches, such as:
- Graph Neural Networks (GNNs)
- message-passing on the pipe adjacency graph
- node classification / node ranking to identify the faulty pipe

### Relation to Dataset B

Dataset C is complementary to Dataset B:
- Dataset B is tabular classification of the overall scenario
- Dataset C is graph localization of the faulty pipe

### Practical advice

1. Start with a simple node classification GNN baseline.
2. Use `y` as the target index and train the model to assign high fault probability to the correct node.
3. Keep in mind that all graphs share the same topology and node count, which simplifies model input preparation.

## Future analysis recommendations

- Visualize the pipe adjacency graph for one sample to confirm edge connectivity.
- Inspect the distribution of `Q_leak` values across normal and fault scenarios.
- Analyze whether the faulty pipe index distribution is balanced across the 34 pipes.
- Compare `H_in`/`H_out` patterns for cracked vs non-cracked pipes.

## Notes

- `water_dataset.pt` is not a standard sensor time-series dataset; it is a graph dataset where each pipe is a node and the label identifies the faulty pipe.
- This file is intended as a companion analysis to `classification_B_analysis.md`.
