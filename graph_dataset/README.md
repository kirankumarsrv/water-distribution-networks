# Graph Dataset Module

This module converts hydraulic simulation results into graph machine learning data.

## Key Ideas

- The dataset is a graph dataset, not tabular ML.
- Water distribution networks are naturally graphs.
- Each pipe becomes a graph node.
- Pipes are connected if they share a junction.

## Node Features

Each pipe/node contains the following feature vector:

- `Q1` (flow before the leak)
- `Q2` (flow after the leak)
- `Q_leak` (leak flow)
- `Hm` (head at leak point)
- `f` (friction factor)
- `Q_EPANET` (EPANET flow)
- `H_in` (inlet head)
- `H_out` (outlet head)

That makes 8 features per pipe.

## Labels

The label is the cracked pipe index, making this dataset suitable for leak localization.

## Output

The module saves:

- `DATASETS/water_dataset.pt`
- `DATASETS/train.pt`
- `DATASETS/val.pt`
- `DATASETS/test.pt`
