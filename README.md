# water-distribution-networks

Modular project structure for EPANET-based leak simulation, physical pipe modeling, graph dataset generation, and train/val/test splitting.

## Pipeline

1. EPANET hydraulic simulation
2. custom leak physics
3. graph generation
4. graph ML dataset creation
5. train/validation/test split

## Folder structure

- `physics/` — physical leak and pipe hydraulics modelling
- `integration/` — EPANET integration and leak scenario generation
- `graph_dataset/` — graph conversion and dataset split logic
- `DATASETS/` — generated graph dataset files
- `notebooks/` — example notebook for inspecting the dataset
