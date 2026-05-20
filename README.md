# water-distribution-networks

Modular project structure for EPANET-based leak simulation, physical pipe modeling, dataset generation, and training for leak detection and fault localisation.

## Pipeline

1. EPANET hydraulic simulation
2. custom leak physics
3. feature extraction for leak detection and localisation
4. dataset generation and train/val/test split
5. model training and inference

## Folder structure

- `physics/` — physical leak and pipe hydraulics modelling
- `integration/` — EPANET integration and leak scenario generation
- `graph_dataset/` — dataset creation and split logic using `2_Extended Hanoi.inp`
- `DATASETS/` — generated numpy and PyTorch dataset files
- `models/` — training code and root model artifacts
- `models/extended_hanoi/` — trained Extended Hanoi production models, metrics, and model notes
- `inference/` — real-time detector class and CLI entrypoint
- `notebooks/` — analysis notebook stubs and performance reports

## Generated dataset structure

- `DATASETS/X_classification.npy` — full training features for Objective 2
- `DATASETS/y_classification.npy` — leak detection labels (0=normal, 1=leak, 2=burst, 3=blockage)
- `DATASETS/X_localization.npy` — localization features for Objective 3
- `DATASETS/y_localization.npy` — zone and normal labels for localisation
- `DATASETS/feature_names.json` — detection feature names
- `DATASETS/localization_feature_names.json` — localization feature names
- `DATASETS/zone_definitions.json` — pipe->zone mapping
- `DATASETS/baseline_pressure_reference.json` — baseline Hm lookup for residual features

## Training artifacts

- `models/stage1_isolation_forest.pkl` — anomaly detector for Objective 2
- `models/stage2_random_forest.pkl` — multi-class fault classifier
- `models/stage2_zone_classifier.pkl` — zone localisation classifier
- `models/baseline_pressure_model.json` — baseline pressure lookup for inference
- `models/leak_detection_metrics.json` — Objective 2 metrics
- `models/localization_metrics.json` — Objective 3 metrics
- `models/extended_hanoi/` — dedicated Extended Hanoi trained models, metrics, and notes

## Usage

Generate datasets from the extended Hanoi input file:

```bash
source .venv/bin/activate
python graph_dataset/dataset.py --inp-file EPANETINPUTFILESFOR7NEWORKS/2_Extended\ Hanoi.inp --samples 3600 --save-dir DATASETS
```

Train leak detection models:

```bash
source .venv/bin/activate
python models/train_leak_detection.py
```

Train localisation models:

```bash
source .venv/bin/activate
python models/train_localization.py
```

Run real-time inference on a sample JSON record:

```bash
source .venv/bin/activate
python inference/real_time_detector.py --sample sample.json
```

Convert the PyTorch classification dataset into NumPy files to avoid pickle loading issues:

```bash
source .venv/bin/activate
python models/convert_classification_pt_to_numpy.py --pt-path DATASETS/classification_B.pt --out-dir DATASETS
```

Example end-to-end sample generation and detection/localisation using the Extended Hanoi trained models:

```bash
source .venv/bin/activate
python inference/example_end_to_end.py --scenario leak --models-dir models/extended_hanoi
```
