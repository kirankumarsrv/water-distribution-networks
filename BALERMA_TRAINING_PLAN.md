# Comprehensive Training Plan: Balerma Network (5_Balerma.inp)

## Executive Summary
This document outlines a complete workflow for training leak detection and localization models on the **Balerma water distribution network**. The key point: **all extracted features are network-specific and cannot be generalized to other networks**. Each WDN has unique topology, pipe dimensions, pressures, and flow dynamics that shape the feature space.

---

## Phase 1: Network Analysis & Dataset Generation

### 1.1 Balerma Network Overview
- **File:** `EPANETINPUTFILESFOR7NEWORKS/5_Balerma.inp`
- **Expected characteristics:** Mid-size urban network with complex topology
- **Key parameters to inspect:**
  - Number of nodes and pipes
  - Pipe materials, diameters, lengths (affect friction and leak signatures)
  - Reservoir/tank configurations
  - Demand patterns and baseline pressures
  - Pump schedules (if any)

**Action:** Load the network and inspect its structure before running full dataset generation.

```bash
python << 'PY'
import wntr
wn = wntr.network.WaterNetworkModel('EPANETINPUTFILESFOR7NEWORKS/5_Balerma.inp')
print(f"Nodes: {len(wn.nodes)}, Pipes: {len([link for name, link in wn.links() if link.link_type == 'Pipe'])}")
print(f"Pumps: {len([link for name, link in wn.links() if link.link_type == 'Pump'])}")
print(f"Tanks: {len(wn.tanks)}, Junctions: {len(wn.junctions)}")
PY
```

### 1.2 Dataset Generation
Run the dataset generation script with appropriate sample count:

```bash
cd /workspaces/water-distribution-networks
python graph_dataset/dataset.py \
  --inp-file EPANETINPUTFILESFOR7NEWORKS/5_Balerma.inp \
  --samples 3600 \
  --save-dir DATASETS \
  --seed 42
```

**Parameters:**
- `--samples 3600`: Generate 3600 total samples (900 per scenario: normal, leak, burst, blockage)
- `--seed 42`: Ensure reproducibility
- **Output files (in DATASETS/):**
  - `X_classification.npy` — pressure/flow features for detection (shape: [n_samples, n_features])
  - `y_classification.npy` — fault labels: 0=Normal, 1=Leak, 2=Burst, 3=Blockage
  - `X_localization.npy` — pressure/flow features for localization (shape: [n_samples, n_features])
  - `y_localization.npy` — zone labels: 1 to n_pipes (which pipe is faulty)
  - `feature_names.json` — list of feature names (e.g., "Pipe_1_H_in", "Pipe_2_Q_leak")
  - `localization_feature_names.json` — feature names for localization
  - `zone_definitions.json` — mapping of pipe ID to zone ID
  - `baseline_pressure_reference.json` — baseline head values per pipe (normal scenario average)
  - `classification_B.pt` — PyTorch tensor version
  - `train.pt`, `val.pt`, `test.pt` — graph-based splits

**⚠️ CRITICAL: Network-Specific Features**
All features extracted depend on:
1. **Network topology** — which pipes are adjacent, how many sensors per pipe
2. **Pipe geometry** — diameter D and length L affect head loss and leak flow
3. **Baseline pressures** — normal operating head values are network-specific
4. **Demand patterns** — peak/off-peak flows shape pressure signatures
5. **Sensor locations** — implicit (one sensor per pipe); real-world networks have sparse sensors

**→ Models trained on Balerma will NOT work on other networks without retraining.**

### 1.3 Feature Extraction Details
**Detection features** (see `physics/FeatureExtractor.py`):
- Head difference (`H_in` – `H_out`) for each pipe
- Flow rate (`Q_EPANET`) for each pipe
- Head at midpoint (`Hm`)
- Friction factor (`f`)
- Leak flow (`Q_leak`) — **this will be removed in cleaning step**

**Localization features** (see `physics/LocalizationFeatureExtractor.py`):
- Pressure deviation from baseline: `(Hm - baseline[pipe])` for each pipe
- Leak share ratio: `(Q_leak / Q_EPANET)` — **also removed in cleaning**
- Flow perturbation indicators

**Number of features = number of pipes × 8 (approx.)**
For Balerma, expect ~500–2000 features depending on pipe count.

---

## Phase 2: Feature Cleaning (Remove Leak-Derived Features)

### 2.1 Why Clean?
In the dataset, `Q_leak` and leak-derived features are **oracle data** — they leak information about the fault directly. A model trained on `Q_leak` learns to detect the artificial leak parameter, not real pressure/flow anomalies. Real sensors don't measure `Q_leak`.

### 2.2 Cleaning Process
Run the two cleaning scripts:

```bash
# Clean detection features
python scripts/clean_leak_features.py

# Clean localization features
python scripts/clean_localization_leak_features.py
```

**Output files:**
- `DATASETS/X_classification_no_leak.npy` — detection features without leak terms
- `DATASETS/feature_names_no_leak.json` — cleaned feature list
- `DATASETS/X_localization_no_leak.npy` — localization features without leak terms
- `DATASETS/localization_feature_names_no_leak.json` — cleaned feature list

**Expected impact:**
- Feature count drops by ~10–20% (removes Q_leak and leak_share features)
- Model accuracy may drop by 5–15% (depends on network complexity)
- Model becomes more realistic (uses only pressure/flow, not leak oracle)

---

## Phase 3: Model Training

### 3.1 Leak Detection Training (Objective 2)

**Goal:** Train a classifier to identify if a fault exists and its type (Normal, Leak, Burst, or Blockage).

```bash
python models/train_leak_detection.py --use-cleaned
```

**What it does:**
1. Loads `X_classification_no_leak.npy` and `y_classification.npy`
2. Splits data: 70% train, 15% val, 15% test (stratified by class)
3. Performs GridSearchCV over RandomForest and ExtraTrees hyperparameters:
   - `n_estimators`: [100, 200, 300]
   - `max_depth`: [10, 20, None]
   - `min_samples_leaf`: [1, 2, 4]
   - `max_features`: ['sqrt', 'log2']
4. Selects best model by cross-validation accuracy
5. Saves:
   - `models/leak_detection_model_cleaned.pkl` — trained classifier
   - `models/leak_detection_metrics_cleaned.json` — performance report

**Expected output structure (metrics JSON):**
```json
{
  "selected_model": "RandomForest or ExtraTrees",
  "best_params": { "n_estimators": 200, "max_depth": 20, ... },
  "feature_importances": [
    { "feature_index": 42, "feature_name": "Pipe_5_H_in", "importance": 0.087 },
    ...
  ],
  "train": { "accuracy": 0.98, "classification_report": {...} },
  "val": { "accuracy": 0.92, "classification_report": {...} },
  "test": { "accuracy": 0.90, "classification_report": {...} }
}
```

**Success criteria:**
- Test accuracy ≥ 80% (depends on network clarity)
- Feature importances identify physically meaningful pipes (high-pressure or boundary regions)
- No overfitting (train acc – test acc < 10%)

### 3.2 Fault Localization Training (Objective 3)

**Goal:** Train a classifier to predict which zone (pipe) is faulty, given that a fault exists.

```bash
python models/train_localization.py --use-cleaned
```

**What it does:**
1. Loads `X_localization_no_leak.npy` and `y_localization.npy`
2. Labels: 0 (normal) + zones 1 to n_pipes for each pipe
3. Splits data: 70% train, 15% val, 15% test (stratified)
4. Trains RandomForest / ExtraTrees with GridSearch
5. Saves:
   - `models/stage2_zone_classifier_cleaned.pkl` — trained classifier
   - `models/localization_metrics_cleaned.json` — performance report
   - `models/baseline_pressure_model.json` — baseline pressures (recomputed)

**Expected output structure:**
```json
{
  "selected_model": "RandomForest or ExtraTrees",
  "best_params": {...},
  "feature_importances": [...],
  "train": { "accuracy": 0.75, "classification_report": {...} },
  "val": { "accuracy": 0.65, "classification_report": {...} },
  "test": { "accuracy": 0.62, "classification_report": {...} }
}
```

**Success criteria:**
- Test accuracy ≥ 50% (harder task than detection; n_pipes choices to pick from)
- Baseline pressures JSON aligns with normal operation (check min/max values)
- Top features are pressure deviations from zones near the fault

### 3.3 Training Tips

**For Balerma specifically:**
1. **Class imbalance:** If leak count << burst count, use `class_weight='balanced'` (already done in scripts).
2. **Feature normalization:** Pressure/flow features span different ranges; tree-based models handle this, but consider standardization if using SVM/NN later.
3. **Hyperparameter tuning:** GridSearch is slow for large feature sets; consider reducing `param_grid` if runtime exceeds 2 hours.
4. **Validation strategy:** StratifiedShuffleSplit preserves class ratios in train/val/test.

---

## Phase 4: Integration & Dashboard Testing

### 4.1 Update Dashboard Configuration
Edit `inference/dashboard.py` to use Balerma network:

```python
# Line ~25
INP_FILE = ROOT_DIR / "EPANETINPUTFILESFOR7NEWORKS" / "5_Balerma.inp"
```

### 4.2 Load Models in Dashboard
The dashboard's `RealTimeLeakDetector` will auto-load:
- `models/leak_detection_model_cleaned.pkl` (detection)
- `models/stage2_zone_classifier_cleaned.pkl` (localization)

Verify they load without errors:

```bash
python << 'PY'
from inference.real_time_detector import RealTimeLeakDetector
detector = RealTimeLeakDetector()
print("✓ Detection model loaded:", detector.classifier is not None)
print("✓ Localization model loaded:", detector.zone_classifier is not None)
PY
```

### 4.3 Test Full Workflow
1. **Start dashboard:**
   ```bash
   cd /workspaces/water-distribution-networks
   python inference/dashboard.py
   ```

2. **Simulate a leak on Balerma:**
   - Open http://localhost:8000
   - Go to "Detection & Localization" tab
   - Select a pipe (e.g., pipe "5")
   - Set Fault type = "Leak"
   - Click "Run Simulation"
   - Check prediction confidence and predicted zone

3. **Test Isolation (Objective 4):**
   - Click "Isolation (Obj 4)" tab
   - Enter faulty pipe ID
   - Click "Compute Isolation"
   - Inspect valve closure set

4. **Test Restoration (Objective 5):**
   - From isolation output, copy isolated pipes/nodes
   - Go to "Restoration (Obj 5)" tab
   - Paste isolated zones
   - Upload/paste customer map (JSON: `{node_id: customer_count}`)
   - Click "Compute Restoration"
   - Check restored customer count

---

## Phase 5: Evaluation & Documentation

### 5.1 Generate Evaluation Report
```bash
python inference/evaluate_saved_models.py
```

Produces JSON report with:
- Per-class precision, recall, F1
- Confusion matrix
- Feature importance rankings
- Comparison to baseline models

### 5.2 Document Results
Create a **results summary** (e.g., `BALERMA_RESULTS.md`) including:

1. **Dataset Stats**
   - Number of pipes, nodes
   - Feature count (before/after cleaning)
   - Class distribution (normal vs leak vs burst vs blockage)

2. **Model Performance**
   ```
   DETECTION (4-way classification):
   - Test Accuracy: XX%
   - Best Model: RandomForest / ExtraTrees
   - Top Feature: [feature name]
   
   LOCALIZATION (n_pipe-way classification):
   - Test Accuracy: XX%
   - Best Model: RandomForest / ExtraTrees
   - Top Feature: [feature name]
   ```

3. **Network Specificity Note**
   ```
   ⚠️ IMPORTANT:
   - These models are trained on Balerma topology and baseline pressures.
   - They will NOT generalize to other networks.
   - To apply to a new network, regenerate dataset and retrain models.
   ```

---

## Phase 6: Troubleshooting & Optimization

### 6.1 Common Issues

**Issue:** Dataset generation stuck on EPANET solver
- **Cause:** Network has convergence issues (extreme pressures, dead-end pipes)
- **Fix:** Reduce sample count to 600; check INP file for physically valid parameters

**Issue:** Low detection accuracy (<70%)
- **Cause:** Features lack discriminative power; network has poor transient dynamics
- **Fix:** Add more features (e.g., 2nd-order derivatives, peak pressure rise); increase sample diversity

**Issue:** Low localization accuracy (<40%)
- **Cause:** Leaks on similar pipes produce similar pressure signatures
- **Fix:** Use more samples (5000+); consider ensemble methods; add node connectivity features

### 6.2 Performance Optimization

If training is slow:
1. **Reduce feature dimensionality:** Use PCA (e.g., keep 95% variance)
2. **Use subset of data:** Train on 1000 samples first, validate, then scale
3. **Parallelize GridSearch:** Already enabled (`n_jobs=-1` in scripts)

---

## Phase 7: Checkpoints & Deliverables

### Checkpoint 1: Dataset Ready
- [ ] `DATASETS/X_classification_no_leak.npy` exists (shape: [n, m])
- [ ] `DATASETS/X_localization_no_leak.npy` exists
- [ ] Feature names JSON files created
- [ ] Baseline pressure JSON created

### Checkpoint 2: Models Trained
- [ ] `models/leak_detection_model_cleaned.pkl` exists
- [ ] `models/stage2_zone_classifier_cleaned.pkl` exists
- [ ] Metrics JSON files contain test accuracy ≥ 50%
- [ ] No NaN/Inf values in metrics

### Checkpoint 3: Integration Complete
- [ ] Dashboard loads models without errors
- [ ] Test simulation runs and produces valid predictions
- [ ] Isolation and restoration compute without crashing

### Checkpoint 4: Documentation Done
- [ ] Results summary written
- [ ] Feature importance analysis documented
- [ ] Network-specific warnings included
- [ ] Training commands recorded in a script for reproducibility

---

## Key Takeaways: Network Specificity

### Why Models Don't Generalize

| Aspect | Impact |
|--------|--------|
| **Topology** | Balerma's pipe layout determines which pressure changes propagate where. A leak on Pipe X will have different pressure signatures than the same leak on an equivalent pipe in another network. |
| **Pipe Properties** | Diameter (D) and length (L) affect Darcy-Weisbach friction and leak orifice flow. Balerma's pipes have specific D/L values; another network will have different ones. |
| **Baseline Pressures** | Normal operating heads are network-specific (depends on elevation, demand, pump schedule). Features use deviations from baseline; baseline changes → features change. |
| **Demand Patterns** | Peak/off-peak flows shape transient behavior. Balerma's demand profile is unique. |

### Generalization Strategy
To apply models to a **new network** (e.g., KLmod):
1. Run `graph_dataset/dataset.py` on the new INP file
2. Clean leak features
3. Retrain both models on the new dataset
4. Test on new network's validation set
5. Update INP_FILE in dashboard and reload

**Time estimate:** ~6–8 hours per new network (depending on size and availability of computing resources)

---

## Quick Reference: Commands

```bash
# Step 1: Generate dataset (2–4 hours)
python graph_dataset/dataset.py --inp-file EPANETINPUTFILESFOR7NEWORKS/5_Balerma.inp --samples 3600

# Step 2: Clean features (~1 min)
python scripts/clean_leak_features.py
python scripts/clean_localization_leak_features.py

# Step 3: Train detection (30–60 min)
python models/train_leak_detection.py --use-cleaned

# Step 4: Train localization (30–60 min)
python models/train_localization.py --use-cleaned

# Step 5: Evaluate
python inference/evaluate_saved_models.py

# Step 6: Test on dashboard
python inference/dashboard.py
# Then open http://localhost:8000
```

---

## Contact & Questions
If stuck, check:
- Solver logs in dataset generation (look for convergence failures)
- Feature names JSON for expected feature count
- Metrics JSON for NaN/Inf values
- Dashboard console for model loading errors

Good luck! 🚀
