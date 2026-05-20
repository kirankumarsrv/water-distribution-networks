# COMPLETE IMPLEMENTATION PLAN
## Objectives 2 & 3: Leak Detection & Fault Localisation

**Document Purpose**: Master plan with all strategic decisions, sample/feature specifications, and step-by-step implementation roadmap.

---

## PART A: STRATEGIC DECISIONS & JUSTIFICATION

### A.1 DECISION 1: Number of Training Samples for Objective 2

| Decision Point | Choice | Reasoning | Risk Mitigation |
|---|---|---|---|
| **Total Samples to Generate** | **3,600 total** (800 baseline + 2,800 leak scenarios) | Random Forest requires 500+ samples (literature); GNN needs 1000+. At 200 scenarios × 4 fault classes × 2.5 samples/scenario = 2000 minimum. 3600 provides buffer for: (1) class imbalance, (2) corrupted simulations, (3) future model complexity. Ref: Joseph et al. 2024 used ~5000. | If dataset too small (<2000), re-run generator with stride=5 instead of stride=10 |
| **Samples per Fault Class** | **900 per class** (Normal: 900, Leak: 900, Burst: 900, Blockage: 900) | Balanced classes → RF can use `class_weight='balanced'` effectively. Imbalanced data (e.g., 70%Normal/10%Leak) causes high false negatives. | Monitor precision/recall per class; if one class <85%, generate 300 more samples for that class |
| **Time-Series Window Size** | **30 samples @ 1Hz** | ~30 seconds real-time window. Captures transient response without over-smoothing. Sliding windows with stride=10 → each scenario yields ~5-10 samples. | Use stride=5 for dense sampling if model underfits; stride=15 if overfitting on redundant windows |
| **Sliding Window Stride** | **Stride=10** (extract every 10 timesteps) | Balances overlap (10s frames) vs computational cost. Stride=5 → 2x samples (CPU expensive); stride=15 → sparser coverage. | Default to 10; adjust based on first-round validation metrics |

**DECISION MATRIX A.1 SUMMARY**:
```
Scenario Generation:
  - Normal operations: 200 scenarios × 1 = 200 baseline samples
  - Fault injection: 200 scenarios × 3 fault types × 1.4 realizations = 840 fault samples
  - Sliding window extraction: ~3.6× multiplier from windows
  - Total: 3,600 samples across 4 classes
  - Hardware cost: ~45 min generation on laptop, ~2 GB disk space
```

---

### A.2 DECISION 2: Feature Engineering Approach for Objective 2

| Feature Category | # Features | Feature Names | Justification |
|---|---|---|---|
| **Statistical (Per Sensor)** | 8 | Mean, Std, Min, Max, Range, Skewness, Kurtosis, IQR | Captures distribution shape; essential for RF feature importance ranking |
| **Temporal (Per Sensor)** | 4 | ΔP (1st diff), Δ²P (2nd diff), Max Rate-of-Change, Autocorr[lag=1] | Detects rapid pressure changes (leak signature) without GPU overhead |
| **Spatial (Sensor Pairs)** | 3 | Pressure gradient (∂P/∂x), Flow imbalance, Elevation-normalized head loss | Identifies spatial anomalies; critical for zone-based fault localization |
| **Frequency Domain (Optional)** | 2 | FFT peak frequency, FFT magnitude at leak band (10-50 Hz) | Leak vibrations have characteristic frequency; may improve RF by 2-3% |
| **Total per Network** | **~40-50** | 5 sensors × 8 features + 3 spatial + 2 frequency | Goldilocks zone: >30 enough for RF to learn patterns; <60 keeps inference fast |

**Feature Extraction Pipeline**:
```python
# Pseudocode
for each sample in dataset:
    for each sensor_node:
        stats = mean, std, min, max, range, skew, kurtosis, iqr
        temporal = delta_P, delta2_P, max_rate, autocorr
        spatial = pressure_gradient, flow_imbalance, normalized_head
        freq = fft_peak, leak_band_magnitude
        feature_vector.append(concat(stats, temporal, spatial, freq))
        
X_shape = (3600 samples, 45 features)  # Final matrix for RandomForest
```

**DECISION MATRIX A.2 SUMMARY**:
```
Final Feature Matrix:
  - Shape: (3600, 45)
  - Data type: float32 (4 MB uncompressed)
  - Normalization: MinMaxScaler [0, 1]
  - Feature selection: All 45 kept initially; top 15-20 kept post-permutation importance
```

---

### A.3 DECISION 3: Train/Validation/Test Split for Objective 2

| Split | Ratio | # Samples | Purpose | Sampling Strategy |
|---|---|---|---|---|
| **Training** | 70% | 2,520 | Learn RF hyperparameters for single-stage 4-class detection | Stratified by class; 630 per class |
| **Validation** | 15% | 540 | Hyperparameter tuning via GridSearchCV (5-fold CV on this set) | Stratified; 135 per class |
| **Test** | 15% | 540 | Final accuracy report (never seen during training) | Stratified; 135 per class |

**Stratified Split Rationale**: Ensures each fold has ~25% Leak, ~25% Burst, ~25% Blockage, ~25% Normal. Prevents the model from overfitting to one class.

**DECISION MATRIX A.3 SUMMARY**:
```
Split Implementation:
  - Use sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
  - Then split training into 70/30 for train/val
  - Expected baseline accuracy on test set: ~92% (Random Forest from literature)
```

---

### A.4 DECISION 4: Single-Stage Pipeline Architecture for Objective 2

```
┌─────────────────────────────────────────────────────────┐
│          REAL-TIME SENSOR DATA STREAM                    │
│        (Pressure, Flow from 5 nodes @ 1 Hz)              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ↓
            ┌─────────────────────────┐
            │   RANDOM FOREST         │
            │  (Supervised, 4-class)  │
            │  - Input: 45 features   │
            │  - Output: 0/1/2/3      │
            │  - Latency: <5 ms       │
            └────────────┬────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                  │
        ↓ Anomaly Detected                ↓ Normal
   ┌─────────────────────┐           [PASS]
   │  FAULT TYPE         │
   │  PREDICTION         │
   │  (Classification)   │
   │ - 4 classes         │
   │ - Latency: <5 ms    │
   └────────────┬────────┘
                │
        ┌───────┴───────┐
        ↓               ↓
   [LEAK/BURST/    [BLOCKAGE]
    BLOCKAGE?]
        │
        ↓
   Fault Type + Confidence
   Score (0-100%)
```

**Single-Stage Random Forest**:
- Train on **all 4 classes** (2,520 train samples)
- Multi-class classification: Normal vs Leak vs Burst vs Blockage
- Hyperparameters: `n_estimators=200, max_depth=None, class_weight='balanced'`
- GridSearchCV tunes: `n_estimators=[100, 200, 300], max_depth=[10, 20, None]`

**DECISION MATRIX A.4 SUMMARY**:
```
Pipeline Benefits:
  - Single-stage detection uses labels directly and avoids failed anomaly gating
  - Direct classification simplifies inference and reduces false normals
  - Expected accuracy: ~92% on the test set
  - Confidence threshold can still gate localization
```

---

### A.5 DECISION 5: Algorithm Selection for Objective 3 (Fault Localisation)

| Stage | Algorithm | Choice | Why Not Alternatives |
|---|---|---|---|
| **Stage 1 (Fast Candidate Zone)** | Pressure Residual Analysis | ✓ Chosen | CNN/GNN require GPU + hours to train. Residual = O(n) lookup in <1ms. Sufficient for zone-level (85% accuracy). |
| **Stage 2 (Zone Refinement)** | Random Forest Zone Classifier | ✓ Chosen | Inverse modeling = too complex for student scale. GNN = overkill & requires GPU. RF is interpretable & fast. |

**NOT choosing**:
- ❌ **Negative Pressure Wave (NPW)**: Requires 1 kHz sampling (infrastructure not present)
- ❌ **Bayesian Network**: Too complex; 5+ manual hyperparameters
- ❌ **GNN**: High complexity; requires GPU; training 2-3 hours
- ❌ **Inverse Hydraulic Model**: Requires numerical solvers + perfect calibration

**DECISION MATRIX A.5 SUMMARY**:
```
Objective 3 Approach:
  - Zones = pipe segments/DMAs defined by EPANET connectivity
  - Number of zones: 30-50 (depends on network; typically 1 zone per 3-5 pipes)
  - Expected accuracy: 82-90% zone identification
  - Real-time: Yes (< 50 ms per prediction)
```

---

### A.6 DECISION 6: Number of Features for Objective 3

| Feature Type | # Features | Per-Sensor? | Purpose |
|---|---|---|---|
| **Pressure Residuals** | 5 | Yes (all sensors) | Baseline error = P_measured - P_baseline; captures hydraulic deviation |
| **Pressure Gradients** | 4 | Pairs of sensors | ∂P between adjacent nodes; identifies pressure drop direction |
| **Flow Imbalance** | 2 | Per zone | Inflow - outflow; detects zone-level blockage/leak |
| **Rate-of-Change** | 3 | All sensors | dP/dt at fault onset; distinguishes sudden vs gradual faults |
| **Total** | **~20-25 features** | — | Small feature set = fast inference + interpretability |

**Justification**:
- Objective 2 (detection) needs 45 features to discriminate 4 classes.
- Objective 3 (localization) needs only 20-25 features to pick 1 zone from 30-50.
- Simpler problem → fewer features needed.

**DECISION MATRIX A.6 SUMMARY**:
```
Feature Matrix for Objective 3:
  - Shape: (N_samples, 25)
  - Labels: Zone IDs (0 to N_zones-1)
  - Multi-class RF: n_estimators=100, max_depth=15
  - Inference latency: ~5 ms per sample
```

---

### A.7 DECISION 7: Dataset Generation Parameters (Summary Table)

| Parameter | Value | Rationale | Impact |
|---|---|---|---|
| **EPANET Network** | 2_Extended Hanoi.inp | Balanced network (13 pipes); good test case; similar to literature | Generalize to other networks in Phase 2 |
| **Simulation Duration** | 24 hours (86,400 steps @ 1 Hz) | Captures diurnal demand pattern (morning/evening peaks) | Realistic leak signatures across demand variability |
| **Demand Variability** | ±15% random noise on EPANET pattern | Real systems have ~10-20% noise; prevents overfitting to clean data | Robustness to sensor noise |
| **Leak Types** | Abrupt (t=12h), Gradual (leak grows 1%/h) | Joseph et al. found gradual leaks hardest to detect | Test detection limits |
| **Fault Injection Pipe** | Random (1 per scenario) | Prevents model from learning a specific "leak pipe" | Generalize to any pipe |
| **Sensor Count** | 5-7 pressure + 2 flow meters | Practical: typical WDN has 1 sensor per 5-10 km pipe | Scalable to real networks |

**DECISION MATRIX A.7 SUMMARY**:
```
Dataset Generation Workflow:
  1. For i = 1 to 200:
       - Scenario i: EPANET simulation 24h, demand pattern + noise
       - Extract normal-class samples (time: 0-11h, no leak)
       - For fault_type in [Leak, Burst, Blockage]:
           - Inject fault at pipe i at t=12h
           - Simulate 12h post-fault response
           - Extract anomaly samples (time: 12-24h)
  2. Apply sliding windows (window=30, stride=10)
  3. Extract features (45 dimensions)
  4. Split train/val/test (70/15/15)
  5. Normalize MinMaxScaler
  6. Save as pickle/HDF5
  
  Total time: ~45 min on laptop (could parallelize via multiprocessing)
  Disk space: ~2 GB
```

---

## PART B: IMPLEMENTATION ROADMAP

### B.1 PHASE 1: DATA GENERATION (Weeks 1-2)

#### Task 1.1: Extend Graph Dataset Generator
**Input**: 7 EPANET .inp files  
**Output**: 3,600-sample dataset with 45 features  
**Steps**:

```
1a. Modify graph_dataset/dataset.py:
    - Add feature extraction function (45 features per sample)
    - Implement sliding window extraction (window=30, stride=10)
    - Add class labels: 0=Normal, 1=Leak, 2=Burst, 3=Blockage
    
1b. Run generation script:
    python graph_dataset/dataset.py \
        --inp-file EPANETINPUTFILESFOR7NEWORKS/2_Extended\ Hanoi.inp \
        --samples 3600 \
        --window-size 30 \
        --window-stride 10 \
        --output DATASETS/leak_detection_dataset.pkl
    
    Expected output:
    - DATASETS/leak_detection_dataset.pkl (X: 3600×45, y: 3600)
    - DATASETS/feature_names.json (for interpretability)
    - Execution time: ~45 min (CPU bound)
    
1c. Verify output:
    - Check class balance: Counter(y) should be ~[900, 900, 900, 900]
    - Check feature ranges: MinMaxScaler should output [0, 1]
    - Check for NaN/Inf: assert np.isfinite(X).all()
```

**Decision Points**:
- If generation fails → check EPANET license (free version OK)
- If class imbalance >10% → add 200 more samples per minority class
- If feature NaN rate >5% → increase sensor count or simulation time

#### Task 1.2: Create Feature Engineering Pipeline
**Input**: Raw EPANET time-series (pressure, flow @ 1 Hz)  
**Output**: 45-dimensional feature vectors  
**Steps**:

```
1d. Create physics/FeatureExtractor.py:
    
    class FeatureExtractor:
        def extract_statistical(self, sensor_window):
            # mean, std, min, max, range, skew, kurt, iqr
            return [mean, std, min, max, range, skew, kurt, iqr]  # 8 features
            
        def extract_temporal(self, sensor_window):
            # delta_P, delta2_P, max_rate, autocorr
            return [delta_P, delta2_P, max_rate, autocorr]  # 4 features
            
        def extract_spatial(self, sensor_pairs):
            # grad_P, flow_imbalance, norm_head_loss
            return [grad_P, flow_imb, norm_head]  # 3 features
            
        def extract_frequency(self, sensor_window):
            # FFT peak freq, leak band magnitude
            return [fft_peak, leak_band]  # 2 features
            
        def __call__(self, df):
            # Input: df with 5 sensors, 30 timesteps
            # Output: 45-dim feature vector
            pass
    
1e. Test on 10 samples:
    python -c "
    from physics.FeatureExtractor import FeatureExtractor
    fe = FeatureExtractor()
    X_test = fe(df_test)
    assert X_test.shape == (10, 45)
    print('Feature extraction OK')
    "
```

**Decision Points**:
- If FFT computation slow → use scipy.fft.rfft (faster)
- If feature correlation >0.95 → remove redundant features
- If features have outliers → apply robust scaling (RobustScaler)

#### Task 1.3: Train/Val/Test Split
**Input**: Full 3,600-sample dataset  
**Output**: 3 CSV files (train: 2,520, val: 540, test: 540)  
**Steps**:

```
1f. Create graph_dataset/split_dataset.py:
    
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import MinMaxScaler
    
    X, y = load_dataset('DATASETS/leak_detection_dataset.pkl')
    
    # Stratified split: maintains class ratio in each fold
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Further split training into train/val (70/30)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.176, random_state=42)
    train_idx2, val_idx2 = next(sss2.split(X_train, y_train))
    
    X_train_final = X_train[train_idx2]  # 2,520
    X_val = X_train[val_idx2]             # 540
    y_train_final = y_train[train_idx2]
    y_val = y_train[val_idx2]
    
    # Normalize (fit on train only, apply to val/test)
    scaler = MinMaxScaler()
    X_train_final = scaler.fit_transform(X_train_final)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Save
    np.save('DATASETS/X_train.npy', X_train_final)
    np.save('DATASETS/X_val.npy', X_val)
    np.save('DATASETS/X_test.npy', X_test)
    np.save('DATASETS/y_train.npy', y_train_final)
    np.save('DATASETS/y_val.npy', y_val)
    np.save('DATASETS/y_test.npy', y_test)
    
1g. Verify splits:
    python -c "
    X_train = np.load('DATASETS/X_train.npy')
    y_train = np.load('DATASETS/y_train.npy')
    print('Train:', X_train.shape, Counter(y_train))
    print('Class balance:', np.bincount(y_train) / len(y_train))
    "
    
    Expected:
      Train: (2520, 45) Counter({0: 630, 1: 630, 2: 630, 3: 630})
      Val:   (540, 45)  Counter({0: 135, 1: 135, 2: 135, 3: 135})
      Test:  (540, 45)  Counter({0: 135, 1: 135, 2: 135, 3: 135})
```

**Decision Points**:
- If class imbalance >5% → increase EPANET scenario count
- If features have outliers after scaling → clip to [0, 1]
- If val/test accuracy <80% → check for data leakage

---

### B.2 PHASE 2: OBJECTIVE 2 IMPLEMENTATION (Weeks 3-5)

#### Task 2.1: Single-Stage Random Forest Detection
**Input**: 2,520 multi-class training samples  
**Output**: Trained Random Forest multi-class classifier  
**Steps**:

```
2a. Create models/train_leak_detection.py:
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib
    
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Train on 4-class labels: 0=Normal, 1=Leak, 2=Burst, 3=Blockage
    classifier.fit(X_train, y_train)
    
    joblib.dump(classifier, 'models/leak_detection_model.pkl')
    
2b. Evaluate detection on validation set:
    
    X_val = np.load('DATASETS/X_val.npy')
    y_val = np.load('DATASETS/y_val.npy')
    
    y_pred = classifier.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_val, y_pred)
    
    print(f"Validation accuracy: {report['accuracy']:.3f}")
    print(f"Confusion matrix:\n{cm}")
    
    Expected: accuracy >92%, recall per class >90%
```

**Decision Points**:
- If overall accuracy <90% → tune `n_estimators`, `max_depth`, and class weights
- If one class recall <85% → generate more samples for that class or add feature engineering
- If the model overfits → reduce tree depth and add regularization

#### Task 2.2: Stage 2 — Random Forest Classifier (Supervised)
**Input**: 2,520 labeled training samples (all 4 classes)  
**Output**: Trained RF classifier; test accuracy report  
**Steps**:

```
2d. Create models/stage2_random_forest.py:
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report, confusion_matrix
    
    class Stage2Classifier:
        def __init__(self):
            self.model = None
            self.best_params = None
        
        def fit(self, X_train, y_train, X_val, y_val):
            # GridSearchCV for hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'class_weight': ['balanced', None]
            }
            
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.best_params = grid_search.best_params_
            self.model = grid_search.best_estimator_
            
            # Evaluate on validation set
            y_pred_val = self.model.predict(X_val)
            y_pred_proba = self.model.predict_proba(X_val)
            
            print(f"\nBest hyperparameters: {self.best_params}")
            print(f"Validation F1-score: {grid_search.best_score_:.3f}")
            print(f"\nValidation Classification Report:")
            print(classification_report(
                y_val, y_pred_val,
                target_names=['Normal', 'Leak', 'Burst', 'Blockage']
            ))
            
            return self
        
        def predict(self, X):
            return self.model.predict(X)
        
        def predict_proba(self, X):
            return self.model.predict_proba(X)
        
        def feature_importance(self):
            return self.model.feature_importances_
    
2e. Train Stage 2:
    
    X_train = np.load('DATASETS/X_train.npy')
    y_train = np.load('DATASETS/y_train.npy')
    X_val = np.load('DATASETS/X_val.npy')
    y_val = np.load('DATASETS/y_val.npy')
    
    classifier = Stage2Classifier()
    classifier.fit(X_train, y_train, X_val, y_val)
    
    with open('models/stage2_random_forest.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    
2f. Evaluate on Test Set (FINAL ACCURACY):
    
    X_test = np.load('DATASETS/X_test.npy')
    y_test = np.load('DATASETS/y_test.npy')
    
    classifier = pickle.load(open('models/stage2_random_forest.pkl', 'rb'))
    y_pred_test = classifier.predict(X_test)
    y_pred_proba_test = classifier.predict_proba(X_test)
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(y_test, y_pred_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average='weighted'
    )
    
    print(f"OBJECTIVE 2 TEST RESULTS:")
    print(f"  Overall Accuracy: {accuracy:.3f}")
    print(f"  Weighted Precision: {precision:.3f}")
    print(f"  Weighted Recall: {recall:.3f}")
    print(f"  Weighted F1: {f1:.3f}")
    print(f"\n{classification_report(y_test, y_pred_test, target_names=['Normal', 'Leak', 'Burst', 'Blockage'])}")
    
    Expected: >92% overall accuracy (literature baseline)
```

**Decision Points**:
- If test accuracy <85% → more samples needed; re-generate with 5,000 samples
- If F1 score imbalanced (e.g., Leak=0.80, Burst=0.95) → adjust class_weight
- If overfitting (train=98%, test=85%) → reduce max_depth or increase min_samples_leaf

#### Task 2.3: Feature Importance Analysis
**Input**: Trained RF model  
**Output**: Top 15 features for interpretability  
**Steps**:

```
2g. Feature Importance Ranking:
    
    classifier = pickle.load(open('models/stage2_random_forest.pkl', 'rb'))
    importances = classifier.feature_importance()
    
    feature_names = [
        'Q1_mean', 'Q1_std', 'Q1_min', 'Q1_max', 'Q1_range', 'Q1_skew', 'Q1_kurt', 'Q1_iqr',
        'Q1_delta', 'Q1_delta2', 'Q1_max_rate', 'Q1_autocorr',
        'Q2_mean', 'Q2_std', ... (repeat for Q2, Q_leak, Hm, f, Q_EPANET, H_in, H_out)
        'grad_P_Q1_Q2', 'grad_P_Q1_H_in', 'grad_P_H_in_H_out',
        'flow_imb_zone1', 'flow_imb_zone2',
        'fft_peak_freq', 'fft_leak_band'
    ]
    
    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]
    
    print("Top 15 Most Important Features:")
    for i, idx in enumerate(sorted_idx[:15]):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Save for next phases
    np.save('models/feature_importance.npy', importances)
    json.dump(feature_names, open('models/feature_names.json', 'w'))
```

**Decision Points**:
- If top feature importance <0.15 → features too weak; increase diversity in data generation
- If 80% importance from 5 features → model may overfit; use only top 15 features in production

---

### B.3 PHASE 3: OBJECTIVE 3 IMPLEMENTATION (Weeks 6-7)

#### Task 3.1: Build Baseline Pressure Model (Stage 1)
**Input**: 24h normal-operation EPANET simulation  
**Output**: Lookup table: P_baseline[node][hour]  
**Steps**:

```
3a. Create models/baseline_pressure_model.py:
    
    class BaselinePressureModel:
        def __init__(self, epanet_network):
            self.epanet = epanet_network
            self.baseline = {}  # node → (24 values for hourly pressure)
        
        def build_baseline(self, simulation_file):
            # Run EPANET for 24h under normal demand
            # Extract pressure at each node every hour
            
            from integration.EPANET_Integration import EPANETIntegrator
            
            integrator = EPANETIntegrator(simulation_file)
            pressures = integrator.run_simulation(24 * 3600)  # 24 hours
            
            # Resample to hourly
            for node_id in integrator.node_list:
                hourly_pressures = []
                for hour in range(24):
                    t_start = hour * 3600
                    t_end = (hour + 1) * 3600
                    p_mean = np.mean(pressures[node_id][t_start:t_end])
                    hourly_pressures.append(p_mean)
                
                self.baseline[node_id] = np.array(hourly_pressures)
        
        def save(self, filepath):
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self.baseline, f)
        
        def load(self, filepath):
            import pickle
            with open(filepath, 'rb') as f:
                self.baseline = pickle.load(f)
    
3b. Build baseline:
    
    model = BaselinePressureModel('EPANETINPUTFILESFOR7NEWORKS/2_Extended Hanoi.inp')
    model.build_baseline('EPANETINPUTFILESFOR7NEWORKS/2_Extended Hanoi.inp')
    model.save('models/baseline_pressure_model.pkl')
    
    print(f"Baseline built: {len(model.baseline)} nodes")
```

**Decision Points**:
- If baseline pressure very noisy → apply 1D median filter (window=5)
- If hourly averaging too coarse → use 30-min or 10-min intervals

#### Task 3.2: Feature Engineering for Localisation (Stage 1 Residuals)
**Input**: Real-time sensor data + baseline model  
**Output**: Residual feature vector (25 dimensions)  
**Steps**:

```
3c. Create physics/LocalizationFeatureExtractor.py:
    
    class LocalizationFeatureExtractor:
        def __init__(self, baseline_model):
            self.baseline = baseline_model.baseline  # P_baseline[node][hour]
        
        def compute_residuals(self, current_pressures, current_hour):
            # residual[i] = P_measured[i] - P_baseline[i][hour]
            residuals = {}
            for node_id, p_measured in current_pressures.items():
                p_baseline = self.baseline[node_id][current_hour % 24]
                residuals[node_id] = p_measured - p_baseline
            return residuals
        
        def compute_pressure_gradients(self, current_pressures, sensor_pairs):
            # grad_P[i,j] = (P[i] - P[j]) / distance[i,j]
            gradients = []
            for (node_i, node_j), distance in sensor_pairs:
                p_i = current_pressures[node_i]
                p_j = current_pressures[node_j]
                grad = (p_i - p_j) / (distance + 1e-6)  # avoid division by zero
                gradients.append(grad)
            return gradients
        
        def compute_flow_imbalance(self, current_flows, zone_definitions):
            # imbalance[z] = Q_inflow[z] - Q_outflow[z]
            imbalances = []
            for zone_id, pipe_list in zone_definitions.items():
                q_in = sum(current_flows[p] for p in pipe_list if current_flows[p] > 0)
                q_out = sum(-current_flows[p] for p in pipe_list if current_flows[p] < 0)
                imbalance = q_in - q_out
                imbalances.append(imbalance)
            return imbalances
        
        def __call__(self, current_pressures, current_flows, current_hour):
            # Extract all 25 features
            residuals = self.compute_residuals(current_pressures, current_hour)
            gradients = self.compute_pressure_gradients(current_pressures, sensor_pairs)
            imbalances = self.compute_flow_imbalance(current_flows, zone_definitions)
            
            feature_vector = np.concatenate([
                list(residuals.values()),      # 5 features
                gradients,                     # 4 features
                imbalances,                    # 2 features
                [rate_of_change_P],           # 1 feature
                [rate_of_change_Q]            # 1 feature
                # ... additional features to reach 25
            ])
            return feature_vector
```

**Decision Points**:
- If gradients dominated by noise → apply low-pass filter to pressures first
- If zones undefined → use EPANET connectivity to auto-define zones

#### Task 3.3: Zone Labels & Training Data for Stage 2
**Input**: 3,600 labelled samples from Objective 2 dataset + zone definitions  
**Output**: X_localization (3600 × 25), y_localization (zone IDs)  
**Steps**:

```
3d. Create localization training dataset:
    
    def create_localization_dataset(
        leak_detection_dataset_file,
        baseline_model_file,
        zone_definitions_file
    ):
        # Load data
        leak_data = pickle.load(open(leak_detection_dataset_file, 'rb'))
        baseline_model = pickle.load(open(baseline_model_file, 'rb'))
        zone_def = json.load(open(zone_definitions_file))
        
        feature_extractor = LocalizationFeatureExtractor(baseline_model)
        
        X_localization = []
        y_localization = []
        
        for sample_idx, (pressures, flows, cracked_pipe_id) in enumerate(leak_data):
            # Determine which zone contains cracked_pipe_id
            zone_id = zone_def['pipe_to_zone'][cracked_pipe_id]
            
            # Extract localization features
            features = feature_extractor(pressures, flows, hour=12)  # At fault time
            
            X_localization.append(features)
            y_localization.append(zone_id)
        
        X_localization = np.array(X_localization)  # Shape: (3600, 25)
        y_localization = np.array(y_localization)  # Shape: (3600,) with zone IDs
        
        np.save('DATASETS/X_localization.npy', X_localization)
        np.save('DATASETS/y_localization.npy', y_localization)
        
        print(f"Localization dataset: X {X_localization.shape}, y unique zones = {len(np.unique(y_localization))}")
    
3e. Run:
    
    create_localization_dataset(
        'DATASETS/leak_detection_dataset.pkl',
        'models/baseline_pressure_model.pkl',
        'models/zone_definitions.json'
    )
```

**Decision Points**:
- If zones too fine-grained (>100 zones) → merge adjacent zones
- If zones too coarse (<5 zones) → subdivide large zones
- If class imbalance >20% → oversample minority zones

#### Task 3.4: Train Random Forest Zone Classifier (Stage 2)
**Input**: X_localization (3600 × 25), y_localization  
**Output**: Trained zone classifier + test accuracy  
**Steps**:

```
3f. Create models/stage2_zone_classifier.py:
    
    class Stage2ZoneClassifier:
        def __init__(self, n_zones):
            self.n_zones = n_zones
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        
        def fit(self, X, y):
            self.model.fit(X, y)
        
        def predict(self, X):
            return self.model.predict(X)
        
        def predict_proba(self, X):
            return self.model.predict_proba(X)
    
3g. Train & evaluate:
    
    X_localization = np.load('DATASETS/X_localization.npy')
    y_localization = np.load('DATASETS/y_localization.npy')
    
    # Train/val/test split (70/15/15)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(sss.split(X_localization, y_localization))
    
    X_train_loc, X_test_loc = X_localization[train_idx], X_localization[test_idx]
    y_train_loc, y_test_loc = y_localization[train_idx], y_localization[test_idx]
    
    n_zones = len(np.unique(y_localization))
    classifier = Stage2ZoneClassifier(n_zones)
    classifier.fit(X_train_loc, y_train_loc)
    
    # Test
    y_pred = classifier.predict(X_test_loc)
    y_pred_proba = classifier.predict_proba(X_test_loc)
    
    accuracy = accuracy_score(y_test_loc, y_pred)
    
    print(f"OBJECTIVE 3 TEST RESULTS:")
    print(f"  Zone Prediction Accuracy: {accuracy:.3f}")
    print(f"  Number of zones: {n_zones}")
    print(f"  Per-zone accuracy:")
    for zone_id in np.unique(y_test_loc):
        mask = y_test_loc == zone_id
        zone_acc = accuracy_score(y_test_loc[mask], y_pred[mask])
        print(f"    Zone {zone_id}: {zone_acc:.3f} ({mask.sum()} samples)")
    
    with open('models/stage2_zone_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    
    Expected: >85% overall accuracy, >80% per-zone accuracy
```

**Decision Points**:
- If accuracy <80% → try n_estimators=200 or max_depth=20
- If one zone <70% accuracy → inspect training samples for that zone; may need more data

---

### B.4 PHASE 4: INTEGRATION & REAL-TIME INFERENCE (Weeks 8-9)

#### Task 4.1: Real-Time Leak Detection Pipeline
**Input**: Streaming sensor data (pressure, flow @ 1 Hz)  
**Output**: Leak alert with confidence + zone location  
**Steps**:

```
4a. Create inference/real_time_detector.py:
    
    import pickle
    import queue
    import threading
    
    class RealTimeLeakDetector:
        def __init__(
            self,
            stage1_model_path,
            stage2_model_path,
            baseline_model_path,
            feature_extractor,
            window_size=30,
            window_stride=10
        ):
            self.stage1 = pickle.load(open(stage1_model_path, 'rb'))
            self.stage2 = pickle.load(open(stage2_model_path, 'rb'))
            self.baseline = pickle.load(open(baseline_model_path, 'rb'))
            self.feature_extractor = feature_extractor
            self.window_size = window_size
            self.window_stride = window_stride
            
            self.sensor_buffer = queue.deque(maxlen=window_size)
            self.alert_history = []
        
        def on_sensor_data(self, pressure_dict, flow_dict, timestamp):
            # Called whenever new sensor reading arrives
            self.sensor_buffer.append({
                'pressures': pressure_dict,
                'flows': flow_dict,
                'timestamp': timestamp
            })
            
            if len(self.sensor_buffer) < self.window_size:
                return None  # Not enough data yet
            
            # Extract features from current window
            features = self.feature_extractor.extract_from_buffer(self.sensor_buffer)
            
            # Stage 1: Anomaly detection
            anomaly_prob = self.stage1.predict_proba([features])[0][1]
            
            if anomaly_prob < 0.5:
                return None  # Normal operation
            
            # Stage 2: Fault classification + localization
            fault_type = self.stage2.predict([features])[0]
            fault_confidence = np.max(self.stage2.predict_proba([features])[0])
            
            zone_id = self.zone_classifier.predict([features])[0]
            zone_confidence = np.max(self.zone_classifier.predict_proba([features])[0])
            
            alert = {
                'timestamp': timestamp,
                'anomaly_prob': anomaly_prob,
                'fault_type': ['Normal', 'Leak', 'Burst', 'Blockage'][fault_type],
                'fault_confidence': fault_confidence,
                'zone_id': zone_id,
                'zone_confidence': zone_confidence,
                'top_3_zones': self.zone_classifier.predict_proba([features])[0].argsort()[-3:][::-1]
            }
            
            self.alert_history.append(alert)
            return alert
        
        def latency(self):
            # Report inference latency
            return f"Stage1: <1ms, Stage2: <5ms, Total: <10ms"
    
4b. Usage example:
    
    detector = RealTimeLeakDetector(
        'models/stage1_isolation_forest.pkl',
        'models/stage2_random_forest.pkl',
        'models/baseline_pressure_model.pkl',
        feature_extractor
    )
    
    # Simulate sensor stream
    for t in range(1000):  # 1000 seconds = 16.7 minutes
        pressure = get_sensor_pressure(t)
        flow = get_sensor_flow(t)
        
        alert = detector.on_sensor_data(pressure, flow, timestamp=t)
        if alert:
            print(f"[{alert['timestamp']}] ALERT: {alert['fault_type']}, "
                  f"Zone {alert['zone_id']}, Confidence {alert['fault_confidence']:.2f}")
```

**Decision Points**:
- If latency >50 ms → optimize feature extraction (use numba JIT compilation)
- If false positive rate >10% → increase anomaly_prob threshold to 0.6
- If detection delay >5 min → decrease window_size to 20 samples

#### Task 4.2: Performance Metrics Dashboard
**Input**: Alert history from 4.1  
**Output**: Summary metrics plot  
**Steps**:

```
4c. Create notebooks/performance_report.ipynb:
    
    # Cell 1: Load alerts
    import pandas as pd
    import matplotlib.pyplot as plt
    
    alerts = detector.alert_history
    df_alerts = pd.DataFrame(alerts)
    
    # Cell 2: Metrics
    print(f"Total Alerts: {len(df_alerts)}")
    print(f"Fault Types: {df_alerts['fault_type'].value_counts()}")
    print(f"Avg Fault Confidence: {df_alerts['fault_confidence'].mean():.3f}")
    print(f"Avg Zone Confidence: {df_alerts['zone_confidence'].mean():.3f}")
    
    # Cell 3: Confidence distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df_alerts['fault_confidence'], bins=20, edgecolor='black')
    axes[0].set_xlabel('Fault Confidence')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Fault Classification Confidence')
    
    axes[1].hist(df_alerts['zone_confidence'], bins=20, edgecolor='black')
    axes[1].set_xlabel('Zone Confidence')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Zone Localization Confidence')
    
    plt.tight_layout()
    plt.savefig('results/performance_metrics.png', dpi=150)
    plt.show()
```

---

## PART C: DECISION SUMMARY TABLE

### C.1 All Strategic Decisions at a Glance

| Phase | Decision | Choice | Rationale | Contingency |
|---|---|---|---|---|
| **Data Gen** | Total samples | 3,600 | Balances model capacity with generation time | Re-gen with 5,000 if accuracy <85% |
| | Samples per class | 900 each | Balanced classes → fair RF training | Add 300 more if one class <85% acc |
| | Window size | 30 samples @ 1 Hz | Captures 30s fault transient | Try 20 (fast) or 50 (slow) if poor results |
| | Sliding stride | 10 samples | ~5-10 samples per scenario | Adjust to 5 (denser) or 15 (sparser) |
| **Feature Eng** | Obj 2 features | 45 (8+4+3+2×5 sensors) | Sufficient for RF; <60 keeps inference fast | Reduce to top-20 if overfitting |
| | Obj 3 features | 25 (5 residuals + 4 grads + 2 imb + 3 rate + 5 other) | Simpler problem → fewer features | Reduce to 20 if class imbalance |
| **Train/Val/Test** | Split ratio | 70/15/15 | Standard ML practice | Use 80/10/10 if fewer samples available |
| | Stratification | Yes | Ensures class balance in each fold | Re-generate if imbalance >10% |
| **Objective 2** | Detection | Random Forest | Supervised multi-class classification using labelled data | Adjust feature set or tree depth if accuracy <90% |
| | Model | Random Forest | Fast; interpretable; no GPU | Switch to XGBoost if RF accuracy <85% |
| | Localization | RF Zone Classifier | Same as Obj 2 detection stage | Same as above |
| **Objective 3** | Stage 1 | Pressure Residual Analysis | O(n) lookup; real-time; intuitive | Switch to Bayesian inference if accuracy <75% |
| | Stage 2 | RF Zone Classifier | Same as Obj 2 Stage 2 | Same as above |
| | Zones | 30-50 | Practical for small-medium networks | Auto-define from EPANET connectivity |
| **Integration** | Real-time Latency | <10 ms | Meets SCADA update rate (1-100 Hz) | Optimize with Cython/numba if >50 ms |
| | Confidence Threshold | 60% | Flags uncertain predictions for operator review | Adjust to 50% (more alerts) or 75% (fewer) |

---

## PART D: DETAILED STEP-BY-STEP ROADMAP (Timeline)

### Week 1: Data Preparation
- **Mon**: Extend dataset.py with 45-feature extraction, 3,600 samples
- **Tue-Wed**: Generate full dataset; verify class balance
- **Thu**: Implement FeatureExtractor class; test on 100 samples
- **Fri**: Create train/val/test split; save as NPY files

### Week 2: Feature Engineering & Baseline
- **Mon-Tue**: Add temporal, spatial, frequency features
- **Wed**: Build baseline pressure model (24h normal operation)
- **Thu**: Feature normalization; correlation analysis
- **Fri**: Finalize feature matrix (3600 × 45); verify no NaN/Inf

### Week 3: Objective 2 Detection
- **Mon-Tue**: Implement Random Forest multi-class classifier
- **Wed**: Tune hyperparameters via GridSearchCV
- **Thu**: Evaluate on validation set; compute precision/recall/F1
- **Fri**: Save trained model; document hyperparameters

### Week 4: Objective 2 Stage 2
- **Mon-Tue**: Implement Random Forest multi-class classifier
- **Wed**: GridSearchCV for n_estimators, max_depth, class_weight
- **Thu**: Cross-validation (5-fold); plot confusion matrices
- **Fri**: Final test set evaluation; compute precision/recall/F1

### Week 5: Objective 2 Analysis & Documentation
- **Mon**: Feature importance ranking; visualize top-15
- **Tue**: ROC curves per class; PR curves
- **Wed**: Error analysis; inspect misclassified samples
- **Thu-Fri**: Write summary report; prepare for Phase 3

### Week 6: Objective 3 Stage 1 + Stage 2 Setup
- **Mon-Tue**: Baseline pressure model (Stage 1); compute residuals
- **Wed**: Zone definitions from EPANET connectivity
- **Thu**: LocalizationFeatureExtractor (25 features)
- **Fri**: Create X_localization, y_localization training data

### Week 7: Objective 3 Stage 2 Training
- **Mon-Tue**: Train zone classifier (RF multi-class, 30-50 zones)
- **Wed**: GridSearchCV; 5-fold CV
- **Thu**: Test set evaluation; per-zone accuracy
- **Fri**: Feature importance for zones; document zones

### Week 8: Integration & Real-Time Inference
- **Mon-Tue**: Build RealTimeLeakDetector class with streaming pipeline
- **Wed**: Simulate sensor stream; test detection latency
- **Thu**: Integrate Objective 2 + Objective 3 alerts
- **Fri**: Test end-to-end pipeline (detection + localization)

### Week 9: Final Testing & Documentation
- **Mon**: Performance metrics dashboard (jupyter notebook)
- **Tue**: Stress test with 1000+ sensor readings
- **Wed**: Sensitivity analysis (vary window_size, stride)
- **Thu**: Prepare presentation/report
- **Fri**: Final review & deployment checklist

---

## PART E: KEY METRICS TO TRACK

### Objective 2 (Leak Detection)
```
✓ Test Accuracy (goal: >92%)
✓ Per-class Precision/Recall/F1 (goal: >90% each)
✓ ROC-AUC (goal: >0.95)
✓ False Positive Rate (goal: <5%)
✓ False Negative Rate (goal: <8%)
✓ Training time (expect: 5-10 min)
✓ Inference latency (expect: <10 ms / sample)
```

### Objective 3 (Fault Localisation)
```
✓ Zone Accuracy (goal: >85%)
✓ Per-zone Accuracy (goal: >80% each)
✓ Top-2 Zone Accuracy (goal: >92%)
✓ Confidence score distribution (goal: mean >80%)
✓ Training time (expect: 2-5 min)
✓ Inference latency (expect: <5 ms / sample)
```

### Combined System
```
✓ End-to-end latency (expect: <50 ms from sensor to alert)
✓ Alert consistency (goal: >90% persistent over 5 windows)
✓ Zone false positive rate (goal: <10%)
✓ Real-world deployment readiness (GPU-free, low memory)
```

---

## PART F: FAILURE MODES & CONTINGENCIES

| Scenario | Symptom | Action |
|---|---|---|
| **Dataset too small** | Accuracy plateaus at 80% | Regenerate with 5,000 samples; use stride=5 |
| **Class imbalance** | Leak recall <70% | Oversample minority; adjust class_weight |
| **Features too noisy** | Validation acc <10% worse than train | Apply median filter; reduce window variance |
| **Baseline model drift** | Zone accuracy <70% | Re-build baseline monthly; allow ±20% tolerance |
| **Sensor outliers** | Detection spikes or false alarms | Add robust filtering (IQR, outlier clipping, feature clean-up) |
| **Inference too slow** | Latency >100 ms | Use ONNX export; compile with numba; reduce n_estimators |
| **Overfitting** | Test acc 15% below validation | Increase max_depth penalty; reduce feature count |
| **Underfitting** | Validation acc <75% | Increase model complexity; more samples; richer features |

---

## CONCLUSION

**This plan provides**:
1. ✅ 3,600 total samples (900 per class) across 4 fault types
2. ✅ 45 features for leak detection + 25 for localization
3. ✅ Two-stage pipeline (anomaly + classification) for both objectives
4. ✅ 70/15/15 train/val/test split with stratification
5. ✅ Week-by-week implementation roadmap (9 weeks total)
6. ✅ All decision points with contingencies
7. ✅ Expected performance targets (>92% detection, >85% localization)
8. ✅ Real-time inference (<50 ms latency, no GPU required)

**Next Step**: Start Week 1 with data generation. Execute Phase 1 to build the 3,600-sample dataset before moving to model training.

---

**Document Version**: 1.0  
**Last Updated**: 2026-05-19  
**Status**: Ready for implementation  
