# QUICK REFERENCE: KEY DECISIONS & CHECKLIST

## OBJECTIVE 2: LEAK DETECTION — CRITICAL DECISIONS

### Data Generation
- [ ] **3,600 total samples** (900 per class)
  - Normal: 900 samples
  - Leak (abrupt): 900 samples
  - Burst: 900 samples
  - Blockage: 900 samples
  
- [ ] **Window size**: 30 samples @ 1 Hz (= 30 seconds)
- [ ] **Sliding stride**: 10 samples (captures 5-10 windows per scenario)
- [ ] **Simulation**: 24 hours EPANET simulation per scenario with ±15% demand noise
- [ ] **Feature extraction**: 45 dimensions per sample

### Feature Breakdown (65 total)
```
Per sensor (5 sensors × 12 features each = 60):
  ├─ Statistical (8): mean, std, min, max, range, skewness, kurtosis, IQR
  ├─ Temporal (4): ΔP, Δ²P, max_rate, autocorr[lag=1]
  └─ (Total per sensor = 8 + 4 = 12 features)

Spatial features (3):
  ├─ Pressure gradients (∂P/∂x between sensor pairs)
  ├─ Flow imbalance (inflow - outflow)
  └─ Normalized head loss

Frequency features (2):
  ├─ FFT peak frequency
  └─ FFT magnitude in leak band (10-50 Hz)

TOTAL: 5 sensors × 12 + 3 + 2 = 65 features
```

### Training Strategy: Single-Stage Classification Pipeline
```
Detection: Random Forest (Supervised, Multi-class)
  ├─ Train on: All 4 classes (2,520 training samples)
  ├─ Hyperparameters: n_estimators=200, max_depth=None, class_weight='balanced'
  ├─ GridSearchCV: Tune n_estimators ∈ [100, 200, 300], max_depth ∈ [10, 20, None]
  ├─ Output: Fault type (0=Normal, 1=Leak, 2=Burst, 3=Blockage) + confidence
  └─ Benefit: Uses labelled data directly, avoids failed unsupervised anomaly gating
```

### Train/Val/Test Split
- **Training**: 2,520 samples (70%) — 630 per class
- **Validation**: 540 samples (15%) — 135 per class (used for GridSearchCV)
- **Test**: 540 samples (15%) — 135 per class (final accuracy report)
- **Stratification**: Yes, maintain class ratio in each fold

### Success Metrics (Objective 2)
```
Expected Performance:
  ✓ Overall Accuracy: >92%
  ✓ Precision (per class): >90%
  ✓ Recall (per class): >90%
  ✓ F1-score (weighted): >90%
  ✓ False Positive Rate: <5%
  ✓ False Negative Rate (Leak): <8%

Real-Time Performance:
  ✓ Detection latency: <5 ms
  ✓ Localization latency: <5 ms
  ✓ Total pipeline: <50 ms
  ✓ Hardware: CPU only (no GPU required)
```

---

## OBJECTIVE 3: FAULT LOCALISATION — CRITICAL DECISIONS

### Two-Stage Localisation
```
Stage 1: Pressure Residual Analysis (Fast Baseline)
  ├─ Method: Compute residuals = P_measured[i] - P_baseline[i][current_hour]
  ├─ Data: Lookup table (P_baseline) built from 24h normal operation
  ├─ Output: Candidate zone with anomalous pressure (>0.5 bar deviation)
  ├─ Latency: <1 ms (table lookup)
  └─ Accuracy: ~75-80% candidate zone

Stage 2: Random Forest Zone Classifier (Refined Prediction)
  ├─ Train on: 25-feature vector + zone labels (3,600 samples)
  ├─ Zones: 30-50 zones (defined by EPANET connectivity)
  ├─ Hyperparameters: n_estimators=100, max_depth=15, class_weight='balanced'
  ├─ Output: Top zone + confidence (goal: >85% accuracy)
  ├─ Latency: <5 ms
  └─ Refinement: Consider top-3 zones if confidence <60%
```

### Localization Features (25 total)
```
Pressure Residuals (5):
  ├─ One per sensor node
  ├─ Captures deviation from baseline
  └─ Normalized to [-1, +1] bar

Pressure Gradients (4):
  ├─ ∂P between sensor pairs
  ├─ Identifies pressure drop direction
  └─ Points toward fault location

Flow Imbalance (2):
  ├─ Per zone: (inflow - outflow)
  ├─ Detects blockage/leak at zone level
  └─ Normalized to [0, 1]

Rate-of-Change (3):
  ├─ dP/dt at fault onset
  ├─ dQ/dt
  └─ Distinguishes sudden vs gradual faults

Other (11):
  ├─ Spectral features (2)
  ├─ Fourier components (3)
  ├─ Temporal lags (3)
  └─ Cross-sensor correlations (3)

TOTAL: 25 features
```

### Zone Definition & Training
- **Zones**: 30-50 pipe segments / DMAs (auto-extracted from EPANET connectivity)
- **Training samples**: 3,600 (reuse from Objective 2 dataset)
- **Label**: y = zone_id containing the cracked/faulty pipe
- **Stratification**: Yes, maintain zone balance in train/val/test
- **Confidence threshold**: 60% (flag uncertain predictions to operator)

### Success Metrics (Objective 3)
```
Expected Performance:
  ✓ Overall Accuracy: >85%
  ✓ Per-zone Accuracy: >80% each
  ✓ Top-2 Zone Accuracy: >92%
  ✓ Confidence (mean): >80%

Real-Time Performance:
  ✓ Stage 1 Latency: <1 ms (baseline lookup)
  ✓ Stage 2 Latency: <5 ms (RF inference)
  ✓ Total Pipeline: <50 ms (including feature extraction)
  ✓ Hardware: CPU only (no GPU required)
```

---

## IMPLEMENTATION CHECKLIST

### Phase 1: Data Generation (Week 1-2)
- [ ] Extend `graph_dataset/dataset.py` with 45-feature extraction
- [ ] Generate 3,600 samples with 4 fault classes (900 each)
- [ ] Implement `FeatureExtractor` class (physics/FeatureExtractor.py)
- [ ] Create train/val/test split (70/15/15 stratified)
- [ ] Normalize with MinMaxScaler, save as NPY files
- [ ] Verify: No NaN/Inf, class balance ±5%, features in [0, 1]

### Phase 2: Objective 2 — Leak Detection (Week 3-5)
- [ ] **Detection: Random Forest**
  - [ ] Train on all 4 classes (2,520 samples)
  - [ ] Hyperparameters: n_estimators=200, max_depth=None, class_weight='balanced'
  - [ ] GridSearchCV for n_estimators, max_depth
  - [ ] 5-fold stratified CV
  - [ ] Test set evaluation: expect >92% accuracy
  - [ ] Save model to `models/leak_detection_model.pkl`

- [ ] Feature importance ranking (top 15 features)
- [ ] ROC curves & confusion matrices
- [ ] Write summary report

### Phase 3: Objective 3 — Fault Localisation (Week 6-7)
- [ ] Build baseline pressure model (24h normal operation)
  - [ ] Save as `models/baseline_pressure_model.pkl`
  - [ ] Create lookup: P_baseline[node][hour]

- [ ] Zone definitions from EPANET connectivity
  - [ ] Define 30-50 zones
  - [ ] Create mapping: pipe_id → zone_id
  - [ ] Save to `models/zone_definitions.json`

- [ ] Create `LocalizationFeatureExtractor` (25 features)
  - [ ] Pressure residuals (5)
  - [ ] Pressure gradients (4)
  - [ ] Flow imbalance (2)
  - [ ] Rate-of-change (3)
  - [ ] Other (11)

- [ ] Train zone classifier (RF multi-class)
  - [ ] Dataset: 3,600 samples, 25 features, 30-50 zone labels
  - [ ] Hyperparameters: n_estimators=100, max_depth=15
  - [ ] Test set evaluation: expect >85% accuracy
  - [ ] Save model to `models/stage2_zone_classifier.pkl`

### Phase 4: Integration & Real-Time (Week 8-9)
- [ ] Create `RealTimeLeakDetector` class (inference/real_time_detector.py)
  - [ ] Streaming pipeline (30-sample window buffer)
  - [ ] Stage 1 + Stage 2 inference
  - [ ] Zone localization
  - [ ] Alert generation with confidence

- [ ] Test end-to-end pipeline
  - [ ] Simulate 1000+ sensor readings
  - [ ] Measure latency: expect <50 ms total
  - [ ] Verify alert consistency (>5 persistent windows)

- [ ] Performance metrics dashboard (jupyter notebook)
- [ ] Sensitivity analysis (vary window_size, stride, thresholds)
- [ ] Final deployment checklist

---

## DECISION CONTINGENCIES

### If Objective 2 Accuracy <85%
1. Check feature distribution (NaN, outliers)
2. Increase training data: 5,000 samples (re-generate with stride=5)
3. Try LSTM instead of RF (Week 10 extension)
4. Inspect misclassified samples (debugging)

### If Objective 3 Accuracy <80%
1. Check zone balance (oversample minority zones)
2. Increase feature count: add 5 more spectral features
3. Try GNN instead of RF (Week 11 extension, requires GPU)
4. Re-build baseline pressure model (may have drifted)

### If Inference Latency >50 ms
1. Reduce feature extraction complexity
2. Export models to ONNX format (inference speedup)
3. Use Cython/numba JIT compilation for bottlenecks
4. Reduce ensemble size: n_estimators from 200 to 100

### If False Positive Rate >10%
1. Increase contamination parameter: 0.05 → 0.02
2. Increase anomaly_prob threshold: 0.5 → 0.65
3. Add temporal confirmation: require 3+ consecutive alerts
4. Check baseline model drift (rebuild every 30 days)

---

## FILE STRUCTURE (After Implementation)

```
water-distribution-networks/
├── IMPLEMENTATION_PLAN.md          ← This file
├── DECISIONS_CHECKLIST.md          ← Quick reference
│
├── DATASETS/
│   ├── leak_detection_dataset.pkl  ← 3,600 samples, 45 features
│   ├── X_train.npy                 ← 2,520 × 45
│   ├── X_val.npy                   ← 540 × 45
│   ├── X_test.npy                  ← 540 × 45
│   ├── y_train.npy                 ← 2,520 labels
│   ├── y_val.npy                   ← 540 labels
│   ├── y_test.npy                  ← 540 labels
│   ├── X_localization.npy          ← 3,600 × 25
│   ├── y_localization.npy          ← 3,600 zone labels
│   └── feature_names.json          ← Feature label mapping
│
├── models/
│   ├── leak_detection_model.pkl      ← Trained 4-class fault classifier
│   ├── stage2_zone_classifier.pkl  ← Trained zone classifier
│   ├── baseline_pressure_model.pkl ← P_baseline[node][hour]
│   ├── zone_definitions.json       ← Zone-to-pipe mapping
│   ├── feature_importance.npy      ← RF feature importance
│   └── model_hyperparameters.json  ← Saved hyperparameters
│
├── inference/
│   └── real_time_detector.py       ← RealTimeLeakDetector class
│
├── physics/
│   ├── FeatureExtractor.py         ← Feature extraction pipeline
│   └── LocalizationFeatureExtractor.py
│
├── notebooks/
│   ├── performance_report.ipynb    ← Metrics & visualization
│   └── water_dataset_inspection.ipynb
│
└── results/
    ├── performance_metrics.png     ← Confidence distribution
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── feature_importance.png
    └── summary_report.md
```

---

## EXPECTED TIMELINE

| Week | Phase | Deliverables | Status |
|---|---|---|---|
| 1-2 | Data Gen | 3,600 samples, 45 features, train/val/test split | ⬜ Not Started |
| 3-5 | Obj 2 | Leak detection: 92%+ accuracy, <10ms latency | ⬜ Not Started |
| 6-7 | Obj 3 | Fault localisation: 85%+ accuracy, <5ms latency | ⬜ Not Started |
| 8-9 | Integration | Real-time pipeline, deployment ready | ⬜ Not Started |

---

## SUCCESS CRITERIA (Green Lights)

### Objective 2 ✓
- [ ] Test accuracy ≥92%
- [ ] Precision/Recall ≥90% per class
- [ ] False positive rate ≤5%
- [ ] Total latency ≤10 ms
- [ ] CPU only, no GPU required

### Objective 3 ✓
- [ ] Zone accuracy ≥85%
- [ ] Per-zone accuracy ≥80%
- [ ] Top-2 accuracy ≥92%
- [ ] Total latency ≤50 ms
- [ ] Confidence mean ≥80%

### System Integration ✓
- [ ] End-to-end pipeline functional
- [ ] Alert consistency >90%
- [ ] No data leakage in train/test splits
- [ ] All models save/load correctly
- [ ] Jupyter notebooks reproducible

---

**Status**: Ready for implementation  
**Last Updated**: 2026-05-19  
