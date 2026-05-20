# VISUAL WORKFLOW ARCHITECTURE

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WATER DISTRIBUTION NETWORK                           │
│                          (5-7 Pressure Sensors)                              │
│                            (2 Flow Meters)                                   │
└────────────────────────────────────────┬────────────────────────────────────┘
                                         │
                    Real-Time Sensor Stream: P[t], Q[t] @ 1 Hz
                                         │
                                         ↓
                    ┌────────────────────────────────────┐
                    │    FEATURE EXTRACTION BUFFER       │
                    │   (Sliding window: size=30, stride=10) │
                    │     45 Features per window         │
                    └────────────────┬────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 │                                       │
                 ↓                                       ↓
        ┌──────────────────────┐            ┌──────────────────────┐
        │   RANDOM FOREST      │            │  BASELINE PRESSURE   │
        │  (Supervised, 4-class)│           │      MODEL           │
        │  (Detection only)     │           │  (Lookup table)      │
        │                      │            │  P_baseline[i][h]    │
        │ Input: 45 features   │            └────────┬─────────────┘
        │ Output: 0/1/2/3      │                     │
        │ Latency: <5 ms       │                     │
        └──────┬───────────────┘                     │
               │                                     │
        ┌──────┴──────┐                              │
        │             │                              │
    NORMAL         PREDICTED                           │
    ├─PASS        FAULT (1/2/3)                         │
    │               ↓                                │
    │      ┌──────────────────────┐                 │
    │      │  FAULT TYPE           │                 │
    │      │  PREDICTION           │                 │
    │      │  (0=Normal, 1=Leak,   │                 │
    │      │   2=Burst, 3=Blockage)│                 │
    │      │ Latency: <5 ms       │                 │
    │      └──────┬───────────────┘                 │
    │             │                                 │
    │      ┌──────┴───────┬──────────┬──────────┐  │
    │      │              │          │          │  │
    │      ↓              ↓          ↓          ↓  │
    │   LEAK          BURST     BLOCKAGE    NORMAL │
    │    │              │          │          │    │
    │    └──────────────┬──────────┴──────────┘    │
    │                   │                          │
    │        ┌──────────┴──────────┐               │
    │        │                     │               │
    │        ↓                     ↓               │
    │   [ALERT]              [CONFIDENCE]          │
    │   Fault Type           Probability           │
    │   Confidence (0-100%)                        │
    │        │                                    │
    │        └───────────────────┬─────────────────┴─────────────────────┐
    │                            │                                       │
    │                      [TRIGGER]                                     │
    │                   Confidence>60%                         Use for Localization
    │                            │                                       │
    │                            ↓                                       ↓
    │                 ┌──────────────────────┐          ┌──────────────────────┐
    │                 │ ALERT TO OPERATOR    │          │ EXTRACT LOCALIZATION │
    │                 │ Fault Type + Time    │          │ FEATURES (25 dim)    │
    │                 │ Severity: HIGH       │          │ - Residuals (5)      │
    │                 └──────────────────────┘          │ - Gradients (4)      │
    │                                                    │ - Imbalance (2)      │
    │                                                    │ - Rate-of-change (3) │
    │                                                    │ - Other (11)         │
    │                                                    └──────┬───────────────┘
    │                                                           │
    │                                                           ↓
    │                                                ┌──────────────────────┐
    │                                                │ ZONE CLASSIFIER      │
    │                                                │ RANDOM FOREST        │
    │                                                │ (Multi-class)        │
    │                                                │                      │
    │                                                │ Input: 25 features  │
    │                                                │ Output: Zone ID      │
    │                                                │ + Confidence         │
    │                                                │ Latency: <5 ms       │
    │                                                └──────┬───────────────┘
    │                                                       │
    │                                            ┌──────────┴──────────┐
    │                                            │                     │
    │                                      Confidence              Not
    │                                        >60%?                Confident
    │                                            │                  │
    │                                            ↓                  ↓
    │                                  [LOCALIZATION]        [TOP-3 ZONES]
    │                                  Primary Zone: Z1       Suggest Z1, Z2, Z3
    │                                  Confidence: 87%        Request Operator
    │                                  Radius: 100m           Confirmation
    │                                            │
    │                                            └────────────┬────────────────┐
    │                                                         │                │
    │                                                         ↓                ↓
    │                                                  [DISPATCH FIELD]   [UPDATE MODEL]
    │                                                   CREW TO ZONE       RECOMPUTE
    │                                                   Estimated Time     BASELINE
    │                                                   to Fix: 2-4 hrs    (Monthly)
    │
    └─────────────────→ [END-TO-END LATENCY: <50ms] ←───────────────────────┘
```

---

## Data Generation & Preparation Flow

```
┌──────────────────────────────────────────────────────────────────┐
│         EPANET NETWORK FILES (7 networks available)              │
│  1_Hanoi.inp, 2_Extended Hanoi.inp, 3_Foss_poly_1.inp, ...      │
└──────────────────────────────────────┬──────────────────────────┘
                                       │
                                       ↓
                    ┌──────────────────────────────────┐
                    │   SCENARIO GENERATOR             │
                    │   (graph_dataset/dataset.py)     │
                    │                                  │
                    │  For i = 1 to 200:              │
                    │   - Run 24h EPANET simulation    │
                    │   - Add ±15% demand noise       │
                    │   - Inject fault at random pipe  │
                    │   - Extract pressure/flow time   │
                    │     series (86,400 samples)      │
                    └──────────────┬───────────────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                │                  │                  │
                ↓                  ↓                  ↓
        Normal Class      Fault Injection      Feature Extraction
        (200 scenarios)   (600 scenarios)       (45 dimensions)
             │            ├─ Leak: 200         ├─ Statistical (8)
             │            ├─ Burst: 200        ├─ Temporal (4)
             │            └─ Blockage: 200     ├─ Spatial (3)
             │                │                └─ Frequency (2)
             └────────────────┴────────────────┬────────────────┘
                                               │
                              Sliding Window Extraction
                           (window=30, stride=10)
                           ~5-10 windows per scenario
                                               │
                                               ↓
                            ┌──────────────────────────┐
                            │  3,600 SAMPLES TOTAL     │
                            │  900 samples per class   │
                            │  Shape: (3600, 45)       │
                            │  Stored: .pkl or .npy    │
                            └──────────┬───────────────┘
                                       │
                      ┌────────────────┼────────────────┐
                      │                │                │
                      ↓                ↓                ↓
                ┌──────────┐      ┌──────────┐    ┌──────────┐
                │  TRAIN   │      │   VAL    │    │  TEST    │
                │ 2,520    │      │   540    │    │  540     │
                │ (70%)    │      │  (15%)   │    │ (15%)    │
                └──────────┘      └──────────┘    └──────────┘
                     │                 │               │
                Stratified Split (class ratio maintained)
                     │                 │               │
        ┌────────────┴─────────┐      │               │
        │                      │      │               │
        ↓                      ↓      ↓               ↓
    [Stage 1]          [GridSearchCV]  [Stage 2]    [FINAL TEST]
    Train IF           Tune Hyperparam  Validate     Report Accuracy
    on Normal only     (5-fold CV)      (F1-score)   92%+ goal
    (630 samples)                                    
```

---

## Model Training Workflow (Objective 2)

```
                      TRAINING DATA: 2,520 × 45
                      (630 per class)
                             │
                ┌────────────┴────────────┐
                │                         │
                ↓                         ↓
        ┌───────────────┐         ┌───────────────┐
        │  STAGE 1:     │         │  STAGE 2:     │
        │  ISO FOREST   │         │  RANDOM       │
        │  (Unlabeled)  │         │  FOREST       │
        │               │         │  (Labeled)    │
        ├───────────────┤         ├───────────────┤
        │ Hyperparams:  │         │ Hyperparams:  │
        │ - n_est=100   │         │ - n_est=200   │
        │ - contamination         │ - max_depth   │
        │   =0.05       │         │   =None       │
        │ - max_samples │         │ - class_weight│
        │   =256        │         │   ='balanced' │
        │               │         │               │
        │ Output:       │         │ Output:       │
        │ Anomaly score │         │4-class labels│
        │ [0, 1]        │         │ + confidence  │
        └───────┬───────┘         └───────┬───────┘
                │                         │
                └────────────┬────────────┘
                             │
                    VALIDATION SET: 540 × 45
                    (135 per class)
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ↓                    ↓                    ↓
    [Anomaly Recall]  [F1-score]         [Precision/Recall]
    Goal: >95%        Goal: >90%          Per-class: >90%
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
    Tuning            Tuning            Final Selection
    IF contamination  RF hyperparams    (Best F1)
        │                    │                   │
        ↓                    ↓                   ↓
     Accept            Accept              TEST SET
   (or retry)          (or retry)          540 × 45
                                          Report: Accuracy,
                                          Precision, Recall,
                                          F1, ROC-AUC
                                          
                                          EXPECTED: >92%
```

---

## Model Training Workflow (Objective 3)

```
              LOCALIZATION DATA: 3,600 × 25
              (Reuse from Objective 2 dataset
               + zone labels)
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ↓             ↓             ↓
    TRAIN        VALIDATION      TEST
    2,520×25      540×25        540×25
    (70%)         (15%)          (15%)
        │             │             │
        ├─ 630 samples per class (stratified)
        │             │             │
        ↓             ↓             ↓
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ RF Train │ │GridSearch│ │ RF Test  │
    │ Multi-   │ │  CV      │ │ Evaluate │
    │ class    │ │ (5-fold) │ │ 30-50    │
    │ 30-50    │ │  Zone    │ │ zones    │
    │ zones    │ │ classifier
    │          │ │          │ │          │
    │Hyperp:  │ │Tune:     │ │Metrics:  │
    │-n_est   │ │-n_est    │ │-Accuracy │
    │ =100    │ │ ∈[50,200]│ │>85%      │
    │-max_d   │ │-max_d    │ │-Per-zone │
    │ =15     │ │ ∈[10,20] │ │>80%      │
    │-class_w │ │ ∈[10,15] │ │-Top-2    │
    │ =balanced               │>92%      │
    └─────────┘ └──────────┘ └──────────┘
```

---

## Real-Time Inference Pipeline

```
                LIVE SENSOR STREAM
                P[t], Q[t] @ 1 Hz
                        │
                        ↓
                ┌───────────────────┐
                │  BUFFER           │
                │ (Last 30 samples) │
                │ Window size=30    │
                │ Rolling update    │
                │ @10 Hz (stride=10)│
                └────────┬──────────┘
                         │
            ┌────────────┴────────────┐
            │                         │
            ↓                         ↓
    ┌──────────────────┐     ┌──────────────────┐
    │ FEATURE          │     │ BASELINE         │
    │ EXTRACTION       │     │ LOOKUP           │
    │ 45 features      │     │ P_baseline[i][h] │
    │ <1 ms            │     │ <0.1 ms          │
    └────────┬─────────┘     └────────┬─────────┘
             │                        │
             └────────────┬───────────┘
                          │
                ┌─────────┴──────────┐
                │ RESIDUAL ANALYSIS │
                │ (Fast Stage 1)     │
                │ <1 ms              │
                └─────────┬──────────┘
                          │
                ┌─────────┴──────────┐
                │                    │
          Normal <60%            >60%
            confidence           (Anomaly
              │                   Likely)
              │                    │
              ↓                    ↓
          [PASS]          ┌───────────────┐
                          │ FAULT         │
                          │ CLASSIFIER    │
                          │ Stage 2       │
                          │ <5 ms         │
                          └────────┬──────┘
                                   │
                ┌──────────────┬────┴────┬──────────────┐
                │              │         │              │
                ↓              ↓         ↓              ↓
            LEAK           BURST     BLOCKAGE       NORMAL
            │              │         │              │
            └──────────────┬─────────┴──────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
        ↓                                     ↓
    CONFIDENCE              ZONE LOCALIZATION
    Score: [0,1]           FEATURES (25 dim)
    Type: LEAK             │
    Time: [t]              ↓
                      ┌────────────────┐
                      │ ZONE           │
                      │ CLASSIFIER     │
                      │ Stage 2        │
                      │ <5 ms          │
                      └────────┬───────┘
                               │
                      ┌────────┴────────┐
                      │                 │
                  Conf>60%          Conf<60%
                      │                 │
                      ↓                 ↓
                  PRIMARY ZONE      TOP-3 ZONES
                  Zone ID: 12       [12, 8, 15]
                  Confidence:       Need operator
                  87%               confirmation
                      │                 │
                      └────────┬────────┘
                               │
                    [ALERT TO OPERATOR]
                    - Fault Type: LEAK
                    - Primary Zone: 12
                    - Confidence: 87%
                    - Radius: ~100m
                    - Estimated time: 2-4h
                    
                    Total Latency: <50ms
```

---

## Data Shapes & Dimensions

```
OBJECTIVE 2 (Leak Detection)
├─ X_train.npy: (2520, 45)
│  └─ 45 = 5 sensors × 8 features + 3 spatial + 2 frequency
├─ y_train.npy: (2520,)
│  └─ 0=Normal, 1=Leak, 2=Burst, 3=Blockage (630 each)
├─ X_val.npy: (540, 45)
├─ y_val.npy: (540,)
├─ X_test.npy: (540, 45)
└─ y_test.npy: (540,)

OBJECTIVE 3 (Fault Localisation)
├─ X_localization.npy: (3600, 25)
│  └─ 25 = 5 residuals + 4 gradients + 2 imbalance + 3 rate-of-change + 11 other
├─ y_localization.npy: (3600,)
│  └─ Zone IDs (0 to 29-49 depending on network)
├─ X_loc_train: (2520, 25)
├─ y_loc_train: (2520,)
├─ X_loc_val: (540, 25)
├─ y_loc_val: (540,)
├─ X_loc_test: (540, 25)
└─ y_loc_test: (540,)

SENSOR TIME-SERIES (Raw)
├─ Pressure data: [P1, P2, P3, P4, P5] @ 1Hz, 86,400 steps/day
├─ Flow data: [Q1, Q2] @ 1Hz, 86,400 steps/day
└─ Simulation duration: 24 hours × 200 scenarios = 4,800 hours = 200 days simulation time

WINDOW EXTRACTION (Intermediate)
├─ Per scenario: 86,400 samples
├─ Sliding window: size=30, stride=10
├─ Windows per scenario: (86,400 - 30) / 10 + 1 ≈ 8,637 windows
├─ Scenarios: 200 × 4 fault types = 800 scenarios
├─ Total raw samples: 800 × 8,637 ≈ 6.9 million
├─ Downsampled to: 3,600 samples (stratified)
└─ Final shape: (3600, 45)
```

---

## Training Timeline (9 Weeks)

```
Week 1-2: Data Generation & Feature Engineering
├─ Generate 3,600 samples (800 scenarios × ~4.5 windows)
├─ Extract 45 features per window
├─ Create train/val/test splits (70/15/15)
└─ [TIME: ~45 min generation + 30 min feature extraction]

Week 3: Objective 2 - Detection
├─ Train Random Forest on all 4 classes (2,520)
├─ Tune hyperparameters via GridSearchCV
├─ Validate: precision/recall/F1 on validation set
└─ [TIME: ~10 min training, 5 min validation]

Week 4: Objective 2 - Finalization
├─ Finalize Random Forest model and hyperparameters
├─ Run 5-fold cross-validation and test evaluation
├─ Plot confusion matrices and feature importance
└─ [TIME: ~10 min training, 5 min CV]

Week 5: Objective 2 - Analysis & Validation
├─ Test set evaluation (final accuracy report)
├─ Feature importance ranking
├─ ROC curves & confusion matrices
└─ [TIME: ~2 min evaluation, ~10 min visualization]

Week 6: Objective 3 - Feature Engineering & Baseline
├─ Build baseline pressure model (24h simulation)
├─ Define zones from EPANET connectivity
├─ Create 25-feature localization dataset
└─ [TIME: ~15 min baseline + 20 min feature extraction]

Week 7: Objective 3 - Zone Classifier Training
├─ Train RF on 30-50 zone labels
├─ GridSearchCV: n_estimators, max_depth
├─ Test set evaluation: 85%+ zone accuracy
└─ [TIME: ~5 min training, 2 min validation]

Week 8: Integration & Real-Time Pipeline
├─ Build RealTimeLeakDetector class (streaming buffer)
├─ Integrate Stage 1 + Stage 2 for both objectives
├─ Test end-to-end: <50 ms latency
└─ [TIME: ~30 min coding, ~20 min testing]

Week 9: Final Validation & Documentation
├─ Performance metrics dashboard (jupyter)
├─ Sensitivity analysis (vary window_size, stride)
├─ Deployment checklist & README
└─ [TIME: ~30 min documentation, ~20 min final tests]

TOTAL: ~4 hours training/inference + ~2 hours feature extraction + ~1 hour testing
       = ~7 hours machine time over 9 weeks
       = ~5-10 hours developer time (mostly coding, not waiting for GPU)
```

---

## Success Metrics (Traffic Light)

```
OBJECTIVE 2 — LEAK DETECTION
🟢 EXCELLENT  │ 🟡 ACCEPTABLE  │ 🔴 NEEDS IMPROVEMENT
≥95% accuracy │ 90-95%         │ <90%
≥92% precision│ 85-92%         │ <85%
≥92% recall   │ 85-92%         │ <85%
≥95% ROC-AUC  │ 90-95%         │ <90%
<2% false pos │ 2-5%           │ >5%
<5% false neg │ 5-10%          │ >10%

OBJECTIVE 3 — FAULT LOCALISATION
🟢 EXCELLENT  │ 🟡 ACCEPTABLE  │ 🔴 NEEDS IMPROVEMENT
≥90% zone acc │ 85-90%         │ <85%
≥88% per-zone │ 80-88%         │ <80%
≥95% top-2    │ 90-95%         │ <90%
≥85% conf(avg)│ 75-85%         │ <75%
<3 min to fix │ 3-5 min        │ >5 min

REAL-TIME PERFORMANCE
🟢 EXCELLENT  │ 🟡 ACCEPTABLE  │ 🔴 NEEDS IMPROVEMENT
<10ms total   │ 10-50ms        │ >50ms
<1% alerts    │ 1-5%           │ >5%
CPU only      │ Needs GPU part │ Needs GPU
24/7 stable   │ Occasional lag │ Frequent timeouts
```

This diagram provides a complete visual reference for the implementation workflow, data shapes, and expected performance targets.
