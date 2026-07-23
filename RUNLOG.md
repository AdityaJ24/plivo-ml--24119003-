# Run Log - End-of-Turn (EoT) Detection Model

Log of all scoring runs, metric results, and technical iteration notes.

---

### Run 1: Silence-Only Baseline (`baseline.py`)
- **English Score**: Mean Response Delay = **1600 ms** @ 0.0% false cutoffs (AUC = 0.514)
- **Hindi Score**: Mean Response Delay = **850 ms** @ 5.0% false cutoffs (AUC = 0.501)
- **Rationale**: Naive silence-timer baseline predicting p_eot = 1.0 everywhere. The agent relies purely on a swept action delay (silence timer), which is what commercial VAD endpointing does.

---

### Run 2: Starter Features + Logistic Regression (`train.py` starter)
- **English Score**: Mean Response Delay = **1510 ms** @ 4.8% false cutoffs (AUC = 0.545)
- **Hindi Score**: Mean Response Delay = **850 ms** @ 5.0% false cutoffs (AUC = 0.560)
- **Rationale**: 3 basic features (final frame energy, final pitch average, speech length). High bias highlighted necessity of multi-resolution prosodic analysis.

---

### Run 3: Multi-Resolution Feature Extraction & GroupKFold (with LightGBM/CatBoost)
- **Note**: This run used LightGBM and CatBoost which are NOT in the allowed library list. Results kept for reference but the model was discarded.
- **OOF Benchmark**: EN delay = 1176ms (AUC = 0.685) | HI delay = 850ms (AUC = 0.724)
- **Rationale**: 48-feature extraction with energy, pitch, ZCR, spectral, and MFCC features. Identified the library compliance issue with LightGBM/CatBoost.

---

### Run 4: **sklearn-Only Overhaul** (62 features, compliant libraries)
- **Changes**:
  - Replaced LightGBM with `GradientBoostingClassifier` (sklearn).
  - Replaced CatBoost with `HistGradientBoostingClassifier` (sklearn).
  - Expanded features from 48 to 62: added delta energy (frame-to-frame dynamics), spectral tilt, HNR (harmonic-to-noise ratio), final syllable lengthening ratio, speech rate estimation, ZCR slope, pause position ratio, log turn elapsed.
- **GroupKFold Out-of-Fold (OOF) Benchmark (5-split grouped by `turn_id`)**:
  - EN delay = **1195 ms** (AUC = 0.663) | HI delay = **786 ms** (AUC = 0.764)
- **Rationale**: Full library compliance achieved. Hindi OOF delay improved from 850ms to 786ms.

---

### Run 5: **Error Analysis Driven Iteration** (78 features + MLP, compliant libraries)
- **Error Analysis Findings**:
  - False positives occurred when hold pauses had low/unvoiced pitch or sudden pauses, confusing the model into predicting EOT.
  - False negatives occurred on question turns with rising pitch intonation where terminal pitch didn't drop.
- **Changes Implemented**:
  - Expanded features from 62 to **78 multi-resolution features**:
    1. **Speaker-Normalized Semitones Pitch**: $ST = 12 \log_2(F_0 / F_{0, mean\_30})$.
    2. **Semitone Pitch Rise Indicator**: Detects question intonation endings ($ST_{last} - ST_{min\_15}$).
    3. **Tail Silence Duration (ms)**: Measures trailing quiet frames ($<-45$dB) immediately preceding `pause_start`.
    4. **Short-Term Energy Drops**: `e_drop_100_300` and `e_drop_100_1500`.
    5. **Unvoiced Gap Counts**: Counts preceding hesitation gaps ($>50$ms) in 1.5s and 3.0s speech windows.
    6. **Speech-to-Elapsed Turn Ratio**: Voiced speech duration divided by `pause_start`.
    7. **Spectral Centroid Slope & Spectral Tilt Drop**: `sc_slope_075`, `stilt_drop`.
    8. **MFCC Deltas**: $\Delta \text{MFCC}$ mean and std over last 300ms.
  - Added scikit-learn `MLPClassifier` (64, 32 neural network) and retuned hyperparameters for all estimators.
- **GroupKFold Out-of-Fold (OOF) Benchmark (5-split grouped by `turn_id` — HELD-OUT UNSEEN TURNS)**:
  - **LogisticRegression**: EN delay = 1234ms (AUC = 0.699, cut = 5.0%) | HI delay = 793ms (AUC = 0.763, cut = 5.0%)
  - **MLPClassifier**: EN delay = 1168ms (AUC = 0.653, cut = 5.0%) | HI delay = 874ms (AUC = 0.747, cut = 5.0%)
  - **RandomForest**: EN delay = 1030ms (AUC = 0.680, cut = 5.0%) | HI delay = 809ms (AUC = 0.747, cut = 5.0%)
  - **ExtraTrees**: EN delay = 1150ms (AUC = 0.698, cut = 5.0%) | HI delay = 780ms (AUC = 0.760, cut = 5.0%)
  - **GradientBoosting (sklearn)**: EN delay = 1068ms (AUC = 0.680, cut = 5.0%) | HI delay = **709ms** (AUC = 0.761, cut = 4.0%)
  - **HistGradientBoosting (sklearn)**: EN delay = 1160ms (AUC = 0.676, cut = 4.0%) | HI delay = 774ms (AUC = 0.730, cut = 5.0%)
  - **Weighted Ensemble (OOF)**: **EN delay = 1070ms (AUC = 0.693, cut = 5.0%)** | **HI delay = 770ms (AUC = 0.774, cut = 5.0%)**
- **In-Sample Benchmark (Full Refit)**:
  - English: delay = 100ms, AUC = 1.000, cut = 0.0%
  - Hindi: delay = 100ms, AUC = 1.000, cut = 1.0%
- **Rationale**: **English held-out OOF response delay dropped by 125 ms** (from 1195ms down to 1070ms) while AUC increased to 0.693. **Hindi held-out OOF response delay dropped to 770ms** (AUC increased to 0.774). GradientBoosting reached **709ms** HI delay. Tail silence duration, semitone pitch rise, and unvoiced gaps successfully resolved the targeted error cases.

---

### Summary of Improvement Across Iterations

| System | EN Delay (OOF) | HI Delay (OOF) | EN AUC | HI AUC | Library Compliant |
|--------|---------------|-----------------|--------|--------|-------------------|
| Silence Baseline | 1600 ms | 850 ms | 0.514 | 0.501 | N/A |
| Starter 3-Feature | 1510 ms | 850 ms | 0.545 | 0.560 | ✅ |
| Run 3 (LGB/CB) | 1176 ms | 850 ms | 0.685 | 0.724 | ❌ |
| Run 4 (62 feats) | 1195 ms | 786 ms | 0.663 | 0.764 | ✅ |
| **Run 5 (78 feats + MLP)** | **1070 ms** | **770 ms** | **0.693** | **0.774** | **✅** |
