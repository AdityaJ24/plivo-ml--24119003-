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
- **GroupKFold Out-of-Fold (OOF) Benchmark (5-split grouped by `turn_id` — HELD-OUT UNSEEN TURNS)**:
  - **LogisticRegression**: EN delay = 1285ms (AUC = 0.678, cut = 5.0%) | HI delay = 783ms (AUC = 0.744, cut = 5.0%)
  - **RandomForest**: EN delay = 1220ms (AUC = 0.642, cut = 5.0%) | HI delay = 783ms (AUC = 0.724, cut = 5.0%)
  - **ExtraTrees**: EN delay = 1225ms (AUC = 0.672, cut = 4.0%) | HI delay = 790ms (AUC = 0.746, cut = 5.0%)
  - **GradientBoosting (sklearn)**: EN delay = 1226ms (AUC = 0.643, cut = 5.0%) | HI delay = 753ms (AUC = 0.731, cut = 5.0%)
  - **HistGradientBoosting (sklearn)**: EN delay = 1230ms (AUC = 0.640, cut = 5.0%) | HI delay = 797ms (AUC = 0.751, cut = 5.0%)
  - **Weighted Ensemble (OOF)**: **EN delay = 1195ms (AUC = 0.663, cut = 5.0%)** | **HI delay = 786ms (AUC = 0.764, cut = 3.0%)**
- **In-Sample Benchmark (Full Refit)**:
  - English: delay = 100ms, AUC = 1.000, cut = 0.0%
  - Hindi: delay = 100ms, AUC = 1.000, cut = 1.0%
- **Rationale**: Full library compliance achieved. Hindi OOF delay improved from 850ms (stuck at baseline) to 786ms (64ms gain). Hindi AUC improved from 0.724 to 0.764. English OOF improved modestly. The new features (HNR, speech rate, final syllable ratio, spectral tilt) helped cross-lingual generalization, especially for Hindi. Top features by ExtraTrees importance: pause_pos_ratio, HNR, spectral flux, f0_last, MFCC coefficients.

---

### Summary of Improvement vs Baseline

| System | EN Delay (OOF) | HI Delay (OOF) | EN AUC | HI AUC | Library Compliant |
|--------|---------------|-----------------|--------|--------|-------------------|
| Silence Baseline | 1600 ms | 850 ms | 0.514 | 0.501 | N/A |
| Starter 3-Feature | 1510 ms | 850 ms | 0.545 | 0.560 | ✅ |
| Run 3 (LGB/CB) | 1176 ms | 850 ms | 0.685 | 0.724 | ❌ |
| **Run 4 (sklearn)** | **1195 ms** | **786 ms** | **0.663** | **0.764** | **✅** |
