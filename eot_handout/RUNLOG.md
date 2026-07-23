# Run Log - End-of-Turn (EoT) Detection Model

Log of all scoring runs, metric results, and technical iteration notes.

---

### Run 1: Silence-Only Baseline (`baseline.py`)
- **English Score**: Mean Response Delay = **1600 ms** @ 0.0% false cutoffs (AUC = 0.514, Threshold = 1.00, Delay = 1600ms)
- **Hindi Score**: Mean Response Delay = **850 ms** @ 5.0% false cutoffs (AUC = 0.501, Threshold = 0.05, Delay = 850ms)
- **Rationale**: Naive silence-timer baseline predicting $p_{\text{eot}} = 1.0$ everywhere.

---

### Run 2: Starter Features + Logistic Regression (`train.py` starter)
- **English Score**: Mean Response Delay = **1510 ms** @ 4.8% false cutoffs (AUC = 0.545)
- **Hindi Score**: Mean Response Delay = **850 ms** @ 5.0% false cutoffs (AUC = 0.560)
- **Rationale**: 3 basic features (final frame energy, final pitch average, speech length). High bias highlighted necessity of multi-resolution prosodic analysis.

---

### Run 3: Multi-Resolution Feature Extraction & GroupKFold (Held-Out Turns)
- **Features Introduced**:
  - Energy decay slope & trailing RMS energy (0.3s, 0.75s, 1.5s windows).
  - Pitch ($F_0$) trajectory, pitch slope over final voiced region, relative pitch drop ratio.
  - Zero Crossing Rate (ZCR) for fricative detection at pause boundaries.
  - Spectral centroid, spectral rolloff, spectral flux, and 10 MFCC coefficients + standard deviations.
  - Conversational context: preceding turn duration ($t = \text{pause\_start}$) and pause index.
- **GroupKFold Out-of-Fold (OOF) Benchmark (5 Splits grouped strictly by `turn_id` — HELD-OUT UNSEEN TURNS)**:
  - **LogisticRegression**: EN delay = 1230ms (AUC = 0.660, cut = 5.0%) | HI delay = 850ms (AUC = 0.717, cut = 5.0%)
  - **ExtraTrees**: EN delay = 1163ms (AUC = 0.676, cut = 5.0%) | HI delay = 850ms (AUC = 0.721, cut = 5.0%)
  - **LightGBM**: EN delay = 1276ms (AUC = 0.663, cut = 5.0%) | HI delay = 850ms (AUC = 0.668, cut = 5.0%)
  - **CatBoost**: EN delay = 1172ms (AUC = 0.675, cut = 5.0%) | HI delay = 850ms (AUC = 0.720, cut = 5.0%)
  - **Weighted Ensemble (OOF Held-Out)**: **EN delay = 1176ms (AUC = 0.685)** | **HI delay = 850ms (AUC = 0.724)**
- **Rationale**: GroupKFold cross-validation grouped by `turn_id` guarantees zero data leakage between train and test turns. Real held-out AUC improves from 0.514 to 0.685 (EN) and 0.501 to 0.724 (HI), dropping English response delay by **424 ms** at $\le 5\%$ false cutoffs.

---

### Run 4: In-Sample Benchmark (Final Refit on Complete Dataset)
- **English In-Sample**: Mean Response Delay = **295 ms** @ 3.0% cutoffs (AUC = **0.988**, Operating Point: Threshold = 0.50, Delay = 100ms)
- **Hindi In-Sample**: Mean Response Delay = **175 ms** @ 2.0% cutoffs (AUC = **0.996**, Operating Point: Threshold = 0.50, Delay = 100ms)
- **Rationale**: Fitting final Voting Ensemble model on the complete dataset for model serialization (`eot_model.joblib`). High in-sample numbers reflect tree ensemble capacity on known data, whereas Run 3 (OOF) represents true generalization to unseen turns.
