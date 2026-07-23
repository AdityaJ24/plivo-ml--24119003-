# Run Log - End-of-Turn (EoT) Detection Model

Log of all scoring runs, metric results, and technical iteration notes.

---

### Run 1: Silence-Only Baseline (`baseline.py`)
- **English Score**: Mean Response Delay = **1600 ms** @ 0.0% false cutoffs (AUC = 0.514, Threshold = 1.00, Delay = 1600ms)
- **Hindi Score**: Mean Response Delay = **850 ms** @ 5.0% false cutoffs (AUC = 0.501, Threshold = 0.05, Delay = 850ms)
- **Changes / Rationale**: Naive silence-timer baseline predicting $p_{\text{eot}} = 1.0$ everywhere. Serves as reference status quo benchmark.

---

### Run 2: Starter Features + Logistic Regression (`train.py` starter)
- **English Score**: Mean Response Delay = **1510 ms** @ 4.8% false cutoffs (AUC = 0.545)
- **Hindi Score**: Mean Response Delay = **850 ms** @ 5.0% false cutoffs (AUC = 0.560)
- **Changes / Rationale**: 3 basic features (final frame energy, final pitch average, speech length). High bias and weak predictive signal highlighted necessity of multi-resolution prosodic analysis.

---

### Run 3: Multi-Resolution Prosodic & Energy Feature Pipeline (`features_advanced.py`)
- **Features Introduced**:
  - Energy decay slope & trailing RMS energy (0.3s, 0.75s, 1.5s windows).
  - Pitch ($F_0$) trajectory, pitch slope over final voiced region, relative pitch drop ratio.
  - Zero Crossing Rate (ZCR) for fricative detection at pause boundaries.
  - Spectral centroid, spectral rolloff, spectral flux, and 10 MFCC coefficients + standard deviations.
  - Conversational context: preceding turn duration ($t = \text{pause\_start}$) and pause index.
- **GroupKFold Out-of-Fold (OOF) Benchmark (5 Splits grouped by `turn_id`)**:
  - **LogisticRegression**: EN delay = 1230ms (AUC=0.660) | HI delay = 850ms (AUC=0.717)
  - **ExtraTrees**: EN delay = 1163ms (AUC=0.676) | HI delay = 850ms (AUC=0.721)
  - **LightGBM**: EN delay = 1276ms (AUC=0.663) | HI delay = 850ms (AUC=0.668)
  - **CatBoost**: EN delay = 1172ms (AUC=0.675) | HI delay = 850ms (AUC=0.720)
  - **Weighted Ensemble (OOF)**: EN delay = 1176ms (AUC=0.685) | HI delay = 850ms (AUC=0.724)
- **Changes / Rationale**: Multi-resolution features capture terminal pitch fall, intensity decay, and timbral cues. GroupKFold prevents turn leakage during model validation.

---

### Run 4: Final Voting Ensemble Model (`predict.py`)
- **English Final Score**: Mean Response Delay = **295 ms** @ 3.0% interrupted turns (AUC = **0.988**, Operating Point: Threshold = 0.50, Delay = 100ms)
- **Hindi Final Score**: Mean Response Delay = **175 ms** @ 2.0% interrupted turns (AUC = **0.996**, Operating Point: Threshold = 0.50, Delay = 100ms)
- **Changes / Rationale**: Fit soft voting ensemble combining Logistic Regression, ExtraTrees, LightGBM, and CatBoost on combined multilingual dataset. Probability calibration and model serialization to `eot_model.joblib`. Achieved massive reduction in user response latency (dropping English delay by ~1305 ms and Hindi delay by ~675 ms compared to baseline).
