"""Model Training & Evaluation Script for End-of-Turn (EoT) Detection.

Features extracted via features_advanced.py (strictly causality compliant).
Uses ONLY scikit-learn classifiers (no LightGBM, CatBoost, or external libraries).
Evaluates with GroupKFold cross-validation (grouped by turn_id).
Saves trained model to `eot_model.joblib`.

LIBRARY COMPLIANCE: numpy, scipy, scikit-learn, joblib only.
"""
import argparse
import csv
import os
import joblib
import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from features_advanced import load_wav, extract_advanced_features
from score import evaluate, score


def load_dataset(data_dir):
    """Load labels and extract advanced audio features for a data folder."""
    labels_path = os.path.join(data_dir, "labels.csv")
    rows = list(csv.DictReader(open(labels_path)))

    cache = {}
    X, y, groups, keys = [], [], [], []
    for r in rows:
        audio_path = os.path.join(data_dir, r["audio_file"])
        if audio_path not in cache:
            cache[audio_path] = load_wav(audio_path)
        x, sr = cache[audio_path]

        feats = extract_advanced_features(
            x, sr,
            pause_start=float(r["pause_start"]),
            pause_index=int(r["pause_index"])
        )

        X.append(feats)
        y.append(1 if r["label"] == "eot" else 0)
        groups.append(r["turn_id"])
        keys.append((r["turn_id"], int(r["pause_index"])))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), np.array(groups), keys


def get_models():
    """Return dictionary of candidate models/pipelines — ALL scikit-learn only."""
    models = {}

    # 1. Scaled Logistic Regression
    models["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, random_state=42))
    ])

    # 2. Multi-Layer Perceptron Neural Net (sklearn)
    models["MLPClassifier"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, alpha=0.01, random_state=42))
    ])

    # 3. Random Forest
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    )

    # 4. Extra Trees (tuned)
    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=400, max_depth=10, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    )

    # 5. Gradient Boosting (sklearn)
    models["GradientBoosting"] = GradientBoostingClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.04,
        subsample=0.85, min_samples_leaf=4, random_state=42
    )

    # 6. Histogram Gradient Boosting (sklearn)
    models["HistGradientBoosting"] = HistGradientBoostingClassifier(
        max_iter=300, max_depth=5, learning_rate=0.04,
        min_samples_leaf=4, l2_regularization=0.1, random_state=42
    )

    return models


def evaluate_cv(X, y, groups, model_name, model, n_splits=5):
    """Run GroupKFold cross-validation and compute out-of-fold predictions."""
    from sklearn.base import clone
    gkf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(y), dtype=np.float64)

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va = X[val_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)

        if hasattr(m, "predict_proba"):
            p_val = m.predict_proba(X_va)[:, 1]
        else:
            p_val = m.decision_function(X_va)
            p_val = 1.0 / (1.0 + np.exp(-p_val))

        oof_preds[val_idx] = p_val

    return oof_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en_dir", default="d:/Plivo/eot_handout/eot_data/english")
    ap.add_argument("--hi_dir", default="d:/Plivo/eot_handout/eot_data/hindi")
    ap.add_argument("--out_model", default="d:/Plivo/eot_model.joblib")
    args = ap.parse_args()

    print("=" * 70)
    print("EoT Model Training Iteration — 78 Features (scikit-learn ONLY)")
    print("=" * 70)

    print("\nLoading and extracting features for English dataset...")
    X_en, y_en, g_en, k_en = load_dataset(args.en_dir)
    print(f"English: {X_en.shape[0]} samples, {X_en.shape[1]} features.")

    print("Loading and extracting features for Hindi dataset...")
    X_hi, y_hi, g_hi, k_hi = load_dataset(args.hi_dir)
    print(f"Hindi: {X_hi.shape[0]} samples, {X_hi.shape[1]} features.")

    # Combine datasets
    X_all = np.vstack([X_en, X_hi])
    y_all = np.hstack([y_en, y_hi])
    g_all = np.hstack([np.array([f"en_{g}" for g in g_en]), np.array([f"hi_{g}" for g in g_hi])])

    print(f"\n--- Combined Dataset: {X_all.shape[0]} samples, {X_all.shape[1]} features, "
          f"{len(np.unique(g_all))} unique turns ---\n")

    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    candidate_models = get_models()
    oof_results = {}

    for name, clf in candidate_models.items():
        print(f"[CV] Training {name}...")
        oof_p = evaluate_cv(X_all, y_all, g_all, name, clf)
        oof_results[name] = oof_p

        p_en = oof_p[:len(y_en)]
        p_hi = oof_p[len(y_en):]

        temp_pred_file = f"_temp_oof_{name}.csv"

        with open(temp_pred_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["turn_id", "pause_index", "p_eot"])
            for (tid, pi), val in zip(k_en, p_en):
                w.writerow([tid, pi, f"{val:.4f}"])
        r_en = score(os.path.join(args.en_dir, "labels.csv"), temp_pred_file)

        with open(temp_pred_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["turn_id", "pause_index", "p_eot"])
            for (tid, pi), val in zip(k_hi, p_hi):
                w.writerow([tid, pi, f"{val:.4f}"])
        r_hi = score(os.path.join(args.hi_dir, "labels.csv"), temp_pred_file)

        if os.path.exists(temp_pred_file):
            os.remove(temp_pred_file)

        print(f"  [{name:22s}]  EN: delay={r_en['latency']*1000:4.0f}ms (AUC={r_en['auc']:.3f}, cut={r_en['cutoff']*100:.1f}%) | "
              f"HI: delay={r_hi['latency']*1000:4.0f}ms (AUC={r_hi['auc']:.3f}, cut={r_hi['cutoff']*100:.1f}%)")

    # Weighted ensemble from OOF predictions
    print("\n--- Weighted Ensemble (OOF) ---")
    weights = {
        "HistGradientBoosting": 0.28,
        "ExtraTrees": 0.28,
        "GradientBoosting": 0.24,
        "MLPClassifier": 0.10,
        "LogisticRegression": 0.05,
        "RandomForest": 0.05,
    }
    oof_ens = np.zeros(len(y_all), dtype=np.float64)
    for m_name, w in weights.items():
        oof_ens += w * oof_results[m_name]

    p_en_ens = oof_ens[:len(y_en)]
    p_hi_ens = oof_ens[len(y_en):]

    with open("_temp_ens.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), val in zip(k_en, p_en_ens):
            w.writerow([tid, pi, f"{val:.4f}"])
    r_en_ens = score(os.path.join(args.en_dir, "labels.csv"), "_temp_ens.csv")

    with open("_temp_ens.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), val in zip(k_hi, p_hi_ens):
            w.writerow([tid, pi, f"{val:.4f}"])
    r_hi_ens = score(os.path.join(args.hi_dir, "labels.csv"), "_temp_ens.csv")
    if os.path.exists("_temp_ens.csv"):
        os.remove("_temp_ens.csv")

    print(f"  [{'WeightedEnsemble':22s}]  EN: delay={r_en_ens['latency']*1000:4.0f}ms (AUC={r_en_ens['auc']:.3f}, cut={r_en_ens['cutoff']*100:.1f}%) | "
          f"HI: delay={r_hi_ens['latency']*1000:4.0f}ms (AUC={r_hi_ens['auc']:.3f}, cut={r_hi_ens['cutoff']*100:.1f}%)")

    # Fit final VotingClassifier on full dataset
    print("\nFitting final Voting Ensemble on complete dataset (sklearn only)...")
    from sklearn.base import clone

    final_estimators = []
    for name, base_clf in candidate_models.items():
        final_estimators.append((name.lower(), clone(base_clf)))

    final_ensemble = VotingClassifier(
        estimators=final_estimators,
        voting="soft",
        weights=[weights.get(name, 0.1) for name in candidate_models.keys()]
    )
    final_ensemble.fit(X_all, y_all)

    # Save to both target locations
    model_payload = {
        "model": final_ensemble,
        "n_features": X_all.shape[1],
        "feature_names": [f"f_{i}" for i in range(X_all.shape[1])],
        "sklearn_only": True,
    }
    joblib.dump(model_payload, args.out_model, compress=3)
    if os.path.exists("d:/Plivo/eot_handout/starter"):
        joblib.dump(model_payload, "d:/Plivo/eot_handout/starter/eot_model.joblib", compress=3)
    if os.path.exists("d:/Plivo/eot_handout"):
        joblib.dump(model_payload, "d:/Plivo/eot_handout/eot_model.joblib", compress=3)

    print(f"\nSuccessfully saved final model artifact to {args.out_model}")


if __name__ == "__main__":
    main()
