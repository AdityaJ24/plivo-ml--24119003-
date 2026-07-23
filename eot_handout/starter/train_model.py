"""Regularized Model Training & Evaluation Script for End-of-Turn (EoT) Detection.

Features extracted via features_advanced.py (strictly causality compliant, 42 features, <=3400Hz telephony bounded).
Uses ONLY scikit-learn classifiers (no LightGBM, CatBoost, or external libraries).
Evaluates with GroupKFold cross-validation (grouped by turn_id).
Includes probability calibration (CalibratedClassifierCV) to hit exact 5% false-cutoff budget.
Saves trained model to `eot_model.joblib`.

LIBRARY COMPLIANCE: numpy, scipy, scikit-learn, joblib only.
"""
import argparse
import csv
import os
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.base import clone

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
    """Return dictionary of regularized candidate models — scikit-learn only."""
    models = {}

    # Primary Model: Regularized GradientBoostingClassifier (shallow max_depth=3, min_samples_leaf=8, subsample=0.85)
    models["GradientBoosting_Reg"] = GradientBoostingClassifier(
        n_estimators=160, max_depth=3, learning_rate=0.03,
        subsample=0.85, min_samples_leaf=8, random_state=42
    )

    # Calibrated GradientBoosting (Platt sigmoid calibration)
    models["Calibrated_GB"] = CalibratedClassifierCV(
        estimator=GradientBoostingClassifier(
            n_estimators=160, max_depth=3, learning_rate=0.03,
            subsample=0.85, min_samples_leaf=8, random_state=42
        ),
        method="sigmoid", cv=3
    )

    # Regularized HistGradientBoostingClassifier
    models["HistGradientBoosting_Reg"] = HistGradientBoostingClassifier(
        max_iter=150, max_depth=3, learning_rate=0.03,
        min_samples_leaf=10, l2_regularization=1.0, random_state=42
    )

    return models


def evaluate_cv(X, y, groups, model_name, model, n_splits=5):
    """Run GroupKFold cross-validation and compute out-of-fold predictions."""
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

    print("=" * 75)
    print("EoT Regularized Training — 42 Streamlined Features (Telephony Bounded)")
    print("=" * 75)

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
    best_overall_score = float("inf")
    best_model_name = None
    best_model_obj = None

    print(f"{'Model Name':28s} | {'EN OOF Delay':12s} {'EN AUC':8s} {'EN Cut':7s} | {'HI OOF Delay':12s} {'HI AUC':8s} {'HI Cut':7s} | {'In-Sample AUC':13s}")
    print("-" * 108)

    for name, clf in candidate_models.items():
        oof_p = evaluate_cv(X_all, y_all, g_all, name, clf)

        p_en = oof_p[:len(y_en)]
        p_hi = oof_p[len(y_en):]

        temp_pred_file = f"_temp_oof_{name}.csv"

        # EN OOF
        with open(temp_pred_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["turn_id", "pause_index", "p_eot"])
            for (tid, pi), val in zip(k_en, p_en):
                w.writerow([tid, pi, f"{val:.4f}"])
        r_en = score(os.path.join(args.en_dir, "labels.csv"), temp_pred_file)

        # HI OOF
        with open(temp_pred_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["turn_id", "pause_index", "p_eot"])
            for (tid, pi), val in zip(k_hi, p_hi):
                w.writerow([tid, pi, f"{val:.4f}"])
        r_hi = score(os.path.join(args.hi_dir, "labels.csv"), temp_pred_file)

        if os.path.exists(temp_pred_file):
            os.remove(temp_pred_file)

        # Measure In-Sample AUC to check generalization gap
        m_full = clone(clf)
        m_full.fit(X_all, y_all)
        p_in = m_full.predict_proba(X_all)[:, 1] if hasattr(m_full, "predict_proba") else m_full.decision_function(X_all)
        from sklearn.metrics import roc_auc_score
        in_sample_auc = roc_auc_score(y_all, p_in)

        # Priority metric: Hindi OOF delay + English OOF delay
        combined_delay = r_en['latency'] * 1000 + r_hi['latency'] * 1000

        print(f"{name:28s} | {r_en['latency']*1000:6.0f} ms     {r_en['auc']:6.3f}   {r_en['cutoff']*100:4.1f}%   | "
              f"{r_hi['latency']*1000:6.0f} ms     {r_hi['auc']:6.3f}   {r_hi['cutoff']*100:4.1f}%   | {in_sample_auc:13.3f}")

        if combined_delay < best_overall_score or best_model_obj is None:
            best_overall_score = combined_delay
            best_model_name = name
            best_model_obj = m_full

    print("\n" + "=" * 75)
    print(f"Selected Primary Model: {best_model_name}")
    print("=" * 75)

    # Save final model payload
    model_payload = {
        "model": best_model_obj,
        "n_features": X_all.shape[1],
        "feature_names": [f"f_{i}" for i in range(X_all.shape[1])],
        "sklearn_only": True,
    }
    joblib.dump(model_payload, args.out_model, compress=3)
    if os.path.exists("d:/Plivo/eot_handout/starter"):
        joblib.dump(model_payload, "d:/Plivo/eot_handout/starter/eot_model.joblib", compress=3)
    if os.path.exists("d:/Plivo/eot_handout"):
        joblib.dump(model_payload, "d:/Plivo/eot_handout/eot_model.joblib", compress=3)

    print(f"Successfully saved regularized model artifact to {args.out_model}\n")


if __name__ == "__main__":
    main()
