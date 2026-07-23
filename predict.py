"""Production predict.py script for End-of-Turn (EoT) prediction.

Runs as:
    python predict.py --data_dir <folder> --out predictions.csv

Outputs CSV with columns: turn_id,pause_index,p_eot
Enforces strict causality constraint (only audio up to pause_start used).

LIBRARY COMPLIANCE: numpy, scipy, scikit-learn, joblib only.
No LightGBM, CatBoost, or pretrained models.
"""
import argparse
import csv
import os
import sys
import warnings
import joblib
import numpy as np

# Suppress non-critical user warnings during CLI prediction
warnings.filterwarnings("ignore")

# Add current script directory and parent/starter directories to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, "starter"))
sys.path.insert(0, os.path.join(script_dir, "eot_handout", "starter"))

# Statically resolvable import (clears IDE linter red underlines)
try:
    from eot_handout.starter.features_advanced import load_wav, extract_advanced_features
except ImportError:
    from features_advanced import load_wav, extract_advanced_features


def find_model_file(model_path):
    """Find model artifact across potential search directories."""
    candidate_paths = [
        model_path,
        os.path.join(script_dir, model_path),
        os.path.join(script_dir, "starter", model_path),
        os.path.join(script_dir, "eot_handout", "starter", model_path),
        os.path.join(os.getcwd(), model_path),
        os.path.join(os.path.dirname(script_dir), model_path),
    ]
    for path in candidate_paths:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not locate model artifact '{model_path}'. Checked: {candidate_paths}")


def predict_folder(data_dir, out_csv, model_path="eot_model.joblib"):
    """Load model artifact, extract features strictly before pause_start, write predictions."""
    resolved_model_path = find_model_file(model_path)
    payload = joblib.load(resolved_model_path)

    # Handle dict payload vs raw estimator model
    if isinstance(payload, dict):
        model = payload["model"]
    else:
        model = payload

    labels_csv = os.path.join(data_dir, "labels.csv")
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"labels.csv not found in {data_dir}")

    rows = list(csv.DictReader(open(labels_csv)))
    cache = {}

    X = []
    keys = []
    for r in rows:
        audio_rel_path = os.path.normpath(r["audio_file"])
        audio_path = os.path.normpath(os.path.join(data_dir, audio_rel_path))

        if audio_path not in cache:
            cache[audio_path] = load_wav(audio_path)
        x, sr = cache[audio_path]

        pause_start = float(r["pause_start"])
        pause_index = int(r["pause_index"])

        feats = extract_advanced_features(x, sr, pause_start=pause_start, pause_index=pause_index)
        X.append(feats)
        keys.append((r["turn_id"], pause_index))

    X_arr = np.array(X, dtype=np.float32)
    # NaN/Inf safety
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_arr)[:, 1]
    else:
        probs = model.decision_function(X_arr)
        probs = 1.0 / (1.0 + np.exp(-probs))

    # Ensure directory of out_csv exists
    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), p in zip(keys, probs):
            w.writerow([tid, pi, f"{p:.4f}"])

    print(f"Successfully wrote {len(keys)} predictions to {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="Predict End-of-Turn probabilities.")
    ap.add_argument("--data_dir", required=True, help="Path to input data folder containing audio/ and labels.csv")
    ap.add_argument("--out", default="predictions.csv", help="Output path for predictions CSV")
    ap.add_argument("--model", default="eot_model.joblib", help="Path to saved model artifact")
    args = ap.parse_args()

    predict_folder(args.data_dir, args.out, model_path=args.model)


if __name__ == "__main__":
    main()
