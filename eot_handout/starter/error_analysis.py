"""Error Analysis Script: Find worst prediction errors and diagnose failure patterns.

Runs OOF (out-of-fold) predictions via GroupKFold to get honest errors,
then analyzes the worst false positives (hold predicted as eot) and
false negatives (eot predicted as hold).
"""
import csv
import os
import sys
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.base import clone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_advanced import load_wav, extract_advanced_features, N_FEATURES
from train_model import load_dataset, get_models


def analyze_errors(data_dir, lang_label="EN"):
    """Load data, compute OOF predictions, analyze worst errors."""
    print(f"\n{'='*70}")
    print(f"ERROR ANALYSIS: {lang_label} ({data_dir})")
    print(f"{'='*70}")

    # Load dataset
    labels_path = os.path.join(data_dir, "labels.csv")
    rows = list(csv.DictReader(open(labels_path)))
    
    X, y, groups, keys = [], [], [], []
    pause_meta = []  # store metadata for analysis
    cache = {}
    
    for r in rows:
        audio_path = os.path.join(data_dir, r["audio_file"])
        if audio_path not in cache:
            cache[audio_path] = load_wav(audio_path)
        x, sr = cache[audio_path]
        
        ps = float(r["pause_start"])
        pe = float(r["pause_end"])
        pi = int(r["pause_index"])
        
        feats = extract_advanced_features(x, sr, pause_start=ps, pause_index=pi)
        X.append(feats)
        y.append(1 if r["label"] == "eot" else 0)
        groups.append(r["turn_id"])
        keys.append((r["turn_id"], pi))
        pause_meta.append({
            "turn_id": r["turn_id"],
            "pause_index": pi,
            "pause_start": ps,
            "pause_end": pe,
            "pause_duration": pe - ps,
            "label": r["label"],
            "audio_file": r["audio_file"],
        })
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    groups = np.array(groups)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Get best model (ExtraTrees tends to be strong)
    models = get_models()
    
    # Run OOF with best ensemble
    print("\nComputing OOF predictions with all models...")
    oof_preds = {}
    for name, clf in models.items():
        gkf = GroupKFold(n_splits=5)
        oof = np.zeros(len(y), dtype=np.float64)
        for train_idx, val_idx in gkf.split(X, y, groups):
            m = clone(clf)
            m.fit(X[train_idx], y[train_idx])
            if hasattr(m, "predict_proba"):
                oof[val_idx] = m.predict_proba(X[val_idx])[:, 1]
            else:
                d = m.decision_function(X[val_idx])
                oof[val_idx] = 1.0 / (1.0 + np.exp(-d))
        oof_preds[name] = oof
    
    # Weighted ensemble
    weights = {"HistGradientBoosting": 0.30, "GradientBoosting": 0.25, 
               "ExtraTrees": 0.25, "LogisticRegression": 0.10, "RandomForest": 0.10}
    oof_ens = np.zeros(len(y), dtype=np.float64)
    for name, w in weights.items():
        oof_ens += w * oof_preds[name]
    
    # Compute errors
    errors = oof_ens - y  # positive = FP (hold predicted as eot), negative = FN (eot predicted as hold)
    abs_errors = np.abs(errors)
    
    # Feature names for interpretation
    feat_names = [
        "e_last", "e_mean_03", "e_mean_075", "e_mean_15", "e_std_075", "e_max_15",
        "e_drop_ratio", "e_slope_03", "e_slope_075", "e_delta_mean", "e_delta_std",
        "voiced_ratio_03", "voiced_ratio_075", "voiced_ratio_15",
        "f0_mean_15", "f0_std_15", "f0_range_15",
        "f0_last", "f0_drop_ratio", "f0_slope_075",
        "f0_min_15", "f0_max_15",
        "final_syl_ratio", "speech_rate_15",
        "hnr_075",
        "zcr_mean_03", "zcr_mean_075", "zcr_last", "zcr_slope",
        "sc_mean_03", "sroll_mean_03", "sflux_03", "stilt_03",
        "sc_mean_075", "sroll_mean_075", "sflux_075", "stilt_075",
        "turn_elapsed", "pause_idx", "pause_pos_ratio", "log_turn_elapsed",
    ]
    feat_names += [f"mfcc_mean_{i}" for i in range(10)]
    feat_names += [f"mfcc_std_{i}" for i in range(11)]
    
    # ===== WORST FALSE POSITIVES (hold predicted as eot) =====
    hold_mask = (y == 0)
    hold_indices = np.where(hold_mask)[0]
    hold_scores = oof_ens[hold_mask]
    worst_fp_order = np.argsort(hold_scores)[::-1]  # highest p_eot for hold pauses
    
    print(f"\n--- TOP 15 WORST FALSE POSITIVES (hold pauses with highest p_eot) ---")
    print(f"{'Turn ID':12s} {'PI':3s} {'p_eot':7s} {'PauseStart':10s} {'PauseDur':8s} {'Label':5s}  Key Features")
    for rank, idx in enumerate(worst_fp_order[:15]):
        real_idx = hold_indices[idx]
        m = pause_meta[real_idx]
        p = oof_ens[real_idx]
        
        # Highlight key features for this sample
        x_i = X[real_idx]
        key_feats = []
        for fi in [37, 38, 39, 6, 7, 8, 17, 19, 22, 23, 24]:  # turn_elapsed, pause_idx, pause_pos, e_drop, e_slope, f0_last, f0_slope, final_syl, speech_rate, hnr
            if fi < len(feat_names):
                key_feats.append(f"{feat_names[fi]}={x_i[fi]:.2f}")
        
        print(f"{m['turn_id']:12s} {m['pause_index']:3d} {p:7.4f} {m['pause_start']:10.1f} {m['pause_duration']:8.2f} {m['label']:5s}  {', '.join(key_feats[:5])}")
    
    # ===== WORST FALSE NEGATIVES (eot predicted as hold) =====
    eot_mask = (y == 1)
    eot_indices = np.where(eot_mask)[0]
    eot_scores = oof_ens[eot_mask]
    worst_fn_order = np.argsort(eot_scores)  # lowest p_eot for eot pauses
    
    print(f"\n--- TOP 15 WORST FALSE NEGATIVES (eot pauses with lowest p_eot) ---")
    print(f"{'Turn ID':12s} {'PI':3s} {'p_eot':7s} {'PauseStart':10s} {'PauseDur':8s} {'Label':5s}  Key Features")
    for rank, idx in enumerate(worst_fn_order[:15]):
        real_idx = eot_indices[idx]
        m = pause_meta[real_idx]
        p = oof_ens[real_idx]
        
        x_i = X[real_idx]
        key_feats = []
        for fi in [37, 38, 39, 6, 7, 8, 17, 19, 22, 23, 24]:
            if fi < len(feat_names):
                key_feats.append(f"{feat_names[fi]}={x_i[fi]:.2f}")
        
        print(f"{m['turn_id']:12s} {m['pause_index']:3d} {p:7.4f} {m['pause_start']:10.1f} {m['pause_duration']:8.2f} {m['label']:5s}  {', '.join(key_feats[:5])}")
    
    # ===== PATTERN ANALYSIS =====
    print(f"\n--- PATTERN ANALYSIS ---")
    
    # FP analysis: what makes hold pauses look like eot?
    fp_threshold = 0.5
    fp_mask = hold_mask & (oof_ens >= fp_threshold)
    fn_mask = eot_mask & (oof_ens < fp_threshold)
    correct_hold = hold_mask & (oof_ens < fp_threshold)
    correct_eot = eot_mask & (oof_ens >= fp_threshold)
    
    print(f"\nAt threshold=0.5:")
    print(f"  Correct holds:  {correct_hold.sum()}")
    print(f"  Correct eots:   {correct_eot.sum()}")
    print(f"  False positives: {fp_mask.sum()} (holds misclassified as eot)")
    print(f"  False negatives: {fn_mask.sum()} (eots misclassified as hold)")
    
    # Compare feature distributions: FP vs correct hold
    if fp_mask.sum() > 0 and correct_hold.sum() > 0:
        print(f"\n  Feature comparison: FALSE POSITIVE holds vs CORRECT holds")
        print(f"  {'Feature':22s} {'FP Mean':>10s} {'Correct Mean':>12s} {'Diff':>8s}")
        for fi in range(min(len(feat_names), X.shape[1])):
            fp_mean = np.mean(X[fp_mask, fi])
            ch_mean = np.mean(X[correct_hold, fi])
            diff = fp_mean - ch_mean
            if abs(diff) > 0.1 * (abs(ch_mean) + 1e-6):  # significant difference
                print(f"  {feat_names[fi]:22s} {fp_mean:10.3f} {ch_mean:12.3f} {diff:+8.3f}")
    
    # Compare: FN vs correct eot
    if fn_mask.sum() > 0 and correct_eot.sum() > 0:
        print(f"\n  Feature comparison: FALSE NEGATIVE eots vs CORRECT eots")
        print(f"  {'Feature':22s} {'FN Mean':>10s} {'Correct Mean':>12s} {'Diff':>8s}")
        for fi in range(min(len(feat_names), X.shape[1])):
            fn_mean = np.mean(X[fn_mask, fi])
            ce_mean = np.mean(X[correct_eot, fi])
            diff = fn_mean - ce_mean
            if abs(diff) > 0.1 * (abs(ce_mean) + 1e-6):
                print(f"  {feat_names[fi]:22s} {fn_mean:10.3f} {ce_mean:12.3f} {diff:+8.3f}")
    
    # Pause duration analysis for FPs
    if fp_mask.sum() > 0:
        fp_durations = [pause_meta[i]["pause_duration"] for i in np.where(fp_mask)[0]]
        print(f"\n  FP pause durations: mean={np.mean(fp_durations):.3f}s, min={np.min(fp_durations):.3f}s, max={np.max(fp_durations):.3f}s")
        # How many FPs have long pauses (>1s)?
        long_fp = sum(1 for d in fp_durations if d > 1.0)
        print(f"  FPs with pause > 1.0s: {long_fp}/{len(fp_durations)} — these are LONG hold pauses the model thinks are eot")
    
    return X, y, groups, keys, oof_ens, pause_meta


def main():
    en_dir = "../eot_data/english"
    hi_dir = "../eot_data/hindi"
    
    X_en, y_en, g_en, k_en, oof_en, meta_en = analyze_errors(en_dir, "English")
    X_hi, y_hi, g_hi, k_hi, oof_hi, meta_hi = analyze_errors(hi_dir, "Hindi")
    
    print(f"\n{'='*70}")
    print("SUMMARY OF ERROR PATTERNS")
    print(f"{'='*70}")
    print("\nKey patterns to fix:")
    print("1. Look at which hold pauses have the highest p_eot (false alarms)")
    print("2. Look at which eot pauses have the lowest p_eot (missed endings)")
    print("3. Feature differences reveal what signals are missing or misleading")


if __name__ == "__main__":
    main()
