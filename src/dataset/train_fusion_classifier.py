"""
train_fusion_classifier.py
-----------------------------
Trains the CNN + Object-Detector late-fusion classifier for attack
detection. This is a genuinely trained meta-classifier (not a hand-tuned
rule), so its output confidence is a real calibrated probability, suitable
to feed directly into a Bayesian trust engine.

REQUIRES TWO INPUT FILES, JOINED ON `path`:
  1. --detection_features: output of extract_detection_features.py
     (num_detections, mean_confidence, max_confidence, std_confidence,
     low_conf_count, is_attacked)
  2. --cnn_probs: a CSV with columns: path, prob_clean, prob_<variation1>,
     prob_<variation2>, ... (your CNN's per-class softmax output for every
     frame in the same set). THIS FILE DOES NOT EXIST YET - it needs to be
     produced by running your trained CNN's inference over the same image
     set. See the bottom of this file for the exact script structure
     needed once your cnn.py / checkpoint loading code is available.

Architecture: Logistic Regression over [CNN class probabilities +
detection-side features]. Chosen deliberately over a deeper meta-model:
with only ~10-15 input features and a binary target, logistic regression
is well-calibrated out of the box (predict_proba is a genuine probability,
not just a relative score), avoids overfitting on a small feature set, and
is directly interpretable (coefficient = how much each signal contributes)
- useful for a paper figure explaining WHY the fusion improves over either
signal alone.

Usage (once cnn_probs.csv exists):
    python train_fusion_classifier.py \
        --detection_features ../../data/final_dataset/detection_features_ssd.csv \
        --cnn_probs ../../data/final_dataset/cnn_probs.csv \
        --out_dir ../../data/final_dataset/fusion_results \
        --out_model ../../data/final_dataset/fusion_classifier.joblib
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler


DETECTION_FEATURE_COLS = [
    "num_detections", "mean_confidence", "max_confidence",
    "std_confidence", "low_conf_count",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detection_features", required=True)
    ap.add_argument("--cnn_probs", required=True)
    ap.add_argument("--out_dir", default="fusion_results")
    ap.add_argument("--out_model", default="fusion_classifier.joblib")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    det_df = pd.read_csv(args.detection_features)
    cnn_df = pd.read_csv(args.cnn_probs)

    merged = det_df.merge(cnn_df, on="path", how="inner", suffixes=("", "_cnn"))
    print(f"Merged {len(merged)} rows (detection features: {len(det_df)}, CNN probs: {len(cnn_df)})")
    if len(merged) < len(det_df) * 0.9:
        print(f"[WARN] significant row loss after merge - check that 'path' values "
              f"match exactly between the two input files (same separators, same split filter).")

    cnn_prob_cols = [c for c in cnn_df.columns if c.startswith("prob_")]
    if not cnn_prob_cols:
        raise ValueError("cnn_probs.csv has no columns starting with 'prob_' - "
                          "expected e.g. prob_clean, prob_freq_low_wide, ...")

    feature_cols = DETECTION_FEATURE_COLS + cnn_prob_cols
    X = merged[feature_cols].fillna(0.0).values
    y = merged["is_attacked"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:, 1]  # P(attacked) - this is the "confidence of sureness"

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print("\n=== Fusion Classifier Test Performance ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n" + classification_report(y_test, y_pred, target_names=["clean", "attacked"]))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))

    # feature importance - which signal (CNN vs detector) actually drives the
    # fusion decision, worth a sentence/figure in the paper
    coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": clf.coef_[0]})
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
    print("\n--- Feature importance (standardized logistic regression coefficients) ---")
    print(coef_df[["feature", "coefficient"]].to_string(index=False))

    pd.DataFrame([metrics]).to_csv(os.path.join(args.out_dir, "fusion_test_metrics.csv"), index=False)
    coef_df.to_csv(os.path.join(args.out_dir, "fusion_feature_importance.csv"), index=False)

    joblib.dump({"model": clf, "scaler": scaler, "feature_cols": feature_cols}, args.out_model)
    print(f"\nTrained fusion model saved to {args.out_model}")
    print("\nInference interface for the safety mechanism:")
    print("  bundle = joblib.load(out_model)")
    print("  x = scaler.transform([[<features in feature_cols order>]])")
    print("  p_attacked = bundle['model'].predict_proba(x)[0, 1]")
    print("  -> feed p_attacked as the per-frame confidence into trust_engine.py")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# WHAT STILL NEEDS TO BE BUILT: cnn_probs.csv
#
# This script cannot produce cnn_probs.csv itself - it needs your actual
# trained CNN's model-loading code (the ResNet18 ensemble mentioned in your
# project notes). Once you share cnn.py (or wherever your model class and
# checkpoint-loading logic live, same as was needed for grad_cam.py earlier),
# the inference script is a short loop:
#
#   for each image in the same label set used by extract_detection_features.py:
#       load + preprocess image (must match YOUR CNN's training preprocessing
#       exactly - resize, normalization)
#       probs = softmax(model(image))
#       write {path, prob_clean, prob_<variation_1>, ...} to cnn_probs.csv
#
# Share that file and this gets filled in precisely rather than guessed.
# ---------------------------------------------------------------------------
