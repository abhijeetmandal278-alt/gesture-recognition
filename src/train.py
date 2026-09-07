"""
Train gesture recognition models.

Trains two parallel Random Forest classifiers on Session 1 data:
    Model A — raw 63-coordinate features
    Model B — 8 invariant geometric features

Both use an 80/20 stratified train/test split with a fixed random seed
so evaluate.py can reproduce the exact same split.

Usage
-----
    python src/train.py                 # defaults to --session 1
    python src/train.py --session 1
"""

import argparse
import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from features import (
    extract_invariant_features_batch,
    extract_raw_features_batch,
)

# ──────────────────────────────────────────────
#  Constants (shared with evaluate.py via training_meta.json)
# ──────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_session_data(session_id):
    """Load and concatenate all CSV files for *session_id*."""
    pattern = os.path.join("data", "raw", f"session_{session_id}_*.csv")
    csv_files = sorted(f for f in glob.glob(pattern) if f.endswith(".csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found for session {session_id}. "
            f"Run:  python src/collect_data.py --session {session_id}"
        )

    dfs = []
    for path in csv_files:
        df = pd.read_csv(path)
        print(f"  Loaded {path}: {len(df)} samples")
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(data)} samples")
    return data


def main():
    parser = argparse.ArgumentParser(description="Train gesture classifiers")
    parser.add_argument(
        "--session", type=int, default=1,
        help="Training session ID (default: 1)",
    )
    args = parser.parse_args()

    # ── Load data ──
    print(f"\n=== Loading Session {args.session} data ===")
    data = load_session_data(args.session)

    labels = data["label"].values
    landmark_cols = [c for c in data.columns if c not in ("session_id", "label")]
    landmarks = data[landmark_cols].values

    # ── Encode labels ──
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f"\nClasses ({len(le.classes_)}):")
    for cls in le.classes_:
        print(f"  {cls}: {np.sum(labels == cls)} samples")

    # ── Feature extraction ──
    print("\nExtracting features …")
    X_raw = extract_raw_features_batch(landmarks)
    X_inv = extract_invariant_features_batch(landmarks)
    print(f"  Raw features shape:      {X_raw.shape}")
    print(f"  Invariant features shape: {X_inv.shape}")

    # ── Train / test split (stratified, fixed seed) ──
    X_raw_train, X_raw_test, y_train, y_test = train_test_split(
        X_raw, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_inv_train, X_inv_test, _, _ = train_test_split(
        X_inv, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\nSplit: {len(y_train)} train / {len(y_test)} test")

    # ── Model A: Raw coordinates (63 features) ──
    print("\n--- Model A  (Raw Coordinates, 63 features) ---")
    model_raw = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1,
    )
    model_raw.fit(X_raw_train, y_train)
    acc_raw = model_raw.score(X_raw_test, y_test)
    print(f"  Same-session test accuracy: {acc_raw * 100:.2f}%")

    # ── Model B: Invariant features (8 features) ──
    print("\n--- Model B  (Invariant Features, 8 features) ---")
    model_inv = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1,
    )
    model_inv.fit(X_inv_train, y_train)
    acc_inv = model_inv.score(X_inv_test, y_test)
    print(f"  Same-session test accuracy: {acc_inv * 100:.2f}%")

    # ── Save artefacts ──
    os.makedirs("models", exist_ok=True)
    joblib.dump(model_raw, os.path.join("models", "model_raw.pkl"))
    joblib.dump(model_inv, os.path.join("models", "model_invariant.pkl"))
    joblib.dump(le, os.path.join("models", "label_encoder.pkl"))

    meta = {
        "train_session": args.session,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "classes": list(le.classes_),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "same_session_acc_raw": float(acc_raw),
        "same_session_acc_inv": float(acc_inv),
    }
    with open(os.path.join("models", "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved to models/:")
    print("  model_raw.pkl")
    print("  model_invariant.pkl")
    print("  label_encoder.pkl")
    print("  training_meta.json")


if __name__ == "__main__":
    main()
