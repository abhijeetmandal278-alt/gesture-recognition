"""
Generate the 2×2 generalization table.

Evaluates both models (raw coordinates vs. invariant features) on both
test sets (same-session held-out split vs. cross-session independent data).

Output
------
    results/generalization_table.csv   — the 2×2 accuracy table
    results/generalization_summary.json — machine-readable accuracy values
    Console: formatted table + auto-generated comparison summary

Usage
-----
    python src/evaluate.py                                     # defaults
    python src/evaluate.py --train-session 1 --test-session 2
"""

import argparse
import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from features import extract_invariant_features_batch, extract_raw_features_batch


def load_session_data(session_id):
    """Load and concatenate all CSV files for *session_id*."""
    pattern = os.path.join("data", "raw", f"session_{session_id}_*.csv")
    csv_files = sorted(f for f in glob.glob(pattern) if f.endswith(".csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No data found for session {session_id}. "
            f"Run:  python src/collect_data.py --session {session_id}"
        )

    dfs = [pd.read_csv(f) for f in csv_files]
    data = pd.concat(dfs, ignore_index=True)
    print(f"  Session {session_id}: {len(data)} samples from {len(csv_files)} file(s)")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Generate the 2×2 generalization accuracy table"
    )
    parser.add_argument("--train-session", type=int, default=1)
    parser.add_argument("--test-session", type=int, default=2)
    args = parser.parse_args()

    # ── Load training metadata ──
    meta_path = os.path.join("models", "training_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("training_meta.json not found. Run train.py first.")

    with open(meta_path) as f:
        meta = json.load(f)

    # ── Load models ──
    model_raw = joblib.load(os.path.join("models", "model_raw.pkl"))
    model_inv = joblib.load(os.path.join("models", "model_invariant.pkl"))
    le = joblib.load(os.path.join("models", "label_encoder.pkl"))

    # ════════════════════════════════════════════
    #  Same-Session Test  (held-out split from training session)
    # ════════════════════════════════════════════
    print(f"\n=== Same-Session Test (Session {args.train_session} held-out split) ===")
    session1 = load_session_data(args.train_session)
    landmark_cols = [c for c in session1.columns if c not in ("session_id", "label")]
    landmarks1 = session1[landmark_cols].values
    y1 = le.transform(session1["label"].values)

    # Reproduce the exact same split used during training
    _, X_raw_test, _, y_test_same = train_test_split(
        extract_raw_features_batch(landmarks1), y1,
        test_size=meta["test_size"],
        random_state=meta["random_state"],
        stratify=y1,
    )
    _, X_inv_test, _, _ = train_test_split(
        extract_invariant_features_batch(landmarks1), y1,
        test_size=meta["test_size"],
        random_state=meta["random_state"],
        stratify=y1,
    )

    acc_raw_same = float(np.mean(model_raw.predict(X_raw_test) == y_test_same))
    acc_inv_same = float(np.mean(model_inv.predict(X_inv_test) == y_test_same))
    print(f"  Raw coordinates:    {acc_raw_same * 100:.2f}%")
    print(f"  Invariant features: {acc_inv_same * 100:.2f}%")

    # ════════════════════════════════════════════
    #  Cross-Session Test  (entire independent session)
    # ════════════════════════════════════════════
    print(f"\n=== Cross-Session Test (Session {args.test_session}) ===")
    session2 = load_session_data(args.test_session)
    labels2 = session2["label"].values
    landmarks2 = session2[
        [c for c in session2.columns if c not in ("session_id", "label")]
    ].values

    # Filter to known classes only
    known = set(le.classes_)
    mask = np.array([lbl in known for lbl in labels2])
    if not mask.all():
        unknown = set(labels2[~mask])
        print(f"  WARNING: Ignoring unknown labels: {unknown}")
        labels2 = labels2[mask]
        landmarks2 = landmarks2[mask]

    y2 = le.transform(labels2)
    X_raw_cross = extract_raw_features_batch(landmarks2)
    X_inv_cross = extract_invariant_features_batch(landmarks2)

    acc_raw_cross = float(np.mean(model_raw.predict(X_raw_cross) == y2))
    acc_inv_cross = float(np.mean(model_inv.predict(X_inv_cross) == y2))
    print(f"  Raw coordinates:    {acc_raw_cross * 100:.2f}%")
    print(f"  Invariant features: {acc_inv_cross * 100:.2f}%")

    # ════════════════════════════════════════════
    #  Build and save the 2×2 table
    # ════════════════════════════════════════════
    table = pd.DataFrame(
        {
            "Feature Type": ["Raw Coordinates", "Invariant Features"],
            "Same-Session Test (%)": [
                f"{acc_raw_same * 100:.2f}",
                f"{acc_inv_same * 100:.2f}",
            ],
            "Cross-Session Test (%)": [
                f"{acc_raw_cross * 100:.2f}",
                f"{acc_inv_cross * 100:.2f}",
            ],
        }
    )

    os.makedirs("results", exist_ok=True)
    csv_out = os.path.join("results", "generalization_table.csv")
    table.to_csv(csv_out, index=False)

    print(f"\n{'═' * 60}")
    print("  GENERALIZATION TABLE")
    print(f"{'═' * 60}")
    print(table.to_string(index=False))
    print(f"{'═' * 60}")

    # ── Auto-generated comparison text ──
    raw_drop = (acc_raw_same - acc_raw_cross) * 100
    inv_drop = (acc_inv_same - acc_inv_cross) * 100
    inv_advantage = (acc_inv_cross - acc_raw_cross) * 100

    print(f"\n--- Summary ---")
    print(
        f"Raw coordinates:    {acc_raw_same*100:.1f}% → {acc_raw_cross*100:.1f}% "
        f"(drop of {raw_drop:.1f}% under session shift)"
    )
    print(
        f"Invariant features: {acc_inv_same*100:.1f}% → {acc_inv_cross*100:.1f}% "
        f"(drop of {inv_drop:.1f}% under session shift)"
    )

    if inv_advantage > 0:
        print(
            f"\n✓ Invariant features generalize {inv_advantage:.1f}% better than "
            f"raw coordinates under session shift."
        )
    elif inv_advantage < 0:
        print(
            f"\n✗ Raw coordinates unexpectedly generalize {-inv_advantage:.1f}% "
            f"better than invariant features under session shift."
        )
    else:
        print("\n= Both feature types generalize equally under session shift.")

    # ── Persist summary for live_demo.py ──
    summary = {
        "acc_raw_same": acc_raw_same,
        "acc_inv_same": acc_inv_same,
        "acc_raw_cross": acc_raw_cross,
        "acc_inv_cross": acc_inv_cross,
        "raw_drop_pct": raw_drop,
        "inv_drop_pct": inv_drop,
        "inv_advantage_pct": inv_advantage,
    }
    with open(os.path.join("results", "generalization_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {csv_out}")
    print(f"       results/generalization_summary.json")


if __name__ == "__main__":
    main()
