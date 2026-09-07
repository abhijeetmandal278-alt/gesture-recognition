"""
Real-time hand gesture recognition — live deployment demo.

Automatically selects the best model based on cross-session test accuracy
from results/generalization_summary.json (falls back to invariant-features
model if no evaluation results exist).

Features
--------
    • Hand bounding box with confidence-coloured border
    • Predicted gesture label + confidence %
    • Live FPS counter
    • Hand skeleton overlay via MediaPipe

Usage
-----
    python src/live_demo.py

Controls
--------
    q — Quit
"""

import json
import os
import sys
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from features import extract_invariant_features, extract_raw_features


# ──────────────────────────────────────────────
#  Model selection
# ──────────────────────────────────────────────

def select_best_model():
    """Pick the model with the highest cross-session accuracy."""
    summary_path = os.path.join("results", "generalization_summary.json")

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            s = json.load(f)
        acc_raw = s["acc_raw_cross"]
        acc_inv = s["acc_inv_cross"]

        if acc_inv >= acc_raw:
            print(
                f"  Selected: Invariant features model "
                f"(cross-session acc {acc_inv*100:.1f}% "
                f"vs raw {acc_raw*100:.1f}%)"
            )
            return "invariant"
        else:
            print(
                f"  Selected: Raw coordinates model "
                f"(cross-session acc {acc_raw*100:.1f}% "
                f"vs invariant {acc_inv*100:.1f}%)"
            )
            return "raw"
    else:
        print("  No evaluation results found — defaulting to invariant features model.")
        return "invariant"


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def get_hand_bbox(hand_landmarks, frame_shape):
    """Pixel bounding box around detected hand landmarks."""
    h, w = frame_shape[:2]
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]

    pad = 20
    return (
        max(0, int(min(xs)) - pad),
        max(0, int(min(ys)) - pad),
        min(w, int(max(xs)) + pad),
        min(h, int(max(ys)) + pad),
    )


# ──────────────────────────────────────────────
#  Main loop
# ──────────────────────────────────────────────

def main():
    print("\n=== Live Gesture Recognition Demo ===\n")

    model_type = select_best_model()

    if model_type == "invariant":
        model = joblib.load(os.path.join("models", "model_invariant.pkl"))
        extract_fn = extract_invariant_features
    else:
        model = joblib.load(os.path.join("models", "model_raw.pkl"))
        extract_fn = extract_raw_features

    le = joblib.load(os.path.join("models", "label_encoder.pkl"))

    # ── MediaPipe ──
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    # ── Webcam ──
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    print("  Press 'q' to quit.\n")

    prev_time = time.perf_counter()
    fps = 0.0
    alpha = 0.9  # exponential smoothing

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            # Extract features → classify
            lm_flat = []
            for lm in hand.landmark:
                lm_flat.extend([lm.x, lm.y, lm.z])

            feats = extract_fn(lm_flat).reshape(1, -1)
            pred = model.predict(feats)[0]
            proba = model.predict_proba(feats)[0]
            confidence = proba[pred]
            label = le.inverse_transform([pred])[0]

            # Bounding box (colour varies with confidence: green→red)
            x1, y1, x2, y2 = get_hand_bbox(hand, frame.shape)
            box_colour = (
                int((1 - confidence) * 255),
                int(confidence * 255),
                0,
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, 2)

            # Label banner
            text = f"{label} ({confidence * 100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - th - 10),
                (x1 + tw + 4, y1),
                box_colour, -1,
            )
            cv2.putText(
                frame, text, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )

        # FPS counter (exponentially smoothed)
        now = time.perf_counter()
        instant_fps = 1.0 / max(now - prev_time, 1e-9)
        fps = alpha * fps + (1 - alpha) * instant_fps
        prev_time = now

        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
        )

        # Model info footer
        cv2.putText(
            frame, f"Model: {model_type} features", (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Demo ended.")


if __name__ == "__main__":
    main()
