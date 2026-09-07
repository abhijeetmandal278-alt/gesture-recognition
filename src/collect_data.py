"""
Webcam-based hand gesture data collection tool.

Uses MediaPipe Hands to detect 21 hand landmarks (x, y, z) per frame.
Supports 5 gesture classes. Each recording burst lasts 5 seconds.
Sessions are tagged with an ID and free-text metadata for reproducibility.

Usage
-----
    python src/collect_data.py --session 1 --note "Normal lighting, 50cm distance"
    python src/collect_data.py --session 2 --note "Dim lighting, 30cm distance, left hand"

Keys
----
    1  Record Open Palm        4  Record Peace Sign
    2  Record Fist             5  Record Pointing
    3  Record Thumbs Up        q  Quit
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────

GESTURE_CLASSES = {
    ord("1"): "open_palm",
    ord("2"): "fist",
    ord("3"): "thumbs_up",
    ord("4"): "peace_sign",
    ord("5"): "pointing",
}

RECORD_DURATION = 5.0  # seconds per recording burst


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect hand gesture landmark data from webcam"
    )
    parser.add_argument(
        "--session", type=int, required=True,
        help="Session ID (use 1 for training, 2 for independent test)",
    )
    parser.add_argument(
        "--note", type=str, default="",
        help="Free-text metadata: lighting, distance, hand, angle, etc.",
    )
    args = parser.parse_args()

    session_id = args.session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Output paths ──
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    csv_path = os.path.join(raw_dir, f"session_{session_id}_{timestamp}.csv")
    meta_path = os.path.join(raw_dir, f"session_{session_id}_{timestamp}_meta.json")

    # Save session metadata
    meta = {
        "session_id": session_id,
        "timestamp": timestamp,
        "note": args.note,
        "gesture_classes": list(GESTURE_CLASSES.values()),
        "record_duration_s": RECORD_DURATION,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # ── CSV header: session_id, label, x0, y0, z0, …, x20, y20, z20 ──
    header = ["session_id", "label"]
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])

    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(header)

    # ── MediaPipe Hands ──
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
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

    recording = False
    current_gesture = None
    record_start = 0.0
    sample_counts = {g: 0 for g in GESTURE_CLASSES.values()}
    total_samples = 0

    print(f"\n{'='*50}")
    print(f"  Gesture Data Collection — Session {session_id}")
    print(f"{'='*50}")
    print(f"  Note : {args.note or '(none)'}")
    print(f"  File : {csv_path}")
    print(f"\n  Press a key to record for {RECORD_DURATION}s:")
    for key_code, name in GESTURE_CLASSES.items():
        print(f"    {chr(key_code)} — {name}")
    print(f"    q — Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS
                )

        # Auto-stop after RECORD_DURATION
        if recording and (time.time() - record_start) >= RECORD_DURATION:
            recording = False
            print(
                f"    ✓ Done recording '{current_gesture}' — "
                f"{sample_counts[current_gesture]} samples total"
            )
            current_gesture = None

        # Save landmarks while recording
        if recording and results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            row = [session_id, current_gesture]
            for lm in hand.landmark:
                row.extend([lm.x, lm.y, lm.z])
            writer.writerow(row)
            sample_counts[current_gesture] += 1
            total_samples += 1

        # ── On-screen status ──
        if recording:
            elapsed = time.time() - record_start
            remaining = max(0, RECORD_DURATION - elapsed)
            cv2.putText(
                frame,
                f"RECORDING: {current_gesture} ({remaining:.1f}s)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
            )
        else:
            cv2.putText(
                frame, "Ready — press 1-5 to record", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
            )

        # Per-class counts
        y_off = 65
        for gesture, count in sample_counts.items():
            cv2.putText(
                frame, f"{gesture}: {count}", (10, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            )
            y_off += 22

        cv2.imshow("Gesture Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in GESTURE_CLASSES and not recording:
            current_gesture = GESTURE_CLASSES[key]
            recording = True
            record_start = time.time()
            print(f"    ● Recording '{current_gesture}' …")

    # ── Cleanup ──
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    print(f"\n{'='*50}")
    print(f"  Collection complete — {total_samples} samples")
    print(f"{'='*50}")
    for gesture, count in sample_counts.items():
        print(f"    {gesture}: {count}")
    print(f"  Saved to {csv_path}\n")


if __name__ == "__main__":
    main()
