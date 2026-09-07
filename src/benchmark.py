"""
Latency and FPS benchmarking for the gesture recognition pipeline.

Measures (separately):
    1. Classifier inference latency (ms) for both models, averaged over 100 calls
    2. Full pipeline FPS with per-stage breakdown over 200+ live frames
    3. Identifies and prints the bottleneck stage

Usage
-----
    python src/benchmark.py
"""

import os
import sys
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from features import extract_invariant_features, extract_raw_features

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────
N_INFERENCE_ITERS = 100
N_PIPELINE_FRAMES = 200


# ──────────────────────────────────────────────
#  Part 1: Classifier inference latency
# ──────────────────────────────────────────────

def benchmark_classifier_latency():
    """Measure model.predict() latency for both models."""
    print("=" * 55)
    print("  Part 1: Classifier Inference Latency")
    print("=" * 55)

    model_raw = joblib.load(os.path.join("models", "model_raw.pkl"))
    model_inv = joblib.load(os.path.join("models", "model_invariant.pkl"))

    # Synthetic single-sample inputs
    dummy_raw = np.random.rand(1, 63)
    dummy_inv = np.random.rand(1, 8)

    # Warm-up
    for _ in range(10):
        model_raw.predict(dummy_raw)
        model_inv.predict(dummy_inv)

    # Benchmark Model A (raw)
    times_raw = []
    for _ in range(N_INFERENCE_ITERS):
        t0 = time.perf_counter()
        model_raw.predict(dummy_raw)
        t1 = time.perf_counter()
        times_raw.append((t1 - t0) * 1000)

    # Benchmark Model B (invariant)
    times_inv = []
    for _ in range(N_INFERENCE_ITERS):
        t0 = time.perf_counter()
        model_inv.predict(dummy_inv)
        t1 = time.perf_counter()
        times_inv.append((t1 - t0) * 1000)

    avg_raw = np.mean(times_raw)
    std_raw = np.std(times_raw)
    avg_inv = np.mean(times_inv)
    std_inv = np.std(times_inv)

    print(f"\n  Model A (Raw, 63 features):      {avg_raw:.3f} ± {std_raw:.3f} ms")
    print(f"  Model B (Invariant, 8 features):  {avg_inv:.3f} ± {std_inv:.3f} ms")
    print(f"  Iterations: {N_INFERENCE_ITERS}\n")

    return {
        "raw_latency_ms": avg_raw,
        "raw_latency_std_ms": std_raw,
        "inv_latency_ms": avg_inv,
        "inv_latency_std_ms": std_inv,
    }


# ──────────────────────────────────────────────
#  Part 2: Full pipeline FPS
# ──────────────────────────────────────────────

def benchmark_pipeline_fps():
    """Measure end-to-end FPS with per-stage timing."""
    print("=" * 55)
    print(f"  Part 2: Full Pipeline FPS ({N_PIPELINE_FRAMES} frames)")
    print("=" * 55)

    model_inv = joblib.load(os.path.join("models", "model_invariant.pkl"))
    le = joblib.load(os.path.join("models", "label_encoder.pkl"))

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\n  ERROR: Cannot open webcam. Skipping pipeline benchmark.\n")
        return None

    times_capture = []
    times_mediapipe = []
    times_features = []
    times_classify = []
    times_render = []

    frame_count = 0
    pipeline_start = time.perf_counter()
    print(f"\n  Running live pipeline for {N_PIPELINE_FRAMES} frames …")

    while frame_count < N_PIPELINE_FRAMES:
        # Stage 1: Webcam capture
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t1 = time.perf_counter()
        if not ret:
            continue
        times_capture.append((t1 - t0) * 1000)

        frame = cv2.flip(frame, 1)

        # Stage 2: MediaPipe landmark detection
        t2 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        t3 = time.perf_counter()
        times_mediapipe.append((t3 - t2) * 1000)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            lm_flat = []
            for lm in hand.landmark:
                lm_flat.extend([lm.x, lm.y, lm.z])

            # Stage 3: Feature extraction
            t4 = time.perf_counter()
            feats = extract_invariant_features(lm_flat).reshape(1, -1)
            t5 = time.perf_counter()
            times_features.append((t5 - t4) * 1000)

            # Stage 4: Classification
            t6 = time.perf_counter()
            pred = model_inv.predict(feats)
            t7 = time.perf_counter()
            times_classify.append((t7 - t6) * 1000)

            label = le.inverse_transform(pred)[0]

            # Stage 5: Overlay rendering
            t8 = time.perf_counter()
            cv2.putText(
                frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
            )
            t9 = time.perf_counter()
            times_render.append((t9 - t8) * 1000)
        else:
            times_features.append(0.0)
            times_classify.append(0.0)
            times_render.append(0.0)

        cv2.imshow("Benchmark", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    pipeline_end = time.perf_counter()
    total_time = pipeline_end - pipeline_start
    fps = frame_count / total_time if total_time > 0 else 0

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # Per-stage averages
    stages = {
        "Webcam capture": np.mean(times_capture) if times_capture else 0,
        "MediaPipe detection": np.mean(times_mediapipe) if times_mediapipe else 0,
        "Feature extraction": np.mean(times_features) if times_features else 0,
        "Classification": np.mean(times_classify) if times_classify else 0,
        "Overlay rendering": np.mean(times_render) if times_render else 0,
    }
    bottleneck = max(stages, key=stages.get)

    print(f"\n  Per-stage average latency:")
    for stage, ms in stages.items():
        tag = " ← BOTTLENECK" if stage == bottleneck else ""
        print(f"    {stage:.<30s} {ms:7.3f} ms/frame{tag}")
    print(f"\n  Total frames : {frame_count}")
    print(f"  Total time   : {total_time:.2f} s")
    print(f"  Pipeline FPS : {fps:.1f}")
    print(
        f"\n  Bottleneck: {bottleneck} "
        f"({stages[bottleneck]:.3f} ms/frame)\n"
    )

    return {
        "fps": fps,
        "total_frames": frame_count,
        "total_time_s": total_time,
        "avg_capture_ms": stages["Webcam capture"],
        "avg_mediapipe_ms": stages["MediaPipe detection"],
        "avg_features_ms": stages["Feature extraction"],
        "avg_classify_ms": stages["Classification"],
        "avg_render_ms": stages["Overlay rendering"],
        "bottleneck": bottleneck,
    }


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    clf_results = benchmark_classifier_latency()
    pipe_results = benchmark_pipeline_fps()

    # ── Save to CSV ──
    os.makedirs("results", exist_ok=True)

    rows = [
        {
            "Metric": "Classifier latency — Raw (ms)",
            "Value": f"{clf_results['raw_latency_ms']:.3f}",
            "StdDev": f"{clf_results['raw_latency_std_ms']:.3f}",
            "Iterations": N_INFERENCE_ITERS,
        },
        {
            "Metric": "Classifier latency — Invariant (ms)",
            "Value": f"{clf_results['inv_latency_ms']:.3f}",
            "StdDev": f"{clf_results['inv_latency_std_ms']:.3f}",
            "Iterations": N_INFERENCE_ITERS,
        },
    ]

    if pipe_results:
        rows.extend(
            [
                {
                    "Metric": "Full pipeline FPS",
                    "Value": f"{pipe_results['fps']:.1f}",
                    "StdDev": "",
                    "Iterations": pipe_results["total_frames"],
                },
                {
                    "Metric": f"Bottleneck: {pipe_results['bottleneck']}",
                    "Value": f"{pipe_results['avg_mediapipe_ms']:.3f} ms",
                    "StdDev": "",
                    "Iterations": "",
                },
                {
                    "Metric": "Webcam capture (ms/frame)",
                    "Value": f"{pipe_results['avg_capture_ms']:.3f}",
                    "StdDev": "",
                    "Iterations": "",
                },
                {
                    "Metric": "MediaPipe detection (ms/frame)",
                    "Value": f"{pipe_results['avg_mediapipe_ms']:.3f}",
                    "StdDev": "",
                    "Iterations": "",
                },
                {
                    "Metric": "Feature extraction (ms/frame)",
                    "Value": f"{pipe_results['avg_features_ms']:.3f}",
                    "StdDev": "",
                    "Iterations": "",
                },
                {
                    "Metric": "Classification (ms/frame)",
                    "Value": f"{pipe_results['avg_classify_ms']:.3f}",
                    "StdDev": "",
                    "Iterations": "",
                },
                {
                    "Metric": "Overlay rendering (ms/frame)",
                    "Value": f"{pipe_results['avg_render_ms']:.3f}",
                    "StdDev": "",
                    "Iterations": "",
                },
            ]
        )

    out_path = os.path.join("results", "latency_benchmark.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
