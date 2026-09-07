# Real-Time Hand Gesture Recognition Pipeline

A complete end-to-end pipeline for recognizing hand gestures in real time using webcam input, MediaPipe hand landmark detection, and scikit-learn classifiers. The project compares **raw coordinate features** against **invariant geometric features** to study how feature engineering affects generalization across different recording conditions.

## Project Overview

This project implements:

1. **Data collection** — webcam-based tool for recording hand landmark data across multiple sessions
2. **Feature engineering** — two parallel extraction paths (raw coordinates vs. geometric invariants)
3. **Model training** — two Random Forest classifiers trained on the same data but different features
4. **Generalization evaluation** — a 2×2 accuracy table comparing both models on same-session and cross-session test data
5. **Latency benchmarking** — per-stage timing of the full pipeline to identify bottlenecks
6. **Live deployment** — real-time gesture recognition with bounding box, label, and confidence overlay

## Gesture Classes

| Key | Class        | Description                                  |
| --- | ------------ | -------------------------------------------- |
| `1` | `open_palm`  | All fingers extended, palm facing camera     |
| `2` | `fist`       | All fingers curled into a fist               |
| `3` | `thumbs_up`  | Thumb extended upward, others curled         |
| `4` | `peace_sign` | Index and middle fingers in V shape          |
| `5` | `pointing`   | Index finger extended, others curled         |

## Project Structure

```
gesture-recognition/
├── data/
│   ├── raw/                    # Saved landmark CSVs per session
│   └── README.md               # Data collection protocol & schema
├── src/
│   ├── collect_data.py         # Webcam-based data collection tool
│   ├── features.py             # Raw vs invariant feature extraction
│   ├── train.py                # Trains + saves both model variants
│   ├── evaluate.py             # Generates the 2×2 generalization table
│   ├── benchmark.py            # Latency + FPS benchmarking
│   └── live_demo.py            # Real-time inference demo
├── models/                     # Saved trained models (.pkl)
├── results/
│   ├── generalization_table.csv
│   └── latency_benchmark.csv
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline (in order)

```bash
# Step 1: Collect training data (Session 1)
python src/collect_data.py --session 1 --note "Normal lighting, 50cm, right hand"

# Step 2: Collect independent test data (Session 2) — different conditions!
python src/collect_data.py --session 2 --note "Dim lighting, 30cm, left hand"

# Step 3: Train both models on Session 1
python src/train.py --session 1

# Step 4: Evaluate generalization (produces the 2×2 table)
python src/evaluate.py --train-session 1 --test-session 2

# Step 5: Run latency benchmarks
python src/benchmark.py

# Step 6: Launch live demo
python src/live_demo.py
```

---

## Feature Engineering

All features are extracted from 21 MediaPipe hand landmarks, each providing `(x, y, z)` coordinates.

### Path A — Raw Coordinates (63 features)

The flattened `(x₀, y₀, z₀, x₁, y₁, z₁, …, x₂₀, y₂₀, z₂₀)` values exactly as reported by MediaPipe.

- **Not** scale-invariant: a hand closer to the camera produces larger coordinate values.
- **Not** translation-invariant: hand position in the frame changes all values.

### Path B — Invariant Geometric Features (8 features)

Scale- and translation-invariant features derived from the landmark geometry:

#### Features 1–5: Normalized Fingertip Distances

$$d_i = \frac{\text{dist}(\text{wrist},\ \text{fingertip}_i)}{\text{dist}(\text{wrist},\ \text{middle\_MCP})}$$

where:
- **wrist** = landmark 0
- **fingertip\_i** ∈ {thumb tip (4), index tip (8), middle tip (12), ring tip (16), pinky tip (20)}
- **middle\_MCP** = landmark 9 (scale normalization anchor)
- **dist(A, B)** = √((Bₓ−Aₓ)² + (Bᵧ−Aᵧ)² + (B_z−A_z)²)

Dividing by the wrist-to-middle-MCP distance normalizes for hand size (scale invariance). Using distances relative to the wrist provides translation invariance.

#### Features 6–8: Inter-Finger Angles

$$\theta = \arccos\!\left(\frac{\vec{v_1} \cdot \vec{v_2}}{|\vec{v_1}| \times |\vec{v_2}|}\right)$$

| Feature | Vector 1 (MCP → Tip) | Vector 2 (MCP → Tip) |
| ------- | -------------------- | -------------------- |
| 6       | Index (lm5 → lm8)   | Middle (lm9 → lm12)  |
| 7       | Middle (lm9 → lm12) | Ring (lm13 → lm16)   |
| 8       | Ring (lm13 → lm16)  | Pinky (lm17 → lm20)  |

Each finger vector points from the MCP joint to the fingertip. Angles are inherently scale- and translation-invariant.

---

## Cross-Session Test Protocol

> **⚠️ Critical:** Collecting data under only one condition does not test generalization.

**Session 1 (training)** — collect under normal, comfortable conditions:
- Normal room lighting
- ~50 cm from camera
- Your dominant hand

**Session 2 (independent test)** — collect under **deliberately different** conditions:
- Different lighting (brighter, dimmer, different direction)
- Different distance from camera
- Different hand angle / orientation
- Optionally a different person's hand

Session 2 data is **never** used for training — it exists solely to measure how well each model generalizes beyond training conditions.

---

## Results

### Generalization Table

*(This table is populated after running `evaluate.py` — replace placeholder values with your actual results)*

|                    | Same-Session Test | Cross-Session Test |
| ------------------ | :---------------: | :----------------: |
| Raw Coordinates    |     XX.XX%        |      XX.XX%        |
| Invariant Features |     XX.XX%        |      XX.XX%        |

**Interpretation:** *(to be written after running the evaluation)*

The invariant features model is expected to show a smaller accuracy drop under session shift because its features are normalized for hand size, position, and scale — all of which change between sessions. The raw coordinates model overfits to the specific pixel-space statistics of the training session.

### Latency / FPS Benchmark

*(Populated after running `benchmark.py`)*

| Stage               | Latency (ms/frame) |
| ------------------- | :-----------------: |
| Webcam capture      |       X.XXX         |
| MediaPipe detection |       X.XXX         |
| Feature extraction  |       X.XXX         |
| Classification      |       X.XXX         |
| Overlay rendering   |       X.XXX         |
| **Full pipeline FPS** |     **XX.X**      |

**Bottleneck:** MediaPipe detection is expected to be the dominant cost, as it runs a neural network per frame. Classification (Random Forest) and feature extraction are negligible in comparison.

---

## Recording a Demo Video

1. Run `python src/live_demo.py`
2. Use a screen recording tool (e.g. OBS Studio, Windows Game Bar `Win+G`, or macOS QuickTime)
3. Demonstrate each gesture class with smooth transitions
4. Show the FPS counter, bounding box, and confidence score in action
5. Save the recording and optionally convert to GIF

### Screenshots / GIF

*(Add screenshots or GIF of the live overlay here after running the demo)*

```
[placeholder — capture a screenshot or GIF from live_demo.py]
```

---

## Technical Details

- **Classifier:** Random Forest (100 estimators, scikit-learn)
- **Hand detection:** MediaPipe Hands (21 3D landmarks, single hand)
- **Train/test split:** 80/20 stratified, random_state=42
- **Model selection for live demo:** automatically picks the model with the highest cross-session accuracy from `results/generalization_summary.json`

## Dependencies

| Package        | Version   |
| -------------- | --------- |
| opencv-python  | 4.10.0.84 |
| mediapipe      | 0.10.14   |
| scikit-learn   | 1.5.1     |
| numpy          | 1.26.4    |
| pandas         | 2.2.2     |
| joblib         | 1.4.2     |
