# Data Directory

This directory stores hand landmark data collected via `src/collect_data.py`.

## Directory Structure

```
data/
├── raw/                          # All collected session data
│   ├── session_1_*.csv           # Session 1 (training data)
│   ├── session_1_*_meta.json     # Session 1 metadata
│   ├── session_2_*.csv           # Session 2 (independent test data)
│   └── session_2_*_meta.json     # Session 2 metadata
└── README.md                     # This file
```

## CSV Schema

Each CSV file contains one row per captured frame with the following columns:

| Column       | Type    | Description                                        |
| ------------ | ------- | -------------------------------------------------- |
| `session_id` | int     | Session identifier (1 = training, 2 = test)        |
| `label`      | string  | Gesture class name (e.g., `open_palm`, `fist`)     |
| `x0`…`z0`   | float   | Wrist landmark (landmark 0) coordinates            |
| `x1`…`z1`   | float   | Landmark 1 coordinates                             |
| …            | …       | …                                                  |
| `x20`…`z20` | float   | Pinky tip (landmark 20) coordinates                |

**Total columns:** 2 metadata + 63 landmark values = 65 columns per row.

Coordinates are MediaPipe normalized values:
- `x`, `y`: normalized to [0, 1] relative to image width/height
- `z`: depth relative to wrist, roughly same scale as `x`

## Metadata JSON

Each CSV has a companion `_meta.json` file recording:
- `session_id`: Integer session identifier
- `timestamp`: Collection timestamp
- `note`: Free-text metadata (lighting conditions, distance, hand orientation, etc.)
- `gesture_classes`: List of gesture class names

## Data Collection Protocol

### Session 1 — Training Data
Collect under your "default" conditions:
- Normal room lighting
- Comfortable webcam distance (~50 cm)
- Your dominant hand
- Aim for **150–200 samples per gesture class**

### Session 2 — Independent Test Data
Collect under **deliberately different** conditions to test generalization:
- Different lighting (brighter, dimmer, or different angle of light)
- Different distance from camera (closer or farther)
- Different hand angle or orientation
- Optionally a different person's hand

> **⚠️ IMPORTANT:** Session 2 data must NEVER be used for training.
> It exists solely to measure how well models generalize beyond training conditions.

## Gesture Classes

| Key | Gesture Class | Description                          |
| --- | ------------- | ------------------------------------ |
| `1` | `open_palm`   | All fingers extended, palm facing camera |
| `2` | `fist`        | All fingers curled into a fist       |
| `3` | `thumbs_up`   | Thumb extended upward, other fingers curled |
| `4` | `peace_sign`  | Index and middle fingers in V shape  |
| `5` | `pointing`    | Index finger extended, others curled |
