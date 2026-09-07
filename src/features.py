"""
Feature extraction for hand gesture recognition.

Two parallel feature extraction paths from 21 MediaPipe hand landmarks:

Path A — Raw Coordinates (63 features):
    Flattened (x, y, z) of all 21 landmarks, as-is.

Path B — Invariant Geometric Features (8 features):
    Scale- and translation-invariant features derived from landmark geometry.

    Features 1–5: Normalized fingertip distances
    ─────────────────────────────────────────────
        d_i = dist(wrist, fingertip_i) / dist(wrist, middle_MCP)

        where:
            wrist       = landmark 0
            fingertip_i ∈ {thumb_tip(4), index_tip(8), middle_tip(12),
                           ring_tip(16), pinky_tip(20)}
            middle_MCP  = landmark 9  (scale normalization anchor)

            dist(A, B) = sqrt( (Bx−Ax)² + (By−Ay)² + (Bz−Az)² )

        Dividing by dist(wrist, middle_MCP) normalizes for hand size
        (scale invariance). Using distances relative to the wrist
        provides translation invariance.

    Features 6–8: Inter-finger angles (radians)
    ─────────────────────────────────────────────
        angle(v1, v2) = arccos( (v1 · v2) / (|v1| × |v2|) )

        Feature 6: angle between index  vector (lm5→lm8)  and middle vector (lm9→lm12)
        Feature 7: angle between middle vector (lm9→lm12) and ring   vector (lm13→lm16)
        Feature 8: angle between ring   vector (lm13→lm16) and pinky  vector (lm17→lm20)

        Each finger vector goes from the finger's MCP joint to its tip.
        Angles are inherently scale- and translation-invariant.
"""

import numpy as np


# ──────────────────────────────────────────────
#  MediaPipe hand landmark indices
# ──────────────────────────────────────────────
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_MCP = 13
RING_TIP = 16
PINKY_MCP = 17
PINKY_TIP = 20

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

# Finger vectors for angle computation: (MCP_index, TIP_index)
FINGER_VECTORS = [
    (INDEX_MCP, INDEX_TIP),      # Index finger
    (MIDDLE_MCP, MIDDLE_TIP),    # Middle finger
    (RING_MCP, RING_TIP),        # Ring finger
    (PINKY_MCP, PINKY_TIP),      # Pinky finger
]

# Human-readable feature names
RAW_FEATURE_NAMES = []
for _i in range(21):
    RAW_FEATURE_NAMES.extend([f"x{_i}", f"y{_i}", f"z{_i}"])

INVARIANT_FEATURE_NAMES = [
    "norm_dist_thumb_tip",       # dist(wrist, lm4)  / dist(wrist, lm9)
    "norm_dist_index_tip",       # dist(wrist, lm8)  / dist(wrist, lm9)
    "norm_dist_middle_tip",      # dist(wrist, lm12) / dist(wrist, lm9)
    "norm_dist_ring_tip",        # dist(wrist, lm16) / dist(wrist, lm9)
    "norm_dist_pinky_tip",       # dist(wrist, lm20) / dist(wrist, lm9)
    "angle_index_middle",        # angle(lm5→lm8, lm9→lm12)
    "angle_middle_ring",         # angle(lm9→lm12, lm13→lm16)
    "angle_ring_pinky",          # angle(lm13→lm16, lm17→lm20)
]


# ──────────────────────────────────────────────
#  Helper functions
# ──────────────────────────────────────────────

def _reshape_landmarks(landmarks_flat):
    """Reshape a flat 63-element array into a (21, 3) landmark matrix."""
    return np.asarray(landmarks_flat, dtype=np.float64).reshape(21, 3)


def _euclidean_dist(a, b):
    """
    Euclidean distance between two 3-D points.

    Formula:  dist = sqrt( (bx-ax)² + (by-ay)² + (bz-az)² )
    """
    return np.linalg.norm(a - b)


def _angle_between_vectors(v1, v2):
    """
    Angle (radians) between two 3-D vectors via the dot-product formula.

    Formula:  θ = arccos( (v1 · v2) / (|v1| × |v2|) )

    The cosine is clamped to [-1, 1] to avoid NaN from floating-point drift.
    Returns 0.0 if either vector has near-zero magnitude.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_angle = np.dot(v1, v2) / (n1 * n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle))


# ──────────────────────────────────────────────
#  Public API — single-sample extraction
# ──────────────────────────────────────────────

def extract_raw_features(landmarks_flat):
    """
    Path A: Return raw coordinate features (63 values, unchanged).

    Parameters
    ----------
    landmarks_flat : array-like of shape (63,)
        Flattened (x0, y0, z0, x1, y1, z1, …, x20, y20, z20).

    Returns
    -------
    numpy.ndarray of shape (63,)
    """
    return np.asarray(landmarks_flat, dtype=np.float64)


def extract_invariant_features(landmarks_flat):
    """
    Path B: Extract 8 scale- and translation-invariant geometric features.

    Parameters
    ----------
    landmarks_flat : array-like of shape (63,)

    Returns
    -------
    numpy.ndarray of shape (8,)
        [0–4] Normalised fingertip distances
        [5–7] Inter-finger angles (radians)
    """
    lm = _reshape_landmarks(landmarks_flat)

    wrist = lm[WRIST]
    middle_mcp = lm[MIDDLE_MCP]

    # Scale normalization anchor: dist(wrist, middle-finger MCP)
    scale_ref = _euclidean_dist(wrist, middle_mcp)
    if scale_ref < 1e-8:
        scale_ref = 1.0  # prevent division by zero for degenerate hands

    # Features 1–5: normalised fingertip distances
    #   d_i = dist(wrist, fingertip_i) / scale_ref
    norm_distances = []
    for tip_idx in FINGERTIPS:
        d = _euclidean_dist(wrist, lm[tip_idx])
        norm_distances.append(d / scale_ref)

    # Finger direction vectors (MCP → TIP)
    finger_vecs = [lm[tip] - lm[mcp] for mcp, tip in FINGER_VECTORS]

    # Features 6–8: angles between adjacent finger vectors
    angles = [
        _angle_between_vectors(finger_vecs[i], finger_vecs[i + 1])
        for i in range(len(finger_vecs) - 1)
    ]

    return np.array(norm_distances + angles, dtype=np.float64)


# ──────────────────────────────────────────────
#  Public API — batch extraction
# ──────────────────────────────────────────────

def extract_raw_features_batch(landmarks_array):
    """
    Batch version of extract_raw_features.

    Parameters
    ----------
    landmarks_array : array-like of shape (N, 63)

    Returns
    -------
    numpy.ndarray of shape (N, 63)
    """
    return np.asarray(landmarks_array, dtype=np.float64)


def extract_invariant_features_batch(landmarks_array):
    """
    Batch version of extract_invariant_features.

    Parameters
    ----------
    landmarks_array : array-like of shape (N, 63)

    Returns
    -------
    numpy.ndarray of shape (N, 8)
    """
    return np.array(
        [extract_invariant_features(row) for row in landmarks_array]
    )
