"""MediaPipe HandLandmarker wrapper (CPU, VIDEO mode, 2 hands).

Outputs per detection:
  * 21 normalized landmarks per hand (x, y, z)
  * Handedness classification (Left / Right)

Finger curls are computed from joint angles and normalised to [0, 1]
where 0 = straight, 1 = fully curled.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = _PROJECT_ROOT / "assets" / "models" / "hand_landmarker.task"

# Landmark indices: [mcp, pip, dip, tip] per finger
_FINGER_JOINTS = {
    "thumb":  (1, 2, 3, 4),
    "index":  (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring":   (13, 14, 15, 16),
    "little": (17, 18, 19, 20),
}


def ensure_model() -> Path:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[hand] downloading model to {MODEL_PATH} ...", flush=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class HandLandmarker:
    def __init__(
        self,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = str(ensure_model())
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> Any:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self._detector.detect_for_video(image, timestamp_ms)


def _joint_curl(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Curl at joint *b* given consecutive points a→b→c. Returns [0, 1].

    Straight finger: segments a→b and b→c point the same way → angle ≈ 0 → 0.
    Curled finger: segments bend up to ~π → 1.
    """
    v1 = b - a
    v2 = c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    angle = np.arccos(cos_a)
    return float(np.clip(angle / np.pi, 0.0, 1.0))


def _compute_curls(lms: list) -> list[float]:
    """Return [thumb, index, middle, ring, little] curl values in [0, 1]."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in lms])
    curls = []
    for name, (mcp, pip, dip, tip) in _FINGER_JOINTS.items():
        if name == "thumb":
            c = _joint_curl(pts[1], pts[2], pts[3]) * 0.5 + \
                _joint_curl(pts[2], pts[3], pts[4]) * 0.5
        else:
            c = _joint_curl(pts[mcp], pts[pip], pts[dip]) * 0.6 + \
                _joint_curl(pts[pip], pts[dip], pts[tip]) * 0.4
        curls.append(c)
    return curls


def extract_hands(result: Any) -> dict[str, dict | None]:
    """Parse HandLandmarkerResult into {left: {curls: [...]}, right: {...}}.

    Returns ``{"left": None, "right": None}`` when no hands are detected.
    """
    out: dict[str, dict | None] = {"left": None, "right": None}
    if not result.hand_landmarks:
        return out

    for i, lms in enumerate(result.hand_landmarks):
        handed = "right"
        if result.handedness and i < len(result.handedness):
            cats = result.handedness[i]
            if cats:
                handed = cats[0].category_name.lower()
        # MediaPipe assumes a mirrored (selfie) view by default; our feed
        # is un-mirrored, so swap Left/Right.
        side = "right" if handed == "left" else "left"
        out[side] = {"curls": _compute_curls(lms)}

    return out
