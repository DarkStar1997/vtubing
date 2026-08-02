"""MediaPipe HandLandmarker wrapper (CPU, VIDEO mode, 2 hands).

Outputs per detection:
  * Per-joint finger angles [0,1] (0=straight, 1=fully curled)
  * Handedness (Left/Right, swapped for un-mirrored feed)

Finger angles are computed from WORLD landmarks (3D, meters, correct scale)
rather than image landmarks (where Z has a different scale, distorting angles).
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

# Landmark indices
_WRIST = 0
_THUMB_CMC, _THUMB_MCP, _THUMB_IP, _THUMB_TIP = 1, 2, 3, 4
_INDEX_MCP, _INDEX_PIP, _INDEX_DIP, _INDEX_TIP = 5, 6, 7, 8
_MIDDLE_MCP, _MIDDLE_PIP, _MIDDLE_DIP, _MIDDLE_TIP = 9, 10, 11, 12
_RING_MCP, _RING_PIP, _RING_DIP, _RING_TIP = 13, 14, 15, 16
_LITTLE_MCP, _LITTLE_PIP, _LITTLE_DIP, _LITTLE_TIP = 17, 18, 19, 20


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


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Flexion angle at joint *b* given consecutive points a→b→c. Returns radians.

    Straight finger: segments a→b and b→c point the same way → 0.
    Curled finger: segments bend up to ~π.
    """
    v1 = b - a
    v2 = c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_a))


def _compute_finger_angles(pts: np.ndarray) -> dict[str, list[float]]:
    """Per-joint flexion angles for each finger. 0=straight, 1=curled.

    *pts* must be 3D points with correct scale (world landmarks, not image).
    Returns {finger: [joint_angles...]} where joint order is proximal→distal.
    """
    w = pts[_WRIST]
    return {
        "thumb": [
            _joint_angle(w, pts[_THUMB_CMC], pts[_THUMB_MCP]),
            _joint_angle(pts[_THUMB_CMC], pts[_THUMB_MCP], pts[_THUMB_IP]),
            _joint_angle(pts[_THUMB_MCP], pts[_THUMB_IP], pts[_THUMB_TIP]),
        ],
        "index": [
            _joint_angle(w, pts[_INDEX_MCP], pts[_INDEX_PIP]),
            _joint_angle(pts[_INDEX_MCP], pts[_INDEX_PIP], pts[_INDEX_DIP]),
            _joint_angle(pts[_INDEX_PIP], pts[_INDEX_DIP], pts[_INDEX_TIP]),
        ],
        "middle": [
            _joint_angle(w, pts[_MIDDLE_MCP], pts[_MIDDLE_PIP]),
            _joint_angle(pts[_MIDDLE_MCP], pts[_MIDDLE_PIP], pts[_MIDDLE_DIP]),
            _joint_angle(pts[_MIDDLE_PIP], pts[_MIDDLE_DIP], pts[_MIDDLE_TIP]),
        ],
        "ring": [
            _joint_angle(w, pts[_RING_MCP], pts[_RING_PIP]),
            _joint_angle(pts[_RING_MCP], pts[_RING_PIP], pts[_RING_DIP]),
            _joint_angle(pts[_RING_PIP], pts[_RING_DIP], pts[_RING_TIP]),
        ],
        "little": [
            _joint_angle(w, pts[_LITTLE_MCP], pts[_LITTLE_PIP]),
            _joint_angle(pts[_LITTLE_MCP], pts[_LITTLE_PIP], pts[_LITTLE_DIP]),
            _joint_angle(pts[_LITTLE_PIP], pts[_LITTLE_DIP], pts[_LITTLE_TIP]),
        ],
    }


def _compute_palm_twist(pts: np.ndarray, is_left: bool) -> float:
    """Pronation/supination from palm normal vertical component.

    Returns angle in [-pi/2, +pi/2]:
      0     = palm sideways
      +pi/2 = palm up
      -pi/2 = palm down
    """
    v1 = pts[_INDEX_MCP] - pts[_WRIST]
    v2 = pts[_LITTLE_MCP] - pts[_WRIST]
    normal = np.cross(v1, v2)
    n = np.linalg.norm(normal)
    if n < 1e-8:
        return 0.0
    normal = normal / n
    # MediaPipe world landmarks: cross(index, little) gives opposite palm-normal
    # directions for left vs right hands. Use same formula for both — the
    # calibration neutral handles the per-hand baseline.
    y = -normal[1]
    return float(np.arcsin(np.clip(y, -1.0, 1.0)))


def extract_hands(result: Any) -> dict[str, dict | None]:
    """Parse HandLandmarkerResult into per-hand finger angles.

    Returns ``{"left": {...}|None, "right": {...}|None}`` where each hand dict
    has ``"fingers"`` (dict of joint angles per finger, 0=straight 1=curled).
    """
    out: dict[str, dict | None] = {"left": None, "right": None}
    if not result.hand_landmarks:
        return out

    world_lms_list = getattr(result, "hand_world_landmarks", None) or []

    for i, lms in enumerate(result.hand_landmarks):
        handed = "right"
        if result.handedness and i < len(result.handedness):
            cats = result.handedness[i]
            if cats:
                handed = cats[0].category_name.lower()
        # MediaPipe's handedness is already correct for un-mirrored feeds
        # (it assumes mirrored/selfie by default, which inverts for our
        # un-mirrored camera — net result: labels match the user's hands).
        side = handed

        # Use world landmarks for angle computation (correct 3D scale).
        # Fall back to image landmarks if world landmarks unavailable.
        if i < len(world_lms_list) and len(world_lms_list[i]) >= 21:
            wl = world_lms_list[i]
            pts = np.array(
                [[lm.x, lm.y, lm.z] for lm in wl], dtype=np.float32
            )
        else:
            pts = np.array(
                [[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32
            )

        out[side] = {
            "fingers": _compute_finger_angles(pts),
            "twist": _compute_palm_twist(pts, is_left=(side == "left")),
        }

    return out
