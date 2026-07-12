"""MediaPipe FaceLandmarker wrapper (CPU, VIDEO mode).

Outputs per detection:
  * 478 normalized landmarks
  * 52 ARKit blendshapes (coefficients in [0, 1])
  * 4x4 facial transformation matrix (head pose)

The blendshape names map 1:1 to VRM 1.0 ARKit blendshape clips.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from scipy.spatial.transform import Rotation

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = _PROJECT_ROOT / "assets" / "models" / "face_landmarker.task"


def ensure_model() -> Path:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[landmarker] downloading model to {MODEL_PATH} ...", flush=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class FaceLandmarker:
    def __init__(
        self,
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = str(ensure_model())
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> Any:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self._detector.detect_for_video(image, timestamp_ms)


def extract_rig(result: Any) -> dict:
    """Convert a FaceLandmarkerResult into a compact rig dict."""
    if not result.face_landmarks:
        return {"detected": False, "blendshapes": {}, "matrix": None, "euler": np.zeros(3, dtype=np.float32)}

    blendshapes: dict[str, float] = {}
    if result.face_blendshapes:
        # mediapipe >=0.10 returns list[list[Category]]; older docs describe
        # list[Classifications] with a .categories field. Handle both.
        fb0 = result.face_blendshapes[0]
        categories = fb0.categories if hasattr(fb0, "categories") else fb0
        for cat in categories:
            name = cat.category_name.lstrip("_")
            blendshapes[name] = float(cat.score)

    euler = np.zeros(3, dtype=np.float32)
    matrix = None
    if result.facial_transformation_matrixes:
        m = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float32).reshape(4, 4)
        matrix = m
        try:
            # [yaw, pitch, roll] degrees; exact convention refined in the M4 rig solver.
            euler = Rotation.from_matrix(m[:3, :3]).as_euler("YXZ", degrees=True).astype(np.float32)
        except ValueError:
            pass

    return {"detected": True, "blendshapes": blendshapes, "matrix": matrix, "euler": euler}
