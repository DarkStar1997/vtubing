"""MediaPipe PoseLandmarker wrapper (CPU, VIDEO mode).

Tracks upper-body skeleton (shoulders, elbows, wrists) and computes
VRM humanoid bone rotation quaternions via direction-vector matching.

Outputs per detection:
  * Upper/lower arm quaternions [x, y, z, w] (local space)
  * Torso forward lean (radians)
  * One-euro filtered for stability
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

from src.rig.one_euro import OneEuroFilter

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = _PROJECT_ROOT / "assets" / "models" / "pose_landmarker_full.task"

_L_SHOULDER, _R_SHOULDER = 11, 12
_L_ELBOW, _R_ELBOW = 13, 14
_L_WRIST, _R_WRIST = 15, 16
_L_HIP, _R_HIP = 23, 24
_L_KNEE, _R_KNEE = 25, 26
_L_ANKLE, _R_ANKLE = 27, 28

_FILTER_CUTOFF = 1.0
_FILTER_BETA = 0.05


def ensure_model() -> Path:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[pose] downloading model to {MODEL_PATH} ...", flush=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class PoseLandmarker:
    def __init__(
        self,
        num_poses: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = str(ensure_model())
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> Any:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self._detector.detect_for_video(image, timestamp_ms)


def _lm_to_arr(lm: Any) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def _dir_to_rotation(rest_dir: np.ndarray, target_dir: np.ndarray) -> Rotation:
    """Rotation that maps rest_dir → target_dir (both 3D vectors).

    Includes a deadzone: if the angle between rest and target is very small
    (< 3°), returns identity to prevent jitter from amplified cross products.
    """
    r = rest_dir / (np.linalg.norm(rest_dir) + 1e-10)
    t = target_dir / (np.linalg.norm(target_dir) + 1e-10)
    cross = np.cross(r, t)
    n = np.linalg.norm(cross)
    dot = float(np.clip(np.dot(r, t), -1.0, 1.0))
    angle = np.arctan2(n, dot)
    # Deadzone: small angles → identity (prevents jitter near rest pose)
    if angle < 0.05:  # ~3 degrees
        return Rotation.identity()
    if n < 1e-6:
        return Rotation.from_rotvec(np.pi * np.array([0.0, 0.0, 1.0])) if dot < 0 \
            else Rotation.identity()
    return Rotation.from_rotvec((angle / n) * cross)


def _torso_lean(l_sh: np.ndarray, r_sh: np.ndarray,
                l_hip: np.ndarray, r_hip: np.ndarray) -> float:
    """Forward lean: positive when shoulders are forward of hips (toward camera)."""
    shoulder_mid = (l_sh + r_sh) / 2
    hip_mid = (l_hip + r_hip) / 2
    d = shoulder_mid - hip_mid
    vert = np.sqrt(d[0] ** 2 + d[1] ** 2)
    if vert < 1e-5:
        return 0.0
    return float(np.arctan2(-d[2], vert))


class PoseTracker:
    """Wraps PoseLandmarker with one-euro filtering for stable arm rotations."""

    # T-pose directions in VRM space: left arm points +X, right arm points -X
    _REST_L = np.array([1.0, 0.0, 0.0])
    _REST_R = np.array([-1.0, 0.0, 0.0])
    # Legs point down in T-pose
    _REST_DOWN = np.array([0.0, -1.0, 0.0])

    # MediaPipe world → VRM model space conversion.
    # MediaPipe: X=camera-right(+)=model-left(+), Y=up(+), Z=toward-camera(+)
    # VRM:       X=model-left(+),               Y=up(+), Z=model-forward(-Z)
    # => only Z needs negating
    _AXIS_FLIP = np.array([1.0, 1.0, -1.0], dtype=np.float32)

    def __init__(self, **kwargs: Any) -> None:
        self._landmarker = PoseLandmarker(**kwargs)
        # Filter rotation vectors (3 components each) — not direction vectors
        names = ["lu", "ll", "ru", "rl", "lul", "lll", "rul", "rll"]
        self._rot_filters: dict[str, list[OneEuroFilter]] = {
            n: [OneEuroFilter(_FILTER_CUTOFF, _FILTER_BETA) for _ in range(3)]
            for n in names
        }
        self._torso_filter = OneEuroFilter(_FILTER_CUTOFF, 0.0)
        self._spine_y_filter = OneEuroFilter(_FILTER_CUTOFF, 0.0)
        self._spine_z_filter = OneEuroFilter(_FILTER_CUTOFF, 0.0)
        self._stand_filter = OneEuroFilter(0.5, 0.0)  # slow for smooth transitions
        self._torso_neutral: float | None = None
        self._spine_y_neutral: float | None = None
        self._spine_z_neutral: float | None = None
        self.last_debug: dict | None = None

    def calibrate_neutral(self) -> None:
        """Capture current filtered spine values as the neutral baseline."""
        self._torso_neutral = self._torso_filter._x_prev
        self._spine_y_neutral = self._spine_y_filter._x_prev
        self._spine_z_neutral = self._spine_z_filter._x_prev

    def _filter_rot(self, name: str, rot: Rotation, dt: float) -> Rotation:
        """Filter a rotation via its rotation vector (axis*angle) with one-euro."""
        rv = rot.as_rotvec()
        filtered = np.array([
            self._rot_filters[name][i].filter(float(rv[i]), dt)
            for i in range(3)
        ], dtype=np.float32)
        return Rotation.from_rotvec(filtered)

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> Any:
        return self._landmarker.detect(rgb_frame, timestamp_ms)

    def extract_angles(self, result: Any, dt: float,
                       hands: dict | None = None) -> dict | None:
        """Extract filtered arm quaternions + torso lean. Returns None if no pose.

        When *hands* (from HandLandmarker) is provided, the hand wrist position
        is used to supplement arm direction — this improves tracking when the
        PoseLandmarker's wrist landmarks are inaccurate (e.g. hands near face).
        """
        if not result.pose_world_landmarks:
            self.last_debug = None
            return None

        lms = result.pose_world_landmarks[0]
        # Also get image-space landmarks (for hand wrist supplementation)
        img_lms = None
        if result.pose_landmarks:
            img_lms = result.pose_landmarks[0]
        ls, rs = _lm_to_arr(lms[_L_SHOULDER]), _lm_to_arr(lms[_R_SHOULDER])
        le, re = _lm_to_arr(lms[_L_ELBOW]), _lm_to_arr(lms[_R_ELBOW])
        lw, rw = _lm_to_arr(lms[_L_WRIST]), _lm_to_arr(lms[_R_WRIST])
        lh, rh = _lm_to_arr(lms[_L_HIP]), _lm_to_arr(lms[_R_HIP])

        # Lower body landmarks (may have low visibility when sitting)
        lk = _lm_to_arr(lms[_L_KNEE])
        rk = _lm_to_arr(lms[_R_KNEE])
        la = _lm_to_arr(lms[_L_ANKLE])
        ra = _lm_to_arr(lms[_R_ANKLE])

        # Check lower-body visibility
        lower_visible = False
        lower_vis_detail = {}
        if result.pose_landmarks:
            plms = result.pose_landmarks[0]
            for idx, name in [(_L_KNEE, "lk"), (_R_KNEE, "rk"),
                              (_L_ANKLE, "la"), (_R_ANKLE, "ra")]:
                lower_vis_detail[name] = plms[idx].visibility
            vis = sum(lower_vis_detail.values()) / 4.0
            lower_visible = vis > 0.3

        # Image-space → VRM: X=right→model-left (keep), Y=down→up (negate),
        # Z=negative=closer-to-camera → VRM -Z=forward (keep, don't negate)
        _IMG_FLIP = np.array([1.0, -1.0, 1.0], dtype=np.float32)

        # Raw direction vectors converted to VRM model space
        # Upper arm: shoulder→elbow, Lower arm: elbow→wrist
        # (PoseLandmarker-only — HandLandmarker supplementation caused twisting
        #  and folding issues due to coordinate system mismatches)
        lu_dir = (le - ls) * self._AXIS_FLIP
        ru_dir = (re - rs) * self._AXIS_FLIP
        ll_dir = (lw - le) * self._AXIS_FLIP
        rl_dir = (rw - re) * self._AXIS_FLIP

        dirs = {"lu": lu_dir, "ll": ll_dir, "ru": ru_dir, "rl": rl_dir}

        # Compute rotations from RAW directions (no pre-filtering to avoid distortion)
        upper_l_raw = _dir_to_rotation(self._REST_L, dirs["lu"])
        upper_r_raw = _dir_to_rotation(self._REST_R, dirs["ru"])
        lower_l_world_raw = _dir_to_rotation(self._REST_L, dirs["ll"])
        lower_r_world_raw = _dir_to_rotation(self._REST_R, dirs["rl"])

        # Lower arm: convert world → local (relative to upper arm) using RAW rotations
        lower_l_local_raw = upper_l_raw.inv() * lower_l_world_raw
        lower_r_local_raw = upper_r_raw.inv() * lower_r_world_raw

        # Filter rotations as rotation vectors (preserves rotation structure)
        upper_l = self._filter_rot("lu", upper_l_raw, dt)
        upper_r = self._filter_rot("ru", upper_r_raw, dt)
        lower_l_local = self._filter_rot("ll", lower_l_local_raw, dt)
        lower_r_local = self._filter_rot("rl", lower_r_local_raw, dt)

        _raw_lean = self._torso_filter.filter(_torso_lean(ls, rs, lh, rh), dt)
        if self._torso_neutral is not None:
            _raw_lean -= self._torso_neutral
        lean = -_raw_lean

        # Spine lateral bend + twist from shoulder line (VRM space)
        # shoulder_vec points from right shoulder to left shoulder
        shoulder_vec = (ls - rs) * self._AXIS_FLIP
        horiz = float(np.sqrt(shoulder_vec[0] ** 2 + shoulder_vec[2] ** 2))
        if horiz > 1e-5:
            # Lateral bend: left shoulder higher → spine bends right (negative Z)
            raw_z = float(-np.arctan2(shoulder_vec[1], horiz))
            # Twist: left shoulder forward (−Z in VRM) → torso twists left (positive Y)
            raw_y = float(np.arctan2(-shoulder_vec[2], horiz))
        else:
            raw_z = 0.0
            raw_y = 0.0
        spine_y = self._spine_y_filter.filter(raw_y, dt)
        spine_z = self._spine_z_filter.filter(raw_z, dt)
        if self._spine_y_neutral is not None:
            spine_y -= self._spine_y_neutral
        if self._spine_z_neutral is not None:
            spine_z -= self._spine_z_neutral

        # --- Leg tracking (DISABLED — lower body rarely visible in facecam) ---
        # Standing detection is handled in main.py via face-size comparison.
        hip_mid = (lh + rh) / 2.0
        shoulder_mid = (ls + rs) / 2.0
        torso_len = float(np.linalg.norm(shoulder_mid - hip_mid))
        standing = 0.0  # set by main.py face-size detection

        # Store debug info
        self.last_debug = {
            "ls": ls.tolist(), "le": le.tolist(), "lw": lw.tolist(),
            "rs": rs.tolist(), "re": re.tolist(), "rw": rw.tolist(),
            "dirs": {k: v.tolist() for k, v in dirs.items()},
            "lean": lean,
            "spine_y": spine_y,
            "spine_z": spine_z,
        }

        result_dict = {
            "leftUpperArm": upper_l.as_quat().tolist(),
            "rightUpperArm": upper_r.as_quat().tolist(),
            "leftLowerArm": lower_l_local.as_quat().tolist(),
            "rightLowerArm": lower_r_local.as_quat().tolist(),
            "torso": float(lean),
            "spine": [float(lean), float(spine_y), float(spine_z)],
        }
        return result_dict
