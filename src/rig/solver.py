"""Rig solver: MediaPipe ARKit data -> VRM 1.0 rig state.

Translates the 52 ARKit blendshapes + head transformation matrix from
MediaPipe into the VRM 1.0 expression weights, head-bone rotation delta,
and eye-gaze angles that the renderer consumes.

Pipeline per frame:
  1. Calibration (first ~1 s of stable detection) — capture neutral baseline
  2. Subtract neutral blendshapes
  3. One-Euro filter on blendshapes + head pose + gaze
  4. Map 52 ARKit -> 18 VRM preset expressions (many-to-one)
  5. Extract head rotation delta from transformation matrix
  6. Compute eye-gaze yaw/pitch from eyeLook* blendshapes + rangeMap

Detection-loss handling: hold last rig, decay to neutral over 200 ms.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from src.avatar.vrm_loader import VRMModel
from src.rig.one_euro import OneEuroFilter

# One-Euro defaults (tunable via config)
_BS_CUTOFF = 3.0
_BS_BETA = 0.0
_HEAD_CUTOFF = 1.5
_HEAD_BETA = 0.05
_GAZE_CUTOFF = 3.0
_GAZE_BETA = 0.0

_CALIB_FRAMES = 30
_LOSS_DECAY_SEC = 0.2


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def map_arkit_to_vrm(bs: dict[str, float]) -> dict[str, float]:
    """Map 52 ARKit blendshapes to 18 VRM preset expression weights [0, 1]."""
    g = lambda n: bs.get(n, 0.0)

    blink = max(g("eyeBlinkLeft"), g("eyeBlinkRight"))
    blink_left = g("eyeBlinkLeft")
    blink_right = g("eyeBlinkRight")

    look_up = max(g("eyeLookUpLeft"), g("eyeLookUpRight"))
    look_down = max(g("eyeLookDownLeft"), g("eyeLookDownRight"))
    look_left = max(g("eyeLookOutLeft"), g("eyeLookInRight"))
    look_right = max(g("eyeLookInLeft"), g("eyeLookOutRight"))

    jaw = g("jawOpen")
    stretch = max(g("mouthStretchLeft"), g("mouthStretchRight"))
    funnel = g("mouthFunnel")
    pucker = g("mouthPucker")
    smile = min(g("mouthSmileLeft"), g("mouthSmileRight"))

    aa = jaw
    ih = min(jaw * 0.3, 0.3) + stretch * 0.7
    ou = max(funnel, pucker)
    ee = stretch * 0.5 + min(g("mouthSmileLeft"), g("mouthSmileRight")) * 0.3
    oh = max(funnel, pucker) * 0.5 + jaw * 0.3

    happy = smile
    angry = max(g("browDownLeft"), g("browDownRight")) * 0.7 + max(
        g("noseSneerLeft"), g("noseSneerRight")
    ) * 0.3
    sad = max(g("mouthFrownLeft"), g("mouthFrownRight")) * 0.7 + g("browInnerUp") * 0.3
    surprised = (
        max(g("browOuterUpLeft"), g("browOuterUpRight")) * 0.3
        + max(g("eyeWideLeft"), g("eyeWideRight")) * 0.3
        + jaw * 0.4
    )

    return {
        "happy": _clamp01(happy),
        "angry": _clamp01(angry),
        "sad": _clamp01(sad),
        "relaxed": 0.0,
        "surprised": _clamp01(surprised),
        "aa": _clamp01(aa),
        "ih": _clamp01(ih),
        "ou": _clamp01(ou),
        "ee": _clamp01(ee),
        "oh": _clamp01(oh),
        "blink": _clamp01(blink),
        "blinkLeft": _clamp01(blink_left),
        "blinkRight": _clamp01(blink_right),
        "lookUp": _clamp01(look_up),
        "lookDown": _clamp01(look_down),
        "lookLeft": _clamp01(look_left),
        "lookRight": _clamp01(look_right),
        "neutral": 0.0,
    }


class RigSolver:
    def __init__(self, model: VRMModel, config: dict | None = None) -> None:
        self.model = model
        cfg = config or {}

        self.head_node = model.humanoid_bone_by_name.get("head")
        self.left_eye_node = model.humanoid_bone_by_name.get("leftEye")
        self.right_eye_node = model.humanoid_bone_by_name.get("rightEye")

        bs_cfg = cfg.get("blendshape", {})
        head_cfg = cfg.get("head", {})
        gaze_cfg = cfg.get("gaze", {})
        self._bs_filters: dict[str, OneEuroFilter] = defaultdict(
            lambda: OneEuroFilter(
                min_cutoff=bs_cfg.get("min_cutoff", _BS_CUTOFF),
                beta=bs_cfg.get("beta", _BS_BETA),
            )
        )
        self._head_filters = [
            OneEuroFilter(
                min_cutoff=head_cfg.get("min_cutoff", _HEAD_CUTOFF),
                beta=head_cfg.get("beta", _HEAD_BETA),
            )
            for _ in range(3)
        ]
        self._gaze_filters = [
            OneEuroFilter(
                min_cutoff=gaze_cfg.get("min_cutoff", _GAZE_CUTOFF),
                beta=gaze_cfg.get("beta", _GAZE_BETA),
            )
            for _ in range(2)
        ]

        calib_cfg = cfg.get("calibration", {})
        self._calib_target = int(calib_cfg.get("frames", _CALIB_FRAMES))
        self._calib_frames = 0
        self._calibrating = True
        self._calib_bs_sum: dict[str, float] = defaultdict(float)
        self._neutral_bs: dict[str, float] = {}
        self._neutral_head_rot: np.ndarray | None = None  # 3x3

        la = model.look_at
        self._look_at_type = la.type
        rm = la.range_maps
        self._gaze_h_scale = rm.get("horizontalInner", {}).get("outputScale", 10.0)
        self._gaze_v_up = rm.get("verticalUp", {}).get("outputScale", 10.0)
        self._gaze_v_down = rm.get("verticalDown", {}).get("outputScale", 10.0)

        self._loss_time = 0.0
        self._last_state: dict[str, Any] | None = None

    @property
    def is_calibrated(self) -> bool:
        return not self._calibrating

    def recalibrate(self) -> None:
        self._calibrating = True
        self._calib_frames = 0
        self._calib_bs_sum.clear()
        self._neutral_bs.clear()
        self._neutral_head_rot = None
        self._last_state = None
        for f in self._bs_filters.values():
            f.reset()
        for f in self._head_filters:
            f.reset()
        for f in self._gaze_filters:
            f.reset()

    def update(self, rig: dict, dt: float) -> dict[str, Any]:
        if not rig["detected"]:
            return self._handle_loss(dt)

        self._loss_time = 0.0
        bs_raw = rig["blendshapes"]
        matrix = rig["matrix"]

        if self._calibrating:
            self._calibrate(bs_raw, matrix)
            return self._empty_state()

        # Subtract neutral blendshapes
        bs_adj = {
            k: _clamp01(v - self._neutral_bs.get(k, 0.0))
            for k, v in bs_raw.items()
        }

        # One-Euro filter on blendshapes
        bs_filt = {k: self._bs_filters[k].filter(v, dt) for k, v in bs_adj.items()}

        # Map to VRM expressions
        expressions = map_arkit_to_vrm(bs_filt)

        # If lookAt type is 'expression', the look* expressions carry the gaze
        # via morph targets. For 'bone' type we compute angles separately.
        if self._look_at_type == "expression":
            # Expression-based lookAt: set look* expression weights directly
            for name in ("lookUp", "lookDown", "lookLeft", "lookRight"):
                expressions[name] = _clamp01(bs_filt.get(name, 0.0)) if name in expressions else 0.0
            eye_yaw = 0.0
            eye_pitch = 0.0
        else:
            # Bone-based lookAt: compute gaze angles
            eye_yaw, eye_pitch = self._compute_gaze_angles(bs_filt)

        # Head pose delta
        head_rot = matrix[:3, :3]
        delta_rot = self._neutral_head_rot.T @ head_rot
        # Filter as Euler angles (YXZ = yaw, pitch, roll)
        euler = Rotation.from_matrix(delta_rot).as_euler("YXZ", degrees=True)
        euler_f = np.array([
            self._head_filters[0].filter(euler[0], dt),
            self._head_filters[1].filter(euler[1], dt),
            self._head_filters[2].filter(euler[2], dt),
        ], dtype=np.float32)
        delta_rot_f = Rotation.from_euler("YXZ", euler_f, degrees=True).as_matrix()

        state: dict[str, Any] = {
            "detected": True,
            "calibrated": True,
            "expressions": expressions,
            "head_delta": delta_rot_f.astype(np.float32),  # 3x3 world-space delta
            "eye_yaw": float(eye_yaw),
            "eye_pitch": float(eye_pitch),
        }
        self._last_state = state
        return state

    def _compute_gaze_angles(self, bs: dict[str, float]) -> tuple[float, float]:
        look_up = max(bs.get("eyeLookUpLeft", 0.0), bs.get("eyeLookUpRight", 0.0))
        look_down = max(bs.get("eyeLookDownLeft", 0.0), bs.get("eyeLookDownRight", 0.0))
        look_left = max(bs.get("eyeLookOutLeft", 0.0), bs.get("eyeLookInRight", 0.0))
        look_right = max(bs.get("eyeLookInLeft", 0.0), bs.get("eyeLookOutRight", 0.0))

        yaw = (look_right - look_left) * self._gaze_h_scale
        pitch = look_up * self._gaze_v_up - look_down * self._gaze_v_down
        return yaw, pitch

    def _calibrate(self, bs: dict[str, float], matrix: np.ndarray | None) -> None:
        for k, v in bs.items():
            self._calib_bs_sum[k] += v
        if matrix is not None:
            self._neutral_head_rot = matrix[:3, :3].copy()
        self._calib_frames += 1
        if self._calib_frames >= self._calib_target:
            self._neutral_bs = {
                k: v / self._calib_frames for k, v in self._calib_bs_sum.items()
            }
            self._calibrating = False
            print(
                f"[rig] calibration complete ({self._calib_frames} frames, "
                f"{len(self._neutral_bs)} blendshapes)"
            )

    def _handle_loss(self, dt: float) -> dict[str, Any]:
        if self._last_state is None or not self._last_state.get("calibrated"):
            return self._empty_state()
        self._loss_time += dt
        decay = max(0.0, 1.0 - self._loss_time / _LOSS_DECAY_SEC)
        if decay <= 0.0:
            self._last_state = None
            return self._empty_state()
        s = self._last_state
        return {
            "detected": False,
            "calibrated": True,
            "expressions": {k: v * decay for k, v in s["expressions"].items()},
            "head_delta": (s["head_delta"] * decay).astype(np.float32),
            "eye_yaw": s["eye_yaw"] * decay,
            "eye_pitch": s["eye_pitch"] * decay,
        }

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "detected": False,
            "calibrated": False,
            "expressions": {},
            "head_delta": np.eye(3, dtype=np.float32),
            "eye_yaw": 0.0,
            "eye_pitch": 0.0,
        }
