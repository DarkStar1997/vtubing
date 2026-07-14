"""Live VTubing pipeline: webcam -> MediaPipe -> rig solver -> WebSocket.

Sends rig state (expressions, head pose, eye gaze) to the browser frontend
which renders the VRM avatar with three-vrm.  Open the printed URL in a
browser (or add it as an OBS Browser Source).

Rate-decoupled: tracking runs only when a new webcam frame arrives; the
solver always runs and broadcasts at the configured target FPS.
"""
from __future__ import annotations

import math
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as _FLC

from src.avatar.vrm_loader import load_vrm
from src.capture.webcam import WebcamCapture
from src.config import load_config
from src.rig.solver import RigSolver
from src.server import RigServer
from src.tracking.landmarker import FaceLandmarker, extract_rig

_CONTOURS = [(c.start, c.end) for c in _FLC.FACE_LANDMARKS_CONTOURS]
_TESSELATION = [(c.start, c.end) for c in _FLC.FACE_LANDMARKS_TESSELATION]


def _draw_landmarks(frame: np.ndarray, result) -> np.ndarray:
    """Draw MediaPipe face mesh (tesselation + contours) on a BGR frame copy."""
    if result is None or not result.face_landmarks:
        return frame
    annotated = frame.copy()
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in result.face_landmarks[0]]
    for s, e in _TESSELATION:
        if s < len(pts) and e < len(pts):
            cv2.line(annotated, pts[s], pts[e], (40, 80, 40), 1)
    for s, e in _CONTOURS:
        if s < len(pts) and e < len(pts):
            cv2.line(annotated, pts[s], pts[e], (0, 255, 0), 1)
    return annotated


def _state_to_ws(state: dict) -> dict:
    """Convert solver state to the JSON format the browser expects."""
    # Head rotation: 3x3 matrix -> YXZ Euler radians [yaw, pitch, roll]
    head_delta = state.get("head_delta")
    if head_delta is not None:
        yaw, pitch, roll = Rotation.from_matrix(head_delta).as_euler(
            "YXZ", degrees=False
        )
        head = [float(yaw), float(-pitch), float(roll)]
    else:
        head = None

    gaze = [
        math.radians(float(state.get("eye_yaw", 0.0))),
        math.radians(float(state.get("eye_pitch", 0.0))),
    ]

    return {
        "detected": bool(state.get("detected", False)),
        "calibrated": bool(state.get("calibrated", False)),
        "expressions": {k: float(v) for k, v in state.get("expressions", {}).items()},
        "head": head,
        "gaze": gaze,
    }


def main() -> None:
    cfg = load_config()
    cam_cfg = cfg.get("camera", {})
    track_cfg = cfg.get("tracking", {})
    avatar_cfg = cfg.get("avatar", {})
    out_cfg = cfg.get("output", {})
    rig_cfg = cfg.get("rig", {})

    vrm_path = avatar_cfg.get("path", "assets/avatars/sample.vrm")
    if not Path(vrm_path).exists():
        print(f"[main] avatar not found: {vrm_path}", file=sys.stderr)
        sys.exit(1)

    cap = WebcamCapture(
        index=cam_cfg.get("index", 0),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        fps=cam_cfg.get("fps", 30),
    )
    cap.start()

    landmarker = FaceLandmarker(
        num_faces=track_cfg.get("num_faces", 1),
        min_detection_confidence=track_cfg.get("min_detection_confidence", 0.5),
        min_tracking_confidence=track_cfg.get("min_tracking_confidence", 0.5),
    )

    model = load_vrm(vrm_path)
    solver = RigSolver(model, rig_cfg)

    port = int(out_cfg.get("port", 8080))
    host = out_cfg.get("host", "localhost")
    server = RigServer(vrm_path, port=port, host=host)
    server.start()

    target_dt = 1.0 / out_cfg.get("fps", 30.0)

    print(f"[main] avatar: {vrm_path}")
    print(f"[main] open {server.url} in a browser, or add as OBS Browser Source.")
    print("[main] calibrating... face the camera neutrally. Ctrl+C to stop.\n", flush=True)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    rgb_buf: np.ndarray | None = None
    last_ts = 0
    last_rig: dict | None = None
    last_result = None
    frame_count = 0
    t0 = time.monotonic()
    t_prev = t0

    try:
        while not stop.is_set():
            frame, new = cap.get_latest()
            if frame is not None:
                if rgb_buf is None or rgb_buf.shape != frame.shape:
                    rgb_buf = np.empty(frame.shape, dtype=np.uint8)

                if new:
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=rgb_buf)
                    ts = int(time.monotonic_ns() // 1_000_000)
                    if ts <= last_ts:
                        ts = last_ts + 1
                    last_ts = ts
                    result = landmarker.detect(rgb_buf, ts)
                    last_rig = extract_rig(result)
                    last_result = result

            now = time.monotonic()
            dt = now - t_prev
            t_prev = now

            if last_rig is None:
                state = solver.update(
                    {"detected": False, "blendshapes": {}, "matrix": None,
                     "euler": np.zeros(3, dtype=np.float32)},
                    dt,
                )
            else:
                state = solver.update(last_rig, dt)

            server.broadcast(_state_to_ws(state))

            if frame is not None:
                annotated = _draw_landmarks(frame, last_result)
                ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    server.set_camera_frame(jpeg.tobytes())

            frame_count += 1
            elapsed = time.monotonic() - t0
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            status = "calibrated" if solver.is_calibrated else "calibrating"
            if state["detected"]:
                exprs = state.get("expressions", {})
                top = sorted(exprs.items(), key=lambda kv: -kv[1])[:3]
                expr_str = " ".join(f"{n}={v:.2f}" for n, v in top if v > 0.01)
                line = f"[{fps:4.1f}fps {status:11s}] {expr_str}"
            else:
                line = f"[{fps:4.1f}fps {status:11s}] no face"
            sys.stdout.write("\r" + line.ljust(80))
            sys.stdout.flush()

            remaining = target_dt - (time.monotonic() - now)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        server.stop()
        cap.stop()
        sys.stdout.write("\n[main] stopped.\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
