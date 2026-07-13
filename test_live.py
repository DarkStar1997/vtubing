"""Live test: webcam with face mesh + 3D avatar mimicking, side by side.

Window 1 "Webcam": camera feed with 478 MediaPipe landmarks + mesh contours.
Window 2 "Avatar": VRM renderer output driven by the rig solver.

ESC or Ctrl+C to stop.
"""
from __future__ import annotations

import signal
import sys
import threading
import time

import cv2
import numpy as np

from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as _FLC

_CONTOURS = [(c.start, c.end) for c in _FLC.FACE_LANDMARKS_CONTOURS]
_TESSELATION = [(c.start, c.end) for c in _FLC.FACE_LANDMARKS_TESSELATION]

from src.avatar.renderer import VRMRenderer
from src.avatar.vrm_loader import load_vrm
from src.capture.webcam import WebcamCapture
from src.config import load_config
from src.rig.solver import RigSolver
from src.tracking.landmarker import FaceLandmarker, extract_rig


def draw_landmarks(frame: np.ndarray, result) -> np.ndarray:
    if result is None or not result.face_landmarks:
        return frame
    annotated = frame.copy()
    h, w = frame.shape[:2]
    lms = result.face_landmarks[0]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

    for s, e in _TESSELATION:
        if s < len(pts) and e < len(pts):
            cv2.line(annotated, pts[s], pts[e], (40, 80, 40), 1)
    for s, e in _CONTOURS:
        if s < len(pts) and e < len(pts):
            cv2.line(annotated, pts[s], pts[e], (0, 255, 0), 1)
    for i in range(468, min(478, len(pts))):
        cv2.circle(annotated, pts[i], 2, (0, 0, 255), -1)

    return annotated


def main() -> None:
    cfg = load_config()
    cam_cfg = cfg["camera"]
    track_cfg = cfg.get("tracking", {})
    out_cfg = cfg.get("output", {})
    rig_cfg = cfg.get("rig", {})

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

    model = load_vrm(cfg["avatar"]["path"])
    out_w = out_cfg.get("width", 1280)
    out_h = out_cfg.get("height", 720)
    renderer = VRMRenderer(model, out_w, out_h)
    solver = RigSolver(model, rig_cfg)
    target_dt = 1.0 / out_cfg.get("fps", 30)

    print("[test] Two windows: Webcam (landmarks) + Avatar (3D model)")
    print("[test] Face camera neutrally for calibration (~1s). ESC / Ctrl+C to stop.\n", flush=True)

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

            rig = last_rig if last_rig is not None else {
                "detected": False, "blendshapes": {}, "matrix": None,
                "euler": np.zeros(3, dtype=np.float32),
            }
            state = solver.update(rig, dt)

            renderer.apply_rig_state(state)
            avatar_rgb = renderer.render()

            if frame is not None:
                annotated = draw_landmarks(frame, last_result)
                cv2.imshow("Webcam", annotated)

            avatar_bgr = cv2.cvtColor(avatar_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("Avatar", avatar_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            elapsed = time.monotonic() - now
            remaining = target_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

            frame_count += 1
            total = time.monotonic() - t0
            fps = frame_count / total if total > 0 else 0.0
            status = "ready" if solver.is_calibrated else "calibrating"
            if rig.get("detected"):
                bs = rig["blendshapes"]
                line = (f"jawOpen={bs.get('jawOpen', 0):.2f}  "
                        f"blink={bs.get('eyeBlinkLeft', 0):.2f}  "
                        f"smile={bs.get('mouthSmileLeft', 0):.2f}")
            else:
                line = "no face"
            sys.stdout.write(f"\r[{fps:4.1f}fps {status:11s}] {line}".ljust(70))
            sys.stdout.flush()
    finally:
        cap.stop()
        cv2.destroyAllWindows()
        print("\n[test] stopped.")


if __name__ == "__main__":
    main()
