"""M4 entry point: webcam -> MediaPipe -> rig solver -> VRM renderer -> sink.

Full real-time VTubing pipeline:
  1. Background-thread webcam capture (latest-frame-only)
  2. MediaPipe FaceLandmarker (CPU, VIDEO mode) on new frames only
  3. RigSolver: ARKit blendshapes -> VRM expressions + head pose + eye gaze
  4. VRMRenderer: moderngl offscreen render with morph targets + skinning
  5. Output sink: preview window / file / virtual camera

Rate-decoupled: tracking runs only when a new webcam frame arrives; the
renderer always outputs at the configured target FPS, reusing the last rig.
"""
from __future__ import annotations

import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from src.avatar.renderer import VRMRenderer
from src.avatar.vrm_loader import load_vrm
from src.capture.webcam import WebcamCapture
from src.config import load_config
from src.output.sink import create_sink
from src.rig.solver import RigSolver
from src.tracking.landmarker import FaceLandmarker, extract_rig


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
    out_w = out_cfg.get("width", 1280)
    out_h = out_cfg.get("height", 720)
    renderer = VRMRenderer(model, out_w, out_h)
    solver = RigSolver(model, rig_cfg)
    sink = create_sink(cfg)
    target_dt = 1.0 / out_cfg.get("fps", 30.0)

    print(f"[main] avatar: {vrm_path}  output: {out_w}x{out_h}@{out_cfg.get('fps', 30)}fps")
    print("[main] calibrating... face the camera neutrally. Ctrl+C to stop.", flush=True)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    rgb_buf: np.ndarray | None = None
    last_ts = 0
    last_rig: dict | None = None
    frame_count = 0
    track_count = 0
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
                    track_count += 1

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

            renderer.apply_rig_state(state)
            out_frame = renderer.render()
            sink.send(out_frame)
            sink.sleep_until_next_frame()
            frame_count += 1

            elapsed = time.monotonic() - t0
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            status = "calibrated" if solver.is_calibrated else "calibrating"
            if state["detected"]:
                exprs = state.get("expressions", {})
                top = sorted(exprs.items(), key=lambda kv: -kv[1])[:3]
                expr_str = " ".join(f"{n}={v:.2f}" for n, v in top if v > 0.01)
                line = f"[{fps:4.1f}fps {status}] {expr_str}"
            else:
                line = f"[{fps:4.1f}fps {status}] no face"
            sys.stdout.write("\r" + line.ljust(80))
            sys.stdout.flush()
    finally:
        sink.close()
        cap.stop()
        sys.stdout.write("\n[main] stopped.\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
