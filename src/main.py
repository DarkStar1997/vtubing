"""M1 entry point: webcam -> MediaPipe face tracking -> console validation.

Runs the full capture + tracking path and prints a compact per-frame summary
(key blendshapes + head yaw/pitch/roll + tracking FPS) to the console. This
validates that tracking works end-to-end before wiring up the avatar renderer.
"""
from __future__ import annotations

import signal
import sys
import threading
import time

import cv2
import numpy as np

from src.capture.webcam import WebcamCapture
from src.config import load_config
from src.tracking.landmarker import FaceLandmarker, extract_rig


def main() -> None:
    cfg = load_config()
    cam_cfg = cfg.get("camera", {})
    track_cfg = cfg.get("tracking", {})

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

    print("[M1] warming up. Face the camera. Ctrl+C to stop.", flush=True)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    rgb_buf: np.ndarray | None = None
    last_ts = 0
    last_rig: dict | None = None
    frame_count = 0
    t0 = time.monotonic()

    try:
        while not stop.is_set():
            frame, new = cap.get_latest()
            if frame is None:
                time.sleep(0.005)
                continue

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
                frame_count += 1
            else:
                time.sleep(0.002)

            if last_rig is not None and last_rig["detected"]:
                bs = last_rig["blendshapes"]
                yaw, pitch, roll = last_rig["euler"]
                line = (
                    f"jawOpen={bs.get('jawOpen', 0.0):.2f} "
                    f"blinkL={bs.get('eyeBlinkLeft', 0.0):.2f} "
                    f"blinkR={bs.get('eyeBlinkRight', 0.0):.2f} "
                    f"smileL={bs.get('mouthSmileLeft', 0.0):.2f} "
                    f"smileR={bs.get('mouthSmileRight', 0.0):.2f} "
                    f"| yaw={yaw:6.1f} pitch={pitch:6.1f} roll={roll:6.1f}"
                )
            else:
                line = "[no face detected]"

            elapsed = time.monotonic() - t0
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            sys.stdout.write("\r" + f"[{fps:4.1f}fps] " + line.ljust(90))
            sys.stdout.flush()
    finally:
        cap.stop()
        sys.stdout.write("\n[M1] stopped.\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
