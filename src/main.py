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
from src.rig.one_euro import OneEuroFilter
from src.rig.solver import RigSolver
from src.server import RigServer
from src.tracking.hand_landmarker import HandLandmarker, extract_hands
from src.tracking.landmarker import FaceLandmarker, extract_rig
from src.tracking.pose_landmarker import PoseTracker

_CONTOURS = [(c.start, c.end) for c in _FLC.FACE_LANDMARKS_CONTOURS]
_TESSELATION = [(c.start, c.end) for c in _FLC.FACE_LANDMARKS_TESSELATION]

_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

_POSE_CONNECTIONS = [
    (11,12),
    (11,13),(13,15),
    (12,14),(14,16),
    (15,17),(15,19),(15,21),(17,19),
    (16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
]


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


def _draw_hand_landmarks(frame: np.ndarray, result) -> np.ndarray:
    """Draw MediaPipe hand landmarks + connections on a BGR frame copy."""
    if result is None or not result.hand_landmarks:
        return frame
    annotated = frame.copy()
    h, w = frame.shape[:2]
    for hand_lms in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
        for s, e in _HAND_CONNECTIONS:
            if s < len(pts) and e < len(pts):
                cv2.line(annotated, pts[s], pts[e], (255, 200, 0), 2)
        for pt in pts:
            cv2.circle(annotated, pt, 3, (0, 100, 255), -1)
    return annotated


def _draw_pose_landmarks(frame: np.ndarray, result) -> np.ndarray:
    """Draw MediaPipe pose skeleton on a BGR frame copy."""
    if result is None or not result.pose_landmarks:
        return frame
    annotated = frame.copy()
    h, w = frame.shape[:2]
    for pose in result.pose_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose]
        for s, e in _POSE_CONNECTIONS:
            if s < len(pts) and e < len(pts):
                cv2.line(annotated, pts[s], pts[e], (0, 255, 255), 3)
        for i, pt in enumerate(pts):
            color = (0, 0, 255) if i in (11, 12, 13, 14, 15, 16) else (0, 180, 180)
            cv2.circle(annotated, pt, 5, color, -1)
    return annotated


_hand_twist_neutral: dict[str, float] = {}
_twist_sin_f: dict[str, OneEuroFilter] = {"left": OneEuroFilter(1.5, 0.01), "right": OneEuroFilter(1.5, 0.01)}
_twist_cos_f: dict[str, OneEuroFilter] = {"left": OneEuroFilter(1.5, 0.01), "right": OneEuroFilter(1.5, 0.01)}
_twist_prev: dict[str, float] = {"left": 0.0, "right": 0.0}


def _process_hand_twist(hands: dict | None, dt: float) -> None:
    """Filter palm twist in-place: circular neutral subtraction + sin/cos smoothing."""
    if not hands:
        return
    for side in ("left", "right"):
        h = hands.get(side)
        if h is None or "twist" not in h:
            _twist_prev[side] = 0.0
            continue
        raw = h["twist"]
        neutral = _hand_twist_neutral.get(side, 0.0)
        if not math.isfinite(raw):
            raw = _twist_prev[side]
        # Circular difference (handles ±π wraparound)
        delta = math.atan2(math.sin(raw - neutral), math.cos(raw - neutral))
        # Filter sin/cos separately for smooth circular smoothing
        s = _twist_sin_f[side].filter(math.sin(delta), dt)
        c = _twist_cos_f[side].filter(math.cos(delta), dt)
        smooth = math.atan2(s, c)
        _twist_prev[side] = smooth
        h["twist"] = smooth


def _state_to_ws(state: dict, hands: dict | None = None, pose: dict | None = None,
                 framing: dict | None = None) -> dict:
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

    # Strip internal keys (prefixed with _) from hands before sending
    clean_hands = None
    if hands:
        clean_hands = {}
        for side, h in hands.items():
            if h is None:
                clean_hands[side] = None
            else:
                clean = {k: v for k, v in h.items() if not k.startswith("_")}
                clean_hands[side] = clean

    return {
        "detected": bool(state.get("detected", False)),
        "calibrated": bool(state.get("calibrated", False)),
        "expressions": {k: float(v) for k, v in state.get("expressions", {}).items()},
        "head": head,
        "gaze": gaze,
        "hands": clean_hands,
        "pose": pose,
        "framing": framing,
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

    hand_landmarker = HandLandmarker(
        num_hands=track_cfg.get("num_hands", 2),
        min_detection_confidence=track_cfg.get("min_hand_detection_confidence", 0.5),
        min_tracking_confidence=track_cfg.get("min_hand_tracking_confidence", 0.5),
    )

    pose_tracker = PoseTracker(
        num_poses=1,
        min_detection_confidence=track_cfg.get("min_pose_detection_confidence", 0.5),
        min_tracking_confidence=track_cfg.get("min_pose_tracking_confidence", 0.5),
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
    last_hands: dict | None = None
    last_hand_result = None
    last_pose: dict | None = None
    last_pose_result = None
    frame_count = 0
    t0 = time.monotonic()
    t_prev = t0

    # Framing calibration — collect head + shoulder positions from webcam
    _calib_face_mins: list[float] = []
    _calib_face_maxs: list[float] = []
    _calib_shoulder_ys: list[float] = []
    _calib_twist: dict[str, list[float]] = {"left": [], "right": []}
    framing: dict | None = None
    _calib_face_h: float = 0.0
    _standing_filter = OneEuroFilter(0.4, 0.0)  # slow, smooth transitions
    _body_extent_filter = OneEuroFilter(0.8, 0.0)  # smooth body extent transitions
    _stand_state = 0  # binary hysteresis state: 0=sit, 1=stand

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
                    hand_result = hand_landmarker.detect(rgb_buf, ts + 1)
                    last_hands = extract_hands(hand_result)
                    _process_hand_twist(last_hands, target_dt)
                    last_hand_result = hand_result
                    pose_result = pose_tracker.detect(rgb_buf, ts + 2)
                    last_pose = pose_tracker.extract_angles(pose_result, target_dt, last_hands)
                    last_pose_result = pose_result
                    last_pose_result = pose_result

                    # Collect framing data during calibration
                    if not solver.is_calibrated and last_result.face_landmarks:
                        face_ys = [lm.y for lm in last_result.face_landmarks[0]]
                        _calib_face_mins.append(min(face_ys))
                        _calib_face_maxs.append(max(face_ys))
                        if pose_result.pose_landmarks:
                            pl = pose_result.pose_landmarks[0]
                            _calib_shoulder_ys.append((pl[11].y + pl[12].y) / 2)

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

            # Compute framing once calibration completes
            if solver.is_calibrated and framing is None and _calib_face_mins:
                face_min = float(np.mean(_calib_face_mins))
                face_max = float(np.mean(_calib_face_maxs))
                face_h = face_max - face_min
                face_center_y = (face_min + face_max) / 2
                if _calib_shoulder_ys:
                    shoulder_y = float(np.mean(_calib_shoulder_ys))
                else:
                    shoulder_y = face_max + face_h * 0.5
                framing = {
                    "face_h": face_h,
                    "face_center_y": face_center_y,
                    "shoulder_y": shoulder_y,
                }
                _calib_face_h = face_h
                pose_tracker.calibrate_neutral()
                # Skip twist calibration — use neutral=0 for both hands.
                # The arcsin-based twist is ≈0 when palms face camera, which
                # is the natural neutral. Calibration caused asymmetric deltas.
                print(f"\n[main] framing: face_h={face_h:.3f} "
                      f"face_center={face_center_y:.3f} shoulder={shoulder_y:.3f}")
            # Standing detection: hips visible in frame → user stepped back
            # Uses hysteresis to prevent rapid flicker at the threshold edge.
            _hip_vis = 0.0
            _body_extent = 0.0  # how many torso-lengths of body visible below shoulders
            if last_pose_result and last_pose_result.pose_landmarks:
                plms = last_pose_result.pose_landmarks[0]
                _hip_vis = (plms[23].visibility + plms[24].visibility) / 2.0
                # Body extent: continuous metric of how far down the body is visible.
                # Uses shoulder→hip distance as a reference unit (torso length).
                #   0 = just shoulders, 1 = to hips, 2 = to knees, 3+ = to ankles
                sh_y = (plms[11].y + plms[12].y) / 2
                hip_y = (plms[23].y + plms[24].y) / 2
                torso_unit = max(hip_y - sh_y, 0.01)
                # Find the lowest visible landmark (highest image Y = bottom of frame)
                lowest_y = sh_y
                for idx in [27, 28, 25, 26, 23, 24]:
                    if plms[idx].visibility > 0.3 and plms[idx].y > lowest_y:
                        lowest_y = plms[idx].y
                _body_extent = (lowest_y - sh_y) / torso_unit
            _body_extent = _body_extent_filter.filter(_body_extent, dt)
            if _stand_state:
                _stand_state = 0 if _hip_vis < 0.30 else 1  # lower threshold to exit stand
            else:
                _stand_state = 1 if _hip_vis > 0.50 else 0  # higher threshold to enter stand
            standing = _standing_filter.filter(float(_stand_state), dt)

            # Inject standing + body coverage into pose dict
            if last_pose is None:
                last_pose = {}
            last_pose["standing"] = standing
            last_pose["body_extent"] = _body_extent

            server.broadcast(_state_to_ws(state, last_hands, last_pose, framing))

            if frame is not None:
                annotated = _draw_landmarks(frame, last_result)
                annotated = _draw_hand_landmarks(annotated, last_hand_result)
                annotated = _draw_pose_landmarks(annotated, last_pose_result)
                ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    server.set_camera_frame(jpeg.tobytes())

            frame_count += 1
            elapsed = time.monotonic() - t0
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            status = "calibrated" if solver.is_calibrated else "calibrating"
            pose_tag = "no-pose"
            if last_pose:
                pose_tag = "stand" if last_pose.get("standing", 0) > 0.5 else "sit"
            hand_tag = "hands:" + "".join(
                s[0].upper() if last_hands and last_hands.get(s) else "-"
                for s in ["left", "right"]
            )
            if state["detected"]:
                exprs = state.get("expressions", {})
                top = sorted(exprs.items(), key=lambda kv: -kv[1])[:3]
                expr_str = " ".join(f"{n}={v:.2f}" for n, v in top if v > 0.01)
                line = f"[{fps:4.1f}fps {status:11s} {pose_tag:8s} {hand_tag:8s}] {expr_str}"
            else:
                line = f"[{fps:4.1f}fps {status:11s} {pose_tag:8s} {hand_tag:8s}] no face"
            if frame_count % 30 == 0 and pose_tracker.last_debug:
                dbg = pose_tracker.last_debug
                d = dbg["dirs"]
                print(f"\n  L: sh→el=({d['lu'][0]:+.2f},{d['lu'][1]:+.2f},{d['lu'][2]:+.2f}) "
                      f"el→wr=({d['ll'][0]:+.2f},{d['ll'][1]:+.2f},{d['ll'][2]:+.2f})")
                print(f"  R: sh→el=({d['ru'][0]:+.2f},{d['ru'][1]:+.2f},{d['ru'][2]:+.2f}) "
                      f"el→wr=({d['rl'][0]:+.2f},{d['rl'][1]:+.2f},{d['rl'][2]:+.2f})")
                print(f"  lean={dbg['lean']:+.3f}  twist={dbg['spine_y']:+.3f}  lateral={dbg['spine_z']:+.3f}")
                print(f"  standing={last_pose.get('standing', 0):.2f}" if last_pose else "")
                # Hand detection debug
                if last_hands:
                    for side in ["left", "right"]:
                        h = last_hands.get(side)
                        if h:
                            f = h.get("fingers", {})
                            curls = [f.get(fing, [0]*3)[0] for fing in ["index", "middle", "ring", "little"]]
                            curl_str = (f"index={curls[0]:.2f} mid={curls[1]:.2f} "
                                  f"ring={curls[2]:.2f} little={curls[3]:.2f}")
                            tw = h.get("twist", 0)
                            tw_n = _hand_twist_neutral.get(side, 0)
                            print(f"  hand[{side}]: {curl_str} twist_delta={tw:+.2f} (neutral={tw_n:.2f})")
                        else:
                            print(f"  hand[{side}]: not detected")
            sys.stdout.write("\r" + line.ljust(100))
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
