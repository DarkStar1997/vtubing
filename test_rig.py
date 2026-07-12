"""M4 test: simulate rig data through RigSolver -> VRMRenderer.

Renders neutral pose and posed frames (expressions + head rotation + eye gaze)
to PNG files and reports pixel diffs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.avatar.renderer import VRMRenderer
from src.avatar.vrm_loader import load_vrm
from src.rig.solver import RigSolver


def make_rig(blendshapes: dict[str, float] | None = None,
             yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> dict:
    rot = Rotation.from_euler("YXZ", [yaw, pitch, roll], degrees=True)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = rot.as_matrix().astype(np.float32)
    return {
        "detected": True,
        "blendshapes": blendshapes or {},
        "matrix": m,
        "euler": np.array([yaw, pitch, roll], dtype=np.float32),
    }


def fg_pixels(arr, bg=(13, 13, 21)) -> int:
    return int(np.any(np.abs(arr.astype(int) - np.array(bg)) > 5, axis=-1).sum())


def diff_pixels(a, b) -> int:
    return int(np.any(np.abs(a.astype(int) - b.astype(int)) > 5, axis=-1).sum())


def main() -> None:
    vrm_path = sys.argv[1] if len(sys.argv) > 1 else "assets/avatars/sample.vrm"
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    model = load_vrm(vrm_path)
    renderer = VRMRenderer(model, 1280, 720)
    solver = RigSolver(model)
    dt = 1.0 / 30.0

    # --- Phase 1: calibration (30 neutral frames) ---
    print("[test] calibrating...")
    for _ in range(35):
        state = solver.update(make_rig(), dt)
    assert solver.is_calibrated, "calibration failed"
    print(f"[test] calibrated: {solver.is_calibrated}")

    # --- Render neutral (post-calibration, no expression) ---
    renderer.apply_rig_state(state)
    renderer.render_to_file(out_dir / "m4_neutral.png")
    neutral_img = renderer.render()
    print(f"[test] neutral: {fg_pixels(neutral_img)} fg px")

    # --- Test 1: 'aa' expression (jawOpen) ---
    rig = make_rig({"jawOpen": 0.7})
    state = solver.update(rig, dt)
    renderer.apply_rig_state(state)
    img_aa = renderer.render()
    renderer.render_to_file(out_dir / "m4_aa.png")
    print(f"[test] aa (jawOpen=0.7): {fg_pixels(img_aa)} fg px, {diff_pixels(img_aa, neutral_img)} diff px")

    # --- Test 2: blink ---
    rig = make_rig({"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9})
    state = solver.update(rig, dt)
    renderer.apply_rig_state(state)
    img_blink = renderer.render()
    renderer.render_to_file(out_dir / "m4_blink.png")
    print(f"[test] blink (0.9): {diff_pixels(img_blink, neutral_img)} diff px")

    # --- Test 3: happy (smile) ---
    rig = make_rig({"mouthSmileLeft": 0.7, "mouthSmileRight": 0.7})
    state = solver.update(rig, dt)
    renderer.apply_rig_state(state)
    img_happy = renderer.render()
    renderer.render_to_file(out_dir / "m4_happy.png")
    print(f"[test] happy (smile=0.7): {diff_pixels(img_happy, neutral_img)} diff px")

    # --- Test 4: head rotation (yaw=20, pitch=10) ---
    rig = make_rig(yaw=20.0, pitch=10.0)
    state = solver.update(rig, dt)
    renderer.apply_rig_state(state)
    img_head = renderer.render()
    renderer.render_to_file(out_dir / "m4_head.png")
    print(f"[test] head (yaw=20,pitch=10): {diff_pixels(img_head, neutral_img)} diff px")

    # --- Test 5: eye gaze (lookRight) ---
    rig = make_rig({"eyeLookInLeft": 0.8, "eyeLookOutRight": 0.8})
    state = solver.update(rig, dt)
    renderer.apply_rig_state(state)
    img_gaze = renderer.render()
    renderer.render_to_file(out_dir / "m4_gaze.png")
    print(f"[test] gaze (lookRight=0.8): {diff_pixels(img_gaze, neutral_img)} diff px")

    # --- Test 6: detection loss ---
    state = solver.update({"detected": False, "blendshapes": {}, "matrix": None,
                           "euler": np.zeros(3, dtype=np.float32)}, dt)
    renderer.apply_rig_state(state)
    img_loss = renderer.render()
    renderer.render_to_file(out_dir / "m4_loss.png")
    print(f"[test] loss (1 frame): {diff_pixels(img_loss, neutral_img)} diff px")

    print("[test] done. outputs in output/m4_*.png")


if __name__ == "__main__":
    main()
