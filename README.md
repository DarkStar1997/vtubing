# VTubing Pipeline

Open-source real-time VTubing pipeline: webcam → MediaPipe face/hand/pose tracking → VRM 3D avatar → OBS browser source.

Lightweight enough to run alongside demanding games. All tracking runs on CPU (no GPU contention with your game); the avatar renders in a browser via three-vrm (WebGL).

## Features

- **Face tracking** — 52 ARKit blendshapes + head pose via MediaPipe FaceLandmarker
- **Hand tracking** — per-joint finger curls + palm pronation/supination via MediaPipe HandLandmarker
- **Body tracking** — arm quaternions, spine lean/twist, standing/sitting detection via MediaPipe PoseLandmarker
- **VRM support** — both VRM 0.x and VRM 1.0 models
- **Real-time avatar** — three.js + three-vrm in the browser, spring bones, eye gaze (lookAt)
- **OBS integration** — browser source with transparent background, webcam PiP overlay
- **Auto-calibration** — neutral face pose captured in the first ~1 second
- **Dynamic camera** — framing adapts to sitting/standing and body coverage in webcam
- **One-Euro filtering** — smooth tracking without lag

## Architecture

```
Webcam (OpenCV) ──► MediaPipe (CPU) ──► Rig Solver ──► WebSocket ──► Browser (three-vrm)
                      │                    │               │
                      ├─ FaceLandmarker    ├─ ARKit→VRM    ├─ HTTP server (aiohttp)
                      ├─ HandLandmarker    │  expressions  ├─ Serves frontend + VRM file
                      └─ PoseLandmarker    ├─ Head pose    └─ MJPEG webcam stream
                                           ├─ Eye gaze
                                           └─ Arm/spine
```

Python handles webcam capture, MediaPipe tracking, and rig solving. A WebSocket broadcasts the rig state as JSON. The browser loads a VRM model and applies the rig in real-time.

## Requirements

- **Python 3.11–3.12** (managed automatically by uv if not installed)
- **uv** — [astral.sh/uv](https://docs.astral.sh/uv/) (Python package manager)
- **Webcam** — any USB webcam or laptop camera
- **Browser** — Chrome/Edge/Firefox (for the avatar renderer)
- **OBS Studio** (optional, for streaming)

### Windows
- OBS Studio with the **Virtual Camera** plugin (bundled with OBS)

### Linux (development only)
- `v4l2loopback-dkms` (only needed for virtual camera output, not for browser-based rendering)

## Quick Start

1. **Clone**
   ```bash
   git clone https://github.com/DarkStar1997/vtubing.git
   cd vtubing
   ```

2. **Add a VRM avatar**

   Place a `.vrm` file in `assets/avatars/`. A few free models:
   - [hinzka/52blendshapes-for-VRoid-face](https://github.com/hinzka/52blendshapes-for-VRoid-face) — VRoid male/female with full 52 ARKit blendshapes
   - [madjin/vrm-samples](https://github.com/madjin/vrm-samples) — CC0 VRoid models
   - [VRoid Hub](https://hub.vroid.com/) — many free models (check license for redistribution)

   Set the path in `config.yaml`:
   ```yaml
   avatar:
     path: assets/avatars/your_model.vrm
   ```

3. **Run**
   ```bash
   uv run python -m src.main
   ```

   uv automatically installs the right Python version and all dependencies on first run.

4. **Open the avatar**

   Navigate to **http://localhost:8080** in your browser. The avatar renders with a transparent background.

5. **Add to OBS** (for streaming)

   - Add a **Browser Source**
   - URL: `http://localhost:8080`
   - Width: 1280, Height: 720
   - Check **"Refresh browser when scene becomes active"**

   The webcam PiP overlay (bottom-right) shows annotated face/hand/pose landmarks.

## Configuration

All settings are in `config.yaml`:

| Section | Key | Description |
|---------|-----|-------------|
| `camera` | `index`, `width`, `height`, `fps` | Webcam device and resolution |
| `avatar` | `path` | Path to `.vrm` model file |
| `tracking` | `min_*_confidence` | Detection thresholds (lower = easier detection, noisier) |
| `rig.calibration` | `frames` | Number of frames to average for neutral pose (default 30 ≈ 1s) |
| `rig.head` | `yaw_gain`, `max_yaw`, etc. | Head rotation gain and clamp limits |
| `output` | `width`, `height`, `fps` | Output resolution and framerate |

## How It Works

### Tracking (Python)

Three MediaPipe models run in VIDEO mode on CPU:
- **FaceLandmarker** — 478 landmarks, 52 ARKit blendshapes, facial transformation matrix
- **HandLandmarker** — 21 landmarks per hand, finger joint angles, palm twist
- **PoseLandmarker** — 33 body landmarks, arm direction quaternions, spine lean/twist

Models auto-download to `assets/models/` on first run (~10MB each).

### Rig Solving (Python)

- ARKit 52 blendshapes → VRM expression presets (happy, blink, mouth shapes, etc.)
- Head pose matrix → head bone rotation (one-euro filtered, gain-clamped)
- Eye blendshapes → VRM lookAt (yaw/pitch)
- Hand landmarks → per-joint finger curl angles (raw radians)
- Pose landmarks → upper/lower arm quaternions via direction-vector matching

### Rendering (Browser)

[three.js](https://threejs.org/) + [@pixiv/three-vrm](https://github.com/pixiv/three-vrm) handle all 3D rendering:
- VRM model loading (0.x and 1.0)
- Morph target (blendshape) animation
- Bone hierarchy with skinning
- Spring bone physics (hair, clothing)
- Eye gaze via lookAt rig
- Transparent background for OBS browser source

## Project Structure

```
vtubing/
├── config.yaml              # All configuration
├── pyproject.toml           # Dependencies (uv virtual project)
├── frontend/
│   ├── index.html           # Browser UI (canvas + webcam PiP)
│   └── avatar.js            # three-vrm renderer + WebSocket client
├── src/
│   ├── main.py              # Entry point: capture → track → solve → broadcast
│   ├── server.py            # aiohttp HTTP + WebSocket server
│   ├── config.py            # YAML config loader
│   ├── constants.py         # ARKit blendshape names, VRM expression presets
│   ├── capture/
│   │   └── webcam.py        # OpenCV webcam capture (daemon thread, latest-frame)
│   ├── tracking/
│   │   ├── landmarker.py    # FaceLandmarker wrapper
│   │   ├── hand_landmarker.py  # HandLandmarker wrapper + finger angles
│   │   └── pose_landmarker.py  # PoseLandmarker wrapper + arm quaternions
│   ├── rig/
│   │   ├── solver.py        # ARKit→VRM mapping, head pose, eye gaze, calibration
│   │   └── one_euro.py      # One-Euro adaptive low-pass filter
│   └── avatar/
│       ├── vrm_loader.py    # VRM 0.x + 1.0 loader (pygltflib)
│       ├── gltf_utils.py    # glTF accessor/matrix utilities
│       ├── renderer.py      # Legacy moderngl renderer (unused, kept for reference)
│       └── shaders.py       # Legacy GLSL shaders (unused)
└── assets/
    ├── avatars/             # .vrm model files (gitignored)
    └── models/              # MediaPipe .task models (auto-downloaded, gitignored)
```

## Performance

- **Tracking**: ~7ms/frame for face, ~5ms for hands, ~8ms for pose (on Ryzen/Intel CPU)
- **Rendering**: GPU-accelerated in browser, trivial for any modern GPU
- **Latency**: dominated by webcam framerate (15-30fps typical USB webcams)

## License

MIT. See bundled VRM model licenses separately — VRM models have their own terms (check `allowRedistribution` and `commercialUsage` in the model's meta).

## Acknowledgements

- [MediaPipe](https://developers.google.com/mediapipe) — Google's on-device ML tracking
- [@pixiv/three-vrm](https://github.com/pixiv/three-vrm) — Pixiv's VRM renderer for three.js
- [three.js](https://threejs.org/) — WebGL 3D library
- [hinzka/52blendshapes-for-VRoid-face](https://github.com/hinzka/52blendshapes-for-VRoid-face) — VRoid Perfect Sync models
