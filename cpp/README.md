# C++ VTuber Pipeline

Real-time VTuber avatar tracking and rendering in pure C++. Uses MediaPipe C API
for face, pose, and hand tracking with a custom software rasterizer (no OpenGL/GPU
compute).

## Prerequisites

- **CMake** 3.20+
- **C++20** compiler (GCC 12+, Clang 15+) with AVX2/FMA support
- **SDL3** (development libraries)
- **Python 3** (for MediaPipe model setup only)

### Install SDL3 (Ubuntu/Debian)

```bash
sudo apt install libsdl3-dev
```

### MediaPipe shared library

The pre-built `libmediapipe.so` must be placed in `cpp/lib/`. You can copy it
from a Python `mediapipe` installation:

```bash
pip install mediapipe
cp .venv/lib/python3.12/site-packages/mediapipe/tasks/c/libmediapipe.so cpp/lib/
```

### Model files

The three MediaPipe `.task` files must be in `assets/models/`:

```
assets/models/face_landmarker.task
assets/models/hand_landmarker.task
assets/models/pose_landmarker_full.task
```

These are auto-downloaded by the Python pipeline on first run, or can be
downloaded manually from `storage.googleapis.com/mediapipe-models/`.

## Building

```bash
cd cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Dependencies (GLM, BS::thread_pool, cgltf, stb) are fetched automatically by CMake.

## Targets

| Target | Description |
|---|---|
| `vtuber_live` | Live webcam tracking + real-time avatar rendering |
| `vtuber_cpu` | Offline benchmark renderer |
| `test_tracker` | Static image inference test |

## Running

```bash
cd cpp/build
./vtuber_live [options] [vrm_file]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--models <dir>` | `../../assets/models` | Path to MediaPipe model directory |
| `--threads <N>` | auto (max 16) | Limit CPU threads (minimum 2) |
| `--fps <N>` | unlimited | Cap frame rate to reduce CPU usage |
| `[vrm_file]` | `../../assets/avatars/male_52blendshapes.vrm` | VRM avatar to load |

### Examples

```bash
# Full quality, all threads
./vtuber_live

# Low CPU usage: 4 threads, 60fps cap
./vtuber_live --threads 4 --fps 60

# Different avatar
./vtuber_live ../../assets/avatars/DefaultSampleAvatar.vrm
```

### Controls

| Key | Action |
|---|---|
| `SPACE` | Calibrate (neutral pose for face, body, hands) |
| `W` | Toggle picture-in-picture webcam overlay |
| `ESC` | Quit |

## GPU usage

MediaPipe creates EGL/GL contexts by default. The `egl_stub.c` compiled into
`vtuber_live` with `-rdynamic` overrides EGL symbols at link time, forcing
pure-CPU inference. The only residual GPU usage (~10MB) is from the SDL/X11
window backing store.
