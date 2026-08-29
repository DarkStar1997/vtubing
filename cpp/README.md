# C++ VTuber Pipeline

Real-time VTuber avatar tracking and rendering in pure C++. Uses MediaPipe C API
for face, pose, and hand tracking with a custom software rasterizer (no OpenGL/GPU
compute).

## Prerequisites

- **CMake** 3.20+
- **C++20** compiler (GCC 12+, Clang 15+, or MSVC 2022) with AVX2/FMA support
- **SDL3** (development libraries)
- **[uv](https://docs.astral.sh/uv/)** (standalone binary; supplies the MediaPipe
  C library — and a managed Python — via the project lockfile)

### Install SDL3 (Ubuntu/Debian)

```bash
sudo apt install libsdl3-dev
```

### Install SDL3 (Arch Linux)

```bash
sudo pacman -S sdl3
```

Arch Linux also requires these packages if not already installed:

```bash
sudo pacman -S cmake gcc base-devel
```

### Setup on Windows (MSVC + CMake + Ninja)

1. **Install tools**: [Visual Studio 2022](https://visualstudio.microsoft.com/)
   or [Build Tools for Visual Studio 2022](https://visualstudio.microsoft.com/downloads/)
   with the *Desktop development with C++* workload (includes MSVC, CMake,
   and Ninja).

2. **Install SDL3** — either via
   [vcpkg](https://learn.microsoft.com/en-us/vcpkg/get_started/overview):
   ```powershell
   vcpkg install sdl3:x64-windows
   ```
   or download the prebuilt `SDL3-devel-3.x.x-VC.zip` from the
   [SDL3 releases](https://github.com/libsdl-org/SDL/releases) page and unzip
   it somewhere local.

3. **Get libmediapipe.dll** — install [uv](https://docs.astral.sh/uv/)
   (`winget install --id=astral-sh.uv -e` or the PowerShell installer from
   <https://docs.astral.sh/uv/getting-started/install/>), then from the repo
   root run:
   ```powershell
   uv sync
   ```
   This creates `.venv\` with the `mediapipe` wheel (uv downloads a managed
   Python automatically — no Python or pip install needed). CMake copies
   `.venv\Lib\site-packages\mediapipe\tasks\c\libmediapipe.dll` into `cpp\lib\`
   automatically at configure time.

4. **Configure and build** from an *x64 Native Tools Command Prompt for
   VS 2022* (or a PowerShell that has run `vcvarsall.bat x64`):
   ```powershell
   cd cpp
   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="C:/path/to/SDL3/devel/cmake"
   cmake --build build
   ```

   With vcpkg, add `-DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake`
   instead of `CMAKE_PREFIX_PATH`.

`mediapipe.dll` (and `SDL3.dll` if shared) are copied next to the executables
automatically after building. For distribution, ship the `.exe` files together
with those DLLs, the `assets/models/*.task` files, and a VRM avatar. Target
machines need the [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
(the `x64` variant).

### MediaPipe shared library

The pre-built MediaPipe C API library must be present in `cpp/lib/`
(`libmediapipe.so` on Linux, `libmediapipe.dll` on Windows). Running
`uv sync` in the repo root installs the `mediapipe` wheel into `.venv/`,
and CMake automatically copies the library from there into `cpp/lib/`
at configure time:

```bash
uv sync
cd cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release
```

No pip or system Python required — uv manages its own interpreter. If you
prefer a manual copy, the library lives at
`.venv/lib/python3.*/site-packages/mediapipe/tasks/c/libmediapipe.so`
(Linux) or `.venv\Lib\site-packages\mediapipe\tasks\c\libmediapipe.dll`
(Windows).

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
