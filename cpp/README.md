# C++ VTuber Pipeline

Real-time VTuber avatar tracking and rendering in pure C++. Uses MediaPipe C API
for face, pose, and hand tracking with a custom software rasterizer (no OpenGL/GPU
compute).

## Prerequisites

- **CMake** 3.20+
- **C++20** compiler (GCC 12+, Clang 15+, or MSVC 2022) with AVX2/FMA support
- **[uv](https://docs.astral.sh/uv/)** (standalone binary; supplies the MediaPipe
  C library — and a managed Python — via the project lockfile)

### SDL3

SDL3 is **not** required to be installed manually: if CMake cannot find an
existing SDL3 (distro package, vcpkg, or `CMAKE_PREFIX_PATH`), the pinned
release [`release-3.4.16`](https://github.com/libsdl-org/SDL/releases/tag/release-3.4.16)
is fetched via FetchContent and built from source as part of the project.

Installing a system package is optional and takes precedence over the source
build (faster incremental builds, distro integration):

```bash
# Ubuntu/Debian
sudo apt install libsdl3-dev

# Arch Linux
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
   and Ninja). SDL3 is fetched and built automatically — nothing to install.

2. **Get libmediapipe.dll** — install [uv](https://docs.astral.sh/uv/)
   (`winget install --id=astral-sh.uv -e` or the PowerShell installer from
   <https://docs.astral.sh/uv/getting-started/install/>), then from the repo
   root run:
   ```powershell
   uv sync
   ```
   This creates `.venv\` with the `mediapipe` wheel (uv downloads a managed
   Python automatically — no Python or pip install needed). CMake copies
   `.venv\Lib\site-packages\mediapipe\tasks\c\libmediapipe.dll` into `cpp\lib\`
   automatically at configure time and generates the import library
   (`libmediapipe.lib`) from its exports.

3. **Configure and build** from an *x64 Native Tools Command Prompt for
   VS 2022* (or a PowerShell that has run `vcvarsall.bat x64`):
   ```powershell
   cd cpp
   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
   cmake --build build
   ```

`mediapipe.dll` (and `SDL3.dll` if shared) are copied next to the executables
automatically after building. For a self-contained Windows release zip
(executables, DLLs, app-local MSVC runtime, models, avatars, README and
third-party notices), run from `cpp/`:

```powershell
powershell -ExecutionPolicy Bypass -File pack_release.ps1 -Version 0.1.0
```

The zip lands in `dist/` and runs on any Windows 10/11 x64 machine — no
VC++ Redistributable install required.

### Signing the release binaries

Unsigned binaries trigger SmartScreen ("Windows protected your PC") and
occasional antivirus false positives. Signing is integrated into packaging:

```powershell
# With a PFX certificate file (password is prompted):
powershell -File pack_release.ps1 -Version 0.2.0 -SignPfx mycert.pfx

# With a certificate in a store (hardware token, Certum SimplySign):
powershell -File pack_release.ps1 -Version 0.2.0 -SignThumbprint <hex>

# Or sign an already-staged package folder:
powershell -File sign_release.ps1 -Path dist\vtuber-cpu-v0.2.0-windows-x64 -Pfx mycert.pfx
```

Signatures are SHA-256 with an RFC 3161 timestamp (DigiCert responder by
default, override with `-TsaUrl`). Certificate routes for open-source
projects:

| Route | Cost | Notes |
|---|---|---|
| [Certum Open Source](https://certum.pl) | ~€69/yr | Code-signing cert for OSS; cloud signing via SimplySign → use `-SignThumbprint` |
| [SignPath Foundation](https://signpath.org) | free | Free signing for OSS projects, GitHub integration; requires application |
| [Azure Trusted Signing](https://azure.microsoft.com/products/trusted-signing) | $9.99/mo | Individual identity validation, CI-friendly |
| DigiCert/Sectigo EV | ~$400+/yr | Instant SmartScreen reputation; hardware token |

Self-signed certificates exercise the pipeline but do not remove SmartScreen
warnings for end users.

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

Dependencies (SDL3 release-3.4.16 when no system copy is found, GLM,
BS::thread_pool, cgltf, stb) are fetched automatically by CMake.

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
| `--models <dir>` | auto-detected | Path to MediaPipe model directory (`assets/models` or `../../assets/models` relative to the executable) |
| `--cam <index>` | 0 | Initial webcam device index |
| `--threads <N>` | 2 | CPU threads, minimum 2 (`0` = automatic, up to 16) |
| `--fps <N>` | 15 | Frame-rate cap in fps (`0` = unlimited) |
| `--no-pip` | off | Start without the webcam picture-in-picture overlay (W toggles it) |
| `[vrm_file]` | auto-detected | VRM avatar to load (`assets/avatars/male_52blendshapes.vrm`) |

By default the app runs lightweight: 2 threads and a 15 fps cap (tracking
included), leaving plenty of CPU headroom for games and OBS.

### Examples

```bash
# Lightweight (default): 2 threads, 15 fps cap
./vtuber_live

# Full quality: automatic threads, uncapped frame rate
./vtuber_live --threads 0 --fps 0

# Balanced: 4 threads, 60 fps cap
./vtuber_live --threads 4 --fps 60

# Different avatar
./vtuber_live ../../assets/avatars/DefaultSampleAvatar.vrm
```

### Controls

| Key | Action |
|---|---|
| `C` | Toggle camera selection menu & keyboard shortcuts legend |
| `1`–`9` | Directly select camera source |
| `R` | Rescan connected camera devices |
| `SPACE` | Calibrate (neutral pose for face, body, hands) |
| `W` | Toggle picture-in-picture webcam & UI overlay |
| `ESC` | Quit |

## GPU usage

MediaPipe creates EGL/GL contexts by default. The `egl_stub.c` compiled into
`vtuber_live` with `-rdynamic` overrides EGL symbols at link time, forcing
pure-CPU inference. The only residual GPU usage (~10MB) is from the SDL/X11
window backing store.
