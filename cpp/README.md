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

### Setup on macOS 14+ (Apple Silicon)

1. **Install tools**: Xcode Command Line Tools plus CMake and Ninja:
   ```bash
   xcode-select --install
   brew install cmake ninja
   ```

2. **Get libmediapipe.dylib** — from the repo root run `uv sync` (see above);
   CMake copies the dylib from `.venv/` into `cpp/lib/` automatically at
   configure time (a `libmediapipe_source.so` symlink is created next to it
   to match the wheel dylib's install name).

3. **Configure and build**:
   ```bash
   cd cpp
   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
   cmake --build build
   ```

Binaries target macOS 14+ on arm64. For a self-contained macOS release zip
(executable, dylibs with `@executable_path` rpaths, models, avatars, README,
third-party notices), run from `cpp/`:

```bash
./pack_release.sh -v 0.4.0
```

The zip lands in `dist/`. It contains both a double-clickable
**VTuber Live.app** bundle (dylibs under `Contents/Frameworks`, with
`NSCameraUsageDescription` set) and a plain `vtuber_live` binary for
terminal use. Binaries are ad-hoc signed (mandatory on arm64); they are
not notarized, so users bypass Gatekeeper once (right-click → *Open*, or
`xattr -dr com.apple.quarantine "VTuber Live.app"`) — see the packaged
README.

#### Linux

For a self-contained Linux release tarball (executable, SDL3/MediaPipe
libraries with an `$ORIGIN` rpath, models, avatars, README, third-party
notices), run from `cpp/` (`patchelf` is required):

```bash
./pack_release_linux.sh -v 0.4.0
```

The tarball lands in `dist/` and runs on any x86-64 Linux with AVX2 and a
glibc as new as the build machine's (CI builds on Ubuntu 24.04 → glibc 2.39).

The app is silent by default: the MediaPipe library's glog diagnostics
(INFO/WARNING lines) go to stderr and ignore `GLOG_minloglevel`, so in
non-verbose mode the launcher redirects stdout/stderr to `/dev/null`
(critical errors keep a private handle on the real stderr). Pass
`--verbose` for the full diagnostic log.

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

### Building releases via GitHub Actions

The [`release` workflow](../.github/workflows/release.yml) builds all three
platforms on GitHub Actions and publishes the bundles as a GitHub release —
no local toolchain needed:

```bash
gh workflow run release.yml -f version=v0.4.0
# optional: -f prerelease=true -f make_latest=false
gh run watch        # monitor progress
```

Each platform job syncs the mediapipe wheel (`uv sync`), downloads the
tracking models and VRM avatars, builds (SDL3 is fetched via FetchContent),
smoke-tests the binaries (renderer benchmark everywhere, plus a headless
`vtuber_live` run on Linux/macOS), and then runs the platform pack script —
`pack_release.ps1` on Windows (exe, DLLs, app-local MSVC CRT), `pack_release.sh`
on macOS (binary, dylibs, `.app` bundle, ad-hoc signed) and
`pack_release_linux.sh` on Linux (binary, libs, `$ORIGIN` rpath). The pack
scripts verify each staged bundle is self-contained before it is uploaded,
and the workflow finally creates the release with the three bundles attached.

macOS is arm64-only: the `mediapipe` wheel — which provides the prebuilt
MediaPipe C library — has no macOS x86_64 build. Windows binaries from CI are
unsigned; for Authenticode signing, build and pack locally with
`pack_release.ps1 -SignPfx` (see above).

### MediaPipe shared library

The pre-built MediaPipe C API library must be present in `cpp/lib/`
(`libmediapipe.so` on Linux, `libmediapipe.dylib` on macOS,
`libmediapipe.dll` on Windows). Running
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
| `test_tracker` | Static image inference test (needs an image argument) |
| `test_calibration` | Rig calibration regression tests |
| `test_rig` | ARKit→VRM expression mapping, gaze, head-knob regression tests |
| `test_springbone` | Springbone physics regression tests |
| `test_render` | Expression-driven render pixel-diff tests (uses the default avatar) |
| `test_vrm_loader` | VRM 1.0 expressions / lookAt / springbone parsing tests (fixture) |

All regression tests are wired into CTest: run `ctest` from the build directory.

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
| `--threads <N>` | 2 | Renderer worker threads, minimum 2 (`0` = automatic, up to 16) |
| `--fps <N>` | 15 | Frame-rate cap in fps (`0` = unlimited) |
| `--bg <mode>` | white | Background: `white`, `black`, `green` (chroma key for OBS), or `transparent` (alpha screenshots; shown as a checkerboard, `P` saves PNGs with real alpha). Also changeable at runtime in the settings panel. |
| `--no-pip` | off | Start without the webcam picture-in-picture overlay (W toggles it) |
| `--no-spring` | off | Start with springbones disabled (hair/clothes physics; toggle at runtime in [S] Settings) |
| `-v`, `--verbose` | off | Print diagnostics to the terminal (default: quiet; fps and calibration status are shown on the window) |
| `[vrm_file]` | auto-detected | VRM avatar to load (`assets/avatars/male_52blendshapes.vrm`) |

By default the app runs lightweight: 2 renderer worker threads and a 15 fps
cap (tracking included), leaving plenty of CPU headroom for games and OBS.

### Thread & CPU usage

The MediaPipe C library sizes its internal thread pools from the CPU count
(look for `Fiber init: ... concurrency = N` in the log) and offers no knob to
change that — `--threads` only bounds the renderer's worker tasks, and the
`OMP_NUM_THREADS` / `TF_NUM_*` environment variables are not honored by
MediaPipe's XNNPACK delegate. This is idle capacity, not consumption: the
pools sit parked between frames. Measured on an Apple M4 (10 cores), whole
process: ~0.85 core at the default 15 fps cap, ~0.9 core fully uncapped
(`--threads 0 --fps 0`) — tracking and rendering are serialized, so real
usage stays around one core regardless of pool size.

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
| `SPACE` | Calibrate / re-calibrate neutral pose (face, body, hands) and camera framing |
| `S` | Toggle settings panel (see below) |
| `P` | Save a PNG screenshot (`screenshot_NNN.png`, real alpha when background = transparent) |
| `W` | Toggle picture-in-picture webcam & UI overlay (incl. fps readout, menus and settings panel) |
| `ESC` | Quit |

### Settings panel (`S`)

Runtime-tunable knobs, adjusted with `UP`/`DOWN` (select) and `LEFT`/`RIGHT`
(adjust). The panel is part of the UI overlay and hides completely with the
`W` visibility toggle.

| Setting | Range | Description |
|---|---|---|
| Head yaw/pitch/roll gain | 0–1 | Tracking-to-bone rotation scale (default 0.65) |
| Head max yaw/pitch/roll | 5–90° | Rotation clamps (defaults 35/20/15°) |
| Gaze scale | 0–2 | Eye-gaze strength multiplier on the model's lookAt range maps |
| Springbones | on/off | Hair/clothes physics |
| Spring stiffness / gravity | 0–2 | Springbone parameter multipliers |
| Background | 4 modes | White / black / green (chroma key) / transparent |

## Avatar feature support

| Feature | VRM 0.x | VRM 1.0 |
|---|---|---|
| ARKit "Perfect Sync" blendshapes | direct passthrough | direct passthrough |
| Standard expressions (`aa`/`ih`/`ou`/`ee`/`oh`, `happy`, `angry`, `sad`, `surprised`, `blink`, `look*`) | via 0.x preset rename (`a`→`aa`, `joy`→`happy`, …) + ARKit formula mapping | `VRMC_vrmExpressions` (preset + custom) |
| Eye gaze | bone-type lookAt (`firstPerson` range maps) or look* expressions | bone-type lookAt (`rangeMap*`) or expressions |
| Springbones (hair/clothes) | `secondaryAnimation` (name chains, node-index chains, and flat root lists) | `VRMC_springBone` (incl. sphere colliders) |

The ARKit→VRM expression mapping mirrors the Python pipeline's
`map_arkit_to_vrm` (visemes from jaw/funnel/pucker/stretch, emotions from
brow/smile/frown/sneer combinations, combined blink, look helpers), so
standard-preset avatars animate without ARKit morph sets. Eye gaze scales the
`eyeLook*` blendshapes by the model's lookAt range maps into leftEye/rightEye
bone rotation (matching the Python renderer's `R_y(-yaw) · R_x(pitch)`
convention).

## GPU usage

MediaPipe creates EGL/GL contexts by default. The `egl_stub.c` compiled into
`vtuber_live` with `-rdynamic` overrides EGL symbols at link time, forcing
pure-CPU inference. The only residual GPU usage (~10MB) is from the SDL/X11
window backing store.
