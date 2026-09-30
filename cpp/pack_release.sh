#!/bin/bash
# Packs a self-contained macOS (arm64) release zip for the C++ VTuber pipeline.
#
# The zip contains: the executable, SDL3/libmediapipe dylibs (app-local, with
# rpaths rewritten to @executable_path), MediaPipe .task models, VRM avatars,
# a README, and third-party notices. Binaries are ad-hoc signed (required on
# arm64); see the release README for the Gatekeeper bypass instructions.
#
# Usage (from cpp/, after a successful build):
#   ./pack_release.sh [-v <version>] [-b <build_dir>]
#
# Options:
#   -v <version>   release version               (default: 0.4.0)
#   -b <dir>       build output directory        (default: cpp/build)
set -euo pipefail

VERSION="0.4.0"
BUILD_DIR=""
while getopts "v:b:" opt; do
    case "$opt" in
        v) VERSION="$OPTARG" ;;
        b) BUILD_DIR="$OPTARG" ;;
        *) echo "Usage: $0 [-v version] [-b build_dir]" >&2; exit 2 ;;
    esac
done

CPP_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$CPP_DIR")"
BUILD_DIR="${BUILD_DIR:-$CPP_DIR/build}"

SDL_DYLIB="$BUILD_DIR/_deps/sdl3-build/libSDL3.0.dylib"
PKG_NAME="vtuber-cpu-v$VERSION-macos-arm64"
DIST_DIR="$REPO_ROOT/dist"
STAGE="$DIST_DIR/$PKG_NAME"
ZIP_PATH="$DIST_DIR/$PKG_NAME.zip"

# --- Verify build outputs ---------------------------------------------------
for f in "$BUILD_DIR/vtuber_live" "$SDL_DYLIB" "$CPP_DIR/lib/libmediapipe.dylib"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing '$f'. Build first: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build" >&2
        exit 1
    fi
done

# --- Verify assets -----------------------------------------------------------
MODEL_DIR="$REPO_ROOT/assets/models"
AVATAR_DIR="$REPO_ROOT/assets/avatars"
task_count=$(ls "$MODEL_DIR"/*.task 2>/dev/null | wc -l | tr -d ' ')
if [ "$task_count" -lt 3 ]; then
    echo "ERROR: expected 3 .task models in '$MODEL_DIR' (download the models first)" >&2
    exit 1
fi
if ! ls "$AVATAR_DIR"/*.vrm >/dev/null 2>&1; then
    echo "ERROR: no .vrm avatars in '$AVATAR_DIR' (run download_models.py first)" >&2
    exit 1
fi

# --- Stage the package -------------------------------------------------------
rm -rf "$STAGE"
mkdir -p "$STAGE/assets/models" "$STAGE/assets/avatars"

# Ship only the live app; vtuber_cpu / test_tracker / test_calibration are
# developer tools and stay out of the release package.
cp "$BUILD_DIR/vtuber_live" "$STAGE/"
cp "$SDL_DYLIB" "$STAGE/"
cp "$CPP_DIR/lib/libmediapipe.dylib" "$STAGE/"
# The wheel dylib's install name is "@rpath/libmediapipe_source.so"; provide
# that name so the executable's dependency resolves next to it.
ln -sf libmediapipe.dylib "$STAGE/libmediapipe_source.so"
cp "$MODEL_DIR"/*.task "$STAGE/assets/models/"
cp "$AVATAR_DIR"/*.vrm "$STAGE/assets/avatars/"
chmod +x "$STAGE/vtuber_live"

# --- Make the executable self-contained --------------------------------------
# Strip the build tree's absolute rpaths and resolve everything via
# @executable_path so the zip runs from any location.
EXE="$STAGE/vtuber_live"
while IFS= read -r rp; do
    [ -n "$rp" ] && install_name_tool -delete_rpath "$rp" "$EXE"
done < <(otool -l "$EXE" | awk '/LC_RPATH/{f=1} f && / path / {print $2; f=0}')
install_name_tool -add_rpath "@executable_path" "$EXE"
# install_name_tool invalidates the signature; re-sign ad-hoc (mandatory on arm64)
codesign -f -s - "$EXE" >/dev/null 2>&1

# --- Release README ----------------------------------------------------------
cat > "$STAGE/README.md" <<EOF
# VTuber CPU - macOS Release v$VERSION (Apple Silicon)

Real-time VTuber avatar tracking and rendering, 100% on the CPU - your GPU
stays free for gaming. Your webcam is tracked (face, hands, upper body) and
mapped onto a VRM avatar rendered by a built-in software rasterizer.

## Requirements

- macOS 14 (Sonoma) or newer, Apple Silicon (M1/M2/M3/M4)
- A webcam (built-in or USB)

No installer, nothing to install - everything needed is inside this zip.

## Quick start

1. Extract the whole zip anywhere (keep the folder structure intact).
2. The first launch needs a one-time Gatekeeper approval because the binary
   is unsigned (see below).
3. Sit in frame, press **SPACE** and hold still for ~1 second - an on-screen
   progress bar shows when calibration is done. That's it.

### Gatekeeper (one-time)

The app is not signed with an Apple Developer certificate. On first run,
either:

- Right-click (or Control-click) \`vtuber_live\` and choose **Open** →
  **Open** in the dialog, or
- Remove the quarantine attribute once from a terminal:

      xattr -d com.apple.quarantine vtuber_live

macOS also asks for camera permission on first launch - click **Allow**.

## Running from a terminal

    ./vtuber_live                                    default (lightweight)
    ./vtuber_live --threads 0 --fps 0                full quality (all cores, uncapped)
    ./vtuber_live --threads 4 --fps 30               something in between
    ./vtuber_live --verbose                          diagnostic log

## Controls

| Key | Action |
|-----|--------|
| C | Toggle camera selection menu & keyboard shortcuts legend |
| 1-9 | Directly select camera source |
| R | Rescan connected camera devices |
| SPACE | Calibrate / re-calibrate neutral pose (face, body, hands) and camera framing |
| W | Toggle webcam picture-in-picture & UI overlay (incl. fps readout and calibration banner) |
| ESC | Quit |

## Try other avatars

Four VRM models are bundled. Any \`.vrm\` file works (VRM 0.x and 1.0):

    ./vtuber_live assets/avatars/DefaultSampleAvatar.vrm
    ./vtuber_live assets/avatars/hair_sample_male.vrm
    ./vtuber_live assets/avatars/masc_vroid.vrm

## Troubleshooting

- **"cannot be opened because the developer cannot be verified"**: see
  *Gatekeeper* above.
- **Avatar does not move**: make sure the webcam works and you are in frame,
  then press SPACE to calibrate. Toggle the PiP overlay with W to see what
  the tracker sees.
- **Choppy tracking**: keep the default 2-thread / 15 fps settings, or lower
  them further with \`--threads 2 --fps 10\`.
- **Intel Macs**: this build is Apple Silicon only.

## Licenses

Code: MIT. Third-party libraries and models are bundled under their own
licenses - see \`THIRD_PARTY_NOTICES.md\`. VRM avatars embed their own usage
terms (check before redistributing a model).
EOF

# --- Third-party notices ------------------------------------------------------
cat > "$STAGE/THIRD_PARTY_NOTICES.md" <<'EOF'
# Third-party notices

This package bundles the following open-source components. Each remains
licensed under its own terms.

| Component | License | Source |
|---|---|---|
| MediaPipe (libmediapipe.dylib, .task models) | Apache License 2.0 | https://github.com/google-ai-edge/mediapipe |
| SDL3 3.4.16 | zlib License | https://github.com/libsdl-org/SDL |
| GLM 1.0.3 | MIT | https://github.com/g-truc/glm |
| BS::thread_pool 5.0.0 | MIT | https://github.com/bshoshany/thread-pool |
| cgltf 1.14 | MIT | https://github.com/jkuhlmann/cgltf |
| stb | Public Domain / Unlicense | https://github.com/nothings/stb |
| font8x8 | Public Domain | https://github.com/dhepper/font8x8 |

## VRM avatar models

- `male_52blendshapes.vrm` (default) - VRoid male with 52 ARKit
  blendshapes ("Perfect Sync"), from
  https://github.com/hinzka/52blendshapes-for-VRoid-face
- `DefaultSampleAvatar.vrm` - VRoid AvatarSample_A, from
  https://github.com/mcreativeIKEP/HolisticMotionCapture
- `hair_sample_male.vrm`, `masc_vroid.vrm` - CC0 samples from
  https://github.com/madjin/vrm-samples

VRM models embed their own license metadata (commercial usage,
redistribution, credit, etc.). Use each model according to its terms; the
CC0 samples have no restrictions.
EOF

# --- Sanity check: the staged app must be self-contained ----------------------
echo "rpath after fixup:"
otool -l "$EXE" | awk '/LC_RPATH/{f=1} f && / path / {print "  " $2; f=0}'
RPATHS=$(otool -l "$EXE" | awk '/LC_RPATH/{f=1} f && / path / {print $2; f=0}')
if [ "$(printf '%s\n' "$RPATHS" | grep -c .)" -ne 1 ] || [ "$RPATHS" != "@executable_path" ]; then
    echo "ERROR: staged executable is not self-contained (unexpected rpaths)" >&2
    exit 1
fi
# Running --help from the stage proves every dylib resolves via
# @executable_path (no build-tree fallbacks remain in the binary).
(cd "$STAGE" && ./vtuber_live --help >/dev/null)

# --- Zip (ditto preserves permissions and symlinks) ---------------------------
rm -f "$ZIP_PATH"
ditto -c -k --sequesterRsrc --keepParent "$STAGE" "$ZIP_PATH"

# --- Summary -------------------------------------------------------------------
ZIP_MB=$(du -m "$ZIP_PATH" | cut -f1)
echo ""
echo "Package: $ZIP_PATH ($ZIP_MB MB)"
echo "Contents:"
(cd "$STAGE" && find . -type f -o -type l | sort | while read -r f; do
    KB=$(du -k "$f" 2>/dev/null | cut -f1)
    printf "%8s KB  %s\n" "$KB" "$f"
done)
