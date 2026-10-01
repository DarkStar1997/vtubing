#!/bin/bash
# Packs a self-contained Linux (x86-64) release tarball for the C++ VTuber
# pipeline.
#
# The tarball contains: the executable, SDL3/libmediapipe shared libraries
# (app-local, with the rpath rewritten to $ORIGIN), MediaPipe .task models,
# VRM avatars, a README, and third-party notices. Executable permissions are
# preserved by the tarball.
#
# Usage (from cpp/, after a successful build):
#   ./pack_release_linux.sh [-v <version>] [-b <build_dir>]
#
# Options:
#   -v <version>   release version               (default: 0.4.0)
#   -b <dir>       build output directory        (default: cpp/build)
#
# Requires: patchelf (rewrite rpaths), tar, ldd.
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

PKG_NAME="vtuber-cpu-v$VERSION-linux-x86_64"
DIST_DIR="$REPO_ROOT/dist"
STAGE="$DIST_DIR/$PKG_NAME"
TARBALL_PATH="$DIST_DIR/$PKG_NAME.tar.gz"

if ! command -v patchelf >/dev/null 2>&1; then
    echo "ERROR: patchelf not found (rewrite rpaths to \$ORIGIN)." >&2
    echo "  Ubuntu/Debian: sudo apt install patchelf" >&2
    echo "  Arch:          sudo pacman -S patchelf" >&2
    exit 1
fi

# --- Verify build outputs ----------------------------------------------------
EXE="$BUILD_DIR/vtuber_live"
if [ ! -f "$EXE" ]; then
    echo "ERROR: missing '$EXE'. Build first: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build" >&2
    exit 1
fi
# Locate libSDL3.so.0 via the dynamic linker: works both for system SDL3
# installs and for the FetchContent build (resolved from _deps/sdl3-build).
SDL_SO="$(ldd "$EXE" | awk '/libSDL3\.so\.0 =>/{print $3}')"
if [ -z "$SDL_SO" ] || [ ! -f "$SDL_SO" ]; then
    echo "ERROR: could not resolve libSDL3.so.0 for '$EXE' via ldd" >&2
    exit 1
fi
MP_SO="$CPP_DIR/lib/libmediapipe.so"
if [ ! -f "$MP_SO" ]; then
    echo "ERROR: missing '$MP_SO' (run 'uv sync' in the repo root, then re-run CMake)" >&2
    exit 1
fi

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
cp "$EXE" "$STAGE/"
# Dereference: the NEEDED entry is the SONAME, so the real file ships under
# the SONAME name.
cp -L "$SDL_SO" "$STAGE/libSDL3.so.0"
cp "$MP_SO" "$STAGE/libmediapipe.so"
cp "$MODEL_DIR"/*.task "$STAGE/assets/models/"
cp "$AVATAR_DIR"/*.vrm "$STAGE/assets/avatars/"
chmod +x "$STAGE/vtuber_live"

# --- Make the executable self-contained --------------------------------------
# Strip the build tree's absolute rpaths (cpp/lib, _deps/...) and resolve
# everything via $ORIGIN so the tarball runs from any location.
patchelf --remove-rpath "$STAGE/vtuber_live"
patchelf --add-rpath '$ORIGIN' "$STAGE/vtuber_live"

# --- Release README ----------------------------------------------------------
cat > "$STAGE/README.md" <<EOF
# VTuber CPU - Linux Release v$VERSION (x86-64)

Real-time VTuber avatar tracking and rendering, 100% on the CPU - your GPU
stays free for gaming. Your webcam is tracked (face, hands, upper body) and
mapped onto a VRM avatar rendered by a built-in software rasterizer.

## Requirements

- x86-64 CPU with AVX2 (Intel Haswell 2013+ / AMD Ryzen, or newer)
- 64-bit Linux with glibc 2.28+ (2018 or newer: Ubuntu 20.04+, Debian 11+,
  RHEL/Rocky/Alma 8 & 9, Amazon Linux 2023, Fedora, openSUSE Leap 15.x,
  Arch and similar). The C++ runtime is statically linked, so no separate
  libstdc++ requirement exists.
- A desktop session (X11 or Wayland) and a webcam

No installer, nothing to install - everything needed is inside this tarball.

## Quick start

1. Extract the tarball anywhere (keep the folder structure intact):

       tar xzf $PKG_NAME.tar.gz

2. Run the app from a terminal:

       cd $PKG_NAME
       ./vtuber_live

3. Sit in frame, press SPACE and hold still for ~1 second - an on-screen
   progress bar shows when calibration is done. That's it.

(The tarball preserves the executable bit; if your archive tool strips it,
run chmod +x vtuber_live once.)

## Options

    ./vtuber_live                                    default (lightweight)
    ./vtuber_live --threads 0 --fps 0                full quality (all cores, uncapped)
    ./vtuber_live --threads 4 --fps 30               something in between
    ./vtuber_live --verbose                          diagnostic log
    ./vtuber_live --help                             all options

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

- **"Illegal instruction" on startup**: the CPU lacks AVX2. There is no
  fallback path for pre-2013 CPUs.
- **"version \`GLIBC_2.xx' not found"**: the distro's glibc is older than
  2.28 (2018). Use a newer distro (or build from source - see the project
  repo).
- **Avatar does not move**: make sure the webcam works and you are in frame,
  then press SPACE to calibrate. Toggle the PiP overlay with W to see what
  the tracker sees.
- **Choppy tracking**: keep the default 2-thread / 15 fps settings, or lower
  them further with \`--threads 2 --fps 10\`.
- **Camera not detected**: press R to rescan, or list/select another source
  with C and the number keys.

## Licenses

Code: MIT. Third-party libraries and models are bundled under their own
licenses - see \`THIRD_PARTY_NOTICES.md\`. VRM avatars embed their own usage
terms (check before redistributing a model).
EOF

# --- Third-party notices ------------------------------------------------------
cat > "$STAGE/THIRD_PARTY_NOTICES.md" <<EOF
# Third-party notices

This package bundles the following open-source components. Each remains
licensed under its own terms.

| Component | License | Source |
|---|---|---|
| MediaPipe (libmediapipe.so, .task models) | Apache License 2.0 | https://github.com/google-ai-edge/mediapipe |
| SDL3 3.4.16 | zlib License | https://github.com/libsdl-org/SDL |
| GLM 1.0.3 | MIT | https://github.com/g-truc/glm |
| BS::thread_pool 5.0.0 | MIT | https://github.com/bshoshany/thread-pool |
| cgltf 1.14 | MIT | https://github.com/jkuhlmann/cgltf |
| stb | Public Domain / Unlicense | https://github.com/nothings/stb |
| font8x8 | Public Domain | https://github.com/dhepper/font8x8 |

## VRM avatar models

- \`male_52blendshapes.vrm\` (default) - VRoid male with 52 ARKit
  blendshapes ("Perfect Sync"), from
  https://github.com/hinzka/52blendshapes-for-VRoid-face
- \`DefaultSampleAvatar.vrm\` - VRoid AvatarSample_A, from
  https://github.com/mcreativeIKEP/HolisticMotionCapture
- \`hair_sample_male.vrm\`, \`masc_vroid.vrm\` - CC0 samples from
  https://github.com/madjin/vrm-samples

VRM models embed their own license metadata (commercial usage,
redistribution, credit, etc.). Use each model according to its terms; the
CC0 samples have no restrictions.
EOF

# --- Sanity check: the staged app must be self-contained ----------------------
echo "rpath after fixup:"
readelf -d "$STAGE/vtuber_live" | awk '/RPATH|RUNPATH/{print "  " $0}'
RPATHS="$(patchelf --print-rpath "$STAGE/vtuber_live")"
if [ "$RPATHS" != '$ORIGIN' ]; then
    echo "ERROR: staged executable is not self-contained (unexpected rpaths: $RPATHS)" >&2
    exit 1
fi

# --- Portability check: the bundle must run on glibc 2.28 systems --------------
# The release is built in the manylinux_2_28 container with a statically
# linked C++ runtime. Verify that held: max referenced GLIBC symbol version
# <= 2.28 and no GLIBCXX_* symbols at all. Warn by default; fail hard when
# LINUX_PORTABILITY_STRICT=1 (the CI release workflow sets it).
check_portability() {
    local bad=0
    local max_glibc glibcxx
    for f in "$STAGE/vtuber_live" "$STAGE/libSDL3.so.0" "$STAGE/libmediapipe.so"; do
        max_glibc="$(readelf --version-info "$f" 2>/dev/null \
            | grep -o 'GLIBC_2\.[0-9]*' | sort -Vu | tail -1 || true)"
        glibcxx="$(readelf --version-info "$f" 2>/dev/null \
            | grep -c 'GLIBCXX_' || true)"
        echo "  $(basename "$f"): max ${max_glibc:-GLIBC_(none)}, GLIBCXX refs: $glibcxx"
        if [ -n "$max_glibc" ]; then
            minor="${max_glibc#GLIBC_2.}"
            if [ "$minor" -gt 28 ]; then
                echo "    -> exceeds the glibc 2.28 baseline" >&2
                bad=1
            fi
        fi
        if [ "$f" = "$STAGE/vtuber_live" ] && [ "$glibcxx" -ne 0 ]; then
            echo "    -> libstdc++ not statically linked (GLIBCXX symbols present)" >&2
            bad=1
        fi
    done
    if [ "$bad" -ne 0 ]; then
        if [ "${LINUX_PORTABILITY_STRICT:-0}" = "1" ]; then
            echo "ERROR: bundle violates the glibc 2.28 / static-libstdc++ baseline" >&2
            exit 1
        fi
        echo "WARNING: bundle does not meet the manylinux_2_28 baseline (see above)" >&2
    fi
}
check_portability
# Every dependency must resolve; run from the stage with a pristine loader
# environment (no LD_LIBRARY_PATH rescue), then confirm via LD_DEBUG that the
# *staged* copies are the ones actually loaded - not system fallbacks.
STAGE_LIBS="$(cd "$STAGE" && env -u LD_LIBRARY_PATH ldd ./vtuber_live)"
if printf '%s\n' "$STAGE_LIBS" | grep -q 'not found'; then
    echo "ERROR: staged executable has unresolved libraries:" >&2
    printf '%s\n' "$STAGE_LIBS" | grep 'not found' >&2
    exit 1
fi
(cd "$STAGE" && env -u LD_LIBRARY_PATH SDL_VIDEODRIVER=dummy \
    LD_DEBUG=libs ./vtuber_live --help >/dev/null 2>"$DIST_DIR/.lddebug.log") \
    || { echo "ERROR: staged executable failed to run" >&2; exit 1; }
for lib in libSDL3.so.0 libmediapipe.so; do
    if ! grep -q "calling init: $STAGE/$lib" "$DIST_DIR/.lddebug.log"; then
        echo "ERROR: $lib was not loaded from the package (missing 'calling init: $STAGE/$lib')" >&2
        exit 1
    fi
done
rm -f "$DIST_DIR/.lddebug.log"

# --- Tarball (preserves the executable bit) -----------------------------------
rm -f "$TARBALL_PATH"
tar -czf "$TARBALL_PATH" -C "$DIST_DIR" "$PKG_NAME"

# --- Summary -------------------------------------------------------------------
TAR_MB=$(du -m "$TARBALL_PATH" | cut -f1)
echo ""
echo "Package: $TARBALL_PATH ($TAR_MB MB)"
echo "Contents:"
(cd "$STAGE" && ls -la && du -sh assets)
