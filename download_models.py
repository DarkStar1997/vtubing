#!/usr/bin/env python3
"""Download free VRM avatar models into assets/avatars/.

Run:  uv run python download_models.py

Each model is downloaded only if missing.  The default model used by
config.yaml (male_52blendshapes.vrm) is downloaded first.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).parent / "assets" / "avatars"

MODELS: list[tuple[str, str, str]] = [
    (
        "male_52blendshapes.vrm",
        "https://raw.githubusercontent.com/hinzka/52blendshapes-for-VRoid-face/main/VRoid_V110_Male_v1.1.3.vrm",
        "VRoid male with 52 ARKit blendshapes (Perfect Sync) — DEFAULT",
    ),
    (
        "DefaultSampleAvatar.vrm",
        "https://raw.githubusercontent.com/creativeIKEP/HolisticMotionCapture/main/Assets/StreamingAssets/DefaultSampleAvatar.vrm",
        "VRoid AvatarSample_A (female, 15 expressions)",
    ),
    (
        "hair_sample_male.vrm",
        "https://raw.githubusercontent.com/madjin/vrm-samples/master/vroid/beta/HairSample_Male.vrm",
        "VRoid HairSample Male (CC0)",
    ),
    (
        "masc_vroid.vrm",
        "https://raw.githubusercontent.com/madjin/vrm-samples/master/vroid/masc_vroid.vrm",
        "VRoid base body (CC0)",
    ),
]


def download(filename: str, url: str, description: str) -> bool:
    dest = DEST / filename
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  [skip] {filename} already exists ({dest.stat().st_size // 1024} KB)")
        return False
    print(f"  [download] {filename}")
    print(f"             {description}")
    print(f"             {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"  [ERROR] failed to download {filename}: {e}")
        if dest.exists():
            dest.unlink()
        return False
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  [ok] {filename} ({size_mb:.1f} MB)")
    return True


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Downloading VRM models to {DEST}/\n")
    downloaded = 0
    for filename, url, desc in MODELS:
        if download(filename, url, desc):
            downloaded += 1
    print(f"\nDone. {downloaded} model(s) downloaded, {len(MODELS)} total available.")
    default = DEST / MODELS[0][0]
    if default.exists():
        print(f"\nDefault avatar: {default}")
        print("Set in config.yaml:  avatar.path: assets/avatars/male_52blendshapes.vrm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
