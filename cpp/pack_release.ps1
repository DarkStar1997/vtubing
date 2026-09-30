# Packs a self-contained Windows release zip for the C++ VTuber pipeline.
#
# The zip contains: executables, SDL3/libmediapipe DLLs, app-local MSVC
# runtime DLLs (no VC++ Redistributable install needed on target machines),
# MediaPipe .task models, VRM avatars, a README, and third-party notices.
#
# Usage (from cpp/, after a successful build):
#   powershell -ExecutionPolicy Bypass -File pack_release.ps1 [-Version 0.1.0]
#
# Optional overrides:
#   -BuildDir <dir>   build output directory       (default: cpp/build)
#   -CrtDir <dir>     MSVC CRT dir to copy from    (default: auto-detect)
[CmdletBinding()]
param(
    [string]$Version = "0.1.0",
    [string]$BuildDir = "",
    [string]$CrtDir = "",
    # Optional Authenticode signing (applied before zipping):
    [string]$SignPfx = "",                          # PFX file
    $SignPassword = $null,                          # PFX password, SecureString or plain
    [string]$SignThumbprint = ""                    # cert in a store (token/SimplySign)
)
$ErrorActionPreference = 'Stop'

$cppDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $cppDir
if (-not $BuildDir) { $BuildDir = Join-Path $cppDir 'build' }

$pkgName = "vtuber-cpu-v$Version-windows-x64"
$distDir = Join-Path $repoRoot 'dist'
$stage   = Join-Path $distDir $pkgName
$zipPath = Join-Path $distDir "$pkgName.zip"

# --- Verify build outputs -------------------------------------------------
foreach ($n in @('vtuber_live.exe', 'SDL3.dll', 'libmediapipe.dll')) {
    if (-not (Test-Path (Join-Path $BuildDir $n))) {
        throw "Missing '$n' in '$BuildDir'. Build first: cmake --build build"
    }
}

# --- Locate the MSVC redistributable CRT (app-local deployment) -----------
if (-not $CrtDir) {
    $crtDirs = @(
        Get-Item "C:\Program Files\Microsoft Visual Studio\*\*\VC\Redist\MSVC\*\x64\Microsoft.VC*.CRT" -ErrorAction SilentlyContinue
        Get-Item "C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Redist\MSVC\*\x64\Microsoft.VC*.CRT" -ErrorAction SilentlyContinue
    )
    if (-not $crtDirs) {
        throw "MSVC CRT redist not found. Pass -CrtDir <path to Microsoft.VC*.CRT>"
    }
    $CrtDir = ($crtDirs | Sort-Object FullName | Select-Object -Last 1).FullName
}
Write-Host "CRT: $CrtDir"

# --- Verify assets ---------------------------------------------------------
$modelDir  = Join-Path $repoRoot 'assets\models'
$avatarDir = Join-Path $repoRoot 'assets\avatars'
$taskFiles = @(Get-ChildItem $modelDir -Filter *.task -ErrorAction SilentlyContinue)
if ($taskFiles.Count -lt 3) {
    throw "Expected 3 .task models in '$modelDir' (download the models first)"
}
$vrmFiles = @(Get-ChildItem $avatarDir -Filter *.vrm -ErrorAction SilentlyContinue)
if (-not $vrmFiles) {
    throw "No .vrm avatars in '$avatarDir' (run download_models.py first)"
}

# --- Stage the package -----------------------------------------------------
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Path (Join-Path $stage 'assets\models'),
                                  (Join-Path $stage 'assets\avatars') -Force | Out-Null

# Ship only the live app; vtuber_cpu / test_tracker / test_calibration are
# developer tools and stay out of the release package.
Copy-Item (Join-Path $BuildDir 'vtuber_live.exe') $stage
Copy-Item (Join-Path $BuildDir 'SDL3.dll'), (Join-Path $BuildDir 'libmediapipe.dll') $stage
Copy-Item (Join-Path $CrtDir '*.dll') $stage
$taskFiles | Copy-Item -Destination (Join-Path $stage 'assets\models')
$vrmFiles | Copy-Item -Destination (Join-Path $stage 'assets\avatars')

# --- Release README --------------------------------------------------------
$readme = @"
# VTuber CPU - Windows Release v$Version

Real-time VTuber avatar tracking and rendering, 100% on the CPU - your GPU
stays free for gaming. Your webcam is tracked (face, hands, upper body) and
mapped onto a VRM avatar rendered by a built-in software rasterizer.

## Requirements

- Windows 10/11 (64-bit)
- A webcam (built-in or USB)
- A CPU with AVX2 (Intel Haswell 2013+ / AMD Ryzen, or newer)

No installer, nothing to install - everything needed is inside this zip.

## Quick start

1. Extract the whole zip anywhere (keep the folder structure intact).
2. Double-click ``vtuber_live.exe``.
3. Sit in frame, press **SPACE** and hold still for ~1 second - an on-screen
   progress bar shows when calibration is done. That's it.

## Controls

| Key | Action |
|-----|--------|
| C | Toggle camera selection menu & keyboard shortcuts legend |
| 1-9 | Directly select camera source |
| R | Rescan connected camera devices |
| SPACE | Calibrate / re-calibrate neutral pose (face, body, hands) and camera framing |
| W | Toggle webcam picture-in-picture & UI overlay (incl. fps readout and calibration banner) |
| ESC | Quit |

## Performance

Runs lightweight by default - a 15 fps cap and roughly a one-core budget -
leaving plenty of headroom for games and OBS. For more responsiveness, run
from a terminal:

    vtuber_live.exe --threads 0 --fps 0     full quality (all cores, uncapped)
    vtuber_live.exe --threads 4 --fps 30    something in between

The app is quiet by default (no console window opens). For a diagnostic log:

    vtuber_live.exe --verbose

## Try other avatars

Four VRM models are bundled. Any ``.vrm`` file works (VRM 0.x and 1.0):

    vtuber_live.exe assets\avatars\DefaultSampleAvatar.vrm
    vtuber_live.exe assets\avatars\hair_sample_male.vrm
    vtuber_live.exe assets\avatars\masc_vroid.vrm

## Troubleshooting

- **Windows protected your PC** (SmartScreen): the binaries are unsigned.
  Click *More info*, then *Run anyway*.
- **Avatar does not move**: make sure the webcam works and you are in frame,
  then press SPACE to calibrate. Toggle the PiP overlay with W to see what
  the tracker sees.
- **Antivirus flags a DLL**: occasional false positives on the unsigned
  ``libmediapipe.dll``; allow it, or re-extract from the zip.
- **Choppy tracking**: keep the default 2-thread / 15 fps settings, or lower
  them further with ``--threads 2 --fps 10``.

## Licenses

Code: MIT. Third-party libraries and models are bundled under their own
licenses - see ``THIRD_PARTY_NOTICES.md``. VRM avatars embed their own usage
terms (check before redistributing a model).
"@
Set-Content -Path (Join-Path $stage 'README.md') -Value $readme -Encoding UTF8

# --- Third-party notices ----------------------------------------------------
$notices = @"
# Third-party notices

This package bundles the following open-source components. Each remains
licensed under its own terms.

| Component | License | Source |
|---|---|---|
| MediaPipe (libmediapipe.dll, .task models) | Apache License 2.0 | https://github.com/google-ai-edge/mediapipe |
| SDL3 3.4.16 | zlib License | https://github.com/libsdl-org/SDL |
| GLM 1.0.3 | MIT | https://github.com/g-truc/glm |
| BS::thread_pool 5.0.0 | MIT | https://github.com/bshoshany/thread-pool |
| cgltf 1.14 | MIT | https://github.com/jkuhlmann/cgltf |
| stb | Public Domain / Unlicense | https://github.com/nothings/stb |
| font8x8 | Public Domain | https://github.com/dhepper/font8x8 |
| MSVC runtime DLLs | Microsoft Visual C++ Redistributable terms | https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist |

## VRM avatar models

- ``male_52blendshapes.vrm`` (default) - VRoid male with 52 ARKit
  blendshapes ("Perfect Sync"), from
  https://github.com/hinzka/52blendshapes-for-VRoid-face
- ``DefaultSampleAvatar.vrm`` - VRoid AvatarSample_A, from
  https://github.com/creativeIKEP/HolisticMotionCapture
- ``hair_sample_male.vrm``, ``masc_vroid.vrm`` - CC0 samples from
  https://github.com/madjin/vrm-samples

VRM models embed their own license metadata (commercial usage,
redistribution, credit, etc.). Use each model according to its terms; the
CC0 samples have no restrictions.
"@
Set-Content -Path (Join-Path $stage 'THIRD_PARTY_NOTICES.md') -Value $notices -Encoding UTF8

# --- Sign binaries (optional, before zipping) --------------------------------
if ($SignPfx -or $SignThumbprint) {
    $signArgs = @{ Path = $stage }
    if ($SignPfx)      { $signArgs.Pfx = $SignPfx; $signArgs.Password = $SignPassword }
    if ($SignThumbprint) { $signArgs.Thumbprint = $SignThumbprint }
    & (Join-Path $cppDir 'sign_release.ps1') @signArgs
    if ($LASTEXITCODE -ne 0) { throw "Signing failed" }
    Write-Host ""
}

# --- Zip -------------------------------------------------------------------
if (Test-Path $zipPath) { Remove-Item $zipPath }
Compress-Archive -Path $stage -DestinationPath $zipPath -CompressionLevel Optimal

# --- Summary ---------------------------------------------------------------
$zipMB = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
Write-Host ""
Write-Host "Package: $zipPath ($zipMB MB)"
Write-Host "Contents:"
Get-ChildItem $stage -Recurse -File | ForEach-Object {
    "{0,10:N1} KB  {1}" -f ($_.Length / 1KB), $_.FullName.Substring($stage.Length + 1)
}
