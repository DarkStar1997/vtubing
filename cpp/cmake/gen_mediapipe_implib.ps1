# Generates an MSVC import library (.lib) for a DLL that ships without one.
# The MediaPipe wheel only provides libmediapipe.dll, and MSVC's link.exe
# cannot link a DLL directly (LNK1107) - it needs an import library.
#
# Requires dumpbin.exe and lib.exe on PATH (run from a VS developer prompt).
#
# Usage: gen_mediapipe_implib.ps1 <path-to-dll> <output-lib>
param(
    [Parameter(Mandatory = $true)][string]$DllPath,
    [Parameter(Mandatory = $true)][string]$LibPath
)
$ErrorActionPreference = 'Stop'

if (-not (Test-Path $DllPath)) { throw "DLL not found: $DllPath" }

# dumpbin /exports prints one line per exported symbol:
#   ordinal  hint   RVA        name
#       1    0    00123AB4    MpFaceLandmarkerCreate
$exports = & dumpbin /exports $DllPath |
    Select-String '^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)\s*$' |
    ForEach-Object { $_.Matches[0].Groups[1].Value }

if (-not $exports) { throw "No exports found in $DllPath" }

$dllName = [System.IO.Path]::GetFileNameWithoutExtension($DllPath)
$defPath = Join-Path $env:TEMP "$dllName.def"
"LIBRARY $dllName"  | Set-Content -Path $defPath -Encoding ASCII
"EXPORTS"           | Add-Content -Path $defPath -Encoding ASCII
$exports | ForEach-Object { "    $_" | Add-Content -Path $defPath -Encoding ASCII }

& lib /NOLOGO /MACHINE:X64 /DEF:"$defPath" /OUT:"$LibPath"
if ($LASTEXITCODE -ne 0) { throw "lib.exe failed with exit code $LASTEXITCODE" }

Write-Host "Generated $LibPath ($($exports.Count) exports)"
