# Authenticode-signs the release binaries with signtool.
#
# Two ways to supply the certificate:
#   -Pfx <file> -Password <secret>   certificate file (PFX/P12)
#   -Thumbprint <hex>                certificate already in a store
#                                    (hardware tokens, Certum SimplySign,
#                                    or a manually imported PFX)
#
# Usage examples:
#   powershell -File sign_release.ps1 -Path dist\vtuber-cpu-v0.1.0-windows-x64 `
#       -Pfx mycert.pfx -Password (Read-Host -AsSecureString "PFX password")
#
#   powershell -File sign_release.ps1 -Path dist\vtuber-cpu-v0.1.0-windows-x64 `
#       -Thumbprint ABC123...
#
# Signatures are RFC 3161 timestamped so they stay valid after the
# certificate expires.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Path,                                   # folder with the binaries, or a single file
    [string]$Pfx = "",
    $Password = $null,                               # SecureString or plain string
    [string]$Thumbprint = "",
    [string]$TsaUrl = "http://timestamp.digicert.com",
    # Binaries we build/ship that are not already Microsoft-signed
    [string[]]$Files = @('vtuber_live.exe', 'vtuber_cpu.exe', 'test_tracker.exe',
                         'SDL3.dll', 'libmediapipe.dll'),
    [string[]]$ExtraArgs = @()                       # e.g. @('/csp','<provider name>')
)
$ErrorActionPreference = 'Stop'

if (-not $Pfx -and -not $Thumbprint) {
    throw "Provide a certificate: -Pfx <file> [+ -Password] or -Thumbprint <hex>"
}

# --- Locate signtool (Windows SDK) -----------------------------------------
$signtool = @(
    Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin\*\x64\signtool.exe" -ErrorAction SilentlyContinue
    Get-ChildItem "C:\Program Files\Windows Kits\10\bin\*\x64\signtool.exe" -ErrorAction SilentlyContinue
) | Sort-Object FullName | Select-Object -Last 1
if (-not $signtool) { throw "signtool.exe not found (install the Windows SDK)" }
$signtool = $signtool.FullName

# --- Resolve the file list ---------------------------------------------------
if (Test-Path $Path -PathType Leaf) {
    $targets = @((Get-Item $Path).FullName)
} else {
    if (-not (Test-Path $Path -PathType Container)) { throw "Path not found: $Path" }
    $targets = @($Files | ForEach-Object {
        $f = Join-Path $Path $_
        if (Test-Path $f) { (Get-Item $f).FullName }
    })
    if (-not $targets) { throw "No files to sign in '$Path' (expected: $($Files -join ', '))" }
}

# --- Build certificate arguments ----------------------------------------------
$certArgs = @()
if ($Pfx) {
    if (-not (Test-Path $Pfx)) { throw "PFX not found: $Pfx" }
    if (-not $Password) { $Password = Read-Host -AsSecureString "PFX password" }
    if ($Password -is [string]) { $Password = ConvertTo-SecureString $Password -AsPlainText -Force }
    $plain = [Net.NetworkCredential]::new('', $Password).Password
    $certArgs = @('/f', (Resolve-Path $Pfx).Path, '/p', $plain)
} else {
    $certArgs = @('/sha1', $Thumbprint)
}

# --- Sign ---------------------------------------------------------------------
foreach ($t in $targets) {
    Write-Host "Signing: $(Split-Path -Leaf $t)"
    & $signtool sign /fd SHA256 /tr $TsaUrl /td SHA256 @certArgs @ExtraArgs $t
    if ($LASTEXITCODE -ne 0) { throw "signtool failed for $t (exit $LASTEXITCODE)" }
}

# --- Report signature status ----------------------------------------------------
Write-Host ""
foreach ($t in $targets) {
    $sig = Get-AuthenticodeSignature -FilePath $t
    $signer = if ($sig.SignerCertificate) { $sig.SignerCertificate.Subject } else { '(none)' }
    "{0,-24} {1}" -f (Split-Path -Leaf $t), $sig.Status
    "  signer: $signer"
}
Write-Host ""
Write-Host "Note: 'NotTrusted' on a self-signed/test certificate is expected."
Write-Host "Signature validity on end-user machines requires a certificate from"
Write-Host "a real CA (Certum OSS, SignPath Foundation, Azure Trusted Signing, ...)."
