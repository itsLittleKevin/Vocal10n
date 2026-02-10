<#
.SYNOPSIS
    Vocal10n — Launch Script
.DESCRIPTION
    Starts the GPT-SoVITS TTS server (subprocess) and the main Vocal10n application.
    On exit, the TTS server process is automatically cleaned up.
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

# ── Validate venvs exist ──────────────────────────────────────────────────
$venvMain = Join-Path $ProjectRoot "venvs\venv_main\Scripts\python.exe"
$venvTTS = Join-Path $ProjectRoot "venvs\venv_tts\Scripts\python.exe"

if (-not (Test-Path $venvMain)) {
    Write-Host "ERROR: venv_main not found. Run setup_env.ps1 first." -ForegroundColor Red
    exit 1
}

# ── Start TTS server (background) ─────────────────────────────────────────
$ttsProcess = $null
$sovitsDir = Join-Path $ProjectRoot "vendor\GPT-SoVITS"
$sovitsApi = Join-Path $sovitsDir "api_v2.py"

if ((Test-Path $venvTTS) -and (Test-Path $sovitsApi)) {
    # Check if port 9880 is already in use
    $portInUse = Get-NetTCPConnection -LocalPort 9880 -ErrorAction SilentlyContinue
    if ($portInUse) {
        Write-Host "WARNING: Port 9880 already in use. Skipping TTS server launch." -ForegroundColor Yellow
    } else {
        Write-Host "Starting GPT-SoVITS TTS server..." -ForegroundColor Cyan
        $sovitsGPT = Join-Path $sovitsDir "GPT_SoVITS"
        $env:PYTHONPATH = "$sovitsDir;$sovitsGPT"
        $ttsProcess = Start-Process -FilePath $venvTTS `
            -ArgumentList "$sovitsApi -a 127.0.0.1 -p 9880" `
            -WorkingDirectory $sovitsDir `
            -WindowStyle Hidden `
            -PassThru
        Write-Host "  TTS server started (PID: $($ttsProcess.Id))" -ForegroundColor Green
        Start-Sleep -Seconds 3
    }
} else {
    Write-Host "WARNING: GPT-SoVITS not found in vendor/. TTS will be unavailable." -ForegroundColor Yellow
}

# ── Start main application ────────────────────────────────────────────────
Write-Host "Starting Vocal10n..." -ForegroundColor Cyan
try {
    & $venvMain -m vocal10n.app
} finally {
    # ── Cleanup TTS server on exit ────────────────────────────────────
    if ($ttsProcess -and -not $ttsProcess.HasExited) {
        Write-Host "Stopping TTS server (PID: $($ttsProcess.Id))..." -ForegroundColor Cyan
        Stop-Process -Id $ttsProcess.Id -Force -ErrorAction SilentlyContinue
    }
}
