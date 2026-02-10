<#
.SYNOPSIS
    Vocal10n — Launch Script
.DESCRIPTION
    Starts the GPT-SoVITS TTS server (subprocess) and the main Vocal10n application.
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
$sovitsApi = Join-Path $ProjectRoot "vendor\GPT-SoVITS\api_v2.py"
if ((Test-Path $venvTTS) -and (Test-Path $sovitsApi)) {
    Write-Host "Starting GPT-SoVITS TTS server..." -ForegroundColor Cyan
    $ttsJob = Start-Job -ScriptBlock {
        param($python, $script, $workdir)
        Set-Location $workdir
        & $python $script
    } -ArgumentList $venvTTS, $sovitsApi, (Join-Path $ProjectRoot "vendor\GPT-SoVITS")
    Write-Host "  TTS server started (Job ID: $($ttsJob.Id))" -ForegroundColor Green
    Start-Sleep -Seconds 3
} else {
    Write-Host "WARNING: GPT-SoVITS not found in vendor/. TTS will be unavailable." -ForegroundColor Yellow
}

# ── Start main application ────────────────────────────────────────────────
Write-Host "Starting Vocal10n..." -ForegroundColor Cyan
& $venvMain -m vocal10n.app
