<#
.SYNOPSIS
    Vocal10n — Environment Setup Script
.DESCRIPTION
    Creates 3 virtual environments (STT, LLM, TTS) and installs dependencies.
    Requires Python 3.11 installed and available via `py -3.11`.
#>

param(
    [switch]$SkipSTT,
    [switch]$SkipLLM,
    [switch]$SkipTTS,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

Write-Host "=== Vocal10n Environment Setup ===" -ForegroundColor Cyan
Write-Host ""

# ── Check Python 3.11 ──────────────────────────────────────────────────────
$py311 = $null
try {
    $py311 = & py -3.11 --version 2>&1
} catch {}

if (-not $py311 -or $py311 -notmatch "3\.11") {
    Write-Host "ERROR: Python 3.11 is required but not found." -ForegroundColor Red
    Write-Host "Install it from https://www.python.org/downloads/ and ensure the py launcher can find it." -ForegroundColor Yellow
    exit 1
}
Write-Host "Found: $py311" -ForegroundColor Green

# ── Helper: Create venv ────────────────────────────────────────────────────
function New-Venv {
    param(
        [string]$Name,
        [string]$RequirementsFile,
        [string[]]$ExtraInstall
    )

    $venvPath = Join-Path $ProjectRoot "venvs\$Name"
    $activate = Join-Path $venvPath "Scripts\Activate.ps1"
    $pip = Join-Path $venvPath "Scripts\pip.exe"

    if ((Test-Path $venvPath) -and -not $Force) {
        Write-Host "  Venv '$Name' already exists. Use -Force to recreate." -ForegroundColor Yellow
        return
    }

    if ((Test-Path $venvPath) -and $Force) {
        Write-Host "  Removing existing venv '$Name'..." -ForegroundColor Yellow
        Remove-Item $venvPath -Recurse -Force
    }

    Write-Host "  Creating venv '$Name'..." -ForegroundColor White
    & py -3.11 -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw "Failed to create venv '$Name'" }

    # Upgrade pip
    & $pip install --upgrade pip --quiet
    if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip in '$Name'" }

    # Install requirements file
    if ($RequirementsFile -and (Test-Path $RequirementsFile)) {
        Write-Host "  Installing requirements from $RequirementsFile..." -ForegroundColor White
        & $pip install -r $RequirementsFile
        if ($LASTEXITCODE -ne 0) { throw "Failed to install requirements for '$Name'" }
    }

    # Install extra packages
    foreach ($pkg in $ExtraInstall) {
        Write-Host "  Installing extra: $pkg" -ForegroundColor White
        & $pip install $pkg
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  WARNING: Failed to install '$pkg'" -ForegroundColor Yellow
        }
    }

    Write-Host "  Venv '$Name' ready." -ForegroundColor Green
}

# ── Create directories if missing ──────────────────────────────────────────
$dirs = @(
    "venvs",
    "models\stt", "models\llm", "models\tts",
    "vendor",
    "reference_audio", "training", "knowledge_base",
    "output\subtitles", "output\audio", "output\training_data"
)
foreach ($dir in $dirs) {
    $full = Join-Path $ProjectRoot $dir
    if (-not (Test-Path $full)) {
        New-Item -ItemType Directory -Path $full -Force | Out-Null
    }
}

# ── venv_stt (Python 3.11) ────────────────────────────────────────────────
if (-not $SkipSTT) {
    Write-Host ""
    Write-Host "[1/3] Setting up venv_stt (STT + UI + OBS)..." -ForegroundColor Cyan
    New-Venv -Name "venv_stt" `
             -RequirementsFile (Join-Path $ProjectRoot "requirements\requirements-stt.txt")
}

# ── venv_llm (Python 3.11) ────────────────────────────────────────────────
if (-not $SkipLLM) {
    Write-Host ""
    Write-Host "[2/3] Setting up venv_llm (LLM Translation)..." -ForegroundColor Cyan
    Write-Host "  NOTE: llama-cpp-python with CUDA requires the right wheel." -ForegroundColor Yellow
    Write-Host "  If installation fails, install manually:" -ForegroundColor Yellow
    Write-Host "    venvs\venv_llm\Scripts\pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121" -ForegroundColor Yellow
    New-Venv -Name "venv_llm" `
             -RequirementsFile (Join-Path $ProjectRoot "requirements\requirements-llm.txt") `
             -ExtraInstall @()
}

# ── venv_tts (Python 3.11) ────────────────────────────────────────────────
if (-not $SkipTTS) {
    Write-Host ""
    Write-Host "[3/3] Setting up venv_tts (GPT-SoVITS TTS Server)..." -ForegroundColor Cyan
    $sovitsReq = Join-Path $ProjectRoot "vendor\GPT-SoVITS\requirements.txt"
    if (Test-Path $sovitsReq) {
        New-Venv -Name "venv_tts" -RequirementsFile $sovitsReq
    } else {
        Write-Host "  WARNING: vendor/GPT-SoVITS not found. Copy GPT-SoVITS into vendor/ first," -ForegroundColor Yellow
        Write-Host "  then run:  .\setup_env.ps1 -SkipSTT -SkipLLM" -ForegroundColor Yellow
        New-Venv -Name "venv_tts" -RequirementsFile $null
    }
}

# ── Summary ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Copy models into models/stt/, models/llm/, models/tts/"
Write-Host "  2. Copy GPT-SoVITS into vendor/GPT-SoVITS/"
Write-Host "  3. Copy reference audio into reference_audio/"
Write-Host "  4. Run: .\start.ps1"
Write-Host ""
