# Vocal10n

Real-time speech translation system built with **FasterWhisper**, **Qwen3-4B**, and **GPT-SoVITS**.

Speak in one language → live subtitles → LLM translation → voice synthesis in another language.

## Features

- **STT** — FasterWhisper large-v3-turbo with hallucination filtering and phonetic correction
- **Translation** — Qwen3-4B-Instruct via llama-cpp-python with RAG support
- **TTS** — GPT-SoVITS voice cloning for natural speech output
- **OBS Integration** — Live HTML/CSS subtitles via Browser Source
- **PySide6 UI** — Native desktop interface with real-time monitoring

## Requirements

- Windows 10/11
- NVIDIA GPU (RTX 3060 12GB or better recommended)
- Python 3.11
- CUDA Toolkit 12.x

## Quick Start

```powershell
# 1. Clone the repo
git clone https://github.com/itsLittleKevin/Vocal10n.git
cd Vocal10n

# 2. Set up environments
.\setup_env.ps1

# 3. Download / copy models into models/ directory

# 4. Launch
.\start.ps1
```

## Project Structure

```
src/vocal10n/
├── stt/        # Speech-to-Text (FasterWhisper)
├── llm/        # Translation Engine (Qwen3-4B)
├── tts/        # Text-to-Speech client (GPT-SoVITS)
├── pipeline/   # Orchestration, events, file I/O
├── ui/         # PySide6 GUI
├── obs/        # OBS subtitle overlay server
└── utils/      # GPU monitoring, logging
```

## License

MIT
