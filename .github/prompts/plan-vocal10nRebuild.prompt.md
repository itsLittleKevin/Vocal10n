# Vocal10n Rebuild Plan

## Decisions Made

| Decision | Choice |
|----------|--------|
| **Venvs** | 2 separate: Main/STT+LLM+UI (Python 3.11), TTS (Python 3.11) |
| **UI Framework** | PySide6 (Qt6, LGPL) |
| **Directory Layout** | `src/vocal10n/` Python package with submodules |
| **TTS Architecture** | GPT-SoVITS as separate subprocess, HTTP API on port 9880 |
| **Git** | GitHub repo `itsLittleKevin/Vocal10n`, exclude `Vocal10n-prebuild/` |

---

## Target Directory Structure

```
Vocal10n/
├── .gitignore
├── README.md
├── LICENSE
├── pyproject.toml                  # Project metadata
├── setup_env.ps1                   # One-click environment setup
├── start.bat                       # Launch script (Windows)
├── start.ps1                       # Launch script (PowerShell)
│
├── src/
│   └── vocal10n/
│       ├── __init__.py
│       ├── app.py                  # Main entry point
│       ├── config.py               # Global config loader (YAML)
│       ├── state.py                # Thread-safe global state (SystemState)
│       ├── constants.py            # Shared constants, enums
│       │
│       ├── stt/                    # Speech-to-Text (FasterWhisper)
│       │   ├── __init__.py
│       │   ├── engine.py           # FasterWhisper wrapper, streaming
│       │   ├── controller.py       # STT controller (lifecycle, events)
│       │   ├── worker.py           # Background STT worker thread
│       │   ├── filters.py          # Hallucination filter, phonetic correction
│       │   ├── audio_capture.py    # Microphone input, VAD
│       │   └── transcript.py       # Segment management, confirmation logic
│       │
│       ├── llm/                    # Translation Engine (Qwen3-4B via llama-cpp)
│       │   ├── __init__.py
│       │   ├── engine.py           # llama-cpp-python model loader
│       │   ├── api_backend.py      # OpenAI-compatible API backend
│       │   ├── controller.py       # LLM controller (lifecycle, events)
│       │   ├── translator.py       # Translation logic, prompt templates
│       │   ├── corrector.py        # Glossary-based STT correction + prompt hints
│       │   └── rag.py              # Knowledge base / RAG integration (planned)
│       │
│       ├── tts/                    # Text-to-Speech (GPT-SoVITS client)
│       │   ├── __init__.py
│       │   ├── client.py           # HTTP client to GPT-SoVITS API
│       │   ├── controller.py       # TTS controller (server lifecycle, queue)
│       │   ├── queue.py            # TTS queue manager, buffering, pruning
│       │   ├── audio_output.py     # Playback, device selection
│       │   └── server_manager.py   # Subprocess launcher for GPT-SoVITS
│       │
│       ├── pipeline/               # Orchestration
│       │   ├── __init__.py
│       │   ├── coordinator.py      # Main pipeline: STT→LLM→TTS flow
│       │   ├── events.py           # Event dispatcher (pub/sub)
│       │   ├── latency.py          # Latency tracker
│       │   └── file_writer.py      # SRT, TXT, WAV output
│       │
│       ├── ui/                     # PySide6 GUI
│       │   ├── __init__.py
│       │   ├── main_window.py      # Main window (A/B split layout)
│       │   ├── section_a.py        # Top: A1 (text streams) + A2 (metrics)
│       │   ├── section_b.py        # Bottom: Tab container
│       │   ├── tabs/
│       │   │   ├── __init__.py
│       │   │   ├── stt_tab.py      # STT settings tab
│       │   │   ├── translation_tab.py  # LLM settings tab
│       │   │   ├── tts_tab.py      # TTS/VITS settings tab
│       │   │   ├── output_tab.py   # Output settings tab
│       │   │   ├── obs_tab.py      # OBS subtitle styling tab
│       │   │   └── training_tab.py # Training placeholder tab
│       │   ├── widgets/            # Reusable custom widgets
│       │   │   ├── __init__.py
│       │   │   ├── param_slider.py # Parameter slider with info tooltip
│       │   │   ├── model_selector.py   # Model dropdown + load/unload
│       │   │   └── stream_text.py  # Streaming text display widget
│       │   └── styles/
│       │       └── theme.qss       # Qt stylesheet
│       │
│       ├── obs/                    # OBS overlay server
│       │   ├── __init__.py
│       │   ├── server.py           # Flask/HTTP server for OBS Browser Source
│       │   └── overlay.html        # HTML/CSS template
│       │
│       └── utils/                  # Shared utilities
│           ├── __init__.py
│           ├── gpu.py              # GPU/VRAM monitoring (pynvml)
│           └── logger.py           # Logging setup
│
├── config/
│   └── default.yaml                # Default pipeline config
│
├── models/                         # Local model storage (git-ignored)
│   ├── stt/                        # FasterWhisper models
│   ├── llm/                        # Qwen3 GGUF files
│   └── tts/                        # GPT-SoVITS pretrained models
│
├── reference_audio/                # TTS reference audio (git-ignored)
│
├── knowledge_base/                 # Glossary files for STT correction
│   ├── .gitkeep
│   └── glossary_general.txt        # Chinese gov/agriculture terms
│
├── output/                         # Generated files (git-ignored)
│   ├── subtitles/
│   ├── audio/
│   └── training_data/
│
├── training/                       # User training data (git-ignored)
│
├── vendor/                         # Vendored dependencies
│   └── GPT-SoVITS/                 # Embedded GPT-SoVITS (git-ignored)
│
├── venvs/                          # Virtual environments (git-ignored)
│   ├── venv_main/                  # Python 3.11 — STT + LLM + UI + Pipeline
│   └── venv_tts/                   # Python 3.11 — GPT-SoVITS server
│
├── requirements/                   # Per-venv requirements
│   ├── requirements-main.txt       # STT + LLM + UI + Pipeline deps
│   └── requirements-tts.txt        # GPT-SoVITS deps (for reference)
│
└── Vocal10n-prebuild/              # Legacy prebuild (git-ignored, local only)
```

---

## .gitignore Plan

```gitignore
# === Environments ===
venvs/
venv*/
__pycache__/
*.pyc
.env

# === Models (large binaries) ===
models/
*.gguf
*.pth
*.ckpt
*.safetensors
*.bin
*.whl

# === Vendored code ===
vendor/

# === User data ===
reference_audio/
training/
knowledge_base/*.db
knowledge_base/*.index

# === Output ===
output/
*.wav
*.srt

# === Legacy prebuild ===
Vocal10n-prebuild/

# === IDE / OS ===
.vscode/
.idea/
*.swp
Thumbs.db
.DS_Store

# === Logs ===
*.log
logs/
```

---

## Master To-Do List

### Phase 0: Project Foundation
- [x] 0.1 — Initialize git repo, connect to GitHub remote
- [x] 0.2 — Create `.gitignore` with all exclusion patterns
- [x] 0.3 — Create directory skeleton (all dirs with `__init__.py` / `.gitkeep`)
- [x] 0.4 — Create `pyproject.toml` with project metadata
- [x] 0.5 — Create `config/default.yaml` (port from prebuild's `pipeline_config.yaml`)
- [x] 0.6 — Create `requirements/` files (derive from prebuild deps)
- [x] 0.7 — Write `setup_env.ps1` to create 2 venvs and install deps
- [x] 0.8 — Initial commit & push

### Phase 1: Core Infrastructure
- [x] 1.1 — Implement `constants.py` — enums (`EventType`, `Language`, `ModelStatus`, `TTSSource`)
- [x] 1.2 — Implement `config.py` — YAML config loader/saver (dot-key access, thread-safe)
- [x] 1.3 — Implement `state.py` — thread-safe `SystemState` (Qt signals for UI binding)
- [x] 1.4 — Implement `pipeline/events.py` — event dispatcher (port from prebuild, sync-only)
- [x] 1.5 — Implement `pipeline/latency.py` — latency tracker (Qt signal on update)
- [x] 1.6 — Implement `utils/gpu.py` — GPU/VRAM monitoring (pynvml)
- [x] 1.7 — Implement `utils/logger.py` — logging config

### Phase 2: PySide6 UI Shell
- [x] 2.1 — `app.py` — QApplication entry, theme loading
- [x] 2.2 — `ui/main_window.py` — Main window with A/B vertical split
- [x] 2.3 — `ui/section_a.py` — A1 (streaming text panels) + A2 (metrics/status)
- [x] 2.4 — `ui/section_b.py` — QTabWidget container
- [x] 2.5 — `ui/widgets/stream_text.py` — Streaming text display widget
- [x] 2.6 — `ui/widgets/param_slider.py` — Slider with info tooltip + reset
- [x] 2.7 — `ui/widgets/model_selector.py` — Model dropdown + load/unload buttons
- [x] 2.8 — `ui/styles/theme.qss` — Base dark theme stylesheet
- [x] 2.9 — Verify UI shell launches and looks correct

### Phase 3: STT Module (FasterWhisper)
- [x] 3.1 — `stt/audio_capture.py` — Microphone capture with device selection
- [x] 3.2 — `stt/engine.py` — FasterWhisper model loader/unloader with VRAM cleanup
- [x] 3.3 — `stt/transcript.py` — Segment management (pending/confirmed)
- [x] 3.4 — `stt/filters.py` — Hallucination filter + phonetic correction (port from prebuild)
- [x] 3.5 — `ui/tabs/stt_tab.py` — STT settings tab (toggle, model select, language, params)
- [x] 3.6 — Wire STT to pipeline events and UI streaming display
- [x] 3.7 — Test: mic → text appearing in A1a with latency metric

### Phase 4: LLM Translation Module (Qwen3)
- [x] 4.1 — `llm/engine.py` — llama-cpp-python model loader (port from prebuild)
- [x] 4.2 — `llm/translator.py` — Translation logic with prompt templates
- [x] 4.3 — `llm/corrector.py` — Glossary-based correction + prompt augmentation
- [x] 4.4 — `llm/rag.py` — Vector retrieval for large glossaries (FAISS + MiniLM)
- [x] 4.5 — `ui/tabs/translation_tab.py` — Translation tab (toggle, lang select, prompt editor, term file list, params)
- [x] 4.6 — Wire LLM to pipeline: STT events → translate → display in A1b
- [x] 4.6a — `llm/api_backend.py` — HTTP API backend for remote LLM servers
- [x] 4.6b — `llm/controller.py` — LLM controller (load/unload/translate lifecycle)
- [x] 4.7 — Test: standalone mode (manual input → translation output)

### Phase 5: TTS Module (GPT-SoVITS)
- [x] 5.1 — Copy GPT-SoVITS into `vendor/GPT-SoVITS/`
- [x] 5.2 — `tts/server_manager.py` — Subprocess launcher for GPT-SoVITS API
- [x] 5.3 — `tts/client.py` — HTTP client (port from prebuild's `tts_client.py`)
- [x] 5.4 — `tts/queue.py` — TTS queue with buffering logic
- [x] 5.5 — `tts/audio_output.py` — Playback with device selection
- [x] 5.6 — `ui/tabs/tts_tab.py` — TTS tab (source/target toggles, ref audio, device select, params)
- [x] 5.7 — Wire TTS to pipeline: translation events → synthesize → play
- [x] 5.7a — `tts/controller.py` — TTS controller (server lifecycle, queue, playback)
- [x] 5.8 — Test: end-to-end STT → LLM → TTS

### Phase 6: Pipeline Orchestration
- [x] 6.1 — `pipeline/coordinator.py` — Full pipeline coordination (port & clean up from prebuild)
- [x] 6.2 — `pipeline/file_writer.py` — Async SRT/TXT/WAV file output
- [x] 6.3 — `ui/tabs/output_tab.py` — Output settings (checkboxes for each output type)
- [x] 6.4 — Wire file outputs to pipeline events
- [x] 6.5 — Test: full pipeline with all outputs enabled

### Phase 7: OBS Integration
- [x] 7.1 — `obs/server.py` — Flask HTTP server for Browser Source
- [x] 7.2 — `obs/overlay.html` — HTML/CSS subtitle overlay (port from prebuild)
- [x] 7.3 — `ui/tabs/obs_tab.py` — OBS tab (font, size, styling, preview)
- [x] 7.4 — Test: OBS Browser Source displays live subtitles

### Phase 8: Launch Scripts & Polish
- [x] 8.1 — `start.ps1` / `start.bat` — Launch TTS server + main app
- [x] 8.2 — A2 section: live GPU metrics, latency display, module status indicators
- [x] 8.3 — Error handling and graceful shutdown
- [x] 8.4 — Manual A1a input mode (standalone translator when STT is off)
- [x] 8.5 — `ui/tabs/training_tab.py` — Training placeholder tab

### Phase 9: Testing & Documentation
- [x] 9.1 — End-to-end test: speech → subtitles → translation → TTS
- [x] 9.2 — VRAM usage validation (target: <12GB total)
- [x] 9.3 — Latency benchmarks (STT <1.5s, total <3.0s)
- [ ] 9.4 — Write README.md with setup instructions
- [ ] 9.5 — Final commit & push

#### Phase 9 Benchmark Results (Chinese newsletter reading)

| Component | Measured | Target | Status |
|-----------|----------|--------|--------|
| STT | 1390ms | <1500ms | **PASS** (borderline) |
| Translation | 486ms | — | OK |
| TTS | 7959ms | — | **BOTTLENECK** |
| **Total** | **9836ms** | **<3000ms** | **FAIL (3.3x over)** |

**VRAM budget**: ~8-9 GB estimated (FasterWhisper ~1.5GB + Qwen3 ~3GB + GPT-SoVITS ~3-4GB) — **PASS** (<12GB)

**Root causes identified**:
1. TTS synthesis (~3-6s per segment) is slower than natural speaking pace; queue backs up to 4-5 items
2. STT phonetic errors on domain terms (科技巧院→科技小院, 驱州→曲周, 能源大学→农业大学)
3. Translation fragmentation — short segments lose context, producing incoherent fragments

**Fixes implemented**:
- TTS queue pruning: `max_pending=3`, drops oldest when queue exceeds depth
- Staleness threshold reduced from 30s to 8s
- Translation context window: last N confirmed translations passed as LLM context
- Sentence buffering: raised `min_clause_chars` from 5→8, configurable `max_buffer_age`
- Glossary-based corrector (`llm/corrector.py`): fuzzy pinyin matching against knowledge_base/*.txt
- Sample glossary with Chinese government/agriculture terms

### Phase 10: Future Improvements (planned)

Priority 1 — TTS latency (must-fix for <3s total target):
- [ ] 10.1 — Chunked audio playback (start playing first chunk while rest downloads)
- [ ] 10.2 — Parallel TTS workers (if VRAM headroom allows)
- [ ] 10.3 — Investigate faster TTS models or reduced-quality mode for live use

Priority 2 — STT accuracy:
- [ ] 10.4 — Expand glossary system with UI for adding/managing domain term files
- [ ] 10.5 — Benchmark alternative Whisper models (medium vs large-v3-turbo)
- [ ] 10.6 — Initial prompt optimization from domain glossaries

Priority 3 — Translation quality:
- [ ] 10.7 — Retranslation-on-extend (cancel in-flight translation when segment grows)
- [x] 10.8 — Phase 4.4: RAG vector retrieval for large glossaries (implemented)

---

## Component Dependencies (per venv)

### venv_main (Python 3.11) — STT + LLM + UI + Pipeline
```
PySide6==6.10.2
pyyaml==6.0.3
numpy==2.4.2
pynvml>=11.5.0              # GPU monitoring (consider nvidia-ml-py)
requests==2.32.5
psutil==7.2.2
faster-whisper==1.2.1
sounddevice==0.5.5
soundfile==0.13.1
scipy==1.17.0
opencc-python-reimplemented==0.1.7
pypinyin==0.55.0
llama-cpp-python==0.3.16    # with CUDA support (--extra-index-url)
sentence-transformers>=2.2.0  # RAG embeddings (CPU-only, optional)
faiss-cpu>=1.7.0              # RAG vector search (optional)
flask==3.1.2
flask-cors==6.0.2
```

### venv_tts (Python 3.11) — GPT-SoVITS server
```
# GPT-SoVITS requirements (see vendor/GPT-SoVITS/requirements.txt)
# Launched as separate subprocess — does not need PySide6
# Key deps: torch 2.x (CUDA), torchaudio, transformers, etc.
```

---

## Models to Copy from Prebuild

| Source | Destination | Notes |
|--------|-------------|-------|
| `Vocal10n-prebuild/Hybrid Translation Dispatcher/models/Qwen3-4B-Instruct-2507.Q4_K_M.gguf` | `models/llm/` | ~4GB |
| FasterWhisper large-v3-turbo (HuggingFace cache) | `models/stt/` | Can auto-download, or copy cache |
| `Vocal10n-prebuild/GPT-SoVITS/` (entire directory) | `vendor/GPT-SoVITS/` | ~3GB+ with pretrained models |
| `Vocal10n-prebuild/live-translation-pipeline/reference_audio/` | `reference_audio/` | Sample audio files |
| `Vocal10n-prebuild/personal-logger/context_gaming.txt` | `knowledge_base/` | Gaming terminology for correction |
| — | `knowledge_base/glossary_general.txt` | Chinese gov/agriculture terms (created in Phase 9) |

---

## Key Architecture Decisions

1. **Event-driven pipeline** — Keep the pub/sub `EventDispatcher` pattern from prebuild. It cleanly decouples STT → LLM → TTS.

2. **PySide6 signals** — Qt signals/slots replace Gradio's polling. Real-time text updates via `QTextEdit.append()` with custom signals from worker threads.

3. **Worker threads** — STT, LLM, and TTS each run in `QThread` workers. The coordinator manages lifecycle and event routing.

4. **Config-driven** — Single `default.yaml` config file controls all parameters. UI changes write back to config. Config is the source of truth.

5. **Graceful VRAM management** — Sequential model loading (TTS first as subprocess, then LLM, then STT). Explicit cleanup on unload with `gc.collect()` + `torch.cuda.empty_cache()`.

6. **Glossary-based correction** — Domain-specific term files in `knowledge_base/` are fuzzy-matched (pinyin similarity) against STT output and injected as glossary hints in the translation prompt; no separate LLM call needed.

7. **TTS queue pruning** — Queue drops oldest items when depth exceeds `max_pending` (default 3). Prevents backlog buildup during fast speech. Combined with larger translation segments (min 8 chars) to reduce TTS call count.

8. **Translation context window** — Last N confirmed translations passed as context in the LLM prompt for cross-segment coherence.

9. **RAG vector retrieval** — When glossary exceeds `rag_threshold` (default 100 terms), `Corrector` auto-switches from O(n) pinyin scan to FAISS vector search via `RAGIndex`. Uses `all-MiniLM-L6-v2` embedding model on CPU (~80 MB, no VRAM cost). Index is cached to `knowledge_base/*.faiss`/`*.npy` for fast reload. Gracefully degrades to pinyin scan if deps not installed.
