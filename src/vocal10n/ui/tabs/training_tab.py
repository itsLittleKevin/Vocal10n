"""Training tab for Section B — STT fine-tuning with SRT + WAV data.

Guides users through fine-tuning FasterWhisper (Whisper large-v3-turbo)
on their own voice recordings so the model produces fewer errors on
domain-specific or speaker-specific vocabulary.

Workflow:
1. Import WAV audio + SRT subtitle file
2. System slices WAV into segments using SRT timestamps
3. Prepares HuggingFace-compatible dataset
4. LoRA fine-tune on local GPU
5. Convert to CTranslate2 format for faster-whisper
6. Load custom model from STT tab
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vocal10n.config import get_config
from vocal10n.state import SystemState

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_TRAINING_DIR = _PROJECT_ROOT / "training"
_MODELS_DIR = _PROJECT_ROOT / "models" / "stt"


class TrainingTab(QWidget):
    """STT fine-tuning tab with SRT-based workflow."""

    training_log = Signal(str)  # emitted with log lines during training

    def __init__(self, state: SystemState, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._state = state
        self._cfg = get_config()

        self._wav_path: Path | None = None
        self._srt_path: Path | None = None
        self._training_thread: threading.Thread | None = None
        self._is_training = False

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── Header ────────────────────────────────────────────────────
        header = QLabel("STT Fine-Tuning")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #e0e0e0;")
        root.addWidget(header)

        desc = QLabel(
            "Fine-tune Whisper on your own voice recordings to reduce "
            "recognition errors. Provide a WAV recording and matching SRT "
            "subtitle file — the system will slice audio by timestamps and "
            "train a LoRA adapter."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #8892a4; font-size: 12px; margin-bottom: 4px;")
        root.addWidget(desc)

        # ── Step 1: Data Import ───────────────────────────────────────
        data_box = QGroupBox("Step 1 — Import Training Data")
        dl = QVBoxLayout(data_box)

        # WAV file
        wav_row = QHBoxLayout()
        wav_row.addWidget(QLabel("WAV Audio:"))
        self._wav_label = QLineEdit()
        self._wav_label.setReadOnly(True)
        self._wav_label.setPlaceholderText("Select a WAV recording...")
        wav_row.addWidget(self._wav_label, stretch=1)
        self._btn_wav = QPushButton("Browse")
        self._btn_wav.setFixedWidth(80)
        self._btn_wav.clicked.connect(self._browse_wav)
        wav_row.addWidget(self._btn_wav)
        dl.addLayout(wav_row)

        # SRT file
        srt_row = QHBoxLayout()
        srt_row.addWidget(QLabel("SRT Subtitle:"))
        self._srt_label = QLineEdit()
        self._srt_label.setReadOnly(True)
        self._srt_label.setPlaceholderText("Select matching SRT file...")
        srt_row.addWidget(self._srt_label, stretch=1)
        self._btn_srt = QPushButton("Browse")
        self._btn_srt.setFixedWidth(80)
        self._btn_srt.clicked.connect(self._browse_srt)
        srt_row.addWidget(self._btn_srt)
        dl.addLayout(srt_row)

        # Data info
        self._data_info = QLabel("No data loaded")
        self._data_info.setStyleSheet("color: #8892a4; font-size: 11px;")
        dl.addWidget(self._data_info)

        # Prepare button
        prep_row = QHBoxLayout()
        self._btn_prepare = QPushButton("Prepare Dataset")
        self._btn_prepare.setFixedWidth(130)
        self._btn_prepare.setEnabled(False)
        self._btn_prepare.clicked.connect(self._prepare_dataset)
        prep_row.addWidget(self._btn_prepare)

        self._prep_status = QLabel("")
        self._prep_status.setStyleSheet("color: #8892a4; font-size: 11px;")
        prep_row.addWidget(self._prep_status)
        prep_row.addStretch()
        dl.addLayout(prep_row)

        data_info_detail = QLabel(
            "<b>Format requirements:</b><br>"
            "• WAV: 16kHz mono recommended (will be converted automatically)<br>"
            "• SRT: Standard subtitle format with timestamps and transcription<br>"
            "• The SRT timestamps are used to slice the WAV into training segments<br>"
            "• Aim for 30+ segments for meaningful improvement"
        )
        data_info_detail.setWordWrap(True)
        data_info_detail.setStyleSheet(
            "color: #6272a4; font-size: 11px; margin-top: 4px;"
        )
        dl.addWidget(data_info_detail)

        root.addWidget(data_box)

        # ── Step 2: Training Configuration ────────────────────────────
        train_box = QGroupBox("Step 2 — Training Configuration")
        tl = QVBoxLayout(train_box)

        # Model name
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Output model name:"))
        self._model_name = QLineEdit()
        self._model_name.setPlaceholderText("my-whisper-finetune")
        self._model_name.setText("whisper-custom")
        name_row.addWidget(self._model_name, stretch=1)
        tl.addLayout(name_row)

        # Training params
        params_row = QHBoxLayout()

        params_row.addWidget(QLabel("Epochs:"))
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 50)
        self._epochs_spin.setValue(3)
        self._epochs_spin.setFixedWidth(60)
        self._epochs_spin.setToolTip(
            "Number of training epochs. 2-5 is typical for fine-tuning.\n"
            "More epochs = better adaptation but risk of overfitting."
        )
        params_row.addWidget(self._epochs_spin)

        params_row.addWidget(QLabel("Batch size:"))
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 32)
        self._batch_spin.setValue(4)
        self._batch_spin.setFixedWidth(60)
        self._batch_spin.setToolTip(
            "Training batch size. Reduce if you run out of VRAM.\n"
            "4 works for RTX 3060 12GB with LoRA."
        )
        params_row.addWidget(self._batch_spin)

        params_row.addWidget(QLabel("LoRA rank:"))
        self._lora_spin = QSpinBox()
        self._lora_spin.setRange(4, 64)
        self._lora_spin.setValue(16)
        self._lora_spin.setFixedWidth(60)
        self._lora_spin.setToolTip(
            "LoRA adapter rank. Higher = more parameters, more capacity.\n"
            "16 is a good default. Increase for complex domain vocabulary."
        )
        params_row.addWidget(self._lora_spin)

        params_row.addStretch()
        tl.addLayout(params_row)

        # Learning rate
        lr_row = QHBoxLayout()
        lr_row.addWidget(QLabel("Learning rate:"))
        self._lr_edit = QLineEdit()
        self._lr_edit.setText("1e-5")
        self._lr_edit.setFixedWidth(80)
        self._lr_edit.setToolTip("1e-5 is standard for Whisper fine-tuning.")
        lr_row.addWidget(self._lr_edit)
        lr_row.addStretch()

        self._fp16_cb = QCheckBox("Use FP16 (recommended for CUDA)")
        self._fp16_cb.setChecked(True)
        lr_row.addWidget(self._fp16_cb)
        tl.addLayout(lr_row)

        root.addWidget(train_box)

        # ── Step 3: Train ─────────────────────────────────────────────
        run_box = QGroupBox("Step 3 — Train & Convert")
        rl = QVBoxLayout(run_box)

        btn_row = QHBoxLayout()
        self._btn_train = QPushButton("Start Training")
        self._btn_train.setProperty("accent", True)
        self._btn_train.setFixedWidth(130)
        self._btn_train.setEnabled(False)
        self._btn_train.clicked.connect(self._start_training)
        btn_row.addWidget(self._btn_train)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setFixedWidth(70)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_training)
        btn_row.addWidget(self._btn_stop)

        btn_row.addStretch()

        self._train_status = QLabel("Ready")
        self._train_status.setStyleSheet("color: #8892a4; font-size: 12px;")
        btn_row.addWidget(self._train_status)
        rl.addLayout(btn_row)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("%v%")
        rl.addWidget(self._progress)

        # Log output
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMinimumHeight(150)
        self._log_view.setMaximumHeight(250)
        self._log_view.setStyleSheet(
            "font-family: 'Consolas', 'Courier New', monospace; "
            "font-size: 11px; background-color: #1a1a2e; color: #c8c8c8;"
        )
        self._log_view.setPlaceholderText(
            "Training log output will appear here..."
        )
        rl.addWidget(self._log_view)

        root.addWidget(run_box)

        # ── Step 4: Output ────────────────────────────────────────────
        out_box = QGroupBox("Step 4 — Use Custom Model")
        ol = QVBoxLayout(out_box)

        out_info = QLabel(
            "After training, the fine-tuned model will be saved as a "
            "CTranslate2 model in <b>models/stt/</b>. Select it from the "
            "Whisper Model dropdown in the <b>STT</b> tab to use it."
        )
        out_info.setWordWrap(True)
        out_info.setStyleSheet("color: #8892a4; font-size: 12px;")
        ol.addWidget(out_info)

        # List existing custom models
        self._models_label = QLabel("")
        self._models_label.setStyleSheet("color: #0f9b8e; font-size: 11px;")
        ol.addWidget(self._models_label)
        self._refresh_models()

        root.addWidget(out_box)
        root.addStretch()

        # Connect signals
        self.training_log.connect(self._append_log)

    # ------------------------------------------------------------------
    # File selection
    # ------------------------------------------------------------------

    @Slot()
    def _browse_wav(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select WAV Recording",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.m4a);;All Files (*)",
        )
        if path:
            self._wav_path = Path(path)
            self._wav_label.setText(path)
            self._update_data_status()

    @Slot()
    def _browse_srt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SRT Subtitle File",
            "",
            "Subtitle Files (*.srt);;All Files (*)",
        )
        if path:
            self._srt_path = Path(path)
            self._srt_label.setText(path)
            self._update_data_status()

    def _update_data_status(self) -> None:
        """Update data info label and enable/disable buttons."""
        has_wav = self._wav_path and self._wav_path.exists()
        has_srt = self._srt_path and self._srt_path.exists()

        if has_wav and has_srt:
            segments = self._count_srt_segments(self._srt_path)
            self._data_info.setText(
                f"WAV: {self._wav_path.name}  |  "
                f"SRT: {self._srt_path.name} ({segments} segments)"
            )
            self._data_info.setStyleSheet("color: #0f9b8e; font-size: 11px;")
            self._btn_prepare.setEnabled(True)
        else:
            parts = []
            if not has_wav:
                parts.append("Missing WAV")
            if not has_srt:
                parts.append("Missing SRT")
            self._data_info.setText(" | ".join(parts))
            self._data_info.setStyleSheet("color: #e94560; font-size: 11px;")
            self._btn_prepare.setEnabled(False)
            self._btn_train.setEnabled(False)

    @staticmethod
    def _count_srt_segments(srt_path: Path) -> int:
        """Count subtitle entries in an SRT file."""
        try:
            text = srt_path.read_text(encoding="utf-8")
            return len(re.findall(r"^\d+\s*$", text, re.MULTILINE))
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    @Slot()
    def _prepare_dataset(self) -> None:
        """Parse SRT and slice WAV into training segments."""
        if not self._wav_path or not self._srt_path:
            return

        self._prep_status.setText("Preparing...")
        self._prep_status.setStyleSheet("color: #e0e0e0; font-size: 11px;")

        def _do_prepare() -> None:
            try:
                segments = self._parse_srt(self._srt_path)
                if not segments:
                    self.training_log.emit("No segments found in SRT file")
                    return

                model_name = self._model_name.text().strip() or "whisper-custom"
                out_dir = _TRAINING_DIR / model_name / "data"
                out_dir.mkdir(parents=True, exist_ok=True)

                success_count = 0
                manifest_lines: list[str] = []

                for i, seg in enumerate(segments):
                    start_s = seg["start"]
                    end_s = seg["end"]
                    duration = end_s - start_s

                    if duration < 0.5 or duration > 30.0:
                        continue  # skip too-short or too-long segments

                    seg_wav = out_dir / f"seg_{i:04d}.wav"
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(self._wav_path),
                        "-ss", f"{start_s:.3f}",
                        "-to", f"{end_s:.3f}",
                        "-ar", "16000",
                        "-ac", "1",
                        "-c:a", "pcm_s16le",
                        str(seg_wav),
                    ]

                    flags = (
                        subprocess.CREATE_NO_WINDOW
                        if hasattr(subprocess, "CREATE_NO_WINDOW")
                        else 0
                    )
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        creationflags=flags,
                    )
                    if result.returncode == 0 and seg_wav.exists():
                        success_count += 1
                        manifest_lines.append(
                            f"{seg_wav.name}\t{seg['text']}"
                        )

                manifest_path = out_dir / "manifest.tsv"
                manifest_path.write_text(
                    "\n".join(manifest_lines), encoding="utf-8"
                )

                self.training_log.emit(
                    f"Dataset ready: {success_count}/{len(segments)} "
                    f"segments in {out_dir}"
                )
                self._prep_status.setText(
                    f"Done: {success_count}/{len(segments)} segments"
                )
                self._prep_status.setStyleSheet(
                    "color: #0f9b8e; font-size: 11px;"
                )
                self._btn_train.setEnabled(success_count >= 5)

            except FileNotFoundError:
                self._prep_status.setText("Error: ffmpeg not found")
                self._prep_status.setStyleSheet(
                    "color: #e94560; font-size: 11px;"
                )
                self.training_log.emit(
                    "ERROR: ffmpeg not found. Install from "
                    "https://ffmpeg.org/ and add to PATH."
                )
            except Exception as e:
                self._prep_status.setText(f"Error: {e}")
                self._prep_status.setStyleSheet(
                    "color: #e94560; font-size: 11px;"
                )
                logger.exception("Dataset preparation failed")

        threading.Thread(target=_do_prepare, daemon=True).start()

    @staticmethod
    def _parse_srt(srt_path: Path) -> list[dict]:
        """Parse SRT file into list of {start, end, text} dicts."""
        text = srt_path.read_text(encoding="utf-8")
        segments: list[dict] = []
        ts_re = (
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
        )

        blocks = re.split(r"\n\n+", text.strip())
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 2:
                continue

            # Try timestamp on the second line (standard SRT)
            ts_match = re.match(ts_re, lines[1] if len(lines) >= 3 else lines[0])
            if ts_match and len(lines) >= 3:
                sub_text = " ".join(lines[2:]).strip()
            elif not ts_match:
                ts_match = re.match(ts_re, lines[0])
                sub_text = " ".join(lines[1:]).strip() if ts_match else ""
            else:
                sub_text = " ".join(lines[1:]).strip()

            if not ts_match or not sub_text:
                continue

            g = ts_match.groups()
            start = (
                int(g[0]) * 3600
                + int(g[1]) * 60
                + int(g[2])
                + int(g[3]) / 1000
            )
            end = (
                int(g[4]) * 3600
                + int(g[5]) * 60
                + int(g[6])
                + int(g[7]) / 1000
            )

            # Strip HTML tags
            sub_text = re.sub(r"<[^>]+>", "", sub_text).strip()

            if sub_text and end > start:
                segments.append(
                    {"start": start, "end": end, "text": sub_text}
                )

        return segments

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @Slot()
    def _start_training(self) -> None:
        """Start the fine-tuning process in a background thread."""
        if self._is_training:
            return

        model_name = self._model_name.text().strip() or "whisper-custom"
        data_dir = _TRAINING_DIR / model_name / "data"
        manifest = data_dir / "manifest.tsv"

        if not manifest.exists():
            QMessageBox.warning(
                self, "No Dataset", "Prepare the dataset first (Step 1)."
            )
            return

        self._is_training = True
        self._btn_train.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._train_status.setText("Training...")
        self._train_status.setStyleSheet("color: #e0e0e0; font-size: 12px;")
        self._progress.setValue(0)
        self._log_view.clear()

        self._training_thread = threading.Thread(
            target=self._run_training,
            args=(model_name, data_dir),
            daemon=True,
        )
        self._training_thread.start()

    @Slot()
    def _stop_training(self) -> None:
        """Request training stop."""
        self._is_training = False
        self._btn_stop.setEnabled(False)
        self._train_status.setText("Stopping...")
        self.training_log.emit("Training stop requested...")

    def _run_training(self, model_name: str, data_dir: Path) -> None:
        """Execute the full training + conversion pipeline."""
        try:
            output_dir = _TRAINING_DIR / model_name / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            ct2_dir = _MODELS_DIR / model_name
            ct2_dir.mkdir(parents=True, exist_ok=True)

            epochs = self._epochs_spin.value()
            batch_size = self._batch_spin.value()
            lora_rank = self._lora_spin.value()
            lr = self._lr_edit.text().strip() or "1e-5"
            fp16 = self._fp16_cb.isChecked()

            self.training_log.emit("=== Fine-tuning Whisper large-v3-turbo ===")
            self.training_log.emit(f"Model name: {model_name}")
            self.training_log.emit(f"Data dir: {data_dir}")
            self.training_log.emit(
                f"Epochs: {epochs}, Batch: {batch_size}, "
                f"LoRA rank: {lora_rank}, LR: {lr}"
            )
            self.training_log.emit("")

            manifest = data_dir / "manifest.tsv"
            sample_count = sum(
                1
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            self.training_log.emit(f"Training samples: {sample_count}")

            # --- Phase 1: Verify dependencies ---
            self.training_log.emit("\n[1/3] Checking dependencies...")
            try:
                import transformers  # noqa: F401
                import peft  # noqa: F401
                import datasets  # noqa: F401

                self.training_log.emit(
                    "  transformers, peft, datasets: OK"
                )
            except ImportError as e:
                self.training_log.emit(
                    f"  Missing dependency: {e}\n"
                    "  Install with:\n"
                    "  pip install transformers peft datasets accelerate"
                )
                self._on_training_done(False, "Missing dependencies")
                return

            if not self._is_training:
                self._on_training_done(False, "Cancelled")
                return

            # --- Phase 2: Training ---
            self.training_log.emit("\n[2/3] Training with LoRA...")
            self._progress.setValue(10)

            success = self._do_lora_training(
                data_dir=data_dir,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                lora_rank=lora_rank,
                learning_rate=float(lr),
                fp16=fp16,
            )
            if not success:
                self._on_training_done(False, "Training failed")
                return

            if not self._is_training:
                self._on_training_done(False, "Cancelled")
                return

            # --- Phase 3: Convert to CTranslate2 ---
            self.training_log.emit("\n[3/3] Converting to CTranslate2...")
            self._progress.setValue(80)

            success = self._convert_to_ct2(output_dir, ct2_dir)
            if not success:
                self._on_training_done(False, "Conversion failed")
                return

            self._progress.setValue(100)
            self._on_training_done(True, f"Model saved to {ct2_dir}")

        except Exception as e:
            logger.exception("Training pipeline failed")
            self.training_log.emit(f"\nERROR: {e}")
            self._on_training_done(False, str(e))

    def _do_lora_training(
        self,
        data_dir: Path,
        output_dir: Path,
        epochs: int,
        batch_size: int,
        lora_rank: int,
        learning_rate: float,
        fp16: bool,
    ) -> bool:
        """Run LoRA fine-tuning on the Whisper model."""
        try:
            import torch
            from datasets import Audio, Dataset
            from peft import LoraConfig, get_peft_model
            from transformers import (
                Seq2SeqTrainer,
                Seq2SeqTrainingArguments,
                WhisperForConditionalGeneration,
                WhisperProcessor,
            )

            base_model = "openai/whisper-large-v3-turbo"
            self.training_log.emit(f"  Loading base model: {base_model}")

            processor = WhisperProcessor.from_pretrained(base_model)
            model = WhisperForConditionalGeneration.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if fp16 else torch.float32,
            )

            # Apply LoRA
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            trainable = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            total = sum(p.numel() for p in model.parameters())
            self.training_log.emit(
                f"  LoRA applied: {trainable:,} trainable / {total:,} total "
                f"({100 * trainable / total:.1f}%)"
            )

            # Load dataset from manifest
            manifest = data_dir / "manifest.tsv"
            records = []
            for line in manifest.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                wav_name, text = parts
                wav_path = data_dir / wav_name
                if wav_path.exists():
                    records.append({"audio": str(wav_path), "text": text})

            if not records:
                self.training_log.emit("  ERROR: No valid training records")
                return False

            ds = Dataset.from_list(records)
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
            self.training_log.emit(f"  Dataset loaded: {len(ds)} samples")

            # Prepare features
            def _prepare(batch):  # noqa: ANN001, ANN202
                audio = batch["audio"]
                inputs = processor(
                    audio["array"],
                    sampling_rate=audio["sampling_rate"],
                    return_tensors="pt",
                )
                batch["input_features"] = inputs.input_features[0]
                batch["labels"] = processor.tokenizer(
                    batch["text"]
                ).input_ids
                return batch

            self.training_log.emit("  Processing audio features...")
            ds = ds.map(_prepare, remove_columns=ds.column_names)

            args = Seq2SeqTrainingArguments(
                output_dir=str(output_dir),
                per_device_train_batch_size=batch_size,
                num_train_epochs=epochs,
                learning_rate=learning_rate,
                fp16=fp16 and torch.cuda.is_available(),
                logging_steps=10,
                save_steps=500,
                save_total_limit=2,
                remove_unused_columns=False,
                predict_with_generate=False,
                report_to="none",
                dataloader_num_workers=0,
            )

            trainer = Seq2SeqTrainer(
                model=model, args=args, train_dataset=ds
            )

            self.training_log.emit("  Starting training...")
            self._progress.setValue(30)

            trainer.train()

            self.training_log.emit("  Training complete!")
            self._progress.setValue(70)

            # Save merged model
            self.training_log.emit("  Merging LoRA weights and saving...")
            merged_dir = output_dir / "merged"
            merged_dir.mkdir(exist_ok=True)
            model = model.merge_and_unload()
            model.save_pretrained(str(merged_dir))
            processor.save_pretrained(str(merged_dir))

            self.training_log.emit(f"  Merged model saved to {merged_dir}")
            return True

        except Exception as e:
            self.training_log.emit(f"  Training error: {e}")
            logger.exception("LoRA training failed")
            return False

    def _convert_to_ct2(self, hf_dir: Path, ct2_dir: Path) -> bool:
        """Convert HuggingFace model to CTranslate2 for faster-whisper."""
        merged_dir = hf_dir / "merged"
        if not merged_dir.exists():
            self.training_log.emit("  ERROR: Merged model not found")
            return False

        try:
            cmd = [
                "ct2-transformers-converter",
                "--model", str(merged_dir),
                "--output_dir", str(ct2_dir),
                "--quantization", "int8_float16",
                "--force",
            ]
            self.training_log.emit(f"  Running: {' '.join(cmd)}")

            flags = (
                subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0
            )
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                creationflags=flags,
            )

            if result.returncode != 0:
                self.training_log.emit(
                    f"  Converter stderr: {result.stderr}"
                )
                return self._convert_to_ct2_python(merged_dir, ct2_dir)

            self.training_log.emit(
                f"  CTranslate2 model saved to {ct2_dir}"
            )
            return True

        except FileNotFoundError:
            self.training_log.emit(
                "  ct2-transformers-converter not found, "
                "trying Python API..."
            )
            return self._convert_to_ct2_python(merged_dir, ct2_dir)
        except Exception as e:
            self.training_log.emit(f"  Conversion error: {e}")
            return False

    def _convert_to_ct2_python(
        self, merged_dir: Path, ct2_dir: Path
    ) -> bool:
        """Fallback: convert using ctranslate2 Python API."""
        try:
            import ctranslate2

            converter = ctranslate2.converters.TransformersConverter(
                str(merged_dir)
            )
            converter.convert(
                str(ct2_dir), quantization="int8_float16", force=True
            )
            self.training_log.emit(
                f"  CTranslate2 model saved to {ct2_dir}"
            )
            return True

        except ImportError:
            self.training_log.emit(
                "  ERROR: ctranslate2 not installed.\n"
                "  Install: pip install ctranslate2"
            )
            return False
        except Exception as e:
            self.training_log.emit(f"  Python conversion error: {e}")
            return False

    # ------------------------------------------------------------------
    # Training lifecycle
    # ------------------------------------------------------------------

    def _on_training_done(self, success: bool, message: str) -> None:
        """Called when training finishes (from background thread)."""
        self._is_training = False
        self._btn_train.setEnabled(True)
        self._btn_stop.setEnabled(False)

        if success:
            self._train_status.setText("Complete!")
            self._train_status.setStyleSheet(
                "color: #0f9b8e; font-size: 12px;"
            )
            self.training_log.emit(
                f"\n=== Training complete: {message} ==="
            )
            self._refresh_models()
        else:
            self._train_status.setText(f"Failed: {message}")
            self._train_status.setStyleSheet(
                "color: #e94560; font-size: 12px;"
            )
            self.training_log.emit(
                f"\n=== Training failed: {message} ==="
            )

    @Slot(str)
    def _append_log(self, text: str) -> None:
        """Append text to the log view (thread-safe via signal)."""
        self._log_view.appendPlainText(text)

    def _refresh_models(self) -> None:
        """List custom models in models/stt/."""
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        custom_models = [
            d.name
            for d in _MODELS_DIR.iterdir()
            if d.is_dir() and (d / "model.bin").exists()
        ]
        if custom_models:
            self._models_label.setText(
                f"Custom models: {', '.join(custom_models)}"
            )
        else:
            self._models_label.setText("No custom models yet")
            self._models_label.setStyleSheet(
                "color: #8892a4; font-size: 11px;"
            )
