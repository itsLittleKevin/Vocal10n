"""OBS subtitle overlay HTTP server.

Serves a lightweight HTML/CSS page for OBS Browser Source at
``http://<host>:<port>/overlay.html``.  The overlay polls two JSON
endpoints for live source and translated text.

Endpoints:
    GET /overlay.html    — main OBS Browser Source page
    GET /live-text       — current source-language subtitle  (JSON)
    GET /live-translation— current target-language subtitle  (JSON)
    GET /style.css       — dynamic CSS driven by OBS tab settings
    GET /health          — simple health-check
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, Response
from flask_cors import CORS

from vocal10n.config import get_config
from vocal10n.constants import EventType
from vocal10n.pipeline.events import (
    Event,
    TextEvent,
    TranslationEvent,
    get_dispatcher,
)

logger = logging.getLogger(__name__)

_OVERLAY_HTML = Path(__file__).with_name("overlay.html")


class OBSSubtitleServer:
    """Flask-based HTTP server for OBS Browser Source subtitles.

    Text is fed by subscribing to pipeline events (STT_PENDING,
    TRANSLATION_PENDING / TRANSLATION_CONFIRMED).
    """

    def __init__(self) -> None:
        self._cfg = get_config()
        self._dispatcher = get_dispatcher()

        # Current subtitle text (updated by event handlers)
        self._source_text: str = ""
        self._source_updated_at: float = 0.0
        self._target_text: str = ""
        self._target_updated_at: float = 0.0
        self._lock = threading.Lock()

        # Auto-clear timeout (seconds of no change → clear)
        self._clear_timeout: float = 2.5

        # Flask app
        self._app = Flask(__name__)
        CORS(self._app)

        # Suppress werkzeug request logs
        wlog = logging.getLogger("werkzeug")
        wlog.setLevel(logging.ERROR)

        self._register_routes()
        self._subscribe_events()

        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the HTTP server in a background thread."""
        if self._running:
            return
        host = self._cfg.get("obs.host", "127.0.0.1")
        port = int(self._cfg.get("obs.port", 5124))
        self._running = True
        self._thread = threading.Thread(
            target=self._run_server,
            args=(host, port),
            daemon=True,
            name="obs-server",
        )
        self._thread.start()
        logger.info("OBS subtitle server started at http://%s:%d", host, port)

    def stop(self) -> None:
        """Signal the server to stop (daemon thread will die with process)."""
        self._running = False
        self._unsubscribe_events()

    # ------------------------------------------------------------------
    # Flask routes
    # ------------------------------------------------------------------

    def _register_routes(self) -> None:
        app = self._app

        @app.route("/overlay.html")
        def serve_overlay():
            if _OVERLAY_HTML.exists():
                return Response(
                    _OVERLAY_HTML.read_text(encoding="utf-8"),
                    mimetype="text/html",
                )
            return "overlay.html not found", 404

        @app.route("/live-text")
        def live_text():
            text = self._get_source_text()
            return jsonify({"text": text, "timestamp": time.time()})

        @app.route("/live-translation")
        def live_translation():
            text = self._get_target_text()
            return jsonify({"text": text, "timestamp": time.time()})

        @app.route("/style.css")
        def dynamic_css():
            css = self._build_dynamic_css()
            return Response(css, mimetype="text/css")

        @app.route("/health")
        def health():
            return jsonify({"status": "ok"})

    # ------------------------------------------------------------------
    # Event subscriptions
    # ------------------------------------------------------------------

    def _subscribe_events(self) -> None:
        d = self._dispatcher
        d.subscribe(EventType.STT_PENDING, self._on_stt_pending)
        d.subscribe(EventType.STT_CONFIRMED, self._on_stt_confirmed)
        d.subscribe(EventType.TRANSLATION_PENDING, self._on_translation_pending)
        d.subscribe(EventType.TRANSLATION_CONFIRMED, self._on_translation_confirmed)

    def _unsubscribe_events(self) -> None:
        d = self._dispatcher
        d.unsubscribe(EventType.STT_PENDING, self._on_stt_pending)
        d.unsubscribe(EventType.STT_CONFIRMED, self._on_stt_confirmed)
        d.unsubscribe(EventType.TRANSLATION_PENDING, self._on_translation_pending)
        d.unsubscribe(EventType.TRANSLATION_CONFIRMED, self._on_translation_confirmed)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_stt_pending(self, event: TextEvent) -> None:
        text = (event.text or "").strip()
        if not text:
            return  # ignore blanks; let the clear timeout handle removal
        with self._lock:
            self._source_text = text
            self._source_updated_at = time.time()

    def _on_stt_confirmed(self, event: TextEvent) -> None:
        text = (event.text or "").strip()
        if not text:
            return
        with self._lock:
            self._source_text = text
            self._source_updated_at = time.time()

    def _on_translation_pending(self, event: TranslationEvent) -> None:
        text = (event.translated_text or "").strip()
        if not text:
            return
        with self._lock:
            self._target_text = text
            self._target_updated_at = time.time()

    def _on_translation_confirmed(self, event: TranslationEvent) -> None:
        text = (event.translated_text or "").strip()
        if not text:
            return
        with self._lock:
            self._target_text = text
            self._target_updated_at = time.time()

    # ------------------------------------------------------------------
    # Text getters (with auto-clear)
    # ------------------------------------------------------------------

    def _get_source_text(self) -> str:
        with self._lock:
            if not self._cfg.get("obs.enable_source_subtitle", True):
                return ""
            if self._source_text and self._source_updated_at > 0:
                if time.time() - self._source_updated_at > self._clear_timeout:
                    return ""
            return self._source_text

    def _get_target_text(self) -> str:
        with self._lock:
            if not self._cfg.get("obs.enable_target_subtitle", True):
                return ""
            if self._target_text and self._target_updated_at > 0:
                if time.time() - self._target_updated_at > self._clear_timeout:
                    return ""
            return self._target_text

    # ------------------------------------------------------------------
    # Dynamic CSS
    # ------------------------------------------------------------------

    def _build_dynamic_css(self) -> str:
        """Generate CSS from current OBS tab settings."""
        cfg = self._cfg
        src_font = cfg.get("obs.font_family_source", "Noto Sans SC")
        tgt_font = cfg.get("obs.font_family_target", "Noto Sans")
        src_size = int(cfg.get("obs.font_size_source", 28))
        tgt_size = int(cfg.get("obs.font_size_target", 28))
        src_color = cfg.get("obs.color_source", "#FFFFFF")
        tgt_color = cfg.get("obs.color_target", "#FFE066")

        src_enabled = cfg.get("obs.enable_source_subtitle", True)
        tgt_enabled = cfg.get("obs.enable_target_subtitle", True)

        src_display = "block" if src_enabled else "none"
        tgt_display = "block" if tgt_enabled else "none"

        # Combined outline (stroke) + drop shadow
        src_shadow = self._combined_shadow_css("source")
        tgt_shadow = self._combined_shadow_css("target")

        return f""":root {{
    --src-font-family: '{src_font}', 'Microsoft YaHei', sans-serif;
    --tgt-font-family: '{tgt_font}', 'Microsoft YaHei', sans-serif;
    --src-font-size: {src_size}px;
    --tgt-font-size: {tgt_size}px;
    --src-color: {src_color};
    --tgt-color: {tgt_color};
}}
.source-text {{
    display: {src_display};
    font-family: var(--src-font-family);
    font-size: var(--src-font-size);
    color: var(--src-color);
    text-shadow: {src_shadow};
    paint-order: stroke fill;
}}
.translation-text {{
    display: {tgt_display};
    font-family: var(--tgt-font-family);
    font-size: var(--tgt-font-size);
    color: var(--tgt-color);
    text-shadow: {tgt_shadow};
    paint-order: stroke fill;
}}
"""

    def _combined_shadow_css(self, which: str) -> str:
        """Build combined outline + drop-shadow text-shadow for *which*.

        Uses multi-directional 0-blur shadows for the outline instead of
        ``-webkit-text-stroke`` so CJK glyphs don't show internal overlaps.
        """
        cfg = self._cfg
        parts: list[str] = []

        # Outline (stroke) — 8-direction shadows at 0 blur
        stroke_w = int(cfg.get(f"obs.stroke_width_{which}", 2))
        stroke_c = cfg.get(f"obs.stroke_color_{which}", "#000000")
        if stroke_w > 0:
            for dx, dy in [
                (-1, -1), (0, -1), (1, -1),
                (-1,  0),          (1,  0),
                (-1,  1), (0,  1), (1,  1),
            ]:
                parts.append(
                    f"{dx * stroke_w}px {dy * stroke_w}px 0 {stroke_c}"
                )

        # Drop shadow
        x = int(cfg.get(f"obs.shadow_x_{which}", 0))
        y = int(cfg.get(f"obs.shadow_y_{which}", 2))
        blur = int(cfg.get(f"obs.shadow_blur_{which}", 4))
        opacity = int(cfg.get(f"obs.shadow_opacity_{which}", 80))
        color = cfg.get(f"obs.shadow_color_{which}", "#000000")
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        a = opacity / 100.0
        parts.append(f"{x}px {y}px {blur}px rgba({r},{g},{b},{a:.2f})")

        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_server(self, host: str, port: int) -> None:
        self._app.run(host=host, port=port, threaded=True, use_reloader=False)
