"""Laptop side: send camera frames to the pipeline server and collect the results.

Only needs OpenCV, NumPy and requests (no LangChain, Neo4j or model on the laptop).
"""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import requests

from thesis.remote.protocol import result_from_json

log = logging.getLogger(__name__)


class RemoteClient:
    """Talk to scripts/serve_pipeline.py. A background thread sends a frame every interval_s."""

    def __init__(
        self,
        server_url: str,
        interval_s: float,
        max_side: int = 768,
        jpeg_quality: int = 90,
        timeout_s: float = 30.0,
    ):
        self.url = server_url.rstrip("/")
        self.interval_s = interval_s
        self.max_side = max_side
        self.jpeg_quality = jpeg_quality
        self.timeout_s = timeout_s
        self.http = requests.Session()
        self._frame: tuple[np.ndarray, float] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="remote-guidance", daemon=True)
        self.latest: SimpleNamespace | None = None
        self.cycles = 0
        self.busy = False
        self.error: str | None = None

    # ---------------------------------------------------------------- session control
    def start_session(self) -> tuple[np.ndarray, SimpleNamespace]:
        """Start a server session. Returns the reference image and its checked result."""
        health = self.http.get(f"{self.url}/health", timeout=10).json()
        log.info("Connected to pipeline server (%s)", health.get("model"))
        sess = self.http.post(f"{self.url}/session", timeout=120).json()
        img = self.http.get(f"{self.url}/reference.jpg", timeout=30).content
        reference = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        ref = result_from_json(sess["reference"], time.perf_counter())
        log.info("Server session %s (log: %s)", sess["session"], sess["log_dir"])
        return reference, ref

    def close_session(self) -> dict:
        try:
            return self.http.post(f"{self.url}/session/close", timeout=30).json()
        except requests.RequestException as exc:
            log.warning("Could not close the server session: %s", exc)
            return {}

    # ------------------------------------------------------------------ frame loop
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.timeout_s + 5)

    def submit(self, frame: np.ndarray) -> None:
        """Offer the newest camera frame (called on every frame)."""
        with self._lock:
            self._frame = (frame, time.perf_counter())

    def report_delivered(self, result: SimpleNamespace, t_shown: float) -> None:
        """Tell the server when the message appeared on screen (for the communication log)."""
        result.end_to_end_ms = (t_shown - result.t_capture) * 1000
        body = {
            "cycle": result.cycle,
            "hop_ms": (t_shown - result.t_response) * 1000,
            "end_to_end_ms": result.end_to_end_ms,
            "network_ms": result.network_ms,
        }
        threading.Thread(target=self._post_delivered, args=(body,), daemon=True).start()

    def _post_delivered(self, body: dict) -> None:
        try:
            self.http.post(f"{self.url}/delivered", json=body, timeout=10)
        except requests.RequestException as exc:
            log.warning("Could not report delivery: %s", exc)

    def _encode(self, frame: np.ndarray) -> bytes:
        h, w = frame.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise ValueError("Could not encode frame")
        return buf.tobytes()

    def _run(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            if time.monotonic() < next_t:
                self._stop.wait(0.02)
                continue
            with self._lock:
                item, self._frame = self._frame, None
            if item is None:
                self._stop.wait(0.02)
                continue
            next_t = time.monotonic() + self.interval_s
            frame, t_capture = item
            self.busy = True
            try:
                body = self._encode(frame)
                t_send = time.perf_counter()
                resp = self.http.post(
                    f"{self.url}/cycle",
                    data=body,
                    headers={"Content-Type": "image/jpeg"},
                    timeout=self.timeout_s,
                )
                t_response = time.perf_counter()
                data = resp.json()
                if resp.status_code != 200:
                    raise RuntimeError(data.get("error", f"HTTP {resp.status_code}"))
                result = result_from_json(data, t_capture)
                result.frame = frame
                result.t_response = t_response
                result.network_ms = max((t_response - t_send) * 1000 - data["server_ms"], 0.0)
                result.upload_kb = len(body) / 1024
                self.latest = result
                self.cycles = result.cycle
                self.error = None
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                self.error = f"server: {exc}"
                log.warning("Cycle failed: %s", exc)
            finally:
                self.busy = False
