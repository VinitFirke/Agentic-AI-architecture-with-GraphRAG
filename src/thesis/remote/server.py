"""HTTP service that runs the guidance pipeline on the server for a remote camera client.

Endpoints (JSON unless noted):

* ``GET  /health``         model and status
* ``POST /session``        start a new logging session; checks the reference image (cycle 0)
* ``GET  /reference.jpg``  the expected-assembly image (JPEG)
* ``POST /cycle``          body: one JPEG frame -> perception + GraphRAG result
* ``POST /delivered``      the client reports when the message was on screen (for the log)
* ``POST /session/close``  write the latency summary and return it

The service binds to 127.0.0.1 only. The laptop reaches it through an SSH tunnel.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from thesis.config import Settings
from thesis.knowledge.graph_store import connect
from thesis.knowledge.graphrag import AssemblyGraphRAG
from thesis.pipeline.comm_log import CommLog
from thesis.pipeline.guidance import CycleResult, GuidanceSystem
from thesis.remote.protocol import result_to_json
from thesis.vision.perception import Perceiver
from thesis.vision.vlm import VLMClient

log = logging.getLogger(__name__)


class PipelineService:
    """Owns the VLM, GraphRAG and the current logging session. One cycle runs at a time."""

    def __init__(self, settings: Settings):
        self.settings = settings
        vcfg = settings["vlm"]
        self.driver = connect(settings)
        self.graphrag = AssemblyGraphRAG(self.driver, settings)
        vlm = VLMClient(vcfg["model"], settings.ollama_host, vcfg)
        self.perceiver = Perceiver(vlm, vcfg["max_image_side"], vcfg["jpeg_quality"])
        self.reference = cv2.imread(str(settings.path(settings["reference_image"])))
        if self.reference is None:
            raise FileNotFoundError(settings.path(settings["reference_image"]))
        self.lock = threading.Lock()
        self.system: GuidanceSystem | None = None
        self.cycle = 0
        self.recent: OrderedDict[int, CycleResult] = OrderedDict()
        log.info("Warming up VLM and Neo4j...")
        GuidanceSystem(self.perceiver, self.graphrag, _NullLog()).warm_up()

    def start_session(self) -> dict:
        with self.lock:
            if self.system is not None:
                self.system.log.close()
            comm = CommLog(self.settings.path(self.settings["logging"]["dir"]))
            self.system = GuidanceSystem(self.perceiver, self.graphrag, comm)
            self.cycle = 0
            self.recent.clear()
            ref = self.system.run_cycle(self.reference, 0, time.perf_counter())
        log.info("Session %s started, reference check: %s", comm.dir.name, ref.decision.status)
        return {
            "session": comm.dir.name,
            "log_dir": str(comm.dir),
            "reference": result_to_json(ref),
        }

    def run_cycle(self, jpeg: bytes) -> dict:
        t_received = time.perf_counter()
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Body is not a decodable image")
        with self.lock:
            if self.system is None:
                raise RuntimeError("No session. POST /session first.")
            self.cycle += 1
            result = self.system.run_cycle(frame, self.cycle, t_received)
            self.recent[result.cycle] = result
            while len(self.recent) > 50:
                self.recent.popitem(last=False)
        out = result_to_json(result)
        out["server_ms"] = (time.perf_counter() - t_received) * 1000
        return out

    def delivered(self, cycle: int, hop_ms: float, end_to_end_ms: float, network_ms: float) -> None:
        with self.lock:
            result = self.recent.get(cycle)
            if self.system is None or result is None:
                return
            self.system.log_worker_message(
                result, hop_ms, end_to_end_ms, f"network round trip {network_ms:.0f} ms"
            )

    def close_session(self) -> dict:
        with self.lock:
            if self.system is None:
                return {}
            comm = self.system.log
            summary = comm.close()
            self.system = None
        return {"log": str(comm.md_path), "summary": summary}


class _NullLog:
    """Stand-in log for the warm-up call."""

    def add(self, *a, **k) -> None: ...

    def add_vlm_stats(self, *a, **k) -> None: ...


def create_app(service: PipelineService) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(ok=True, model=service.settings["vlm"]["model"], cycle=service.cycle)

    @app.post("/session")
    def session():
        return jsonify(service.start_session())

    @app.get("/reference.jpg")
    def reference():
        ok, buf = cv2.imencode(".jpg", service.reference, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(buf.tobytes(), mimetype="image/jpeg")

    @app.post("/cycle")
    def cycle():
        try:
            return jsonify(service.run_cycle(request.get_data()))
        except (ValueError, RuntimeError) as exc:
            return jsonify(error=str(exc)), 400
        except Exception as exc:  # VLM or Neo4j failure: report it, keep serving
            log.exception("Cycle failed")
            return jsonify(error=f"{type(exc).__name__}: {exc}"), 500

    @app.post("/delivered")
    def delivered():
        d = request.get_json(force=True)
        service.delivered(
            int(d["cycle"]), float(d["hop_ms"]), float(d["end_to_end_ms"]), float(d["network_ms"])
        )
        return jsonify(ok=True)

    @app.post("/session/close")
    def close():
        return jsonify(service.close_session())

    return app
