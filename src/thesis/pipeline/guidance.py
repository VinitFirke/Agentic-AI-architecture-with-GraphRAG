"""One guidance cycle: frame -> VLM (eyes) -> GraphRAG (brain) -> worker message.

The worker message is logged by ``log_delivery`` when the dashboard has actually shown it,
so the VLM -> Worker latency includes rendering.

``GuidanceLoop`` runs the cycles on a background thread every ``interval_s`` seconds, so
the dashboard keeps showing the live camera while the VLM is busy.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from thesis.knowledge.graphrag import AssemblyGraphRAG, Decision
from thesis.pipeline.comm_log import CommLog
from thesis.vision.perception import Observation, Perceiver

log = logging.getLogger(__name__)


@dataclass
class CycleResult:
    cycle: int
    frame: np.ndarray  # the frame the VLM analysed (markings belong to this frame)
    observation: Observation
    decision: Decision
    perception_ms: float
    graphrag_ms: float
    t_capture: float  # perf_counter when the frame was read
    t_ready: float  # perf_counter when GraphRAG answered
    end_to_end_ms: float = 0.0  # frame capture -> message on screen (set on delivery)


class GuidanceSystem:
    """Connect the perceiver (VLM) and GraphRAG, and log every message."""

    def __init__(self, perceiver: Perceiver, graphrag: AssemblyGraphRAG, comm_log: CommLog):
        self.perceiver = perceiver
        self.graphrag = graphrag
        self.log = comm_log

    def warm_up(self) -> None:
        """Load the VLM into memory and let Neo4j plan the query before timing starts."""
        self.graphrag.verify([])
        self.perceiver.perceive(np.full((64, 64, 3), 127, np.uint8))

    def run_cycle(self, frame: np.ndarray, cycle: int, t_capture: float) -> CycleResult:
        """Analyse one frame. ``t_capture`` is time.perf_counter() when the frame was read."""
        t0 = time.perf_counter()
        obs = self.perceiver.perceive(frame)
        perception_ms = (time.perf_counter() - t0) * 1000
        for call in obs.calls.values():
            self.log.add_vlm_stats(call.wall_ms, call.output_tokens, call.tokens_per_s)

        triples = obs.triples()
        triple_txt = ", ".join(
            f"({t['subject']}, {t['predicate']}, {t['object']})" for t in triples
        )
        call_txt = "; ".join(
            f"{name}: {c.wall_ms:.0f} ms, {c.output_tokens} tok @ {c.tokens_per_s:.0f} tok/s"
            for name, c in obs.calls.items()
        )
        self.log.add(
            cycle,
            "VLM",
            "GraphRAG",
            f"{obs.as_text()} Triples: [{triple_txt or 'none'}]",
            perception_ms,
            (time.perf_counter() - t_capture) * 1000,
            f"{call_txt}; VLM's own grid answer: {obs.vlm_grid_position}",
        )

        decision = self.graphrag.verify(triples)
        self.log.add(
            cycle,
            "GraphRAG",
            "VLM",
            decision.as_text(),
            decision.latency_ms,
            (time.perf_counter() - t_capture) * 1000,
            f"Cypher verify; status={decision.status}",
        )

        return CycleResult(
            cycle,
            frame,
            obs,
            decision,
            perception_ms,
            decision.latency_ms,
            t_capture=t_capture,
            t_ready=time.perf_counter(),
        )

    def log_delivery(self, result: CycleResult, t_shown: float) -> None:
        """Log the worker message once it is actually on screen (called by the dashboard)."""
        result.end_to_end_ms = (t_shown - result.t_capture) * 1000
        self.log_worker_message(result, (t_shown - result.t_ready) * 1000, result.end_to_end_ms)

    def log_worker_message(
        self, result: CycleResult, hop_ms: float, end_to_end_ms: float, extra: str = ""
    ) -> None:
        """Write the VLM -> Worker row. The remote client passes latencies it measured itself."""
        d = result.decision
        marks = []
        if d.highlight_part:
            marks.append(f"mark part {d.highlight_part['id']}")
        if d.target_position and d.status != "success":
            marks.append(f"mark target {d.target_position['id']}")
        if extra:
            marks.append(extra)
        self.log.add(
            result.cycle,
            "VLM",
            "Worker",
            f"[{d.status.upper()}] {d.message}",
            hop_ms,
            end_to_end_ms,
            ", ".join(marks),
        )


class GuidanceLoop:
    """Run GuidanceSystem.run_cycle on a background thread every interval_s seconds."""

    def __init__(self, system: GuidanceSystem, interval_s: float):
        self.system = system
        self.interval_s = interval_s
        self._frame: tuple[np.ndarray, float] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.latest: CycleResult | None = None
        self.cycles = 0
        self.busy = False
        self.error: str | None = None
        self._thread = threading.Thread(target=self._run, name="guidance", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        """Offer the newest camera frame (called by the dashboard on every frame)."""
        with self._lock:
            self._frame = (frame, time.perf_counter())

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=30)

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
            self.busy = True
            try:
                self.cycles += 1
                self.latest = self.system.run_cycle(item[0], self.cycles, item[1])
                self.error = None
            except Exception as exc:  # keep the dashboard alive if Ollama/Neo4j hiccup
                log.exception("Guidance cycle failed")
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                self.busy = False
