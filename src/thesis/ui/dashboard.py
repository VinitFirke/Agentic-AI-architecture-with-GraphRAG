"""OpenCV dashboard: expected assembly (left), live camera with markings (right), guidance below.

Markings on the live view:
* thin cyan box: the pegboard the VLM found
* orange box "PICK"/"CHECK": the part GraphRAG says the current step acts on
* green target "PLACE HERE": the position GraphRAG says the part must go to
* red box + arrow: a wrong placement (error), pointing to the correct position
* green box "OK": the part in its goal position (success)
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from thesis.pipeline.guidance import CycleResult
from thesis.vision.perception import Box, cell_center, to_pixels

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (32, 32, 32)
WHITE = (240, 240, 240)
GREY = (150, 150, 150)
CYAN = (230, 200, 40)
ORANGE = (0, 140, 255)
GREEN = (60, 200, 60)
RED = (40, 40, 230)
AMBER = (0, 180, 230)
STATUS_COLORS = {"success": GREEN, "error": RED, "in_progress": AMBER}


def _fit(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * height / h), height), interpolation=cv2.INTER_AREA)


def _label(img, text, org, color, scale=0.6, thickness=2):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    y = max(y, th + 4)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + base - 2), color, -1)
    cv2.putText(img, text, (x + 3, y - 3), FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def _wrap(text: str, width_px: int, scale: float, thickness: int = 1) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cv2.getTextSize(trial, FONT, scale, thickness)[0][0] > width_px and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def _target_px(peg: Box, pos: dict, w: int, h: int) -> tuple[int, int, int]:
    cx, cy = cell_center(peg, pos["row"], pos["col"])
    radius = int(min((peg[2] - peg[0]) * w, (peg[3] - peg[1]) * h) / 1000 / 8)
    return int(cx * w / 1000), int(cy * h / 1000), max(radius, 8)


def draw_markings(frame: np.ndarray, result: CycleResult | None) -> np.ndarray:
    """Draw GraphRAG's guidance onto a copy of the frame (boxes come from the VLM)."""
    img = frame.copy()
    if result is None:
        return img
    h, w = img.shape[:2]
    obs, dec = result.observation, result.decision
    boxes = {"pegboard": obs.pegboard_box, "yellow_cylinder": obs.cylinder_box}

    if obs.pegboard_box:
        x1, y1, x2, y2 = to_pixels(obs.pegboard_box, w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), CYAN, 2)

    target = None
    if dec.target_position and obs.pegboard_box:
        target = _target_px(obs.pegboard_box, dec.target_position, w, h)

    if dec.status == "success":
        if obs.cylinder_box:
            x1, y1, x2, y2 = to_pixels(obs.cylinder_box, w, h)
            cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 4)
            _label(img, "OK", (x1, y1 - 6), GREEN)
        return img

    part = dec.highlight_part
    part_box = boxes.get(part["vlm_label"]) if part else None
    if dec.status == "error" and part_box:
        x1, y1, x2, y2 = to_pixels(part_box, w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), RED, 4)
        _label(img, f"WRONG: {(dec.observed_position or '').replace('_', ' ')}", (x1, y1 - 6), RED)
        if target:
            cv2.arrowedLine(
                img, ((x1 + x2) // 2, (y1 + y2) // 2), target[:2], RED, 3, tipLength=0.08
            )
    elif part_box:
        x1, y1, x2, y2 = to_pixels(part_box, w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), ORANGE, 4)
        verb = "PICK" if dec.target_position else "CHECK"
        _label(img, f"{verb}: {part['name']}", (x1, y1 - 6), ORANGE)
    elif part:
        _label(img, f"LOOKING FOR: {part['name']}", (14, 70), ORANGE, 0.8)

    if target:
        cx, cy, r = target
        cv2.circle(img, (cx, cy), r, GREEN, 3)
        cv2.line(img, (cx - r - 8, cy), (cx + r + 8, cy), GREEN, 2)
        cv2.line(img, (cx, cy - r - 8), (cx, cy + r + 8), GREEN, 2)
        _label(img, "PLACE HERE", (cx + r + 6, cy - r), GREEN)
    return img


def draw_reference(reference: np.ndarray, ref_result: CycleResult | None) -> np.ndarray:
    """Expected assembly with the goal position marked."""
    img = reference.copy()
    if ref_result is None:
        return img
    h, w = img.shape[:2]
    obs, dec = ref_result.observation, ref_result.decision
    if obs.pegboard_box and dec.target_position:
        cx, cy, r = _target_px(obs.pegboard_box, dec.target_position, w, h)
        cv2.circle(img, (cx, cy), int(r * 1.6), GREEN, max(3, w // 300))
    return img


class DashboardRenderer:
    """Compose the full dashboard image."""

    def __init__(self, reference: np.ndarray, view_height: int = 540, panel_height: int = 230):
        self.reference = reference
        self.vh = view_height
        self.ph = panel_height
        self.ref_result: CycleResult | None = None
        self._last_e2e_ms: float | None = None

    def render(
        self,
        live: np.ndarray,
        result: CycleResult | None,
        source_name: str,
        cycles: int,
        busy: bool,
        error: str | None = None,
        fps: float = 0.0,
    ) -> np.ndarray:
        # draw after resizing so line widths and labels have the same size on any camera
        left = draw_reference(_fit(self.reference, self.vh), self.ref_result)
        right = draw_markings(_fit(live, self.vh), result)
        top = np.hstack([left, np.full((self.vh, 6, 3), 80, np.uint8), right])
        width = top.shape[1]
        _label(top, "EXPECTED ASSEMBLY", (8, 26), WHITE)
        _label(top, f"LIVE: {source_name}  {fps:.0f} fps", (left.shape[1] + 14, 26), WHITE)
        if result is not None:
            age = time.perf_counter() - result.t_capture
            _label(
                top,
                f"markings from frame {age:.1f}s ago",
                (left.shape[1] + 14, self.vh - 10),
                GREY,
                0.5,
                1,
            )
        panel = self._panel(width, result, cycles, busy, error, self._last_e2e_ms)
        return np.vstack([top, panel])

    def delivered(self, result: CycleResult) -> None:
        """Remember the end-to-end latency of the last message shown to the worker."""
        self._last_e2e_ms = result.end_to_end_ms

    def _panel(self, width, result, cycles, busy, error, last_e2e_ms) -> np.ndarray:
        p = np.full((self.ph, width, 3), BG, np.uint8)
        if error:
            status, msg = "error", f"System error: {error}"
        elif result is None:
            status, msg = "in_progress", "Waiting for the first VLM inspection..."
        else:
            status, msg = result.decision.status, result.decision.message
        color = STATUS_COLORS[status]
        cv2.rectangle(p, (0, 0), (width, 70), color, -1)
        cv2.putText(p, status.replace("_", " ").upper(), (14, 30), FONT, 0.8, (0, 0, 0), 2)
        for i, line in enumerate(_wrap(msg, width - 240, 0.65, 2)[:2]):
            cv2.putText(p, line, (220, 28 + 28 * i), FONT, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

        # step checklist (from the graph)
        y = 100
        cv2.putText(p, "Manual steps (from knowledge graph)", (14, y), FONT, 0.55, GREY, 1)
        if result is not None:
            cur = result.decision.current_step
            for s in result.decision.steps:
                y += 26
                if s["satisfied"] and (cur is None or s["order"] < cur["order"]):
                    mark, col = "[x]", GREEN
                elif cur and s["id"] == cur["id"]:
                    mark, col = "-->", color
                else:
                    mark, col = "[ ]", GREY
                cv2.putText(
                    p,
                    f"{mark} {s['order']}. {s['instruction']}",
                    (14, y),
                    FONT,
                    0.55,
                    col,
                    1,
                    cv2.LINE_AA,
                )

        # latency / throughput
        x = width // 2 + 40
        y = 100
        cv2.putText(p, "Latency / throughput", (x, y), FONT, 0.55, GREY, 1)
        lines = [f"cycles: {cycles}   VLM: {'busy' if busy else 'idle'}"]
        if result is not None:
            calls = result.observation.calls
            pres, grd = calls.get("presence"), calls.get("grounding")
            lines.append(
                f"VLM perception: {result.perception_ms:.0f} ms"
                + (
                    f" (presence {pres.wall_ms:.0f} + grounding {grd.wall_ms:.0f})"
                    if pres and grd
                    else ""
                )
            )
            tps = [c.tokens_per_s for c in calls.values() if c.tokens_per_s]
            lines.append(
                f"GraphRAG (Cypher): {result.graphrag_ms:.1f} ms   VLM speed: "
                f"{(sum(tps) / len(tps)) if tps else 0:.0f} tok/s"
            )
            if last_e2e_ms is not None:
                lines.append(f"end-to-end (frame -> worker, last): {last_e2e_ms:.0f} ms")
        for line in lines:
            y += 26
            cv2.putText(p, line, (x, y), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
        cv2.putText(p, "q / ESC: quit", (width - 140, self.ph - 10), FONT, 0.45, GREY, 1)
        return p
