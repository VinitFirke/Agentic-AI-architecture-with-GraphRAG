"""The VLM as the "eyes": turn a camera frame into observed triples for GraphRAG.

Each frame takes two VLM calls (design measured on data/test_frames, see the notebook):

1. **Presence** (JSON-schema output): is the green pegboard / yellow cylinder visible?
   This was correct on 7/7 test frames.
2. **Grounding** (Qwen ``bbox_2d`` output, 0-1000 coordinates): where are they?
   The boxes are precise, but the model also returns boxes for objects that are absent,
   so a box is only used when step 1 says the object is visible.

The grid cell of the cylinder is computed from the boxes, because the VLM's own grid
answer was wrong on 2/7 test frames while the boxes were right.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np

from thesis.knowledge.ontology import GRID
from thesis.vision.vlm import VLMCall, VLMClient, encode_image

GRID_IDS = [pid for row in GRID for pid in row]

SYSTEM_PROMPT = (
    "You are the eyes of an assembly assistant. The camera is egocentric (the worker's own "
    "view). Only consider the GREEN PEGBOARD (a square board with a 3x3 grid of positions) "
    "and the YELLOW CYLINDER. Ignore every other object, colour, tool or background."
)

PRESENCE_PROMPT = (
    "Look at the image and report:\n"
    "- pegboard_visible: is a green pegboard visible?\n"
    "- cylinder_visible: is a yellow cylinder visible?\n"
    "- cylinder_state: on_pegboard, in_hand, off_pegboard (visible but not on the board) "
    "or not_visible\n"
    "- cylinder_grid_position: if on the pegboard, which of the 9 grid cells it occupies ("
    + ", ".join(GRID_IDS)
    + "), otherwise none"
)

PRESENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "pegboard_visible": {"type": "boolean"},
        "cylinder_visible": {"type": "boolean"},
        "cylinder_state": {
            "type": "string",
            "enum": ["on_pegboard", "in_hand", "off_pegboard", "not_visible"],
        },
        "cylinder_grid_position": {"type": "string", "enum": [*GRID_IDS, "none"]},
    },
    "required": [
        "pegboard_visible",
        "cylinder_visible",
        "cylinder_state",
        "cylinder_grid_position",
    ],
}

GROUNDING_PROMPT = (
    "Locate the green pegboard and the yellow cylinder. Output JSON: "
    '[{"bbox_2d": [x1, y1, x2, y2], "label": "pegboard"}, '
    '{"bbox_2d": [...], "label": "yellow_cylinder"}]. Omit objects that are not visible.'
)

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in 0-1000 image coordinates


@dataclass
class Observation:
    """What the VLM saw in one frame."""

    pegboard_visible: bool
    cylinder_visible: bool
    vlm_cylinder_state: str  # VLM's own answer (logged for comparison)
    vlm_grid_position: str  # VLM's own answer (logged for comparison)
    pegboard_box: Box | None
    cylinder_box: Box | None
    cylinder_position: str | None  # grid cell from geometry, None if not on the board
    calls: dict[str, VLMCall] = field(default_factory=dict)

    def triples(self) -> list[dict[str, str]]:
        """Observed triples in the ontology vocabulary, sent to GraphRAG."""
        t: list[tuple[str, str, str]] = []
        if self.pegboard_visible:
            t.append(("GreenPegboard", "IS_PRESENT", "Workspace"))
        if self.cylinder_visible:
            t.append(("YellowCylinder", "IS_PRESENT", "Workspace"))
            if self.cylinder_position:
                t.append(("YellowCylinder", "PLACED_AT", self.cylinder_position))
            elif self.cylinder_box is not None and self.pegboard_box is not None:
                t.append(("YellowCylinder", "OFF_PEGBOARD", "Workspace"))
        return [{"subject": s, "predicate": p, "object": o} for s, p, o in t]

    def as_text(self) -> str:
        """Natural-language summary (VLM -> GraphRAG message)."""
        parts = [f"Pegboard {'visible' if self.pegboard_visible else 'not visible'}"]
        if not self.cylinder_visible:
            parts.append("yellow cylinder not visible")
        elif self.cylinder_position:
            parts.append(
                f"yellow cylinder on the pegboard at {self.cylinder_position.replace('_', ' ')}"
            )
        elif self.cylinder_box is not None and self.pegboard_box is not None:
            parts.append("yellow cylinder visible but not on the pegboard")
        else:
            parts.append("yellow cylinder visible")
        return "; ".join(parts) + "."


# ---------------------------------------------------------------------- geometry helpers
def grid_cell(cylinder: Box, pegboard: Box) -> str | None:
    """3x3 grid cell of the cylinder centre inside the pegboard box, or None if outside."""
    cx, cy = (cylinder[0] + cylinder[2]) / 2, (cylinder[1] + cylinder[3]) / 2
    x1, y1, x2, y2 = pegboard
    if x2 <= x1 or y2 <= y1:
        return None
    u, v = (cx - x1) / (x2 - x1), (cy - y1) / (y2 - y1)
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return None
    return GRID[min(int(v * 3), 2)][min(int(u * 3), 2)]


def cell_center(pegboard: Box, row: int, col: int) -> tuple[float, float]:
    """Centre of a grid cell in 0-1000 image coordinates."""
    x1, y1, x2, y2 = pegboard
    return x1 + (col + 0.5) * (x2 - x1) / 3, y1 + (row + 0.5) * (y2 - y1) / 3


def to_pixels(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    """Convert a 0-1000 box to pixel coordinates."""
    return (
        int(box[0] * width / 1000),
        int(box[1] * height / 1000),
        int(box[2] * width / 1000),
        int(box[3] * height / 1000),
    )


def parse_json(text: str):
    """Parse JSON from a model reply, tolerating ```json fences."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return json.loads(m.group(1) if m else text)


class PerceptionError(RuntimeError):
    """The VLM reply could not be used (e.g. it ran out of tokens while thinking)."""


def parse_grounding(text: str) -> dict[str, Box] | None:
    """Map 'pegboard' / 'yellow_cylinder' to their boxes from a bbox_2d reply.

    Returns None if the reply is not JSON at all (unusable), {} if it lists no boxes.
    """
    try:
        items = parse_json(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(items, dict):
        items = [items]
    boxes: dict[str, Box] = {}
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        label = str(it.get("label", "")).lower()
        bb = it.get("bbox_2d")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        box = tuple(float(v) for v in bb)
        if "cylinder" in label or "yellow" in label:
            boxes.setdefault("yellow_cylinder", box)
        elif "pegboard" in label or "board" in label or "green" in label:
            boxes.setdefault("pegboard", box)
    return boxes


# ---------------------------------------------------------------------------- perceiver
class Perceiver:
    """Run the presence and grounding prompts on a frame and build an Observation."""

    def __init__(self, vlm: VLMClient, max_side: int = 768, jpeg_quality: int = 85):
        self.vlm = vlm
        self.max_side = max_side
        self.jpeg_quality = jpeg_quality
        self._presence_llm = vlm.with_schema(PRESENCE_SCHEMA)

    def perceive(self, frame_bgr: np.ndarray) -> Observation:
        img = encode_image(frame_bgr, self.max_side, self.jpeg_quality)

        presence_call = self.vlm.ask(SYSTEM_PROMPT, PRESENCE_PROMPT, img, self._presence_llm)
        try:
            presence = parse_json(presence_call.text)
            if not isinstance(presence, dict) or "pegboard_visible" not in presence:
                raise ValueError("missing keys")
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            # Never guess: an unusable reply would otherwise send the worker back to step 1.
            raise PerceptionError(
                f"Presence reply unusable (done_reason={presence_call.done_reason}): "
                f"{presence_call.text[:80]!r}"
            ) from exc
        peg = bool(presence.get("pegboard_visible", False))
        cyl = bool(presence.get("cylinder_visible", False))

        calls = {"presence": presence_call}
        boxes: dict[str, Box] = {}
        if peg or cyl:  # nothing to locate otherwise
            grounding_call = self.vlm.ask(SYSTEM_PROMPT, GROUNDING_PROMPT, img)
            calls["grounding"] = grounding_call
            parsed = parse_grounding(grounding_call.text)
            if parsed is None:
                raise PerceptionError(
                    f"Grounding reply unusable (done_reason={grounding_call.done_reason}): "
                    f"{grounding_call.text[:80]!r}"
                )
            boxes = parsed

        peg_box = boxes.get("pegboard") if peg else None
        cyl_box = boxes.get("yellow_cylinder") if cyl else None
        position = grid_cell(cyl_box, peg_box) if (peg_box and cyl_box) else None
        return Observation(
            pegboard_visible=peg,
            cylinder_visible=cyl,
            vlm_cylinder_state=str(presence.get("cylinder_state", "?")),
            vlm_grid_position=str(presence.get("cylinder_grid_position", "?")),
            pegboard_box=peg_box,
            cylinder_box=cyl_box,
            cylinder_position=position,
            calls=calls,
        )
