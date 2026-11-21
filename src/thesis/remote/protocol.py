"""JSON format exchanged between the pipeline server and the laptop client.

The server turns a ``CycleResult`` into JSON. The client turns it back into a lightweight
object with the same attribute names, so the OpenCV dashboard can draw it without importing
LangChain or Neo4j.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from thesis.pipeline.guidance import CycleResult

DEFAULT_PORT = 8765


def result_to_json(result: CycleResult) -> dict[str, Any]:
    """Serialise a server-side cycle result (everything the dashboard and log need)."""
    obs, dec = result.observation, result.decision
    return {
        "cycle": result.cycle,
        "perception_ms": result.perception_ms,
        "graphrag_ms": result.graphrag_ms,
        "observation": {
            "pegboard_visible": obs.pegboard_visible,
            "cylinder_visible": obs.cylinder_visible,
            "vlm_cylinder_state": obs.vlm_cylinder_state,
            "vlm_grid_position": obs.vlm_grid_position,
            "pegboard_box": obs.pegboard_box,
            "cylinder_box": obs.cylinder_box,
            "cylinder_position": obs.cylinder_position,
            "triples": obs.triples(),
            "text": obs.as_text(),
            "calls": {
                name: {
                    "wall_ms": c.wall_ms,
                    "server_ms": c.server_ms,
                    "output_tokens": c.output_tokens,
                    "tokens_per_s": c.tokens_per_s,
                    "done_reason": c.done_reason,
                }
                for name, c in obs.calls.items()
            },
        },
        "decision": {
            "status": dec.status,
            "message": dec.message,
            "current_step": dec.current_step,
            "steps": dec.steps,
            "expected": dec.expected,
            "matched": dec.matched,
            "error_id": dec.error_id,
            "highlight_part": dec.highlight_part,
            "target_position": dec.target_position,
            "observed_position": dec.observed_position,
            "latency_ms": dec.latency_ms,
            "n_steps": dec.n_steps,
            "text": dec.as_text(),
        },
    }


def result_from_json(data: dict[str, Any], t_capture: float) -> SimpleNamespace:
    """Rebuild a result on the client. ``t_capture`` is the client's perf_counter at capture."""
    o, d = data["observation"], data["decision"]
    observation = SimpleNamespace(
        **{k: v for k, v in o.items() if k != "calls"},
        calls={name: SimpleNamespace(**c) for name, c in o["calls"].items()},
    )
    for key in ("pegboard_box", "cylinder_box"):
        box = getattr(observation, key)
        setattr(observation, key, tuple(box) if box else None)
    return SimpleNamespace(
        cycle=data["cycle"],
        perception_ms=data["perception_ms"],
        graphrag_ms=data["graphrag_ms"],
        observation=observation,
        decision=SimpleNamespace(**d),
        t_capture=t_capture,
        end_to_end_ms=0.0,
    )
