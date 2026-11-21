import json

from thesis.knowledge.graphrag import Decision
from thesis.pipeline.guidance import CycleResult
from thesis.remote.protocol import result_from_json, result_to_json
from thesis.vision.perception import Observation
from thesis.vision.vlm import VLMCall


def _result():
    call = VLMCall("[]", 200.0, 190.0, 1100, 70, 3.0, 150.0, "stop")
    obs = Observation(
        pegboard_visible=True,
        cylinder_visible=True,
        vlm_cylinder_state="on_pegboard",
        vlm_grid_position="top_center",
        pegboard_box=(300.0, 200.0, 680.0, 850.0),
        cylinder_box=(364.0, 275.0, 420.0, 375.0),
        cylinder_position="top_left",
        calls={"presence": call, "grounding": call},
    )
    dec = Decision(
        status="error",
        message="Wrong move: ...",
        current_step={"id": "S3", "order": 3, "instruction": "Pick up ..."},
        steps=[{"id": "S3", "order": 3, "instruction": "Pick up ...", "satisfied": False}],
        expected=("YellowCylinder", "PLACED_AT", "center"),
        matched=False,
        error_id="E3_WRONG_POSITION",
        highlight_part={
            "id": "YellowCylinder",
            "name": "yellow cylinder",
            "vlm_label": "yellow_cylinder",
        },
        target_position={"id": "center", "name": "center", "row": 1, "col": 1},
        observed_position="top_left",
        latency_ms=5.0,
        n_steps=3,
    )
    return CycleResult(7, None, obs, dec, 400.0, 5.0, t_capture=1.0, t_ready=1.4)


def test_round_trip_keeps_what_the_dashboard_needs():
    data = json.loads(json.dumps(result_to_json(_result())))  # must survive real JSON
    r = result_from_json(data, t_capture=42.0)
    assert r.cycle == 7 and r.t_capture == 42.0
    assert r.observation.cylinder_box == (364.0, 275.0, 420.0, 375.0)
    assert r.observation.calls["presence"].tokens_per_s == 70 / 0.15
    assert r.decision.status == "error"
    assert r.decision.target_position["row"] == 1
    assert r.decision.highlight_part["vlm_label"] == "yellow_cylinder"
    assert "YellowCylinder" in json.dumps(data["observation"]["triples"])
