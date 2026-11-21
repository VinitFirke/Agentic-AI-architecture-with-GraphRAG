import pytest

from thesis.vision.perception import Observation, cell_center, grid_cell, parse_grounding

BOARD = (300.0, 200.0, 600.0, 800.0)  # 300 wide, 600 high


@pytest.mark.parametrize(
    "cyl, expected",
    [
        ((440, 480, 460, 520), "center"),
        ((310, 210, 330, 250), "top_left"),
        ((570, 760, 590, 790), "bottom_right"),
        ((700, 700, 750, 750), None),  # outside the board
    ],
)
def test_grid_cell(cyl, expected):
    assert grid_cell(cyl, BOARD) == expected


def test_cell_center_is_inverse_of_grid_cell():
    cx, cy = cell_center(BOARD, 1, 1)
    assert grid_cell((cx - 1, cy - 1, cx + 1, cy + 1), BOARD) == "center"


def test_parse_grounding_handles_fences_and_label_variants():
    text = (
        '```json\n[{"bbox_2d": [1, 2, 3, 4], "label": "green pegboard"},'
        ' {"bbox_2d": [5, 6, 7, 8], "label": "yellow cylinder"}]\n```'
    )
    assert parse_grounding(text) == {
        "pegboard": (1.0, 2.0, 3.0, 4.0),
        "yellow_cylinder": (5.0, 6.0, 7.0, 8.0),
    }
    assert parse_grounding("```json\n[]\n```") == {}
    assert parse_grounding("<think> still thinking") is None  # unusable reply


def _obs(**kw):
    base = dict(
        pegboard_visible=True,
        cylinder_visible=True,
        vlm_cylinder_state="on_pegboard",
        vlm_grid_position="center",
        pegboard_box=BOARD,
        cylinder_box=(440, 480, 460, 520),
        cylinder_position="center",
    )
    base.update(kw)
    return Observation(**base)


def test_triples_for_placed_cylinder():
    preds = {(t["subject"], t["predicate"], t["object"]) for t in _obs().triples()}
    assert preds == {
        ("GreenPegboard", "IS_PRESENT", "Workspace"),
        ("YellowCylinder", "IS_PRESENT", "Workspace"),
        ("YellowCylinder", "PLACED_AT", "center"),
    }


def test_triples_for_cylinder_off_board():
    obs = _obs(cylinder_box=(700, 700, 750, 750), cylinder_position=None)
    assert {"subject": "YellowCylinder", "predicate": "OFF_PEGBOARD", "object": "Workspace"} in (
        obs.triples()
    )


def test_no_triples_when_nothing_visible():
    obs = _obs(
        pegboard_visible=False,
        cylinder_visible=False,
        pegboard_box=None,
        cylinder_box=None,
        cylinder_position=None,
    )
    assert obs.triples() == []
