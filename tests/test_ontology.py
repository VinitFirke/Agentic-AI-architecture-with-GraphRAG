from thesis.knowledge import ontology as onto


def test_manual_has_three_ordered_steps():
    assert [s.order for s in onto.STEPS] == [1, 2, 3]
    assert onto.STEPS[0].instruction == "Check for green pegboard."
    assert onto.STEPS[1].instruction == "Check for yellow cylinder."
    assert "center of the pegboard" in onto.STEPS[2].instruction


def test_expected_triples_use_observable_predicates_and_known_ids():
    part_ids = {p["id"] for p in onto.PARTS}
    object_ids = part_ids | {p["id"] for p in onto.POSITIONS} | {"Workspace"}
    for step in onto.STEPS:
        subj, pred, obj = step.expects
        assert subj in part_ids
        assert pred in onto.OBSERVABLE_PREDICATES
        assert obj in object_ids


def test_goal_is_cylinder_at_center():
    assert onto.GOAL == ("YellowCylinder", "PLACED_AT", "center")
    center = next(p for p in onto.POSITIONS if p["id"] == "center")
    assert (center["row"], center["col"]) == (1, 1)


def test_relation_types_cover_all_manual_triples():
    for _, pred, _ in onto.manual_triples():
        assert pred in onto.RELATION_TYPES
