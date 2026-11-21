"""Needs the Neo4j container (scripts/start_neo4j.sh) with the graph built."""

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def graphrag():
    from thesis.config import load_settings
    from thesis.knowledge.graph_store import connect
    from thesis.knowledge.graphrag import AssemblyGraphRAG

    try:
        st = load_settings()
        driver = connect(st)
    except Exception as exc:  # no .env or Neo4j not running
        pytest.skip(f"Neo4j not available: {exc}")
    yield AssemblyGraphRAG(driver, st)
    driver.close()


def T(s, p, o):
    return {"subject": s, "predicate": p, "object": o}


BOARD = T("GreenPegboard", "IS_PRESENT", "Workspace")
CYL = T("YellowCylinder", "IS_PRESENT", "Workspace")


@pytest.mark.parametrize(
    "observed, status, step",
    [
        ([], "in_progress", 1),
        ([BOARD], "in_progress", 2),
        ([BOARD, CYL, T("YellowCylinder", "OFF_PEGBOARD", "Workspace")], "in_progress", 3),
        ([BOARD, CYL, T("YellowCylinder", "PLACED_AT", "top_left")], "error", 3),
        ([BOARD, CYL, T("YellowCylinder", "PLACED_AT", "center")], "success", None),
    ],
)
def test_verify(graphrag, observed, status, step):
    d = graphrag.verify(observed)
    assert d.status == status
    assert (d.current_step["order"] if d.current_step else None) == step


def test_error_message_comes_from_graph(graphrag):
    d = graphrag.verify([BOARD, CYL, T("YellowCylinder", "PLACED_AT", "top_left")])
    assert d.error_id == "E3_WRONG_POSITION"
    assert "top left" in d.message and "center" in d.message
    assert d.target_position["id"] == "center"
