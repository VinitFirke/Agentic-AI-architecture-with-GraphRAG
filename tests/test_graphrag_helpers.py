from thesis.knowledge.graphrag import extract_cypher, is_read_only


def test_extract_cypher_strips_fences_and_repeats():
    assert extract_cypher("```cypher\nMATCH (n) RETURN n\n```") == "MATCH (n) RETURN n"
    repeated = "MATCH (a) RETURN a.x MATCH (a) RETURN a.x MATCH (b {"
    assert extract_cypher(repeated) == "MATCH (a) RETURN a.x"
    assert extract_cypher("MATCH (n) // note\nRETURN n;") == "MATCH (n) RETURN n"


def test_is_read_only():
    assert is_read_only("MATCH (s:Step) RETURN s.instruction")
    assert not is_read_only("MATCH (n) DETACH DELETE n")
    assert not is_read_only("MERGE (n:Step {id: 'x'})")
    assert not is_read_only("MATCH (n) SET n.x = 1")
    assert not is_read_only("")
