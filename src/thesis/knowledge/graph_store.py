"""Write the assembly ontology (schema + manual instructions) into Neo4j."""

from __future__ import annotations

import logging

from neo4j import Driver, GraphDatabase

from thesis.config import Settings
from thesis.knowledge import ontology as onto

log = logging.getLogger(__name__)


def connect(settings: Settings) -> Driver:
    """Open a Neo4j driver and check that the server is reachable."""
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        notifications_min_severity="WARNING",
    )
    driver.verify_connectivity()
    return driver


def _label_of(node_id: str) -> str:
    if node_id == onto.WORKSPACE["id"]:
        return "Workspace"
    if node_id in {p["id"] for p in onto.POSITIONS}:
        return "Position"
    if node_id in {p["id"] for p in onto.PARTS}:
        return "Part"
    raise ValueError(f"Unknown node id in ontology: {node_id}")


def build_graph(driver: Driver, database: str = "neo4j", reset: bool = True) -> dict[str, int]:
    """Create the knowledge graph. With reset=True the database is emptied first.

    Returns the node and relationship counts after the build.
    """
    with driver.session(database=database) as s:
        if reset:
            s.run("MATCH (n) DETACH DELETE n").consume()
        for label in onto.NODE_CLASSES:
            s.run(
                f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
            ).consume()

        s.run("MERGE (t:Task {id: $id}) SET t += $props", id=onto.TASK["id"], props=onto.TASK)
        s.run("MERGE (w:Workspace {id: $id}) SET w += $props", id="Workspace", props=onto.WORKSPACE)
        for a in onto.ACTIONS:
            s.run("MERGE (a:Action {id: $id}) SET a += $props", id=a["id"], props=a)
        for p in onto.PARTS:
            s.run(
                "MERGE (p:Part {id: $id}) SET p += $props "
                "WITH p MATCH (w:Workspace {id: 'Workspace'}) MERGE (p)-[:LOCATED_IN]->(w)",
                id=p["id"],
                props=p,
            )
        for pos in onto.POSITIONS:
            s.run(
                "MERGE (pos:Position {id: $id}) SET pos += $props "
                "WITH pos MATCH (b:Part {id: 'GreenPegboard'}) MERGE (b)-[:HAS_POSITION]->(pos)",
                id=pos["id"],
                props=pos,
            )

        prev = None
        for st in onto.STEPS:
            s.run(
                "MATCH (t:Task {id: $task}) "
                "MERGE (s:Step {id: $id}) "
                "SET s.order = $order, s.instruction = $instruction, s.guidance = $guidance "
                "MERGE (t)-[:HAS_STEP]->(s)",
                task=onto.TASK["id"],
                id=st.id,
                order=st.order,
                instruction=st.instruction,
                guidance=st.guidance,
            )
            if prev is None:
                s.run(
                    "MATCH (t:Task {id: $task}), (s:Step {id: $id}) MERGE (t)-[:FIRST_STEP]->(s)",
                    task=onto.TASK["id"],
                    id=st.id,
                )
            else:
                s.run(
                    "MATCH (a:Step {id: $a}), (b:Step {id: $b}) MERGE (a)-[:NEXT]->(b)",
                    a=prev,
                    b=st.id,
                )
            prev = st.id
            for seq, act in enumerate(st.actions, start=1):
                s.run(
                    "MATCH (s:Step {id: $id}), (a:Action {id: $act}) "
                    "MERGE (s)-[r:PERFORMS]->(a) SET r.seq = $seq",
                    id=st.id,
                    act=act,
                    seq=seq,
                )
            s.run(
                "MATCH (s:Step {id: $id}), (p:Part {id: $part}) MERGE (s)-[:ACTS_ON]->(p)",
                id=st.id,
                part=st.acts_on,
            )
            if st.targets:
                s.run(
                    "MATCH (s:Step {id: $id}), (pos:Position {id: $pos}) "
                    "MERGE (s)-[:TARGETS]->(pos)",
                    id=st.id,
                    pos=st.targets,
                )

            subj, pred, obj = st.expects
            s.run(
                f"MATCH (s:Step {{id: $step}}), (sub:Part {{id: $subj}}), "
                f"(o:{_label_of(obj)} {{id: $obj}}) "
                "MERGE (e:ExpectedRelation {id: $eid}) "
                "SET e.subject = $subj, e.predicate = $pred, e.object = $obj, "
                "    e.description = $desc "
                "MERGE (s)-[:EXPECTS]->(e) MERGE (e)-[:SUBJECT]->(sub) MERGE (e)-[:OBJECT]->(o)",
                step=st.id,
                eid=f"{st.id}_EXPECTS",
                subj=subj,
                pred=pred,
                obj=obj,
                desc=f"({subj})-[:{pred}]->({obj})",
            )
            for err in st.errors:
                s.run(
                    "MATCH (s:Step {id: $step}), (sub:Part {id: $props.subject}) "
                    "MERGE (e:ErrorCondition {id: $props.id}) SET e += $props "
                    "MERGE (s)-[:HAS_ERROR]->(e) MERGE (e)-[:SUBJECT]->(sub)",
                    step=st.id,
                    props=err,
                )

        goal_subj, _, goal_obj = onto.GOAL
        s.run(
            "MATCH (t:Task {id: $task}), (e:ExpectedRelation {id: $eid}) "
            "MERGE (t)-[:HAS_GOAL]->(e)",
            task=onto.TASK["id"],
            eid=f"{onto.STEPS[-1].id}_EXPECTS",
        )
        s.run(
            "MATCH (p:Part {id: $p}), (pos:Position {id: $pos}) "
            "MERGE (p)-[:MUST_BE_PLACED_AT]->(pos)",
            p=goal_subj,
            pos=goal_obj,
        )

        nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    log.info("Knowledge graph built: %d nodes, %d relationships", nodes, rels)
    return {"nodes": nodes, "relationships": rels}
