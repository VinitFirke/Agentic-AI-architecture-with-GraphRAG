"""GraphRAG over the assembly knowledge graph (the "brain" of the system).

Two ways to query the graph:

* ``verify(observed)`` is used on every cycle. The VLM reports what it sees as triples. A
  fixed, parameterised Cypher query matches them against each step's ExpectedRelation and
  ErrorConditions. The query returns the current step, whether the expected triple
  matches, and the guidance, error or success text. All of that text is stored in the
  graph, so the VLM never writes recommendations itself.
* ``ask(question)`` answers free-form questions with LangChain: the VLM, used as a plain
  LLM, turns the question into Cypher. The query runs read-only on Neo4j, and the VLM
  turns the rows back into a natural-language answer.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama
from neo4j import Driver

from thesis.config import Settings

NO_THINK_PREFILL = "<think>\n\n</think>\n\n"

VERIFY_CYPHER = """
MATCH (t:Task {id: $task_id})-[:HAS_STEP]->(s:Step)-[:EXPECTS]->(e:ExpectedRelation)
WITH t, s, e,
     any(o IN $observed WHERE o.subject = e.subject AND o.predicate = e.predicate
                          AND o.object = e.object) AS satisfied
ORDER BY s.order
WITH t, collect({id: s.id, order: s.order, instruction: s.instruction, guidance: s.guidance,
                 expected: [e.subject, e.predicate, e.object], satisfied: satisfied}) AS steps
WITH t, steps, [x IN steps WHERE NOT x.satisfied] AS pending
WITH t, steps, CASE WHEN size(pending) = 0 THEN null ELSE pending[0] END AS current
OPTIONAL MATCH (:Step {id: current.id})-[:HAS_ERROR]->(err:ErrorCondition)
  WHERE any(o IN $observed WHERE o.subject = err.subject AND o.predicate = err.predicate
                             AND o.object <> err.object_not)
WITH t, steps, current, collect(err {.id, .message, .subject, .predicate}) AS errors
OPTIONAL MATCH (:Step {id: current.id})-[:ACTS_ON]->(part:Part)
OPTIONAL MATCH (:Step {id: current.id})-[:TARGETS]->(pos:Position)
OPTIONAL MATCH (t)-[:HAS_GOAL]->(:ExpectedRelation)-[:OBJECT]->(goal:Position)
RETURN t.success_message AS success_message, steps, current, errors,
       part {.id, .name, .vlm_label} AS part,
       pos {.id, .name, .row, .col} AS target,
       goal {.id, .name, .row, .col} AS goal
"""


@dataclass
class Decision:
    """GraphRAG's answer for one set of observed triples."""

    status: str  # "in_progress" | "error" | "success"
    message: str  # text shown to the worker (always from the graph)
    current_step: dict[str, Any] | None
    steps: list[dict[str, Any]]
    expected: tuple[str, str, str] | None
    matched: bool  # does the observation contain the current step's expected triple?
    error_id: str | None = None
    highlight_part: dict[str, Any] | None = None  # part to mark (has vlm_label)
    target_position: dict[str, Any] | None = None  # position to mark (row/col)
    observed_position: str | None = None  # where the part actually is (error case)
    latency_ms: float = 0.0
    n_steps: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    def as_text(self) -> str:
        """Natural-language summary of the decision (GraphRAG -> VLM message)."""
        if self.status == "success":
            return f"All {self.n_steps} expected triples observed. Goal state reached."
        s = self.current_step
        exp = "(" + ", ".join(self.expected) + ")" if self.expected else "-"
        text = (
            f"Current step {s['order']}/{self.n_steps}: '{s['instruction']}' "
            f"Expected {exp}: {'MATCH' if self.matched else 'NO MATCH'}."
        )
        if self.error_id:
            text += f" Error {self.error_id} (observed at {self.observed_position})."
        return text


class AssemblyGraphRAG:
    """Query the assembly knowledge graph in Neo4j."""

    def __init__(self, driver: Driver, settings: Settings):
        self.driver = driver
        self.settings = settings
        self.database = settings["graphrag"]["database"]
        self.task_id = settings["task_id"]
        self._qa_chain = None

    # ---------------------------------------------------------------- per-cycle verify
    def verify(self, observed: list[dict[str, str]]) -> Decision:
        """Match observed triples [{subject, predicate, object}, ...] against the manual."""
        t0 = time.perf_counter()
        with self.driver.session(database=self.database) as s:
            rec = s.run(VERIFY_CYPHER, task_id=self.task_id, observed=observed).single()
        latency = (time.perf_counter() - t0) * 1000
        if rec is None:
            raise RuntimeError(f"Task '{self.task_id}' not found. Run scripts/build_graph.py.")

        steps, current, errors = rec["steps"], rec["current"], rec["errors"]
        if current is None:
            return Decision(
                status="success",
                message=rec["success_message"],
                current_step=None,
                steps=steps,
                expected=None,
                matched=True,
                target_position=rec["goal"],
                latency_ms=latency,
                n_steps=len(steps),
                raw=dict(rec),
            )

        expected = tuple(current["expected"])
        decision = Decision(
            status="in_progress",
            message=current["guidance"],
            current_step=current,
            steps=steps,
            expected=expected,
            matched=False,
            highlight_part=rec["part"],
            target_position=rec["target"],
            latency_ms=latency,
            n_steps=len(steps),
            raw=dict(rec),
        )
        if errors:
            err = errors[0]
            seen = next(
                (
                    o["object"]
                    for o in observed
                    if o["subject"] == err["subject"] and o["predicate"] == err["predicate"]
                ),
                "?",
            )
            decision.status = "error"
            decision.error_id = err["id"]
            decision.observed_position = seen
            decision.message = err["message"].format(observed=seen.replace("_", " "))
        return decision

    # ---------------------------------------------------------- free-form NL questions
    def _build_qa_chain(self):
        vlm_cfg = self.settings["vlm"]
        llm = ChatOllama(
            model=vlm_cfg["model"],
            base_url=self.settings.ollama_host,
            temperature=0,
            reasoning=False,
            num_predict=160,
        )
        graph = Neo4jGraph(
            url=self.settings.neo4j_uri,
            username=self.settings.neo4j_user,
            password=self.settings.neo4j_password,
            database=self.database,
            enhanced_schema=False,
        )
        prefill = ("ai", NO_THINK_PREFILL) if vlm_cfg.get("disable_thinking", True) else None
        cypher_msgs = [("system", TEXT2CYPHER_SYSTEM), ("human", "Question: {question}")]
        qa_msgs = [("system", QA_SYSTEM), ("human", QA_HUMAN)]
        if prefill:
            cypher_msgs.append(prefill)
            qa_msgs.append(prefill)
        cypher_chain = ChatPromptTemplate.from_messages(cypher_msgs) | llm | StrOutputParser()
        qa_chain = ChatPromptTemplate.from_messages(qa_msgs) | llm | StrOutputParser()
        return graph, cypher_chain, qa_chain

    def ask(self, question: str) -> dict[str, Any]:
        """Answer a natural-language question from the graph (NL -> Cypher -> rows -> NL).

        Returns the question, generated Cypher, rows, answer, and per-stage latency in ms.
        """
        if self._qa_chain is None:
            self._qa_chain = self._build_qa_chain()
        graph, cypher_chain, qa_chain = self._qa_chain
        out: dict[str, Any] = {"question": question}

        t0 = time.perf_counter()
        cypher = extract_cypher(
            cypher_chain.invoke({"question": question, "schema": graph.get_schema})
        )
        out["cypher"] = cypher
        out["t_text2cypher_ms"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        if not is_read_only(cypher):
            out.update(rows=[], answer="Refused: generated Cypher is not read-only.", ok=False)
            out["t_query_ms"] = 0.0
        else:
            try:
                rows = graph.query(cypher)[: self.settings["graphrag"]["text2cypher_top_k"]]
                out.update(rows=rows, ok=True)
            except Exception as exc:  # invalid Cypher from the LLM
                out.update(rows=[], ok=False, error=str(exc).splitlines()[0])
            out["t_query_ms"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        if out["ok"] and out["rows"]:
            out["answer"] = qa_chain.invoke({"question": question, "context": out["rows"]}).strip()
        else:
            out.setdefault("answer", "I don't know: the knowledge graph returned no result.")
        out["t_answer_ms"] = (time.perf_counter() - t2) * 1000
        out["t_total_ms"] = (time.perf_counter() - t0) * 1000
        return out


# ------------------------------------------------------------------------- helpers
_WRITE_CLAUSES = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|FOREACH|CALL\s+\{|CALL\s+dbms|CALL\s+apoc)\b",
    re.IGNORECASE,
)


def is_read_only(cypher: str) -> bool:
    """True if the query contains no write or admin clauses."""
    return bool(cypher.strip()) and not _WRITE_CLAUSES.search(cypher)


def extract_cypher(text: str) -> str:
    """Pull one Cypher query out of an LLM reply.

    Handles ```cypher fences, drops // comments, and keeps only the first statement
    (small models sometimes repeat the query until they run out of tokens).
    """
    m = re.search(r"```(?:cypher)?\s*(.*?)(?:```|$)", text, re.DOTALL | re.IGNORECASE)
    q = m.group(1) if m else text
    q = "\n".join(line.split("//")[0] for line in q.splitlines())
    q = q.split(";")[0]
    ret = re.search(r"\bRETURN\b", q, re.IGNORECASE)
    if ret:
        nxt = re.search(r"\b(MATCH|OPTIONAL\s+MATCH)\b", q[ret.end() :], re.IGNORECASE)
        if nxt:
            q = q[: ret.end() + nxt.start()]
    return " ".join(q.split())


TEXT2CYPHER_SYSTEM = """You translate questions about an assembly task into Neo4j Cypher.
Use only the node labels, relationship types and properties in this schema:
{schema}

Facts about the data:
- Step nodes have properties: id, order (1, 2, 3), instruction, guidance.
- Part nodes: match them by id, which is 'GreenPegboard' or 'YellowCylinder'
  (their name property is lower case text, e.g. 'yellow cylinder').
- Position ids: 'top_left', 'top_center', 'top_right', 'middle_left', 'center',
  'middle_right', 'bottom_left', 'bottom_center', 'bottom_right'.
- ExpectedRelation nodes have properties: subject, predicate, object.
- ErrorCondition nodes have properties: id, description, message.

Examples:
Question: What is the first step?
MATCH (:Task)-[:FIRST_STEP]->(s:Step) RETURN s.order, s.instruction
Question: What comes after step 2?
MATCH (:Step {{order: 2}})-[:NEXT]->(s:Step) RETURN s.order, s.instruction
Question: Which part does step 3 act on?
MATCH (:Step {{order: 3}})-[:ACTS_ON]->(p:Part) RETURN p.name
Question: Which triple must be observed to complete step 1?
MATCH (:Step {{order: 1}})-[:EXPECTS]->(e:ExpectedRelation) RETURN e.subject, e.predicate, e.object
Question: Which positions does the green pegboard have?
MATCH (:Part {{id: 'GreenPegboard'}})-[:HAS_POSITION]->(p:Position) RETURN p.name
Question: How many actions does step 1 perform?
MATCH (:Step {{order: 1}})-[:PERFORMS]->(a:Action) RETURN count(a) AS actions
Question: What can go wrong in step 3?
MATCH (:Step {{order: 3}})-[:HAS_ERROR]->(e:ErrorCondition) RETURN e.description

Return only one read-only Cypher query, with no explanation."""

QA_SYSTEM = """You answer a worker's question using only the database rows given.
Do not add facts, advice or steps that are not in the rows. Answer in one short sentence."""

QA_HUMAN = "Question: {question}\nDatabase rows: {context}"
