"""Assembly ontology and the manual instructions encoded as knowledge-graph triples.

The ontology has two layers:

* **Schema (T-Box):** the node classes and relation types, with their domain and range.
* **Instances (A-Box):** the worker manual from docs/prompt_rough.md, written as nodes and
  (subject, predicate, object) triples:

      1. Check for green pegboard.
      2. Check for yellow cylinder.
      3. Pick up the yellow cylinder and place on the center of the pegboard.

Every step has an *ExpectedRelation*: a reified triple the VLM must observe for the step
to count as done (e.g. ``(YellowCylinder, PLACED_AT, Center)``). Wrong moves are
*ErrorConditions* on a step. GraphRAG compares the triples the VLM observes with these
expectations, so all guidance and error/success text comes from the graph, never from the VLM.

The 3x3 grid of pegboard positions is taken from the reference image
(data/simple_assembly.jpeg). The manual itself only names the center position. The other
eight positions exist so that a wrong placement can be reported by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# Schema (T-Box)
# --------------------------------------------------------------------------------------

NODE_CLASSES: dict[str, str] = {
    "Task": "An assembly task the worker must complete.",
    "Step": "One ordered instruction of the task, taken from the worker manual.",
    "Action": "An atomic operation performed in a step (CHECK, PICK, PLACE).",
    "Part": "A physical object involved in the assembly.",
    "Position": "A named location on the pegboard (3x3 grid).",
    "Workspace": "The area in front of the worker seen by the egocentric camera.",
    "ExpectedRelation": "Reified triple (subject, predicate, object) that must be observed "
    "for a step to be complete.",
    "ErrorCondition": "An observed triple pattern that counts as a wrong move in a step.",
}

# relation type -> (domain, range, description)
RELATION_TYPES: dict[str, tuple[str, str, str]] = {
    "HAS_STEP": ("Task", "Step", "The task consists of this step."),
    "FIRST_STEP": ("Task", "Step", "The step the task starts with."),
    "NEXT": ("Step", "Step", "The step that follows."),
    "PERFORMS": ("Step", "Action", "The step performs this action (ordered by 'seq')."),
    "ACTS_ON": ("Step", "Part", "The part the worker handles or checks in this step."),
    "TARGETS": ("Step", "Position", "The position the part must end up in."),
    "EXPECTS": ("Step", "ExpectedRelation", "The triple that must hold when the step is done."),
    "HAS_ERROR": ("Step", "ErrorCondition", "A wrong move that can happen in this step."),
    "HAS_GOAL": ("Task", "ExpectedRelation", "The triple that defines a completed assembly."),
    "SUBJECT": ("ExpectedRelation|ErrorCondition", "Part", "Subject of the reified triple."),
    "OBJECT": (
        "ExpectedRelation",
        "Part|Position|Workspace",
        "Object of the reified triple.",
    ),
    "HAS_POSITION": ("Part", "Position", "The pegboard offers this position."),
    "MUST_BE_PLACED_AT": ("Part", "Position", "Goal placement of a part (from the manual)."),
    "LOCATED_IN": ("Part", "Workspace", "The part is expected in the workspace."),
}

# Predicates the VLM can report in observed triples (subject, predicate, object).
OBSERVABLE_PREDICATES: dict[str, str] = {
    "IS_PRESENT": "Part is visible in the workspace (object = 'Workspace').",
    "PLACED_AT": "Part sits on the pegboard at a Position (object = position id).",
    "OFF_PEGBOARD": "Part is visible but not on the pegboard (object = 'Workspace').",
}

# --------------------------------------------------------------------------------------
# Instances (A-Box) from the worker manual
# --------------------------------------------------------------------------------------

TASK = {
    "id": "peg_assembly",
    "name": "Place the yellow cylinder on the green pegboard",
    "success_message": "Assembly complete: the yellow cylinder is placed on the center "
    "of the green pegboard.",
}

WORKSPACE = {"id": "Workspace", "name": "Worker workspace (egocentric camera view)"}

# vlm_label links a part to the label the VLM uses in its grounding output.
PARTS = [
    {
        "id": "GreenPegboard",
        "name": "green pegboard",
        "color": "green",
        "shape": "square board with a 3x3 grid of positions",
        "role": "base",
        "vlm_label": "pegboard",
    },
    {
        "id": "YellowCylinder",
        "name": "yellow cylinder",
        "color": "yellow",
        "shape": "cylinder",
        "role": "component",
        "vlm_label": "yellow_cylinder",
    },
]

GRID = [
    ["top_left", "top_center", "top_right"],
    ["middle_left", "center", "middle_right"],
    ["bottom_left", "bottom_center", "bottom_right"],
]
POSITIONS = [
    {"id": pid, "row": r, "col": c, "name": pid.replace("_", " ")}
    for r, row in enumerate(GRID)
    for c, pid in enumerate(row)
]

ACTIONS = [
    {"id": "CHECK", "verb": "check"},
    {"id": "PICK", "verb": "pick up"},
    {"id": "PLACE", "verb": "place"},
]


@dataclass(frozen=True)
class StepSpec:
    """One manual step and the triples that define it."""

    id: str
    order: int
    instruction: str  # verbatim from the manual
    guidance: str  # what the worker is told while this step is pending
    actions: tuple[str, ...]
    acts_on: str
    expects: tuple[str, str, str]  # (subject, predicate, object)
    targets: str | None = None
    errors: tuple[dict, ...] = field(default_factory=tuple)


STEPS: tuple[StepSpec, ...] = (
    StepSpec(
        id="S1",
        order=1,
        instruction="Check for green pegboard.",
        guidance="Step 1: Check for the green pegboard. It must be visible in the workspace.",
        actions=("CHECK",),
        acts_on="GreenPegboard",
        expects=("GreenPegboard", "IS_PRESENT", "Workspace"),
    ),
    StepSpec(
        id="S2",
        order=2,
        instruction="Check for yellow cylinder.",
        guidance="Step 2: Check for the yellow cylinder. It must be visible in the workspace.",
        actions=("CHECK",),
        acts_on="YellowCylinder",
        expects=("YellowCylinder", "IS_PRESENT", "Workspace"),
    ),
    StepSpec(
        id="S3",
        order=3,
        instruction="Pick up the yellow cylinder and place on the center of the pegboard.",
        guidance="Step 3: Pick up the yellow cylinder and place it on the center of the "
        "green pegboard.",
        actions=("PICK", "PLACE"),
        acts_on="YellowCylinder",
        targets="center",
        expects=("YellowCylinder", "PLACED_AT", "center"),
        errors=(
            {
                "id": "E3_WRONG_POSITION",
                "subject": "YellowCylinder",
                "predicate": "PLACED_AT",
                "object_not": "center",
                "description": "The yellow cylinder is on the pegboard but not at the center.",
                "message": "Wrong move: the yellow cylinder is at the {observed} position. "
                "Move it to the center of the green pegboard.",
            },
        ),
    ),
)

GOAL = STEPS[-1].expects


def manual_triples() -> list[tuple[str, str, str]]:
    """All A-Box facts as (subject, predicate, object) id triples, for display and tests."""
    triples: list[tuple[str, str, str]] = []
    for i, s in enumerate(STEPS):
        triples.append((TASK["id"], "HAS_STEP", s.id))
        if i == 0:
            triples.append((TASK["id"], "FIRST_STEP", s.id))
        else:
            triples.append((STEPS[i - 1].id, "NEXT", s.id))
        triples += [(s.id, "PERFORMS", a) for a in s.actions]
        triples.append((s.id, "ACTS_ON", s.acts_on))
        if s.targets:
            triples.append((s.id, "TARGETS", s.targets))
        triples.append((s.id, "EXPECTS", "(" + ", ".join(s.expects) + ")"))
        triples += [(s.id, "HAS_ERROR", e["id"]) for e in s.errors]
    triples += [("GreenPegboard", "HAS_POSITION", p["id"]) for p in POSITIONS]
    triples.append((GOAL[0], "MUST_BE_PLACED_AT", GOAL[2]))
    triples.append((TASK["id"], "HAS_GOAL", "(" + ", ".join(GOAL) + ")"))
    triples += [(p["id"], "LOCATED_IN", WORKSPACE["id"]) for p in PARTS]
    return triples
