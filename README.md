# Agentic AI architecture with GraphRAG

Guidance for a manual assembly task. A vision-language model (`qwen3-vl:2b`) acts as the **eyes**, and GraphRAG over a Neo4j knowledge graph of the worker manual acts as the **brain**. An OpenCV dashboard shows the expected assembly next to the live Intel RealSense D435 view, with guidance, error and success messages, and markings for the part to pick and the position to place it.

```
RealSense D435 ──frame──▶ VLM (qwen3-vl:2b, Ollama)  ──observed triples──▶  GraphRAG (Neo4j)
                           presence + grounding                           match against the manual's
                                  ▲                                       expected triples
                                  └───────── step, match/no match, guidance/error/success ◀──┘
                                                     │
                                 Dashboard (OpenCV) ◀┘  + communication log with latencies
```

The VLM never writes recommendations itself. All instruction, error and success text is stored in the knowledge graph and retrieved by GraphRAG.

## Task (worker manual)

1. Check for green pegboard.
2. Check for yellow cylinder.
3. Pick up the yellow cylinder and place on the center of the pegboard.

The expected result is `data/simple_assembly.jpeg`.

## Ontology

| Step | Expected triple (must be observed) | Error condition |
|---|---|---|
| 1 | `(GreenPegboard, IS_PRESENT, Workspace)` | |
| 2 | `(YellowCylinder, IS_PRESENT, Workspace)` | |
| 3 | `(YellowCylinder, PLACED_AT, center)` | `E3_WRONG_POSITION`: `PLACED_AT` any other cell |

Classes: `Task`, `Step`, `Action`, `Part`, `Position` (3x3 pegboard grid), `Workspace`, `ExpectedRelation` (reified triple), `ErrorCondition`. The full schema is in `src/thesis/knowledge/ontology.py`.

## Layout

```
configs/assembly.yaml          model, camera, interval and dashboard settings
src/thesis/knowledge/          ontology, Neo4j graph builder, GraphRAG (verify + NL->Cypher->NL)
src/thesis/vision/             VLM client (LangChain + Ollama), perception, camera sources
src/thesis/pipeline/           guidance cycle, background loop, communication log
src/thesis/ui/dashboard.py     OpenCV dashboard
scripts/                       start_neo4j.sh, build_graph.py, run_dashboard.py, ask_graph.py,
                               make_test_frames.py
notebooks/01_graphrag_vlm_assembly.ipynb   walkthrough with measurements
tests/                         unit tests (+ integration tests that need Neo4j)
```

## Setup on the server (Ollama + Neo4j)

```bash
source myenv/bin/activate
uv pip install -r requirements.txt && uv pip install -e .
cp .env.example .env              # then set NEO4J_PASSWORD
./scripts/start_neo4j.sh          # Neo4j 5 + APOC in Docker, bound to 127.0.0.1
python scripts/build_graph.py     # write the ontology and manual into Neo4j
python scripts/make_test_frames.py
pytest                            # unit + integration tests
```

Ollama must serve `qwen3-vl:2b` on port 11434.

## Run the dashboard on the laptop (camera attached)

1. Get the code and the reference image: `data/` is git-ignored, so copy `data/simple_assembly.jpeg` to the laptop.
2. Create a Python 3.11 env, then run `pip install -r requirements.txt && pip install -e .`.
   `pyrealsense2` has pip wheels for Linux and Windows x86_64.
3. Create `.env` with the same `NEO4J_PASSWORD` as on the server.
4. Open an SSH tunnel to the server's Ollama and Neo4j, and keep it running:
   ```bash
   ssh -N -L 11434:localhost:11434 -L 7687:localhost:7687 <user>@<server>
   ```
5. Start the dashboard:
   ```bash
   python scripts/run_dashboard.py                      # RealSense D435
   python scripts/run_dashboard.py --source webcam:0    # any webcam
   python scripts/run_dashboard.py --source data/demo_sequence   # recorded frames
   ```
   Press `q` or `ESC` to quit. On a machine without a display, add `--headless` to save the dashboard frames to the log folder instead.

Ask the graph questions in natural language:
```bash
python scripts/ask_graph.py "Where must the yellow cylinder be placed?"
```

## Communication log

Each run writes `logs/<timestamp>/communication.md` (a Markdown table) and `communication.csv`. There is one row per message, `VLM → GraphRAG`, `GraphRAG → VLM` and `VLM → Worker`, with the hop latency and the end-to-end latency since frame capture. A latency and VLM throughput summary is appended when the session ends.

## Measured on the server (H100, synthetic test frames)

| | |
|---|---|
| Perception (presence + grounding) | about 0.35 s per frame |
| GraphRAG `verify` (Cypher) | about 6 ms |
| End-to-end, frame to worker message | about 0.4 s |
| VLM generation speed | about 460 output tokens/s, about 5.6 requests/s |
| Presence / grid cell correct | 7/7 and 7/7 test frames |
| NL → Cypher retrieval correct | 9/10 held-out questions |

These are measurements on synthetic frames. Accuracy on real D435 frames still has to be measured: see the limitations at the end of the notebook.
