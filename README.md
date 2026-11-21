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
scripts/                       start_neo4j.sh, build_graph.py, serve_pipeline.py, run_client.py,
                               run_dashboard.py, ask_graph.py, make_test_frames.py
src/thesis/remote/             HTTP server + laptop client for the remote camera setup
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

## Run with the camera on the laptop (processing on the server)

The camera and the OpenCV dashboard run on the laptop. The VLM, GraphRAG, Neo4j and the
communication log run on the server. The laptop sends one frame every `interval_s` seconds
and gets back the guidance and the boxes to draw.

```
Laptop: D435 → run_client.py ──JPEG──▶ [SSH tunnel :8765] ──▶ Server: serve_pipeline.py
        OpenCV dashboard  ◀──JSON (status, message, boxes, latencies)──┘   VLM + GraphRAG + log
```

**On the server** (e.g. in `tmux`, so it keeps running when you disconnect):
```bash
cd ~/thesis && source myenv/bin/activate
curl -s localhost:11434/api/tags > /dev/null || (OLLAMA_MODELS=~/hf-cache ~/.local/bin/ollama serve > /dev/null 2>&1 &)
./scripts/start_neo4j.sh
python scripts/serve_pipeline.py
```

**On the laptop, once:** get the code (push the branch and clone it, or copy `src/`,
`scripts/`, `configs/`, `pyproject.toml` and `requirements-client.txt`), then:
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-client.txt
pip install --no-deps -e .
```
The laptop needs no `.env`, model, Neo4j or reference image: the client downloads the
reference image from the server.

**On the laptop, every session:**
```bash
ssh -N -L 8765:localhost:8765 firke@134.169.80.170   # terminal 1: keep open
python scripts/run_client.py                          # terminal 2: RealSense D435
python scripts/run_client.py --source webcam:0        # or any webcam
```
Press `q` or `ESC` to quit. The communication log is written on the server in
`logs/<timestamp>/`. The worker rows include the network round trip measured on the laptop.

### Everything on one machine

`scripts/run_dashboard.py` runs the camera, pipeline and dashboard in one process. Use it
when the camera is attached to the machine that runs the models, or with `--headless` to
test on the server (dashboard frames are saved to the log folder).

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
