"""Build the assembly knowledge graph in Neo4j from the ontology.

Usage: python scripts/build_graph.py [--no-reset]
"""

import argparse
import logging

from thesis.config import load_settings
from thesis.knowledge import ontology
from thesis.knowledge.graph_store import build_graph, connect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-reset", action="store_true", help="keep existing nodes")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    settings = load_settings()
    with connect(settings) as driver:
        counts = build_graph(driver, settings["graphrag"]["database"], reset=not args.no_reset)
    print(f"Built graph: {counts['nodes']} nodes, {counts['relationships']} relationships")
    print("Manual instructions as triples:")
    for s, p, o in ontology.manual_triples():
        print(f"  ({s})-[:{p}]->({o})")


if __name__ == "__main__":
    main()
