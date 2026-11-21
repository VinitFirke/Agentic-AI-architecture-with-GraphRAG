"""Ask the assembly knowledge graph a question in natural language (NL -> Cypher -> NL).

Usage: python scripts/ask_graph.py "Where must the yellow cylinder be placed?"
"""

import argparse
import json

from thesis.config import load_settings
from thesis.knowledge.graph_store import connect
from thesis.knowledge.graphrag import AssemblyGraphRAG


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question", nargs="+")
    args = ap.parse_args()
    st = load_settings()
    with connect(st) as driver:
        out = AssemblyGraphRAG(driver, st).ask(" ".join(args.question))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
