#!/usr/bin/env bash
# Start (or reuse) the Neo4j container that stores the assembly knowledge graph.
# Ports are bound to 127.0.0.1 only; reach them from the laptop through an SSH tunnel.
# Credentials are read from .env (NEO4J_USER / NEO4J_PASSWORD). APOC is enabled because
# LangChain's Neo4jGraph reads the graph schema with apoc.meta.data().
set -euo pipefail

cd "$(dirname "$0")/.."
set -a; source .env; set +a

NAME="${NEO4J_CONTAINER:-thesis-neo4j}"

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "Neo4j container '$NAME' is already running."
elif docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker start "$NAME"
else
  docker run -d --name "$NAME" --restart unless-stopped \
    -p 127.0.0.1:7474:7474 -p 127.0.0.1:7687:7687 \
    -e NEO4J_AUTH="${NEO4J_USER}/${NEO4J_PASSWORD}" \
    -e NEO4J_PLUGINS='["apoc"]' \
    -v thesis-neo4j-data:/data \
    neo4j:5-community
fi

# Rootless Docker accepts TCP on the port before Neo4j is up, so poll the HTTP API instead.
echo -n "Waiting for Neo4j on 127.0.0.1:7474 "
for _ in $(seq 1 90); do
  if curl -sf -o /dev/null http://127.0.0.1:7474; then echo " ready."; exit 0; fi
  echo -n "."; sleep 2
done
echo " timed out. Check: docker logs $NAME" >&2
exit 1
