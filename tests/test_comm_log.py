import csv

from thesis.pipeline.comm_log import CommLog


def test_comm_log_writes_csv_markdown_and_summary(tmp_path):
    log = CommLog(tmp_path, "s1")
    log.add(0, "VLM", "GraphRAG", "reference check", 100, 100)
    for cycle in (1, 2):
        log.add(cycle, "VLM", "GraphRAG", "seen | pipe", 300, 300)
        log.add(cycle, "GraphRAG", "VLM", "step 3", 5, 305)
        log.add(cycle, "VLM", "Worker", "[ERROR] move it", 20, 325)
        log.add_vlm_stats(150, 70, 450)
    summary = log.close()

    rows = list(csv.DictReader(open(log.csv_path)))
    assert len(rows) == 7
    md = log.md_path.read_text()
    assert "seen \\| pipe" in md  # pipes escaped in the table
    assert "## Latency summary" in md and "## VLM throughput" in md
    assert summary["VLM → GraphRAG"]["n"] == 2  # cycle 0 excluded
    assert summary["end-to-end (frame → worker)"]["mean"] == 325
    assert summary["VLM throughput"]["cycles"] == 2
