"""Session log of the messages VLM <-> GraphRAG and VLM -> Worker, with latencies.

Each session writes to logs/<timestamp>/:

* ``communication.csv``: one row per message (machine-readable, for plots).
* ``communication.md``: the same rows as a Markdown table, appended live, plus a latency
  and throughput summary when the session closes.
"""

from __future__ import annotations

import csv
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

FIELDS = [
    "cycle",
    "time",
    "t_session_s",
    "sender",
    "receiver",
    "message",
    "hop_latency_ms",
    "end_to_end_ms",
    "details",
]


@dataclass
class _Row:
    cycle: int
    time: str
    t_session_s: float
    sender: str
    receiver: str
    message: str
    hop_latency_ms: float
    end_to_end_ms: float
    details: str


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


class CommLog:
    """Write communication rows to CSV and Markdown, and summarise latency at the end."""

    def __init__(self, root: str | Path, session_name: str | None = None):
        name = session_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / "communication.csv"
        self.md_path = self.dir / "communication.md"
        self._t0 = time.perf_counter()
        self._rows: list[_Row] = []
        self._vlm_stats: list[dict[str, float]] = []
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._csv_file, fieldnames=FIELDS)
        self._csv.writeheader()
        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write(f"# Communication log: session {name}\n\n")
            f.write(
                "`hop latency` is the time the sender needed to produce the message. "
                "`end-to-end` is the time from frame capture to this message.\n\n"
            )
            f.write(
                "| cycle | time | sender → receiver | message | hop latency (ms) "
                "| end-to-end (ms) | details |\n"
            )
            f.write("|---:|---|---|---|---:|---:|---|\n")

    def add(
        self,
        cycle: int,
        sender: str,
        receiver: str,
        message: str,
        hop_latency_ms: float,
        end_to_end_ms: float,
        details: str = "",
    ) -> None:
        row = _Row(
            cycle=cycle,
            time=datetime.now().strftime("%H:%M:%S.%f")[:-3],
            t_session_s=round(time.perf_counter() - self._t0, 3),
            sender=sender,
            receiver=receiver,
            message=message,
            hop_latency_ms=round(hop_latency_ms, 1),
            end_to_end_ms=round(end_to_end_ms, 1),
            details=details,
        )
        self._rows.append(row)
        self._csv.writerow(row.__dict__)
        self._csv_file.flush()
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(
                f"| {row.cycle} | {row.time} | {row.sender} → {row.receiver} | "
                f"{_md_escape(row.message)} | {row.hop_latency_ms:.1f} | "
                f"{row.end_to_end_ms:.1f} | {_md_escape(row.details)} |\n"
            )

    def add_vlm_stats(self, wall_ms: float, output_tokens: int, tokens_per_s: float) -> None:
        """Record one VLM request for the throughput summary."""
        self._vlm_stats.append(
            {"wall_ms": wall_ms, "tokens": output_tokens, "tokens_per_s": tokens_per_s}
        )

    # ------------------------------------------------------------------------ summary
    def summary(self) -> dict[str, dict[str, float]]:
        """Latency statistics per hop (ms) and VLM throughput."""
        by_hop: dict[str, list[float]] = defaultdict(list)
        e2e: list[float] = []
        cycles = set()
        for r in self._rows:
            if r.cycle <= 0:  # startup/reference rows are not part of the live loop
                continue
            cycles.add(r.cycle)
            by_hop[f"{r.sender} → {r.receiver}"].append(r.hop_latency_ms)
            if r.receiver == "Worker":
                e2e.append(r.end_to_end_ms)
        out: dict[str, dict[str, float]] = {k: _stats(v) for k, v in by_hop.items()}
        if e2e:
            out["end-to-end (frame → worker)"] = _stats(e2e)
        live = [r for r in self._rows if r.cycle > 0]
        duration = (live[-1].t_session_s - live[0].t_session_s) if len(live) > 1 else 0.0
        if self._vlm_stats:
            wall = [s["wall_ms"] for s in self._vlm_stats]
            out["VLM throughput"] = {
                "requests": len(wall),
                "requests_per_s": 1000 / statistics.mean(wall),
                "mean_output_tokens_per_s": statistics.mean(
                    s["tokens_per_s"] for s in self._vlm_stats
                ),
                "cycles": len(cycles),
                "cycles_per_min": 60 * (len(cycles) - 1) / duration if duration > 0 else 0.0,
            }
        return out

    def close(self) -> dict[str, dict[str, float]]:
        """Append the summary to the Markdown log and close the files."""
        summary = self.summary()
        self._csv_file.close()
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write("\n## Latency summary (live cycles only)\n\n")
            f.write("| hop | n | mean (ms) | median (ms) | p95 (ms) | min (ms) | max (ms) |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for hop, s in summary.items():
                if hop == "VLM throughput":
                    continue
                f.write(
                    f"| {hop} | {s['n']:.0f} | {s['mean']:.1f} | {s['median']:.1f} | "
                    f"{s['p95']:.1f} | {s['min']:.1f} | {s['max']:.1f} |\n"
                )
            if "VLM throughput" in summary:
                t = summary["VLM throughput"]
                f.write("\n## VLM throughput\n\n| metric | value |\n|---|---:|\n")
                f.write(f"| VLM requests | {t['requests']:.0f} |\n")
                f.write(
                    f"| VLM requests per second (1 / mean latency) | {t['requests_per_s']:.2f} |\n"
                )
                f.write(
                    "| Mean generation speed (output tokens/s) | "
                    f"{t['mean_output_tokens_per_s']:.1f} |\n"
                )
                f.write(f"| Guidance cycles completed | {t['cycles']:.0f} |\n")
                f.write(f"| Guidance cycles per minute | {t['cycles_per_min']:.1f} |\n")
        return summary


def _stats(values: list[float]) -> dict[str, float]:
    v = sorted(values)
    return {
        "n": len(v),
        "mean": statistics.mean(v),
        "median": statistics.median(v),
        "p95": v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))],
        "min": v[0],
        "max": v[-1],
    }
