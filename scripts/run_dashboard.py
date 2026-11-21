"""Run the live assembly-guidance dashboard.

Examples
--------
Live RealSense D435 (on the laptop, with the SSH tunnel to the server open):
    python scripts/run_dashboard.py

Webcam or a recorded file instead:
    python scripts/run_dashboard.py --source webcam:0
    python scripts/run_dashboard.py --source data/demo_sequence

Without a display (e.g. on the server): save dashboard frames instead of showing a window:
    python scripts/run_dashboard.py --source data/demo_sequence --headless
"""

from __future__ import annotations

import argparse
import logging
import time

import cv2

from thesis.config import DEFAULT_CONFIG, load_settings
from thesis.knowledge.graph_store import connect
from thesis.knowledge.graphrag import AssemblyGraphRAG
from thesis.pipeline.comm_log import CommLog
from thesis.pipeline.guidance import GuidanceLoop, GuidanceSystem
from thesis.ui.dashboard import DashboardRenderer
from thesis.vision.camera import open_source
from thesis.vision.perception import Perceiver
from thesis.vision.vlm import VLMClient

log = logging.getLogger("dashboard")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--source", help="realsense | webcam:<n> | image/folder/video path")
    ap.add_argument("--interval", type=float, help="seconds between VLM inspections")
    ap.add_argument("--headless", action="store_true", help="save frames instead of a window")
    ap.add_argument("--max-cycles", type=int, default=0, help="stop after N cycles (0 = no limit)")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    st = load_settings(args.config)
    vcfg, dcfg, ccfg = st["vlm"], st["dashboard"], st["camera"]
    interval = args.interval or st["pipeline"]["interval_s"]
    source_spec = args.source or ccfg["source"]

    driver = connect(st)
    graphrag = AssemblyGraphRAG(driver, st)
    vlm = VLMClient(vcfg["model"], st.ollama_host, vcfg)
    perceiver = Perceiver(vlm, vcfg["max_image_side"], vcfg["jpeg_quality"])
    comm = CommLog(st.path(st["logging"]["dir"]))
    system = GuidanceSystem(perceiver, graphrag, comm)
    log.info("Session log: %s", comm.dir)

    log.info("Warming up VLM (%s at %s) and Neo4j...", vcfg["model"], st.ollama_host)
    system.warm_up()

    reference = cv2.imread(str(st.path(st["reference_image"])))
    if reference is None:
        raise FileNotFoundError(st.path(st["reference_image"]))
    renderer = DashboardRenderer(reference, dcfg["view_height"], dcfg["panel_height"])
    # Cycle 0: check that the reference image is the goal state in the graph.
    ref_result = system.run_cycle(reference, 0, time.perf_counter())
    renderer.ref_result = ref_result
    if ref_result.decision.status != "success":
        log.warning(
            "Reference image does NOT match the graph goal: %s", ref_result.decision.message
        )
    else:
        log.info("Reference image matches the graph goal state.")

    source = open_source(source_spec, ccfg)
    loop = GuidanceLoop(system, interval)
    loop.start()
    frames_dir = comm.dir / "frames"
    if args.headless:
        frames_dir.mkdir(exist_ok=True)
    else:
        cv2.namedWindow(dcfg["window_name"], cv2.WINDOW_NORMAL)

    delivered = None
    t_prev, fps = time.perf_counter(), 0.0
    try:
        while True:
            frame = source.read()
            if frame is None:
                log.info("Source ended.")
                break
            loop.submit(frame)
            result = loop.latest
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now
            img = renderer.render(
                frame, result, source.name, loop.cycles, loop.busy, loop.error, fps
            )

            if args.headless:
                if result is not None and result is not delivered:
                    cv2.imwrite(str(frames_dir / f"cycle_{result.cycle:04d}.jpg"), img)
            else:
                cv2.imshow(dcfg["window_name"], img)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            if result is not None and result is not delivered:
                system.log_delivery(result, time.perf_counter())
                renderer.delivered(result)
                delivered = result
                if args.max_cycles and result.cycle >= args.max_cycles:
                    break
            if args.headless or source.name.startswith("file"):
                time.sleep(1 / 30)  # file sources: pace like a 30 fps camera
    except KeyboardInterrupt:
        pass
    finally:
        loop.stop()
        source.close()
        if not args.headless:
            cv2.destroyAllWindows()
        summary = comm.close()
        driver.close()

    log.info("Communication log: %s", comm.md_path)
    for hop, s in summary.items():
        if hop == "VLM throughput":
            log.info(
                "VLM throughput: %.2f req/s, %.0f tok/s, %d cycles",
                s["requests_per_s"],
                s["mean_output_tokens_per_s"],
                s["cycles"],
            )
        else:
            log.info("%-30s mean %7.1f ms  p95 %7.1f ms  (n=%d)", hop, s["mean"], s["p95"], s["n"])


if __name__ == "__main__":
    main()
