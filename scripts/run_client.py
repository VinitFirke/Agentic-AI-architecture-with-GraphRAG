"""Laptop client: camera + OpenCV dashboard locally, all processing on the server.

1. On the server:   python scripts/serve_pipeline.py
2. On the laptop:   ssh -N -L 8765:localhost:8765 <user>@<server>     (keep it open)
3. On the laptop:   python scripts/run_client.py

Examples:
    python scripts/run_client.py                          # RealSense D435
    python scripts/run_client.py --source webcam:0        # any webcam
    python scripts/run_client.py --source some/folder --headless   # no window, save frames
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import requests

from thesis.config import DEFAULT_CONFIG, load_settings
from thesis.remote.client import RemoteClient
from thesis.remote.protocol import DEFAULT_PORT
from thesis.ui.dashboard import DashboardRenderer
from thesis.vision.camera import open_source

log = logging.getLogger("client")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    # 127.0.0.1, not "localhost": on Windows "localhost" tries IPv6 first, which can add ~2 s.
    ap.add_argument("--server", default=f"http://127.0.0.1:{DEFAULT_PORT}")
    ap.add_argument("--source", help="realsense | webcam:<n> | image/folder/video path")
    ap.add_argument("--interval", type=float, help="seconds between VLM inspections")
    ap.add_argument("--headless", action="store_true", help="save frames instead of a window")
    ap.add_argument("--max-side", type=int, help="resize frames to this longest side before upload")
    ap.add_argument("--jpeg-quality", type=int, help="upload JPEG quality (1-100)")
    ap.add_argument("--max-cycles", type=int, default=0, help="stop after N cycles (0 = no limit)")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    st = load_settings(args.config)
    vcfg, dcfg, ccfg = st["vlm"], st["dashboard"], st["camera"]
    upcfg = st.raw.get("client", {})
    client = RemoteClient(
        args.server,
        args.interval or st["pipeline"]["interval_s"],
        max_side=args.max_side or upcfg.get("max_side", 640),
        jpeg_quality=args.jpeg_quality or upcfg.get("jpeg_quality", 70),
        timeout_s=vcfg["timeout_s"],
    )
    log.info(
        "Upload: frames resized to %d px, JPEG quality %d", client.max_side, client.jpeg_quality
    )
    try:
        reference, ref_result = client.start_session()
    except requests.RequestException as exc:
        raise SystemExit(
            f"Cannot reach the pipeline server at {args.server} ({type(exc).__name__}).\n"
            "Is scripts/serve_pipeline.py running on the server, and is the SSH tunnel open?\n"
            "  ssh -N -L 8765:localhost:8765 <user>@<server>"
        ) from None
    if ref_result.decision.status != "success":
        log.warning(
            "Reference image does NOT match the graph goal: %s", ref_result.decision.message
        )
    renderer = DashboardRenderer(reference, dcfg["view_height"], dcfg["panel_height"])
    renderer.ref_result = ref_result

    source = open_source(args.source or ccfg["source"], ccfg)
    client.start()
    out_dir = Path("client_frames") / time.strftime("%Y-%m-%d_%H-%M-%S")
    if args.headless:
        out_dir.mkdir(parents=True, exist_ok=True)
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
            client.submit(frame)
            result = client.latest
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now
            img = renderer.render(
                frame, result, source.name, client.cycles, client.busy, client.error, fps
            )
            if args.headless:
                if result is not None and result is not delivered:
                    cv2.imwrite(str(out_dir / f"cycle_{result.cycle:04d}.jpg"), img)
            else:
                cv2.imshow(dcfg["window_name"], img)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            if result is not None and result is not delivered:
                client.report_delivered(result, time.perf_counter())
                renderer.delivered(result)
                delivered = result
                if args.max_cycles and result.cycle >= args.max_cycles:
                    break
            if args.headless or source.name.startswith("file"):
                time.sleep(1 / 30)  # file sources: pace like a 30 fps camera
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()
        source.close()
        if not args.headless:
            cv2.destroyAllWindows()
        time.sleep(0.5)  # let the last delivery report reach the server
        closed = client.close_session()

    if closed:
        log.info("Communication log on the server: %s", closed.get("log"))
        for hop, s in closed.get("summary", {}).items():
            if hop == "VLM throughput":
                log.info(
                    "VLM throughput: %.2f req/s, %.0f tok/s, %d cycles",
                    s["requests_per_s"],
                    s["mean_output_tokens_per_s"],
                    s["cycles"],
                )
            else:
                log.info(
                    "%-30s mean %7.1f ms  p95 %7.1f ms  (n=%d)", hop, s["mean"], s["p95"], s["n"]
                )


if __name__ == "__main__":
    main()
