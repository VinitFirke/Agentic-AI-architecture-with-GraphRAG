"""Run the guidance pipeline (VLM + GraphRAG + log) as a service for a remote camera client.

Start this on the server, then run scripts/run_client.py on the laptop with the camera.
The service listens on 127.0.0.1 only; the laptop reaches it through an SSH tunnel:

    ssh -N -L 8765:localhost:8765 <user>@<server>

Usage: python scripts/serve_pipeline.py [--port 8765]
"""

import argparse
import logging

from thesis.config import DEFAULT_CONFIG, load_settings
from thesis.remote.protocol import DEFAULT_PORT
from thesis.remote.server import PipelineService, create_app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # no line per HTTP request
    logging.getLogger("httpx").setLevel(logging.WARNING)

    service = PipelineService(load_settings(args.config))
    print(f"Pipeline server ready on http://127.0.0.1:{args.port} (Ctrl+C to stop)")
    try:
        create_app(service).run(host="127.0.0.1", port=args.port, threaded=True)
    finally:
        service.close_session()
        service.driver.close()


if __name__ == "__main__":
    main()
