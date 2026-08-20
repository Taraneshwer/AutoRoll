"""
AutoRoll Worker Main Entry Point CLI with Signal Handling for Graceful Shutdown.
Usage: python -m worker.main [--server-url http://localhost:8000] [--video path_to_test.mp4]
"""

import argparse
import signal
import sys
import time

from app.core.logger import get_logger
from app.workers.config import WorkerSettings
from app.workers.service import WorkerService

logger = get_logger("worker_main")


def parse_args():
    parser = argparse.ArgumentParser(description="AutoRoll ML Worker Process CLI")
    parser.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help="Central AutoRoll server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Path to prerecorded video file for local worker test mode",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device selection (default: auto)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = WorkerSettings(SERVER_URL=args.server_url, DEVICE=args.device)
    worker = WorkerService(config=cfg)

    # Signal Handler for Graceful Shutdown
    def handle_signal(sig, frame):
        logger.info(f"Signal {sig} received. Initiating graceful worker shutdown...")
        worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if args.video:
        logger.info(f"Running worker in local video test mode on '{args.video}'...")
        events = worker.process_local_video_test(args.video)
        print("\n" + "=" * 60)
        print("         AUTOROLL WORKER LOCAL TEST COMPLETE        ")
        print("=" * 60)
        print(f"Events Detected : {len(events)}")
        print(f"Worker State    : {worker.state}")
        print("=" * 60 + "\n")
        worker.stop()
        return

    worker.start()
    logger.info("AutoRoll Worker Process running. Press Ctrl+C to terminate.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Stopping worker...")
        worker.stop()


if __name__ == "__main__":
    main()
