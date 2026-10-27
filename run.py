#!/usr/bin/env python3
"""
One-command launcher for Particles.

Starts:
  1. the hand-tracking inference server (backend/server.py, WebSocket :8765)
  2. a static HTTP server for the frontend (:8000)
  3. the default web browser

Examples:
  python run.py                    # MediaPipe on CPU (demo mode)
  python run.py --backend mock     # synthetic hands, no webcam needed
  python run.py --backend hailo --hef models/hand_landmark_lite.hef
  python run.py --camera 1 --no-browser

Any extra arguments are forwarded to backend/server.py (e.g. --smoothing 0.25).
"""

from __future__ import annotations

import argparse
import functools
import http.server
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Static file handler without per-request logging."""

    def log_message(self, *args) -> None:
        pass

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")  # always serve fresh files
        super().end_headers()


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Particles (inference server + web UI)")
    parser.add_argument("--backend", choices=["cpu", "hailo", "mock"], default="cpu")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--hef", default="models/hand_landmark_lite.hef")
    parser.add_argument("--ws-port", type=int, default=8765)
    parser.add_argument("--http-port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    args, passthrough = parser.parse_known_args()

    # Treat SIGTERM like Ctrl+C so the child server is always cleaned up.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    # 1) Inference server as a child process (same Python interpreter / venv).
    server_cmd = [
        sys.executable, str(ROOT / "backend" / "server.py"),
        "--backend", args.backend, "--camera", str(args.camera),
        "--hef", args.hef, "--port", str(args.ws_port), *passthrough,
    ]
    server = subprocess.Popen(server_cmd, cwd=ROOT)

    # 2) Static HTTP server for the frontend, in a background thread.
    handler = functools.partial(QuietHandler, directory=str(ROOT / "frontend"))
    try:
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", args.http_port), handler)
    except OSError as exc:
        print(f"[run] cannot bind HTTP port {args.http_port}: {exc}", flush=True)
        server.terminate()
        return 1
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    url = f"http://localhost:{args.http_port}/?ws=ws://localhost:{args.ws_port}"
    print(f"[run] UI:        {url}", flush=True)
    print(f"[run] backend:   {args.backend}  (Ctrl+C to stop)", flush=True)

    # 3) Browser.
    if not args.no_browser:
        time.sleep(0.8)  # give the WebSocket server a moment to start
        webbrowser.open(url)

    try:
        while server.poll() is None:
            time.sleep(0.5)
        print(f"[run] inference server exited with code {server.returncode}", flush=True)
        return server.returncode or 0
    except KeyboardInterrupt:
        print("\n[run] shutting down...", flush=True)
        return 0
    finally:
        httpd.shutdown()
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    sys.exit(main())
