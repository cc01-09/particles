"""
Particles inference server.

Pipeline (runs in a background thread at ~30 Hz):

    webcam frame -> mirror -> backend.process() -> 21 keypoints per hand
                 -> HandFeatureTracker (openness, distance, EMA smoothing)
                 -> JSON (+ small JPEG preview with skeleton)
                 -> WebSocket broadcast to every connected browser

Usage:
    python backend/server.py --backend cpu          # MediaPipe on CPU (demo)
    python backend/server.py --backend hailo --hef models/hand_landmark_lite.hef
    python backend/server.py --backend mock         # synthetic hands, no camera
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import errno
import json
import logging
import signal
import threading
import time

import cv2
import numpy as np
from websockets.asyncio.server import broadcast, serve

from hand_features import HandFeatureTracker, draw_skeleton

log = logging.getLogger("particles")


def create_backend(args: argparse.Namespace):
    """Instantiate the requested inference backend (imports are lazy so that
    e.g. the CPU mode never needs hailo_platform installed)."""
    if args.backend == "cpu":
        from backend_cpu import MediaPipeBackend
        return MediaPipeBackend(max_hands=2)
    if args.backend == "hailo":
        from backend_hailo import HailoHandBackend
        return HailoHandBackend(hef_path=args.hef)
    if args.backend == "mock":
        from backend_mock import MockBackend
        return MockBackend()
    raise ValueError(f"unknown backend: {args.backend}")


class CaptureWorker(threading.Thread):
    """Grabs frames, runs inference and publishes JSON messages."""

    def __init__(self, backend, args: argparse.Namespace, publish):
        super().__init__(daemon=True, name="capture")
        self.backend = backend
        self.args = args
        self.publish = publish  # thread-safe callable(str)
        self.stop_event = threading.Event()
        self.tracker = HandFeatureTracker(alpha=args.smoothing)
        self._fps = 0.0

    def _open_camera(self):
        cap = cv2.VideoCapture(self.args.camera)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        cap.set(cv2.CAP_PROP_FPS, self.args.fps)
        log.info("camera %s opened", self.args.camera)
        return cap

    def run(self) -> None:
        cap = None
        period = 1.0 / self.args.fps
        last_warn = 0.0
        while not self.stop_event.is_set():
            t_start = time.perf_counter()

            # 1) Acquire a frame (or a blank canvas for the mock backend).
            if self.backend.needs_camera:
                if cap is None:
                    cap = self._open_camera()
                    if cap is None:
                        if t_start - last_warn > 10:
                            log.warning("camera %s not available, retrying...", self.args.camera)
                            last_warn = t_start
                        self.publish(self._message(None, self.tracker.update([], 1, 1, t_start), 0.0))
                        self.stop_event.wait(1.0)
                        continue
                ok, frame = cap.read()
                if not ok:
                    log.warning("camera read failed, reopening")
                    cap.release()
                    cap = None
                    continue
                frame = cv2.flip(frame, 1)  # selfie view: move hand right -> right
            else:
                frame = np.zeros((self.args.height, self.args.width, 3), dtype=np.uint8)

            # 2) Inference.
            t_infer = time.perf_counter()
            try:
                hands = self.backend.process(frame)
            except Exception:  # keep streaming even if one frame fails
                log.exception("inference failed")
                hands = []
            infer_ms = (time.perf_counter() - t_infer) * 1000

            # 3) Features + smoothing, 4) preview image, 5) publish.
            h, w = frame.shape[:2]
            features = self.tracker.update(hands, w, h, time.perf_counter())
            draw_skeleton(frame, hands)
            self.publish(self._message(frame, features, infer_ms))

            # Pace the loop to the target rate and measure the achieved FPS.
            elapsed = time.perf_counter() - t_start
            self.stop_event.wait(max(0.0, period - elapsed))
            dt = time.perf_counter() - t_start
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt

        if cap is not None:
            cap.release()

    def _message(self, frame, features: dict, infer_ms: float) -> str:
        msg = {
            "type": "hands",
            "timestamp": time.time(),
            "backend": self.backend.name,
            "backend_label": self.backend.label,
            "camera": frame is not None,
            "fps": round(self._fps, 1),
            "infer_ms": round(infer_ms, 1),
            **features,
            "image": self._encode_preview(frame) if frame is not None else None,
        }
        return json.dumps(msg, separators=(",", ":"))

    def _encode_preview(self, frame: np.ndarray) -> str | None:
        if self.args.preview_width <= 0:
            return None
        h, w = frame.shape[:2]
        pw = self.args.preview_width
        small = cv2.resize(frame, (pw, int(h * pw / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii") if ok else None


async def main(args: argparse.Namespace) -> None:
    backend = create_backend(args)
    clients: set = set()
    loop = asyncio.get_running_loop()

    def publish(message: str) -> None:
        # Called from the capture thread; hop onto the event loop to send.
        loop.call_soon_threadsafe(lambda: broadcast(clients, message))

    async def handler(ws):
        clients.add(ws)
        log.info("client connected (%d total)", len(clients))
        try:
            await ws.send(json.dumps({"type": "hello", "backend": backend.name,
                                      "backend_label": backend.label}))
            async for raw in ws:
                handle_client_message(raw)
        finally:
            clients.discard(ws)
            log.info("client disconnected (%d total)", len(clients))

    worker = CaptureWorker(backend, args, publish)

    def handle_client_message(raw) -> None:
        """Clients may send {"type": "config", "fist_ratio": f, "open_ratio": o}
        to recalibrate the openness mapping (used by the debug panel)."""
        try:
            msg = json.loads(raw)
            if msg.get("type") == "config":
                worker.tracker.set_thresholds(float(msg["fist_ratio"]), float(msg["open_ratio"]))
                log.info("thresholds set to fist=%.3f open=%.3f", *worker.tracker.thresholds)
        except (ValueError, KeyError, TypeError, AttributeError):
            log.warning("ignored malformed client message: %.80r", raw)
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows: rely on KeyboardInterrupt
            pass

    async with serve(handler, args.host, args.port, max_size=2**16):
        log.info("backend=%s  ws://%s:%d", backend.label, args.host, args.port)
        worker.start()
        try:
            await stop.wait()
        finally:
            worker.stop_event.set()
            worker.join(timeout=2)
            backend.close()
            log.info("server stopped")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Particles hand-tracking inference server")
    p.add_argument("--backend", choices=["cpu", "hailo", "mock"], default="cpu",
                   help="cpu = MediaPipe (demo), hailo = UGen300 prototype, mock = synthetic hands")
    p.add_argument("--hef", default="models/hand_landmark_lite.hef", help="HEF path for --backend hailo")
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=30.0, help="target processing rate")
    p.add_argument("--smoothing", type=float, default=0.35, help="EMA alpha (0..1, lower = smoother)")
    p.add_argument("--preview-width", type=int, default=240, help="preview JPEG width, 0 = disable")
    p.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 to allow LAN clients)")
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        log.error("%s\nIs another Particles server already running? Try --port.", exc)
        raise SystemExit(1)
