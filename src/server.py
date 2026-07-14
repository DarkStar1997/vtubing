"""Lightweight aiohttp server for the three-vrm browser frontend.

Serves static frontend files + the VRM avatar, and exposes a WebSocket
endpoint that the tracking loop broadcasts rig state to.

The server runs in a background daemon thread with its own asyncio event
loop so the main thread can run MediaPipe + the rig solver uninterrupted.
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from aiohttp import web, WSMsgType

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class RigServer:
    """HTTP + WebSocket server running in a background thread."""

    def __init__(self, avatar_path: str | Path, port: int = 8080, host: str = "localhost") -> None:
        self._avatar_path = Path(avatar_path)
        self._port = port
        self._host = host
        self._clients: set[web.WebSocketResponse] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._camera_frame: bytes | None = None

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="rig-server")
        self._thread.start()
        self._ready.wait(timeout=5)

    def stop(self) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=3)

    # -- public API (called from main thread) --------------------------------

    def broadcast(self, state: dict) -> None:
        """Send a rig-state dict to every connected browser (thread-safe)."""
        if not self._clients or self._loop is None:
            return
        msg = json.dumps(state, default=_json_default)
        asyncio.run_coroutine_threadsafe(self._broadcast(msg), self._loop)

    def set_camera_frame(self, jpeg_bytes: bytes) -> None:
        """Store the latest annotated webcam JPEG for the MJPEG stream."""
        self._camera_frame = jpeg_bytes

    # -- background thread ----------------------------------------------------

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as exc:
            print(f"[server] error: {exc}")
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/avatar.js", self._avatar_js)
        app.router.add_get("/avatar", self._vrm_file)
        app.router.add_get("/camera", self._camera_stream)
        app.router.add_get("/ws", self._websocket)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        await site.start()
        self._ready.set()
        print(f"[server] listening on {self.url}")
        # Block until stopped
        stop_event = asyncio.Event()
        await stop_event.wait()

    async def _broadcast(self, msg: str) -> None:
        dead: list[web.WebSocketResponse] = []
        for ws in self._clients:
            try:
                await ws.send_str(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    # -- request handlers -----------------------------------------------------

    async def _index(self, _request: web.Request) -> web.Response:
        return web.FileResponse(_FRONTEND_DIR / "index.html")

    async def _avatar_js(self, _request: web.Request) -> web.Response:
        return web.FileResponse(
            _FRONTEND_DIR / "avatar.js",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    async def _vrm_file(self, _request: web.Request) -> web.Response:
        if not self._avatar_path.exists():
            return web.Response(status=404, text=f"avatar not found: {self._avatar_path}")
        return web.FileResponse(
            self._avatar_path,
            headers={
                "Content-Type": "model/gltf-binary",
                "Cache-Control": "no-cache, no-store, must-revalidate",
            },
        )

    async def _camera_stream(self, request: web.Request) -> web.StreamResponse:
        """MJPEG stream of the annotated webcam feed."""
        response = web.StreamResponse()
        response.content_type = "multipart/x-mixed-replace; boundary=frame"
        await response.prepare(request)
        try:
            while True:
                frame = self._camera_frame
                if frame is not None:
                    await response.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    )
                await asyncio.sleep(1 / 30)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return response

    async def _websocket(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._clients.add(ws)
        print(f"[server] client connected ({len(self._clients)} total)")
        try:
            async for _msg in ws:
                pass  # browser sends nothing; we push rig state
        finally:
            self._clients.discard(ws)
            print(f"[server] client disconnected ({len(self._clients)} total)")
        return ws


def _json_default(obj):
    """Fallback for objects json can't serialise natively."""
    import numpy as np
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not serialisable: {type(obj)}")
