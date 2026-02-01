import asyncio
import json
import threading
from typing import Any, Callable

from module.IpcProtocol import IpcResponse


class IpcServer:
    def __init__(
        self,
        host: str,
        port: int,
        dispatch: Callable[[str, dict[str, Any], str], IpcResponse],
    ) -> None:
        self.host = host
        self.port = int(port)
        self._dispatch = dispatch
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: dict[str, asyncio.StreamWriter] = {}
        self._sub_lock = threading.Lock()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        def run() -> None:
            asyncio.run(self._run_async())

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def publish(self, event: str, data: Any) -> None:
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._publish_in_loop, event, data)

    def _publish_in_loop(self, event: str, data: Any) -> None:
        payload = {"type": "event", "event": event, "data": data}
        line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        stale: list[str] = []
        with self._sub_lock:
            items = list(self._subscribers.items())
        for sid, w in items:
            try:
                if w.is_closing():
                    stale.append(sid)
                    continue
                w.write(line)
            except Exception:
                stale.append(sid)
        for sid in stale:
            with self._sub_lock:
                self._subscribers.pop(sid, None)

    async def _run_async(self) -> None:
        self._loop = asyncio.get_running_loop()
        server = await asyncio.start_server(self._handle_client, host=self.host, port=self.port)
        async with server:
            while not self._stop.is_set():
                await asyncio.sleep(0.2)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        keep_open = False
        try:
            line = await reader.readline()
            if not line:
                return
            try:
                req = json.loads(line.decode("utf-8", errors="replace"))
            except Exception:
                resp = IpcResponse(id="unknown", ok=False, error="invalid_json")
                writer.write((json.dumps(resp.to_dict(), ensure_ascii=False) + "\n").encode("utf-8"))
                await writer.drain()
                return

            req_id = str(req.get("id", ""))
            method = str(req.get("method", ""))
            params = req.get("params", {})
            if not isinstance(params, dict):
                params = {}

            if method == "subscribe":
                sid = req_id or "sub"
                with self._sub_lock:
                    self._subscribers[sid] = writer
                ack = {"type": "subscribed", "id": sid}
                writer.write((json.dumps(ack, ensure_ascii=False) + "\n").encode("utf-8"))
                await writer.drain()
                keep_open = True
                while not self._stop.is_set() and not writer.is_closing():
                    await asyncio.sleep(0.5)
                return

            resp = self._dispatch(method, params, req_id)
            if not isinstance(resp, IpcResponse):
                resp = IpcResponse(id=req_id or "unknown", ok=False, error="invalid_response")

            if not resp.id:
                resp.id = req_id or "unknown"

            writer.write((json.dumps(resp.to_dict(), ensure_ascii=False) + "\n").encode("utf-8"))
            await writer.drain()
        finally:
            if keep_open:
                return
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
