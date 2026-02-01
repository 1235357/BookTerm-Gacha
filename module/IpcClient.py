import json
import socket
import uuid
from typing import Any


class IpcClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout_seconds: float = 0.6) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_seconds = float(timeout_seconds)

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        req = {
            "id": uuid.uuid4().hex,
            "method": method,
            "params": params or {},
        }
        payload = (json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8")
        with socket.create_connection((self.host, self.port), timeout=self.timeout_seconds) as s:
            s.settimeout(self.timeout_seconds)
            s.sendall(payload)
            buf = b""
            while b"\n" not in buf:
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
            line = buf.split(b"\n", 1)[0]
            if not line:
                return {"ok": False, "error": "empty_response"}
            return json.loads(line.decode("utf-8", errors="replace"))

