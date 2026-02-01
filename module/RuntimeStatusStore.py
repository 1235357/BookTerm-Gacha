import json
import os
import threading
import time
from typing import Any, Callable


def status_file_path() -> str:
    return os.path.join("log", "runtime_status.json")


def write_status_snapshot(data: dict[str, Any], path: str | None = None) -> None:
    p = path or status_file_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, p)


def read_status_snapshot(path: str | None = None) -> dict[str, Any] | None:
    p = path or status_file_path()
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None


class RuntimeStatusWriter:
    def __init__(self, get_status: Callable[[], dict[str, Any]], interval_seconds: float = 1.0) -> None:
        self._get_status = get_status
        self._interval = max(0.2, float(interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    data = self._get_status()
                    data = dict(data) if isinstance(data, dict) else {"status": str(data)}
                    data["timestamp"] = time.time()
                    write_status_snapshot(data)
                except Exception:
                    pass
                self._stop.wait(self._interval)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
