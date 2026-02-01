from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class IpcResponse:
    id: str
    ok: bool
    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "ok": bool(self.ok)}
        if self.ok:
            d["result"] = self.result
        else:
            d["error"] = self.error or "unknown_error"
        return d


def sanitize_updates(updates: Any) -> dict[str, Any]:
    if not isinstance(updates, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in updates.items():
        if not isinstance(k, str) or not k:
            continue
        out[k] = v
    return out

