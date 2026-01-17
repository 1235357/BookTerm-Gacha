import json
import os
import re
import threading
from datetime import datetime
from typing import Any


class ErrorLogger:
    _lock = threading.Lock()
    _enabled: bool = True
    _max_chars: int = 20000
    _log_file = "log/error_detail.log"
    _re_secret_tokens = [
        re.compile(r"\b(nvapi-[A-Za-z0-9_\-]{20,})\b"),
        re.compile(r"\b(sk-[A-Za-z0-9_\-]{20,})\b"),
        re.compile(r"\b(AIza[0-9A-Za-z\-_]{20,})\b"),
        re.compile(r"\b(ms-[0-9a-fA-F\-]{20,})\b"),
        re.compile(r"\b([0-9a-fA-F]{16,}\.[0-9a-fA-F]{16,})\b"),
        re.compile(r"\b(xox[baprs]-[0-9A-Za-z\-]{10,})\b"),
        re.compile(r"(?i)\bBearer\s+([A-Za-z0-9\.\-_]{20,})\b"),
    ]
    _secret_keys = {
        "api_key",
        "apikey",
        "api-key",
        "authorization",
        "auth",
        "token",
        "access_token",
        "access-token",
        "secret",
        "password",
    }

    @classmethod
    def configure(cls, enabled: bool | None = None, max_chars: int | None = None, log_file: str | None = None) -> None:
        if enabled is not None:
            cls._enabled = bool(enabled)
        if isinstance(max_chars, int) and max_chars > 0:
            cls._max_chars = max_chars
        if isinstance(log_file, str) and log_file.strip():
            cls._log_file = log_file.strip()

    @classmethod
    def _redact_token(cls, s: Any) -> Any:
        if not isinstance(s, str):
            return s
        if len(s) <= 12:
            return "***"
        return f"{s[:6]}…{s[-4:]}"

    @classmethod
    def _redact_secrets_in_text(cls, s: str) -> str:
        redacted = s
        for pat in cls._re_secret_tokens:
            def repl(m):
                token = m.group(1)
                return cls._redact_token(token)
            redacted = pat.sub(repl, redacted)
        return redacted

    @classmethod
    def _truncate_text(cls, s: str) -> str:
        if cls._max_chars <= 0 or len(s) <= cls._max_chars:
            return s
        return s[: cls._max_chars] + f"...(truncated, len={len(s)})"

    @classmethod
    def _sanitize(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            out: dict[str, Any] = {}
            for k, v in obj.items():
                lk = str(k).lower()
                if lk in cls._secret_keys:
                    if isinstance(v, list):
                        out[k] = [cls._redact_token(x) for x in v]
                    else:
                        out[k] = cls._redact_token(v)
                    continue
                out[k] = cls._sanitize(v)
            return out
        if isinstance(obj, list):
            return [cls._sanitize(v) for v in obj]
        if isinstance(obj, str):
            return cls._truncate_text(cls._redact_secrets_in_text(obj))
        return obj

    @classmethod
    def log(cls, error_type: str, message: str, context: dict[str, Any] | None = None) -> None:
        if cls._enabled is not True:
            return

        if context is None:
            context = {}
        safe_context = cls._sanitize(context)

        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error_type": error_type,
            "message": message,
            "context": safe_context,
        }

        os.makedirs(os.path.dirname(cls._log_file), exist_ok=True)
        with cls._lock:
            try:
                if not os.path.exists(cls._log_file) or os.path.getsize(cls._log_file) == 0:
                    f = open(cls._log_file, "w", encoding="utf-8-sig")
                else:
                    f = open(cls._log_file, "a", encoding="utf-8")
                with f:
                    f.write(json.dumps(entry, ensure_ascii=False, indent=2))
                    f.write("\n" + "-" * 80 + "\n")
            except Exception:
                return

