import json
import os
from typing import Any


def config_path() -> str:
    if os.path.isfile("config_dev.json"):
        return "config_dev.json"
    return "config.json"


def load_raw(path: str | None = None) -> tuple[str, dict[str, Any]]:
    p = path or config_path()
    with open(p, "r", encoding="utf-8-sig") as f:
        return p, json.load(f)


def save_raw(path: str, raw: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8-sig") as f:
        json.dump(raw, f, ensure_ascii=False, indent=4)
        f.write("\n")


def get_value(raw: dict[str, Any], key: str, default: Any = None) -> Any:
    v = raw.get(key, default)
    if isinstance(v, list) and len(v) > 0:
        return v[0]
    return v


def set_value(raw: dict[str, Any], key: str, value: Any) -> None:
    old = raw.get(key)
    if isinstance(old, list) and len(old) > 0:
        old[0] = value
        raw[key] = old
    else:
        raw[key] = value


def platform_summary(raw: dict[str, Any]) -> tuple[str, int, int]:
    platforms = raw.get("platforms", [])
    activate_id = raw.get("activate_platform", 0)
    name = "Unknown"
    key_count = 0
    for p in platforms:
        if p.get("id") == activate_id:
            name = p.get("name", name)
            keys = p.get("api_key", [])
            if isinstance(keys, list):
                key_count = len(keys)
            break
    return name, int(activate_id), int(key_count)

