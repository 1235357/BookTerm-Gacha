from __future__ import annotations

from typing import Iterable

from module.LogHelper import LogHelper


def read_text_lines_any_encoding(abs_path: str, encodings: Iterable[str] | None = None) -> list[str]:
    if encodings is None:
        encodings = (
            "utf-8-sig",
            "utf-8",
            "cp932",
            "shift_jis",
            "gb18030",
            "gbk",
            "big5",
        )

    last_error: Exception | None = None
    for enc in encodings:
        try:
            with open(abs_path, "r", encoding=enc) as reader:
                return [line.rstrip("\r\n") for line in reader.readlines()]
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            break

    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as reader:
            lines = [line.rstrip("\r\n") for line in reader.readlines()]
        LogHelper.warning(f"[编码警告] 文本解码失败，已使用 replace 模式读取: {abs_path} ({last_error})")
        return lines
    except Exception:
        return []

