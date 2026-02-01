import re
from module.File.TextIO import read_text_lines_any_encoding

class SRT():

    # 1
    # 00:00:08,120 --> 00:00:10,460
    # にゃにゃにゃ

    # 2
    # 00:00:14,000 --> 00:00:15,880
    # えーこの部屋一人で使

    # 3
    # 00:00:15,880 --> 00:00:17,300
    # えるとか最高じゃん

    def __init__(self) -> None:
        super().__init__()

    # 读取
    def read_from_path(self, abs_paths: list[str]) -> list[str]:
        items: list[str] = []
        for abs_path in set(abs_paths):
            # 数据处理
            text = "\n".join(read_text_lines_any_encoding(abs_path)).strip()
            chunks = re.split(r"\n{2,}", text)
            for chunk in chunks:
                lines = [line.strip() for line in chunk.splitlines()]

                if len(lines) < 3 or not lines[0].isdecimal():
                    continue

                if lines[-1] != "":
                    items.append("\n".join(lines[2:]))

        return items
