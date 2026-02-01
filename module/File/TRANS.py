import json
from module.File.TextIO import read_text_lines_any_encoding

class TRANS():

    def __init__(self) -> None:
        super().__init__()

    # 读取
    def read_from_path(self, abs_paths: list[str]) -> list[str]:
        items: list[str] = []
        for abs_path in set(abs_paths):
            # 数据处理
            json_data = json.loads("\n".join(read_text_lines_any_encoding(abs_path)))

            if not isinstance(json_data, dict):
                continue

            project: dict = json_data.get("project", {})

            for path, entry in project.get("files", {}).items():
                for data in entry.get("data", []):
                    if not isinstance(data, list) or len(data) == 0 or not isinstance(data[0], str):
                        continue

                    items.append(data[0])

        return items
