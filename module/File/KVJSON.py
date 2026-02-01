import json
from module.File.TextIO import read_text_lines_any_encoding

class KVJSON():

    # {
    #     "「あ・・」": "「あ・・」",
    #     "「ごめん、ここ使う？」": "「ごめん、ここ使う？」",
    #     "「じゃあ・・私は帰るね」": "「じゃあ・・私は帰るね」",
    # }

    def __init__(self) -> None:
        super().__init__()

    # 读取
    def read_from_path(self, abs_paths: list[str]) -> list[str]:
        items: list[str] = []
        for abs_path in set(abs_paths):
            # 数据处理
            json_data: dict[str, str] = json.loads("\n".join(read_text_lines_any_encoding(abs_path)))

            if not isinstance(json_data, dict):
                continue

            for k, v in json_data.items():
                if isinstance(k, str) and isinstance(v, str):
                    src = k
                    dst = v
                    if src == "":
                        items.append(src)
                    elif dst != "" and src != dst:
                        items.append(src)
                    else:
                        items.append(src)

        return items
