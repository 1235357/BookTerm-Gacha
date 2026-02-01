from module.File.TextIO import read_text_lines_any_encoding

class MD():

    def __init__(self) -> None:
        super().__init__()

    # 读取
    def read_from_path(self, abs_paths: list[str]) -> list[str]:
        items: list[str] = []
        for abs_path in set(abs_paths):
            for line in read_text_lines_any_encoding(abs_path):
                items.append(line)

        return items
