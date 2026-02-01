from module.File.TextIO import read_text_lines_any_encoding

class ASS():

    # [Script Info]
    # ; This is an Advanced Sub Station Alpha v4+ script.
    # Title:
    # ScriptType: v4.00+
    # PlayDepth: 0
    # ScaledBorderAndShadow: Yes

    # [V4+ Styles]
    # Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
    # Style: Default,Arial,20,&H00FFFFFF,&H0000FFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,1,2,10,10,10,1

    # [Events]
    # Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
    # Dialogue: 0,0:00:08.12,0:00:10.46,Default,,0,0,0,,にゃにゃにゃ
    # Dialogue: 0,0:00:14.00,0:00:15.88,Default,,0,0,0,,えーこの部屋一人で使\Nえるとか最高じゃん
    # Dialogue: 0,0:00:15.88,0:00:17.30,Default,,0,0,0,,えるとか最高じゃん

    def __init__(self) -> None:
        super().__init__()

    # 读取
    def read_from_path(self, abs_paths: list[str]) -> list[str]:
        items: list[str] = []
        for abs_path in set(abs_paths):
            lines = [line.strip() for line in read_text_lines_any_encoding(abs_path)]

            in_event = False
            format_field_num = -1
            for line in lines:
                if line == "[Events]":
                    in_event = True
                if in_event == True and line.startswith("Format:"):
                    format_field_num = len(line.split(",")) - 1
                    break

            for line in lines:
                content = ",".join(line.split(",")[format_field_num:]) if line.startswith("Dialogue:") else ""
                items.append(content.replace("\\N", "\n"))

        return items
