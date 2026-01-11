"""
RubyCleaner - 日文注音（振り仮名/ルビ）清理器

用于清理日文文本中的各种注音标记格式，提取基础汉字文本。
支持的格式包括：
- (漢字/かんじ) - 括号斜杠格式
- [漢字/かんじ] - 方括号斜杠格式  
- |漢字[かんじ] - 竖线方括号格式（常见于 EPUB/小说）
- \\r[漢字,かんじ] - 反斜杠r格式
- \\rb[漢字,かんじ] - 反斜杠rb格式
- [r_かんじ][ch_漢字] - r_ch格式
- [ch_漢字] - ch格式
- <ruby = かんじ>漢字</ruby> - HTML ruby简化格式
- <ruby><rb>漢字</rb>...</ruby> - HTML ruby完整格式
- [ruby text=かんじ] - ruby text格式
"""

import re


class RubyCleaner:
    """日文注音清理器"""

    # 注音规则列表：(正则模式, 替换内容)
    # 注意：更具体的规则需要放在前面；使用 [^<>/\[\]] 避免误匹配HTML标签
    RULE: tuple[tuple[re.Pattern, str], ...] = (
        # HTML ruby格式（需要最先处理，避免被其他规则误匹配）
        # <ruby = かんじ>漢字</ruby> - HTML ruby简化格式
        (re.compile(r'<ruby\s*=\s*[^>]*>([^<]*)</ruby>', flags=re.IGNORECASE), r"\1"),
        
        # <ruby><rb>漢字</rb>...</ruby> - HTML ruby带rb标签格式
        (re.compile(r'<ruby>\s*<rb>([^<]*)</rb>.*?</ruby>', flags=re.IGNORECASE | re.DOTALL), r"\1"),
        
        # <ruby>漢字<rp>(</rp><rt>かんじ</rt><rp>)</rp></ruby> - 带括号的HTML ruby格式
        (re.compile(r'<ruby>([^<]*)<rp>[^<]*</rp><rt>[^<]*</rt><rp>[^<]*</rp></ruby>', flags=re.IGNORECASE), r"\1"),
        
        # <ruby>漢字<rt>かんじ</rt></ruby> - 最常见的HTML ruby格式
        (re.compile(r'<ruby>([^<]*)<rt>[^<]*</rt></ruby>', flags=re.IGNORECASE), r"\1"),
        
        # (漢字/かんじ) - 括号斜杠格式（限制不含<>以避免误匹配HTML）
        (re.compile(r'\(([^<>/\(\)]+)/[^<>/\(\)]+\)', flags=re.IGNORECASE), r"\1"),
        
        # [漢字/かんじ] - 方括号斜杠格式
        (re.compile(r'\[([^<>/\[\]]+)/[^<>/\[\]]+\]', flags=re.IGNORECASE), r"\1"),
        
        # |漢字[かんじ] - 竖线方括号格式（最常见的格式）
        (re.compile(r'\|([^<>\[\]]+?)\[[^<>\[\]]+?\]', flags=re.IGNORECASE), r"\1"),
        
        # \r[漢字,かんじ] - 反斜杠r格式
        (re.compile(r'\\r\[([^,\[\]]+?),[^\[\]]+?\]', flags=re.IGNORECASE), r"\1"),
        
        # \rb[漢字,かんじ] - 反斜杠rb格式
        (re.compile(r'\\rb\[([^,\[\]]+?),[^\[\]]+?\]', flags=re.IGNORECASE), r"\1"),
        
        # [r_かんじ][ch_漢字] - r_ch组合格式
        (re.compile(r'\[r_[^\[\]]+?\]\[ch_([^\[\]]+?)\]', flags=re.IGNORECASE), r"\1"),
        
        # [ch_漢字] - 单独ch格式
        (re.compile(r'\[ch_([^\[\]]+?)\]', flags=re.IGNORECASE), r"\1"),
        
        # [ruby text=かんじ] [ruby text = かんじ] [ruby text="かんじ"] - ruby text标签
        (re.compile(r'\[ruby text\s*=\s*[^\]]*\]', flags=re.IGNORECASE), ""),
    )

    @classmethod
    def clean(cls, text: str) -> str:
        """
        清理文本中的注音标记，只保留基础文本
        
        Args:
            text: 包含注音标记的原始文本
            
        Returns:
            清理后的纯文本
        """
        for pattern, replacement in cls.RULE:
            text = re.sub(pattern, replacement, text)
        return text

    @classmethod
    def has_ruby(cls, text: str) -> bool:
        """
        检测文本是否包含注音标记
        
        Args:
            text: 待检测的文本
            
        Returns:
            是否包含注音标记
        """
        for pattern, _ in cls.RULE:
            if pattern.search(text):
                return True
        return False


# 测试代码
if __name__ == "__main__":
    test_cases = [
        "(漢字/かんじ)",
        "[漢字/かんじ]",
        "|漢字[かんじ]",
        "\\r[漢字,かんじ]",
        "\\rb[漢字,かんじ]",
        "[r_かんじ][ch_漢字]",
        "[ch_漢字]",
        "<ruby = かんじ>漢字</ruby>",
        "<ruby><rb>漢字</rb><rtc><rt>かんじ</rt></rtc></ruby>",
        "[ruby text=かんじ]漢字",
        "これは|魔法[まほう]の|世界[せかい]です。",
    ]
    
    print("RubyCleaner 测试:")
    print("-" * 50)
    for text in test_cases:
        cleaned = RubyCleaner.clean(text)
        print(f"原文: {text}")
        print(f"清理: {cleaned}")
        print()
