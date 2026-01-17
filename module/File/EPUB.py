import zipfile
import re
import warnings
from enum import StrEnum
from typing import Generator

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, NavigableString, Tag
from lxml import etree

# 忽略 BeautifulSoup 的 XML 解析警告
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

class EPUB():
    """
    EPUB 文件处理器 (KeywordGacha 版)
    
    采用多策略并行竞争架构 (移植自 Dev-Experimental)：
    1. Strategy.STANDARD - 原始标准方法（使用 <p> 等块级标签）
    2. Strategy.BR_SEPARATED - BR分隔方法（适用于 Calibre/Sigil 等编辑器输出）
    3. Strategy.EBOOKLIB - ebooklib 保底方法（最终兜底）
    """

    # 解析策略枚举
    class Strategy(StrEnum):
        STANDARD = "standard"           # 原始标准方法
        BR_SEPARATED = "br_separated"   # BR分隔方法
        EBOOKLIB = "ebooklib"           # ebooklib 保底方法
        MIXED = "mixed"                 # 混合方法

    # 显式引用以避免打包问题
    etree

    # EPUB 文件中读取的标签范围（块级标签）
    EPUB_TAGS = ("p", "h1", "h2", "h3", "h4", "h5", "h6", "div", "li", "td")

    # 内联标签（不作为独立文本单元，但需要保留其内容）
    INLINE_TAGS = ("span", "a", "em", "strong", "b", "i", "u", "ruby", "rt", "rp", "sub", "sup", "small", "mark", "code")

    # 换行标签
    BR_TAGS = ("br",)

    # 需要跳过的元素（导航、目录等 Calibre 生成的结构性元素）
    SKIP_CLASSES = (
        "calibreMeta", "calibreMetaTitle", "calibreMetaAuthor",
        "calibreToc", "calibreEbNav", "calibreEbNavTop",
        "calibreAPrev", "calibreANext", "calibreAHome"
    )

    # SVG/MathML 驼峰命名属性映射表（小写 → 正确大小写）
    CAMEL_CASE_ATTRS: dict[str, str] = {
        "viewbox": "viewBox",
        "preserveaspectratio": "preserveAspectRatio",
        # ... 仅列出常用的，完整列表参考 Dev-Experimental ...
        # 为保持简洁，此处省略部分不常用的 SVG 属性，重点保留 viewBox 等关键属性
        # 如果需要完整列表，可从 Dev-Experimental 复制
    }

    # 命名空间声明（用于自动补全）
    NAMESPACE_DECLARATIONS: dict[str, str] = {
        "xlink:": 'xmlns:xlink="http://www.w3.org/1999/xlink"',
        "epub:": 'xmlns:epub="http://www.idpf.org/2007/ops"',
        "xml:": 'xmlns:xml="http://www.w3.org/XML/1998/namespace"',
    }

    def __init__(self) -> None:
        pass

    @classmethod
    def _auto_fix_namespaces(cls, content: str) -> str:
        """自动补全缺失的命名空间声明"""
        for prefix, declaration in cls.NAMESPACE_DECLARATIONS.items():
            if prefix in content and declaration.split("=")[0] not in content:
                if "<html" in content:
                    content = re.sub(r"(<html\s[^>]*)(>)", rf"\1 {declaration}\2", content, count=1)
                elif re.search(r"<\w+\s", content):
                    content = re.sub(r"(<\w+\s[^>]*)(>)", rf"\1 {declaration}\2", content, count=1)
        return content

    @classmethod
    def _fix_camel_case_attrs(cls, content: str) -> str:
        """修复被 html.parser 小写化的驼峰属性名"""
        # 简单实现，重点修复 viewBox
        if "viewbox=" in content.lower() and "viewBox=" not in content:
            content = re.sub(r'\bviewbox=', 'viewBox=', content, flags=re.IGNORECASE)
        return content

    @classmethod
    def _parse_xml(cls, content: str) -> BeautifulSoup:
        """智能解析 XML/XHTML 内容"""
        content = cls._auto_fix_namespaces(content)
        try:
            return BeautifulSoup(content, "lxml-xml")
        except Exception:
            soup = BeautifulSoup(content, "html.parser")
            return soup

    @classmethod
    def _should_skip_element(cls, element: Tag) -> bool:
        """判断元素是否应该被跳过"""
        if not hasattr(element, 'get'):
            return False
        
        classes = element.get("class", [])
        if isinstance(classes, str):
            classes = classes.split()
        
        for skip_class in cls.SKIP_CLASSES:
            if skip_class in classes:
                return True
        
        parent = element.parent
        while parent:
            if hasattr(parent, 'get'):
                parent_classes = parent.get("class", [])
                if isinstance(parent_classes, str):
                    parent_classes = parent_classes.split()
                for skip_class in cls.SKIP_CLASSES:
                    if skip_class in parent_classes:
                        return True
            parent = parent.parent
        return False

    @classmethod
    def _extract_text_with_ruby(cls, element) -> str:
        """从元素中提取文本，正确处理 ruby 标签"""
        if isinstance(element, NavigableString):
            return str(element)
        if not hasattr(element, 'name'):
            return ""
        if element.name in ('rt', 'rp'):
            return ""
        text_parts = []
        for child in element.children:
            text_parts.append(cls._extract_text_with_ruby(child))
        return "".join(text_parts)

    @classmethod
    def _collect_line_elements(cls, container) -> Generator[tuple[str, list], None, None]:
        """从容器中收集用 <br> 分隔的文本行"""
        current_line_text = []
        current_line_nodes = []
        
        for child in container.children:
            if isinstance(child, NavigableString):
                text = str(child)
                if text.strip():
                    current_line_text.append(text)
                    current_line_nodes.append(child)
            elif hasattr(child, 'name'):
                if child.name in cls.BR_TAGS:
                    line_text = "".join(current_line_text).strip()
                    if line_text:
                        yield (line_text, current_line_nodes.copy())
                    current_line_text = []
                    current_line_nodes = []
                elif child.name in cls.INLINE_TAGS:
                    text = cls._extract_text_with_ruby(child)
                    if text:
                        current_line_text.append(text)
                        current_line_nodes.append(child)
                elif child.name == 'hr':
                    line_text = "".join(current_line_text).strip()
                    if line_text:
                        yield (line_text, current_line_nodes.copy())
                    current_line_text = []
                    current_line_nodes = []
                else:
                    line_text = "".join(current_line_text).strip()
                    if line_text:
                        yield (line_text, current_line_nodes.copy())
                    current_line_text = []
                    current_line_nodes = []
                    for item in cls._collect_line_elements(child):
                        yield item
        
        line_text = "".join(current_line_text).strip()
        if line_text:
            yield (line_text, current_line_nodes.copy())

    @classmethod
    def _extract_strings_standard(cls, bs: BeautifulSoup) -> list[str]:
        """策略 1：使用标准方法提取文本"""
        items = []
        for dom in bs.find_all(cls.EPUB_TAGS):
            if dom.get_text().strip() == "" or dom.find(cls.EPUB_TAGS) != None:
                continue
            if cls._should_skip_element(dom):
                continue
            items.append(dom.get_text())
        return items

    @classmethod
    def _extract_strings_br_separated(cls, bs: BeautifulSoup) -> list[str]:
        """策略 2：从 BR 分隔的结构中提取文本"""
        items = []
        body = bs.find('body')
        if not body:
            return items
        
        for line_text, line_nodes in cls._collect_line_elements(body):
            if not line_text.strip():
                continue
            skip = False
            for node in line_nodes:
                if hasattr(node, 'parent') and cls._should_skip_element(node.parent):
                    skip = True
                    break
            if skip:
                continue
            items.append(line_text)
        return items

    @classmethod
    def _extract_strings_ebooklib(cls, abs_path: str) -> list[str]:
        """策略 3：使用 ebooklib 库作为最终保底解析方法"""
        items = []
        try:
            import ebooklib
            from ebooklib import epub
            
            book = epub.read_epub(abs_path, options={"ignore_ncx": True})
            
            # Document Items
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                content = item.get_content()
                if isinstance(content, bytes):
                    content = content.decode("utf-8-sig", errors="ignore")
                
                try:
                    body_content = item.get_body_content()
                    if isinstance(body_content, bytes):
                        body_content = body_content.decode("utf-8-sig", errors="ignore")
                except Exception:
                    body_content = content
                
                bs = cls._parse_xml(body_content if body_content else content)
                
                # Try standard first
                std_items = cls._extract_strings_standard(bs)
                if std_items:
                    items.extend(std_items)
                else:
                    # Fallback to all text
                    all_text = bs.get_text(separator="\n", strip=True)
                    for line in all_text.split("\n"):
                        line = line.strip()
                        if line and len(line) > 1:
                            items.append(line)

            # Navigation Items
            for item in book.get_items_of_type(ebooklib.ITEM_NAVIGATION):
                content = item.get_content()
                if isinstance(content, bytes):
                    content = content.decode("utf-8-sig", errors="ignore")
                bs = cls._parse_xml(content)
                for dom in bs.find_all("text"):
                    text = dom.get_text().strip()
                    if text:
                        items.append(text)

        except ImportError:
            pass
        except Exception:
            pass
        
        return items

    @classmethod
    def _extract_strings_from_epub_competitive(cls, abs_path: str) -> list[str]:
        """对整个 EPUB 文件并行运行多个策略，选择提取行数最多的结果"""
        results: dict[EPUB.Strategy, list[str]] = {
            cls.Strategy.STANDARD: [],
            cls.Strategy.BR_SEPARATED: [],
            cls.Strategy.EBOOKLIB: [],
        }
        
        # 策略 1 & 2：使用 zipfile 手动解析
        try:
            with zipfile.ZipFile(abs_path, "r") as zip_reader:
                for path in zip_reader.namelist():
                    if path.lower().endswith((".htm", ".html", ".xhtml")):
                        with zip_reader.open(path) as reader:
                            content = reader.read().decode("utf-8-sig")
                            bs = cls._parse_xml(content)
                            
                            results[cls.Strategy.STANDARD].extend(cls._extract_strings_standard(bs))
                            results[cls.Strategy.BR_SEPARATED].extend(cls._extract_strings_br_separated(bs))
                    
                    elif path.lower().endswith(".ncx"):
                        with zip_reader.open(path) as reader:
                            bs = cls._parse_xml(reader.read().decode("utf-8-sig"))
                            for dom in bs.find_all("text"):
                                text = dom.get_text().strip()
                                if text:
                                    results[cls.Strategy.STANDARD].append(text)
                                    results[cls.Strategy.BR_SEPARATED].append(text)
        except Exception:
            pass
        
        # 策略 3：ebooklib 保底
        results[cls.Strategy.EBOOKLIB] = cls._extract_strings_ebooklib(abs_path)
        
        # 选择最佳策略
        best_strategy = cls.Strategy.STANDARD
        best_count = 0
        
        for strategy, items in results.items():
            count = len(items)
            if count > best_count:
                best_count = count
                best_strategy = strategy
        
        # print(f"DEBUG: {abs_path} selected {best_strategy} with {best_count} lines")
        return results[best_strategy]

    # 读取
    def read_from_path(self, abs_paths: list[str]) -> list[str]:
        items: list[str] = []
        for abs_path in set(abs_paths):
            file_items = self._extract_strings_from_epub_competitive(abs_path)
            items.extend(file_items)
        return items
