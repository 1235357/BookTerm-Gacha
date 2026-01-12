"""
BookTerm Gacha - Main Application Entry Point
==============================================

A specialized term extraction tool for books (EPUB/TXT/MD) that uses
BERT-based NER and LLM semantic analysis to generate translation glossaries.

Main Workflow:
    1. Read input files (EPUB, TXT, MD, etc.) from the input folder
    2. Extract named entities using BERT NER model
    3. Analyze terms with LLM to determine categories and translations
    4. Generate output files (JSON, MD, XLSX) for use in translation tools

Key Features:
    - Multi-language support: Chinese, Japanese, Korean, English
    - GPU acceleration for NER (CUDA)
    - Configurable LLM backends (OpenAI-compatible APIs)
    - Quality checks for kana residue and similarity issues
    - Traditional/Simplified Chinese output options

Configuration:
    All settings are loaded from config.json in the application directory.
    See README.md for detailed configuration options.

Usage:
    python app.py

Based on KeywordGacha v0.13.1 by neavo
https://github.com/neavo/KeywordGacha
"""

import os
import sys
import copy
import json
import asyncio
import subprocess
from types import SimpleNamespace

# ============== Windows 控制台 UTF-8 编码设置（必须在最开始） ==============
if sys.platform == 'win32':
    try:
        import ctypes
        # 设置控制台输出代码页为 UTF-8
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except:
        pass
    
    # 设置标准输出编码
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except:
            pass


def run_environment_check() -> bool:
    """
    运行环境检查和自动修复
    
    Returns:
        True 如果环境就绪，False 如果失败
    """
    try:
        from module.EnvChecker import check_environment
        return check_environment(auto_repair=True)
    except ImportError:
        # Rich 未安装，先安装基础包
        print("=" * 60)
        print("  首次运行，正在初始化环境...")
        print("=" * 60)
        
        mirrors = [
            "https://pypi.tuna.tsinghua.edu.cn/simple",
            "https://mirrors.aliyun.com/pypi/simple",
        ]
        
        for mirror in mirrors:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-i", mirror, 
                     "--trusted-host", mirror.split("//")[1].split("/")[0],
                     "rich", "loguru"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0:
                    print(f"  ✓ 基础包安装成功")
                    break
            except Exception:
                continue
        
        # 重新尝试
        try:
            from module.EnvChecker import check_environment
            return check_environment(auto_repair=True)
        except Exception as e:
            print(f"\n❌ 环境初始化失败: {e}")
            print("   请手动运行: pip install -r requirements.txt")
            return False


# ============== 环境检查（必须在导入其他模块之前） ==============
if __name__ == "__main__":
    if not run_environment_check():
        print("\n❌ 环境检测失败，请手动安装依赖后重试")
        print("   参考命令: pip install -r requirements.txt")
        os.system("pause")
        sys.exit(1)


# ============== 环境检查通过后，导入所有依赖 ==============
from rich import box
from rich.table import Table
from rich.prompt import Prompt
from rich.traceback import install

from model.LLM import LLM
from model.NER import NER
from model.Word import Word
from module.LogHelper import LogHelper
from module.ProgressHelper import ProgressHelper
from module.TestHelper import TestHelper
from module.FileManager import FileManager
from module.Text.TextHelper import TextHelper


# ============== 配置常量（默认值，会被 config.json 覆盖） ==============
SCORE_THRESHOLD = 0.60          # 置信度阈值
MAX_DISPLAY_LENGTH = 32         # 术语最大显示长度

# 合并词语
def merge_words(words: list[Word]) -> list[Word]:
    words_unique = {}
    for word in words:
        words_unique.setdefault(word.surface, []).append(word)

    words_merged = []
    for v in words_unique.values():
        word = v[0]
        word.score = min(0.9999, max(w.score for w in v))
        words_merged.append(word)

    return sorted(words_merged, key = lambda x: x.count, reverse = True)


# 过滤超长术语（借鉴自 V0.20.2）
def filter_by_display_length(words: list[Word], max_length: int = MAX_DISPLAY_LENGTH) -> list[Word]:
    """过滤显示长度超过阈值的术语"""
    filtered = []
    for word in words:
        display_length = TextHelper.get_display_lenght(word.surface)
        if display_length <= max_length:
            filtered.append(word)
        else:
            LogHelper.debug(f"[长度过滤] 过滤超长术语: {word.surface} (长度: {display_length})")
    return filtered

# 搜索参考文本，并按出现次数排序
def search_for_context(words: list[Word], input_lines: list[str]) -> list[Word]:
    # 复制一份，避免后续的修改影响原始数据
    input_lines_ex = copy.copy(input_lines)

    # 按实体词语的长度降序排序
    words = sorted(words, key = lambda v: len(v.surface), reverse = True)

    LogHelper.print("")
    with ProgressHelper.get_progress() as progress:
        pid = progress.add_task("搜索参考文本", total = len(words))

        # 搜索参考文本
        for word in words:
            # 找出匹配的行
            index = {i for i, line in enumerate(input_lines_ex) if word.surface in line}

            # 获取匹配的参考文本，去重，并按长度降序排序
            word.context = {line for i, line in enumerate(input_lines) if i in index}
            word.context = sorted(list(word.context), key = lambda v: len(v), reverse = True)
            word.count = len(word.context)
            word.group = "未知类型"

            # 掩盖已命中的实体词语文本，避免其子串错误的与父串匹配
            input_lines_ex = [
                line.replace(word.surface, len(word.surface) * "#")  if i in index else line
                for i, line in enumerate(input_lines_ex)
            ]

            # 更新进度条
            progress.update(pid, advance = 1)
    LogHelper.print("")

    # 按出现次数降序排序
    return sorted(words, key = lambda x: x.count, reverse = True)

# 按置信度过滤词语
def filter_words_by_score(words: list[Word], threshold: float) -> list[Word]:
    return [word for word in words if word.score >= threshold]

# 按出现次数过滤词语
def filter_words_by_count(words: list[Word], threshold: float) -> list[Word]:
    return [word for word in words if word.count >= max(1, threshold)]

# 获取指定类型的词
def get_words_by_type(words: list[Word], group: str) -> list[Word]:
    return [word for word in words if word.group == group]

# 移除指定类型的词
def remove_words_by_type(words: list[Word], group: str) -> list[Word]:
    return [word for word in words if word.group != group]

# 开始处理文本
async def process_text(llm: LLM, ner: NER, file_manager: FileManager, config: SimpleNamespace, language: int) -> None:
    # 初始化
    words = []

    # 读取输入文件
    input_lines, names, nicknames = file_manager.read_lines_from_input_file(language)

    # 查找实体词语
    LogHelper.info("即将开始执行 [查找实体词语] ...")
    words, fake_name_mapping = ner.search_for_entity(input_lines, names, nicknames, language)

    # 合并相同词条
    words = merge_words(words)

    # 调试功能
    TestHelper.check_score_threshold(words, "log_score_threshold.log")

    # 置信度阈值过滤
    LogHelper.info(f"即将开始执行 [置信度阈值]，当前置信度的阈值为 {SCORE_THRESHOLD:.4f} ...")
    words = filter_words_by_score(words, SCORE_THRESHOLD)
    LogHelper.info("[置信度阈值] 已完成 ...")

    # 搜索参考文本
    LogHelper.info("即将开始执行 [搜索参考文本] ...")
    words = search_for_context(words, input_lines)

    # 出现次数阈值过滤
    LogHelper.info(f"即将开始执行 [出现次数阈值]，当前出现次数的阈值为 {config.count_threshold} ...")
    words = filter_words_by_count(words, config.count_threshold)
    LogHelper.info("[出现次数阈值] 已完成 ...")

    # 长度过滤（过滤显示长度超过32的超长术语，借鉴 V0.20.2）
    LogHelper.info(f"即将开始执行 [长度过滤]，过滤显示长度超过 {MAX_DISPLAY_LENGTH} 的术语 ...")
    original_count = len(words)
    words = filter_by_display_length(words, MAX_DISPLAY_LENGTH)
    filtered_count = original_count - len(words)
    if filtered_count > 0:
        LogHelper.info(f"[长度过滤] 已过滤 {filtered_count} 个超长术语 ...")
    LogHelper.info("[长度过滤] 已完成 ...")

    # 设置 LLM 对象
    llm.set_language(language)
    llm.set_request_limiter()

    # ============== 新工作流程：先翻译参考文本，再进行词义分析 ==============
    # 核心思路：让 LLM 在翻译过程中充分理解上下文，然后基于翻译结果进行校对审查
    
    # 步骤1：参考文本翻译（非中文时执行，让 LLM 理解上下文）
    if language != NER.Language.ZH:
        LogHelper.info("即将开始执行 [参考文本翻译]（第一阶段：翻译上下文，理解语境）...")
        words = await llm.context_translate_batch(words)
    
    # 步骤2：词义分析（基于翻译结果进行校对、审查、语义分析，给出最终译名）
    LogHelper.info("即将开始执行 [词义分析]（第二阶段：校对审查，确定最终译名）...")
    words = await llm.surface_analysis_batch(words, fake_name_mapping)
    words = remove_words_by_type(words, "")

    # 步骤3：问题修复（第三阶段：检测并修复问题词条）
    LogHelper.info("")
    LogHelper.info("即将开始执行 [问题修复]（第三阶段：检测问题词条，自动修复）...")
    words = await llm.fix_translation_batch(words)

    # 调试功能
    TestHelper.save_surface_analysis_log(words, "log_surface_analysis.log")
    TestHelper.check_result_duplication(words, "log_result_duplication.log")
    TestHelper.save_context_translate_log(words, "log_context_translate.log")

    # 还原伪名
    for word in words:
        for k, v in fake_name_mapping.items():
            word.context_summary = word.context_summary.replace(v, k)
            word.context = [line.replace(v, k) for line in word.context]
            word.context_translation = [line.replace(v, k) for line in word.context_translation]

    # 将结果写入文件
    LogHelper.info("")
    file_manager.write_result_to_file(words, language)

    # 执行结果检查
    LogHelper.info("")
    from module.ResultChecker import ResultChecker
    checker = ResultChecker(words, language)
    checker.check_all()

    # 等待用户退出
    LogHelper.info("")
    LogHelper.info("工作流程已结束 ... 请检查生成的数据文件 ...")
    LogHelper.info("")
    LogHelper.info("")
    os.system("pause")

# 接口测试
async def test_api(llm: LLM) -> None:
    # 设置请求限制器
    llm.set_request_limiter()

    # 等待接口测试结果
    if await llm.api_test():
        LogHelper.print("")
        LogHelper.info("接口测试 [green]执行成功[/] ...")
    else:
        LogHelper.print("")
        LogHelper.warning("接口测试 [red]执行失败[/], 请检查配置文件 ...")

    LogHelper.print("")
    os.system("pause")
    os.system("cls")

# 打印应用信息
def print_app_info(config: SimpleNamespace, version: str) -> None:
    LogHelper.print()
    LogHelper.print()
    LogHelper.rule(f"📚 BookTerm Gacha {version}", style = "light_goldenrod2")
    LogHelper.rule("[blue]An LLM-Powered Agent for Book Terminology Extraction", style = "light_goldenrod2")
    LogHelper.rule("专为书籍（EPUB/TXT/MD）优化的 LLM 术语表生成工具", style = "light_goldenrod2")
    LogHelper.print()

    table = Table(
        box = box.ASCII2,
        expand = True,
        highlight = True,
        show_lines = True,
        show_header = False,
        border_style = "light_goldenrod2",
    )
    table.add_column("", style = "white", ratio = 2, overflow = "fold")
    table.add_column("", style = "white", ratio = 5, overflow = "fold")

    rows = []
    rows.append(("模型名称", str(config.model_name)))
    rows.append(("接口密钥", str(config.api_key)))
    rows.append(("接口地址", str(config.base_url)))
    rows.append(("网络请求超时时间", f"{config.request_timeout} 秒"))
    rows.append(("网络请求频率阈值", f"{config.request_frequency_threshold} 次/秒"))
    rows.append(("参考文本翻译模式", "新流程：先翻译后分析（强制启用）"))

    for row in rows:
        table.add_row(*row)
    LogHelper.print(table)

    LogHelper.print()
    LogHelper.print("请编辑 [green]config.json[/] 文件来修改应用设置 ...")
    LogHelper.print()

# 打印菜单
def print_menu_main() -> int:
    LogHelper.print("请选择功能：")
    LogHelper.print("")
    LogHelper.print("\t--> 1. 开始处理 [green]中文文本[/]")
    LogHelper.print("\t--> 2. 开始处理 [green]英文文本[/]")
    LogHelper.print("\t--> 3. 开始处理 [green]日文文本[/]")
    LogHelper.print("\t--> 4. 开始处理 [green]韩文文本[/]")
    LogHelper.print("\t--> 5. 开始执行 [green]接口测试[/]")
    LogHelper.print("")
    choice = int(Prompt.ask("请输入选项前的 [green]数字序号[/] 来使用对应的功能，默认为 [green][3][/] ",
        choices = ["1", "2", "3", "4", "5"],
        default = "3",
        show_choices = False,
        show_default = False
    ))
    LogHelper.print("")

    return choice

# 主函数
async def begin(llm: LLM, ner: NER, file_manager: FileManager, config: SimpleNamespace, version: str) -> None:
    choice = -1
    while choice not in (1, 2, 3, 4):
        print_app_info(config, version)

        choice = print_menu_main()
        if choice == 1:
            await process_text(llm, ner, file_manager, config, NER.Language.ZH)
        elif choice == 2:
            await process_text(llm, ner, file_manager, config, NER.Language.EN)
        elif choice == 3:
            await process_text(llm, ner, file_manager, config, NER.Language.JA)
        elif choice == 4:
            await process_text(llm, ner, file_manager, config, NER.Language.KO)
        elif choice == 5:
            await test_api(llm)

# 一些初始化步骤
def load_config() -> tuple[LLM, NER, FileManager, SimpleNamespace, str]:
    global SCORE_THRESHOLD, MAX_DISPLAY_LENGTH
    
    with LogHelper.status("正在初始化 [green] BookTerm Gacha [/] 引擎 ..."):
        config = SimpleNamespace()
        version = ""

        try:
            # 优先使用开发环境配置文件
            if not os.path.isfile("config_dev.json"):
                path = "config.json"
            else:
                path = "config_dev.json"

            # 读取配置文件
            with open(path, "r", encoding = "utf-8-sig") as reader:
                for k, v in json.load(reader).items():
                    setattr(config, k, v[0])

            # 读取版本号文件
            with open("version.txt", "r", encoding = "utf-8-sig") as reader:
                version = reader.read().strip()
        except Exception:
            LogHelper.error("配置文件读取失败 ...")

        # ============== 从配置加载全局参数 ==============
        # 置信度阈值
        SCORE_THRESHOLD = getattr(config, 'score_threshold', 0.60)
        # 术语最大显示长度
        MAX_DISPLAY_LENGTH = getattr(config, 'max_display_length', 32)
        # Word 类的上下文采样配置
        Word.set_config(
            max_context_samples=getattr(config, 'max_context_samples', 10),
            tokens_per_sample=getattr(config, 'tokens_per_sample', 512)
        )

        # 初始化 LLM 对象
        llm = LLM(config)
        llm.load_prompt()
        llm.load_llm_config()

        # 初始化 NER 对象
        ner = NER()
        ner.load_blacklist()
        # 设置 NER 目标实体类型（从配置加载）
        ner_target_types = getattr(config, 'ner_target_types', ["PER", "LOC"])
        ner.set_target_types(ner_target_types)

        # 初始化 FileManager 对象（传入简繁转换配置）
        traditional_chinese_enable = getattr(config, 'traditional_chinese_enable', False)
        file_manager = FileManager(
            traditional_chinese_enable=traditional_chinese_enable
        )
        
        # 打印配置状态
        LogHelper.info(f"置信度阈值: {SCORE_THRESHOLD}")
        LogHelper.info(f"术语最大长度: {MAX_DISPLAY_LENGTH}")
        LogHelper.info(f"上下文采样数: {Word.MAX_CONTEXT_SAMPLES}")
        LogHelper.info(f"每样本Token数: {Word.TOKENS_PER_SAMPLE}")
        LogHelper.info(f"NER目标类型: {', '.join(ner_target_types)}")
        if traditional_chinese_enable:
            LogHelper.info("繁体中文输出已启用 ...")

    return llm, ner, file_manager, config, version

# 确保程序出错时可以捕捉到错误日志
async def main() -> None:
    try:
        # 注册全局异常追踪器
        install()

        # 加载配置
        llm, ner, file_manager, config, version = load_config()

        # 开始处理
        await begin(llm, ner, file_manager, config, version)
    except EOFError:
        LogHelper.error("EOFError - 程序即将退出 ...")
    except KeyboardInterrupt:
        LogHelper.error("KeyboardInterrupt - 程序即将退出 ...")
    except Exception as e:
        LogHelper.error(f"{LogHelper.get_trackback(e)}")
        LogHelper.print()
        LogHelper.error("出现严重错误，程序即将退出，错误信息已保存至日志文件 ...")
        LogHelper.print()
        os.system("pause")

# 入口函数
if __name__ == "__main__":
    asyncio.run(main())