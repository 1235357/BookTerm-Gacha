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
import re
from types import SimpleNamespace
import argparse
import threading
import time

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


def _resolve_platform_api_keys(platform: dict) -> list[str]:
    keys: list[str] = []

    env_name = platform.get("api_key_env", "") if isinstance(platform, dict) else ""
    if isinstance(env_name, str) and env_name.strip():
        env_val = os.environ.get(env_name.strip(), "") or ""
        for part in re.split(r"[,\n;]+", env_val):
            m = re.search(r"(nvapi-[A-Za-z0-9_\-]{20,})", part.strip())
            if m:
                keys.append(m.group(1))

    file_path = platform.get("api_key_file", "") if isinstance(platform, dict) else ""
    if isinstance(file_path, str) and file_path.strip():
        abs_path = file_path.strip()
        if not os.path.isabs(abs_path):
            abs_path = os.path.join(os.getcwd(), abs_path)
        try:
            with open(abs_path, "r", encoding="utf-8-sig") as reader:
                for line in reader.read().splitlines():
                    m = re.search(r"(nvapi-[A-Za-z0-9_\-]{20,})", line.strip())
                    if m:
                        keys.append(m.group(1))
        except Exception:
            pass

    api_key_value = platform.get("api_key", []) if isinstance(platform, dict) else []
    if isinstance(api_key_value, str):
        m = re.search(r"(nvapi-[A-Za-z0-9_\-]{20,})", api_key_value.strip())
        if m:
            keys.append(m.group(1))
    elif isinstance(api_key_value, list):
        for item in api_key_value:
            if not isinstance(item, str):
                continue
            m = re.search(r"(nvapi-[A-Za-z0-9_\-]{20,})", item.strip())
            if m:
                keys.append(m.group(1))

    deduped: list[str] = []
    seen: set[str] = set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            deduped.append(k)
    return deduped


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

    # ============== 新工作流程：整合为单一任务流（翻译→校对审查）=============
    LogHelper.info("即将开始执行 [翻译+校对]（同一批任务并行推进，避免阶段切割）...")
    words = await llm.translate_and_surface_analysis_batch(words, fake_name_mapping)
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
    await llm.set_request_limiter()

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
    
    # 显示平台名称（如果有）
    platform_name = getattr(config, 'platform_name', None)
    if platform_name:
        rows.append(("当前平台", f"[bold cyan]{platform_name}[/]"))
    
    rows.append(("模型名称", str(config.model_name)))
    
    # API Key 显示优化（支持多 Key）
    api_key = config.api_key
    if isinstance(api_key, list):
        if len(api_key) > 1:
            rows.append(("API Key", f"[green]{len(api_key)} 个 Key (轮询模式)[/]"))
        elif len(api_key) == 1:
            rows.append(("API Key", f"{api_key[0][:20]}..."))
        else:
            rows.append(("API Key", "[red]未配置[/]"))
    else:
        rows.append(("API Key", str(api_key)[:40] + "..." if len(str(api_key)) > 40 else str(api_key)))
    
    rows.append(("接口地址", str(config.base_url)))
    rows.append(("网络请求超时时间", f"{config.request_timeout} 秒"))
    rows.append(("网络请求频率阈值", f"{config.request_frequency_threshold} 次/秒"))
    rows.append(("最大并发请求数", f"{getattr(config, 'max_concurrent_requests', 5)} 个"))
    rows.append(("参考文本翻译模式", "新流程：先翻译后分析（强制启用）"))

    for row in rows:
        table.add_row(*row)
    LogHelper.print(table)

    LogHelper.print()
    LogHelper.print("请编辑 [green]config.json[/] 文件来修改应用设置 ...")
    LogHelper.print("提示: 修改 [cyan]activate_platform[/] 字段来切换不同的 API 平台")
    LogHelper.print()

# 打印菜单
async def print_menu_main() -> int:
    LogHelper.print("请选择功能：")
    LogHelper.print("")
    LogHelper.print("\t--> 1. 开始处理 [green]中文文本[/]")
    LogHelper.print("\t--> 2. 开始处理 [green]英文文本[/]")
    LogHelper.print("\t--> 3. 开始处理 [green]日文文本[/]")
    LogHelper.print("\t--> 4. 开始处理 [green]韩文文本[/]")
    LogHelper.print("\t--> 5. 开始执行 [green]接口测试[/]")
    LogHelper.print("\t--> 6. 打开 [green]配置面板[/]（交互式编辑）")
    LogHelper.print("\t--> 7. 查看 [green]运行状态面板[/]")
    LogHelper.print("")
    choice_text = await asyncio.to_thread(
        Prompt.ask,
        "请输入选项前的 [green]数字序号[/] 来使用对应的功能，默认为 [green][3][/] ",
        choices=["1", "2", "3", "4", "5", "6", "7"],
        default="3",
        show_choices=False,
        show_default=False,
    )
    choice = int(choice_text)
    LogHelper.print("")

    return choice

# 主函数
async def begin(llm: LLM, ner: NER, file_manager: FileManager, config: SimpleNamespace, version: str) -> None:
    choice = -1
    while choice not in (1, 2, 3, 4):
        print_app_info(config, version)

        choice = await print_menu_main()
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
        elif choice == 6:
            from module.ConsolePanels import interactive_config_edit
            changed = await asyncio.to_thread(interactive_config_edit)
            if changed:
                _hot_reload_config(llm, config)
        elif choice == 7:
            from module.ConsolePanels import show_status_live
            await asyncio.to_thread(show_status_live, llm)


def _start_runtime_status_writer(llm: LLM):
    try:
        from module.RuntimeStatusStore import RuntimeStatusWriter
        w = RuntimeStatusWriter(get_status=llm.get_runtime_status, interval_seconds=1.0)
        w.start()
        return w
    except Exception:
        return None


def _update_namespace_in_place(dst: SimpleNamespace, src: SimpleNamespace) -> None:
    dst.__dict__.clear()
    dst.__dict__.update(src.__dict__)


def _select_config_path() -> str:
    if not os.path.isfile("config_dev.json"):
        return "config.json"
    return "config_dev.json"


def _load_config_namespace() -> tuple[SimpleNamespace, str]:
    config = SimpleNamespace()
    version = ""
    raw_config = {}
    path = _select_config_path()

    try:
        with open(path, "r", encoding="utf-8-sig") as reader:
            raw_config = json.load(reader)

        if "platforms" in raw_config and "activate_platform" in raw_config:
            platforms = raw_config.get("platforms", [])
            activate_id = raw_config.get("activate_platform", 0)

            active_platform = None
            for platform in platforms:
                if platform.get("id") == activate_id:
                    active_platform = platform
                    break

            if active_platform is None and platforms:
                active_platform = platforms[0]
                LogHelper.warning(f"[配置警告] 未找到 ID={activate_id} 的平台，使用默认平台: {active_platform.get('name', 'Unknown')}")

            if active_platform:
                config.api_key = _resolve_platform_api_keys(active_platform)
                config.base_url = active_platform.get("api_url", "")
                config.model_name = active_platform.get("model", "")
                config.platform_name = active_platform.get("name", "Unknown")
                config.thinking = active_platform.get("thinking", True)
                config.top_p = active_platform.get("top_p", 0.95)
                config.temperature = active_platform.get("temperature", 0.05)

            for k, v in raw_config.items():
                if k not in ("platforms", "activate_platform"):
                    if isinstance(v, list) and len(v) > 0 and not k.startswith("api"):
                        setattr(config, k, v[0])
                    elif not isinstance(v, list):
                        setattr(config, k, v)
        else:
            for k, v in raw_config.items():
                setattr(config, k, v[0] if isinstance(v, list) else v)

        with open("version.txt", "r", encoding="utf-8-sig") as reader:
            version = reader.read().strip()
    except Exception as e:
        LogHelper.error(f"配置文件读取失败: {e}")

    return config, version


def _apply_global_settings(config: SimpleNamespace) -> None:
    global SCORE_THRESHOLD, MAX_DISPLAY_LENGTH
    SCORE_THRESHOLD = getattr(config, "score_threshold", 0.60)
    MAX_DISPLAY_LENGTH = getattr(config, "max_display_length", 32)
    Word.set_config(
        max_context_samples=getattr(config, "max_context_samples", 10),
        tokens_per_sample=getattr(config, "tokens_per_sample", 512),
    )
    task_timeout = getattr(config, "task_timeout_threshold", 430)
    LLM.TASK_TIMEOUT_THRESHOLD = task_timeout

    try:
        from module.ErrorLogger import ErrorLogger

        ErrorLogger.configure(
            enabled=getattr(config, "error_detail_log_enable", True),
            max_chars=getattr(config, "error_detail_log_max_chars", 20000),
            log_file=getattr(config, "error_detail_log_file", "log/error_detail.log"),
        )
    except Exception:
        pass


def _hot_reload_config(llm: LLM, config_obj: SimpleNamespace) -> None:
    new_config, _ = _load_config_namespace()
    old_keys = list(getattr(llm, "api_keys", []) or [])
    old_url = getattr(llm, "base_url", "")
    old_model = getattr(llm, "model_name", "")
    new_keys = list(getattr(new_config, "api_key", []) or []) if isinstance(getattr(new_config, "api_key", []), list) else []
    if old_url != getattr(new_config, "base_url", "") or old_model != getattr(new_config, "model_name", "") or old_keys != new_keys:
        LLM.reset_api_state()
    _apply_global_settings(new_config)
    _update_namespace_in_place(config_obj, new_config)
    llm.apply_runtime_config(config_obj)

# 一些初始化步骤
def load_config() -> tuple[LLM, NER, FileManager, SimpleNamespace, str]:
    with LogHelper.status("正在初始化 [green] BookTerm Gacha [/] 引擎 ..."):
        config, version = _load_config_namespace()
        _apply_global_settings(config)
        LLM.reset_api_state()

        # 初始化 LLM 对象
        llm = LLM(config)
        llm.load_prompt()
        llm.load_llm_config()
        llm.set_request_limiter()

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
        LogHelper.info(f"任务超时阈值: {LLM.TASK_TIMEOUT_THRESHOLD}s")
        LogHelper.info(f"NER目标类型: {', '.join(ner_target_types)}")
        if traditional_chinese_enable:
            LogHelper.info("繁体中文输出已启用 ...")

    return llm, ner, file_manager, config, version

# 确保程序出错时可以捕捉到错误日志
async def main() -> None:
    try:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--no-status-writer", action="store_true")
        parser.add_argument("--no-ipc", action="store_true")
        args, _ = parser.parse_known_args()

        # 注册全局异常追踪器
        install()

        # 加载配置
        llm, ner, file_manager, config, version = load_config()
        writer = None if args.no_status_writer else _start_runtime_status_writer(llm)

        ipc_server = None
        if not args.no_ipc and bool(getattr(config, "ipc_enable", True)):
            from module.IpcServer import IpcServer
            from module.IpcProtocol import IpcResponse, sanitize_updates
            from module.ConfigStore import load_raw, save_raw, set_value, get_value, platform_summary

            ipc_lock = threading.Lock()

            def dispatch(method: str, params: dict, req_id: str) -> IpcResponse:
                rid = str(req_id or "req")
                try:
                    with ipc_lock:
                        if method == "get_status":
                            return IpcResponse(id=rid, ok=True, result=llm.get_runtime_status())
                        if method == "reload_platform":
                            _hot_reload_config(llm, config)
                            return IpcResponse(id=rid, ok=True, result={"status": llm.get_runtime_status()})
                        if method == "get_config":
                            path, raw = load_raw()
                            name, pid, key_count = platform_summary(raw)
                            result = {
                                "config_path": path,
                                "activate_platform": pid,
                                "platform_name": name,
                                "platform_key_count": key_count,
                                "multi_key_default_enable": bool(get_value(raw, "multi_key_default_enable", True)),
                                "multi_key_default_per_key_rpm": float(get_value(raw, "multi_key_default_per_key_rpm", 1) or 1),
                                "api_key_blacklist_ttl_seconds": int(get_value(raw, "api_key_blacklist_ttl_seconds", 3600) or 3600),
                                "max_concurrent_requests": int(get_value(raw, "max_concurrent_requests", 0) or 0),
                                "request_frequency_threshold": float(get_value(raw, "request_frequency_threshold", 1) or 1),
                                "ipc_host": str(getattr(config, "ipc_host", "127.0.0.1")),
                                "ipc_port": int(getattr(config, "ipc_port", 8765)),
                            }
                            return IpcResponse(id=rid, ok=True, result=result)
                        if method == "set_config":
                            updates = sanitize_updates(params.get("updates"))
                            path, raw = load_raw()
                            for k, v in updates.items():
                                set_value(raw, k, v)
                            save_raw(path, raw)
                            _hot_reload_config(llm, config)
                            return IpcResponse(id=rid, ok=True, result={"status": llm.get_runtime_status()})
                        return IpcResponse(id=rid, ok=False, error="unknown_method")
                except Exception as e:
                    return IpcResponse(id=rid, ok=False, error=str(e))

            host = str(getattr(config, "ipc_host", "127.0.0.1") or "127.0.0.1")
            port = int(getattr(config, "ipc_port", 8765) or 8765)
            ipc_server = IpcServer(host=host, port=port, dispatch=dispatch)
            ipc_server.start()

            def publish_loop() -> None:
                while True:
                    try:
                        with ipc_lock:
                            data = llm.get_runtime_status()
                        ipc_server.publish("status", data)
                    except Exception:
                        pass
                    time.sleep(0.25)

            threading.Thread(target=publish_loop, daemon=True).start()

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
