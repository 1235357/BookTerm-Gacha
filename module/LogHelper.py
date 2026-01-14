"""
BookTerm Gacha - Log Helper Module (重构版)
============================================

参照兄弟项目 Dev-Experimental/base/LogManager.py 完全重写。

核心改动：
1. 使用 Rich RichHandler 实现彩色日志输出（用颜色区分级别，而非 INFO 文字）
2. 使用 SafeTimedRotatingFileHandler 实现按日轮转的文件日志
3. 【强制启用专家模式】- 始终显示详细日志，不区分专家模式
4. 日志分离：控制台日志用于用户查看，文件日志用于调试

日志级别颜色（由 RichHandler 自动处理）：
    - DEBUG: 灰色/暗色
    - INFO: 绿色
    - WARNING: 黄色
    - ERROR: 红色

Based on LinguaGacha's LogManager.py
"""

import logging
import os
import shutil
import threading
import time
import traceback
import sys
from logging.handlers import TimedRotatingFileHandler
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.status import Status


# ==================== Windows VT100 支持 ====================
def enable_windows_vt100():
    """
    在 Windows 上启用 VT100 转义序列支持
    这是让 Rich Live 正确原地更新的关键！！
    """
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # 获取标准输出句柄
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            # 获取当前控制台模式
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            # 启用 ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x0004)
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass

# 在模块加载时启用
enable_windows_vt100()


class SafeTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    Windows 下更稳健的按日轮转 Handler（直接借鉴 Dev-Experimental）

    现象：TimedRotatingFileHandler 在 doRollover() 中会尝试 os.rename(app.log -> app.log.YYYY-MM-DD)。
    若该文件正被其他进程占用（常见：编辑器/杀软/另一个程序实例），rename 会抛 WinError 32，
    且之后每次写日志都会重复触发 rollover，导致控制台刷屏 "--- Logging error ---"。

    策略：rename 失败时退化为"复制到目标文件 + 截断源文件"，并吞掉异常以避免刷屏。
    """

    def rotate(self, source: str, dest: str) -> None:
        try:
            return super().rotate(source, dest)
        except PermissionError:
            last_exc: Optional[Exception] = None
            for attempt in range(5):
                try:
                    if not os.path.exists(source):
                        return

                    if os.path.exists(dest):
                        try:
                            os.remove(dest)
                        except OSError:
                            pass

                    with open(source, "rb") as sf, open(dest, "wb") as df:
                        shutil.copyfileobj(sf, df, length=1024 * 1024)

                    try:
                        with open(source, "wb"):
                            pass
                    except OSError:
                        pass

                    return
                except Exception as e:
                    last_exc = e
                    time.sleep(0.15 * (attempt + 1))

            _ = last_exc
            return


class LogHelper:
    """
    日志辅助类 - BookTerm Gacha 日志系统（重构版）
    
    核心特点：
    1. 控制台日志：使用 RichHandler，用颜色区分日志级别
    2. 文件日志：按日轮转，保留 3 天
    3. 【强制专家模式】：始终显示详细信息
    4. 分离控制：file/console 参数控制输出目标
    """

    LOG_PATH: str = "./log"
    _LOCK: threading.Lock = threading.Lock()
    _initialized: bool = False

    # 实例变量
    _console: Optional[Console] = None
    _console_logger: Optional[logging.Logger] = None
    _file_logger: Optional[logging.Logger] = None
    _file_handler: Optional[SafeTimedRotatingFileHandler] = None
    
    # 兼容旧代码
    console_highlight: Optional[Console] = None
    console_no_highlight: Optional[Console] = None
    logger: Optional[logging.Logger] = None

    @classmethod
    def _ensure_init(cls) -> None:
        """确保日志系统已初始化（延迟初始化）"""
        if cls._initialized:
            return
        
        with cls._LOCK:
            if cls._initialized:
                return
            
            # ========== 设置控制台编码（修复 Windows PyInstaller 打包后的 Unicode 问题） ==========
            import sys
            if sys.platform == 'win32':
                try:
                    # 尝试设置控制台为 UTF-8 编码
                    import ctypes
                    ctypes.windll.kernel32.SetConsoleOutputCP(65001)
                except:
                    pass
            
            # ========== 控制台实例 ==========
            # 使用 force_terminal=True
            # 配合 legacy_windows=False，强制使用 ANSI 转义序列（前提是 enable_windows_vt100 已成功）
            cls._console = Console(highlight=True, tab_size=4, force_terminal=True, legacy_windows=False)
            cls.console_highlight = cls._console
            cls.console_no_highlight = Console(highlight=False, tab_size=4, force_terminal=True, legacy_windows=False)

            # ========== 文件日志实例 ==========
            os.makedirs(cls.LOG_PATH, exist_ok=True)
            cls._file_handler = SafeTimedRotatingFileHandler(
                f"{cls.LOG_PATH}/app.log",
                when="midnight",
                interval=1,
                encoding="utf-8",
                backupCount=3,
            )
            cls._file_handler.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
            )
            cls._file_logger = logging.getLogger("booktermgacha_file")
            cls._file_logger.propagate = False
            cls._file_logger.setLevel(logging.DEBUG)
            cls._file_logger.handlers.clear()
            cls._file_logger.addHandler(cls._file_handler)

            # ========== 控制台日志实例 ==========
            console_handler = RichHandler(
                console=cls._console,  # 【关键】复用同一个 Console 实例
                markup=True,
                show_path=False,
                rich_tracebacks=False,
                tracebacks_extra_lines=0,
                log_time_format="[%X]",
                omit_repeated_times=False,
            )
            cls._console_logger = logging.getLogger("booktermgacha_console")
            cls._console_logger.propagate = False
            # 【强制专家模式】：始终使用 DEBUG 级别
            cls._console_logger.setLevel(logging.DEBUG)
            cls._console_logger.handlers.clear()
            cls._console_logger.addHandler(console_handler)
            
            # 兼容旧代码
            cls.logger = cls._console_logger

            cls._initialized = True

    @classmethod
    def get_console(cls) -> Console:
        """获取全局唯一的 Console 实例"""
        cls._ensure_init()
        return cls._console

    
    # ==================== 核心日志方法 ====================

    @classmethod
    def print(cls, msg: str = "", e: Optional[Exception] = None, file: bool = True, console: bool = True, **kwargs) -> None:
        """
        通用打印方法（不带日志级别前缀，支持 Rich markup）
        """
        cls._ensure_init()
        msg_e = f"{msg} {e}" if e and msg else f"{e}" if e else msg
        
        if e is None:
            if file:
                cls._file_logger.info(msg)
            if console:
                cls._console.print(msg, **kwargs)
        else:
            tb = cls.get_trackback(e)
            if file:
                cls._file_logger.info(f"{msg_e}\n{tb}\n")
            if console:
                cls._console.print(f"{msg_e}\n{tb}\n", **kwargs)

    @classmethod
    def debug(cls, msg: str, e: Optional[Exception] = None, file: bool = True, console: bool = True) -> None:
        """DEBUG 级别日志（灰色/暗色）"""
        cls._ensure_init()
        msg_e = f"{msg} {e}" if e and msg else f"{e}" if e else msg
        
        if e is None:
            if file:
                cls._file_logger.debug(msg)
            if console:
                cls._console_logger.debug(msg)
        else:
            tb = cls.get_trackback(e)
            if file:
                cls._file_logger.debug(f"{msg_e}\n{tb}\n")
            if console:
                cls._console_logger.debug(f"{msg_e}\n{tb}\n")

    @classmethod
    def info(cls, msg: str, e: Optional[Exception] = None, file: bool = True, console: bool = True) -> None:
        """INFO 级别日志（绿色）"""
        cls._ensure_init()
        msg_e = f"{msg} {e}" if e and msg else f"{e}" if e else msg
        
        if e is None:
            if file:
                cls._file_logger.info(msg)
            if console:
                cls._console_logger.info(msg)
        else:
            tb = cls.get_trackback(e)
            if file:
                cls._file_logger.info(f"{msg_e}\n{tb}\n")
            if console:
                cls._console_logger.info(f"{msg_e}\n{tb}\n")

    @classmethod
    def warning(cls, msg: str, e: Optional[Exception] = None, file: bool = True, console: bool = True) -> None:
        """WARNING 级别日志（黄色）"""
        cls._ensure_init()
        msg_e = f"{msg} {e}" if e and msg else f"{e}" if e else msg
        
        if e is None:
            if file:
                cls._file_logger.warning(msg)
            if console:
                cls._console_logger.warning(msg)
        else:
            tb = cls.get_trackback(e)
            if file:
                cls._file_logger.warning(f"{msg_e}\n{tb}\n")
            if console:
                cls._console_logger.warning(f"{msg_e}\n{tb}\n")

    @classmethod
    def error(cls, msg: str, e: Optional[Exception] = None, file: bool = True, console: bool = True) -> None:
        """ERROR 级别日志（红色）"""
        cls._ensure_init()
        msg_e = f"{msg} {e}" if e and msg else f"{e}" if e else msg
        
        if e is None:
            if file:
                cls._file_logger.error(msg)
            if console:
                cls._console_logger.error(msg)
        else:
            tb = cls.get_trackback(e)
            if file:
                cls._file_logger.error(f"{msg_e}\n{tb}\n")
            if console:
                cls._console_logger.error(f"{msg_e}\n{tb}\n")

    @classmethod
    def critical(cls, msg: str, e: Optional[Exception] = None, file: bool = True, console: bool = True) -> None:
        """CRITICAL 级别日志（等同于 error）"""
        cls.error(msg, e, file, console)

    # ==================== 辅助方法 ====================

    @classmethod
    def rule(cls, *args, **kwargs) -> None:
        """打印分隔线"""
        cls._ensure_init()
        cls.console_no_highlight.rule(*args, **kwargs)

    @classmethod
    def input(cls, *args, **kwargs) -> str:
        """获取用户输入"""
        cls._ensure_init()
        return cls.console_no_highlight.input(*args, **kwargs)

    @classmethod
    def status(cls, *args, **kwargs) -> Status:
        """创建状态指示器"""
        cls._ensure_init()
        return cls.console_no_highlight.status(*args, **kwargs)

    @staticmethod
    def is_debug() -> bool:
        """
        检查是否为调试模式
        
        【强制专家模式】：始终返回 True
        """
        return True

    @staticmethod
    def get_trackback(e: Exception) -> str:
        """获取异常的完整堆栈追踪"""
        return "".join(traceback.format_exception(type(e), e, e.__traceback__)).strip()