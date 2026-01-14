"""
BookTerm Gacha - Log Table Module (重构版)
==========================================

完全参照兄弟项目 Dev-Experimental/module/Engine/Translator/TranslatorTask.py 重写。

核心改动：
1. 【强制专家模式】- 移除所有专家模式判断，始终显示完整内容
2. 使用 Rich Table 实现表格化日志输出
3. 颜色区分任务状态（绿色=成功，黄色=警告，红色=失败）
4. 完整显示：请求内容、模型思考、响应内容
5. 打开工作流的黑盒子，让用户看到一切！

Based on LinguaGacha's TranslatorTask.py
"""

import sys
import itertools
import time
from typing import Optional

import rich
from rich import box
from rich import markup
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from module.LogHelper import LogHelper


class LogTable:
    """
    LLM 操作详细日志打印器
    
    【强制专家模式】- 始终显示完整内容，打开黑盒子让用户看到一切！
    
    完全模仿 TranslatorTask.py 的 print_log_table 方法
    """
    
    # 控制台宽度限制
    CONSOLE_WIDTH = 120
    
    # Console 实例（延迟初始化）
    _console: Optional[Console] = None
    
    @classmethod
    def get_console(cls) -> Console:
        """获取控制台实例 (代理到 LogHelper)"""
        # 始终尽可能复用 LogHelper 中的全局 Console
        return LogHelper.get_console()
    
    # ==================== 阶段标题 ====================
    
    @classmethod
    def print_stage_header(cls, stage_name: str, stage_num: int = 0) -> None:
        """打印阶段标题（醒目的分隔线）"""
        console = cls.get_console()
        if stage_num > 0:
            title = f"阶段 {stage_num}: {stage_name}"
        else:
            title = stage_name
        LogHelper.print("")
        console.rule(f"[bold cyan]{title}[/]", style="cyan")
        LogHelper.print("")
    
    # ==================== 批量任务汇总 ====================
    
    @classmethod
    def print_batch_summary(
        cls,
        task_name: str,
        total: int,
        success: int,
        failed: int,
        elapsed_time: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """
        打印批量任务汇总
        
        参照 TranslatorTask.py 的风格
        """
        console = cls.get_console()
        
        # 计算成功率
        success_rate = (success / total * 100) if total > 0 else 0
        
        # 选择颜色
        if failed == 0:
            status_color = "green"
            status_icon = "✓"
        elif success > 0:
            status_color = "yellow"
            status_icon = "⚠"
        else:
            status_color = "red"
            status_icon = "✗"
        
        # 构建汇总消息
        token_info = f" | Token: {input_tokens}+{output_tokens}" if input_tokens or output_tokens else ""
        summary = (
            f"[{status_color}]{status_icon}[/] [{task_name}] 完成 | "
            f"总计: {total} | 成功: [green]{success}[/] | 失败: [red]{failed}[/] | "
            f"成功率: {success_rate:.1f}% | 耗时: {elapsed_time:.1f}s{token_info}"
        )
        
        LogHelper.print("")
        console.rule(summary, style=status_color)
        LogHelper.print("")
    
    # ==================== 核心：LLM 任务日志表格 ====================
    
    @classmethod
    def print_log_table(
        cls,
        task_name: str,
        word_surface: str,
        status: str,  # "success", "warning", "error"
        message: str,
        srcs: list[str],
        dsts: list[str],
        request_content: Optional[str] = None,
        response_think: Optional[str] = None,
        response_result: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        elapsed_time: float = 0,
        extra_info: Optional[dict] = None,
    ) -> None:
        """
        打印 LLM 任务日志表格
        
        完全模仿 TranslatorTask.py 的 print_log_table 方法
        
        【强制专家模式】- 始终显示完整内容
        """
        console = cls.get_console()
        
        # 状态颜色映射
        style_map = {
            "success": "green",
            "warning": "yellow",
            "error": "red",
        }
        style = style_map.get(status, "white")
        
        # 构建日志行
        rows = []
        
        # 第一行：任务信息
        time_info = f"{elapsed_time:.2f}s" if elapsed_time > 0 else ""
        token_info = f"Token: {input_tokens}+{output_tokens}" if input_tokens or output_tokens else ""
        info_parts = [f"[{task_name}]", word_surface]
        if time_info:
            info_parts.append(time_info)
        if token_info:
            info_parts.append(token_info)
        rows.append(f"{message} ({' | '.join(info_parts)})")
        
        # 额外信息
        if extra_info:
            info_str = " | ".join(f"{k}: {v}" for k, v in extra_info.items() if v)
            if info_str:
                rows.append(info_str)
        
        # 请求内容（【强制显示】）
        if request_content:
            rows.append(f"[bold blue]【请求内容】[/]\n{markup.escape(request_content)}")
        
        # 模型思考（【强制显示】- 这是打开黑盒子的关键！）
        if response_think:
            rows.append(f"[bold magenta]【模型思考】[/]\n{markup.escape(response_think)}")
        
        # 响应内容（【强制显示】）
        if response_result:
            rows.append(f"[bold green]【模型回复】[/]\n{markup.escape(response_result)}")
        
        # 原文译文对比（如果有）
        if srcs and dsts:
            pair = ""
            for src, dst in itertools.zip_longest(srcs, dsts, fillvalue=""):
                pair = pair + "\n" + f"{markup.escape(src)} [bright_blue]-->[/] {markup.escape(dst)}"
            rows.append(pair.strip())
        
        # 生成并打印表格
        table = cls._generate_log_table(rows, style)
        console.print(table)
        
        # 同时写入文件日志（简化版）
        file_log = f"[{task_name}] {word_surface} - {message}"
        if response_think:
            file_log += f"\n[思考] {response_think[:500]}..."
        if response_result:
            file_log += f"\n[回复] {response_result[:500]}..."
        LogHelper.debug(file_log, file=True, console=False)
    
    @classmethod
    def _generate_log_table(cls, rows: list, style: str) -> Table:
        """
        生成日志表格（完全模仿 TranslatorTask.py）
        """
        table = Table(
            box=box.ASCII2,
            expand=True,
            title=" ",
            caption=" ",
            highlight=True,
            show_lines=True,
            show_header=False,
            show_footer=False,
            collapse_padding=True,
            border_style=style,
        )
        table.add_column("", style="white", ratio=1, overflow="fold")
        
        for row in rows:
            if isinstance(row, str):
                table.add_row(row)
            else:
                table.add_row(*row)
        
        return table
    
    # ==================== 简化版任务日志 ====================
    
    @classmethod
    def print_llm_task(
        cls,
        task_name: str,
        word_surface: str,
        status: str,
        message: str,
        request_content: Optional[str] = None,
        response_content: Optional[str] = None,
        response_think: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        elapsed_time: float = 0,
        extra_info: Optional[dict] = None,
    ) -> None:
        """
        打印 LLM 任务日志（简化版，调用 print_log_table）
        """
        cls.print_log_table(
            task_name=task_name,
            word_surface=word_surface,
            status=status,
            message=message,
            srcs=[],
            dsts=[],
            request_content=request_content,
            response_think=response_think,
            response_result=response_content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_time=elapsed_time,
            extra_info=extra_info,
        )
    
    # ==================== API 请求/响应日志 ====================
    
    @classmethod
    def print_api_request(
        cls,
        model: str,
        base_url: str,
        messages: list,
        thinking_enabled: bool = False,
        stream_enabled: bool = False,
    ) -> None:
        """
        打印 API 请求详情
        
        【强制显示】- 不再判断专家模式
        """
        console = cls.get_console()
        
        # 构建请求信息行
        rows = [
            f"[bold cyan]【API 请求】[/]",
            f"模型: [green]{model}[/] | 地址: [dim]{base_url}[/]",
            f"思考模式: [{'green' if thinking_enabled else 'red'}]{'启用' if thinking_enabled else '禁用'}[/] | "
            f"流式输出: [{'green' if stream_enabled else 'red'}]{'启用' if stream_enabled else '禁用'}[/]",
        ]
        
        # 添加消息内容
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # 截断过长内容
            if len(content) > 1000:
                content = content[:500] + f"\n... [dim](省略 {len(content) - 1000} 字符)[/dim] ...\n" + content[-500:]
            rows.append(f"\n[bold]消息 {i+1} ({role}):[/]\n{markup.escape(content)}")
        
        # 打印表格
        table = cls._generate_log_table(rows, "blue")
        console.print(table)
    
    @classmethod
    def print_api_response(
        cls,
        response_content: str,
        response_think: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        elapsed_time: float = 0,
    ) -> None:
        """
        打印 API 响应详情
        
        【强制显示】- 不再判断专家模式
        """
        console = cls.get_console()
        
        rows = [
            f"[bold green]【API 响应】[/]",
            f"耗时: {elapsed_time:.2f}s | 输入Token: {input_tokens} | 输出Token: {output_tokens}",
        ]
        
        # 思考内容（【强制显示】- 这是打开黑盒子的关键！）
        if response_think:
            # 截断过长内容
            think_display = response_think
            if len(think_display) > 1500:
                think_display = think_display[:750] + f"\n... [dim](省略 {len(think_display) - 1500} 字符)[/dim] ...\n" + think_display[-750:]
            rows.append(f"\n[bold magenta]【思考过程】[/]\n{markup.escape(think_display)}")
        
        # 响应内容
        response_display = response_content
        if len(response_display) > 2000:
            response_display = response_display[:1000] + f"\n... [dim](省略 {len(response_display) - 2000} 字符)[/dim] ...\n" + response_display[-1000:]
        rows.append(f"\n[bold white]【输出内容】[/]\n{markup.escape(response_display)}")
        
        # 打印表格
        table = cls._generate_log_table(rows, "green")
        console.print(table)
    
    # ==================== 流式输出进度（无刷屏版本） ====================
    
    @classmethod
    def create_stream_live(cls) -> Live:
        """
        创建流式输出的 Live 实时显示对象
        
        使用方式:
            live = LogTable.create_stream_live()
            with live:
                # 更新进度...
                LogTable.update_stream_live(live, phase, chunk_count, think_len, reply_len)
        """
        console = cls.get_console()
        return Live(
            cls._build_stream_status("准备中", 0, 0, 0),
            console=console,
            refresh_per_second=4,  # 每秒刷新4次，避免闪烁
            transient=True,  # 完成后自动清除
        )
    
    @classmethod
    def _build_stream_status(cls, phase: str, chunk_count: int, think_len: int, reply_len: int) -> Text:
        """
        构建流式状态显示文本
        """
        # 根据阶段选择颜色和图标
        if phase == "思考中":
            icon = "🧠"
            color = "magenta"
        elif phase == "接收回复":
            icon = "📝"
            color = "cyan"
        elif phase == "完成":
            icon = "✓"
            color = "green"
        else:
            icon = "⏳"
            color = "yellow"
        
        # 构建状态文本
        status_text = Text()
        status_text.append(f"  {icon} ", style=f"bold {color}")
        status_text.append(f"[流式] ", style="dim")
        status_text.append(f"{phase}", style=f"bold {color}")
        status_text.append(f" | ", style="dim")
        status_text.append(f"数据块: ", style="dim")
        status_text.append(f"{chunk_count}", style="bold white")
        
        if think_len > 0:
            status_text.append(f" | ", style="dim")
            status_text.append(f"思考: ", style="dim")
            status_text.append(f"{think_len} 字", style="magenta")
        
        if reply_len > 0:
            status_text.append(f" | ", style="dim")
            status_text.append(f"回复: ", style="dim")
            status_text.append(f"{reply_len} 字", style="cyan")
        
        return status_text
    
    @classmethod
    def update_stream_live(
        cls,
        live: Live,
        phase: str,
        chunk_count: int,
        think_len: int = 0,
        reply_len: int = 0,
    ) -> None:
        """
        更新流式输出的实时进度（不刷屏）
        
        Args:
            live: Live 对象
            phase: 当前阶段 ("思考中", "接收回复", "完成")
            chunk_count: 已接收数据块数量
            think_len: 思考内容长度
            reply_len: 回复内容长度
        """
        live.update(cls._build_stream_status(phase, chunk_count, think_len, reply_len))
    
    @classmethod
    def print_stream_complete(
        cls,
        chunk_count: int,
        think_len: int,
        reply_len: int,
        elapsed: float = 0,
    ) -> None:
        """
        打印流式输出完成信息（单行，不刷屏）
        """
        time_info = f" | 耗时: {elapsed:.1f}s" if elapsed > 0 else ""
        LogHelper.info(
            f"[流式] 完成 | 数据块: {chunk_count} | "
            f"思考: {think_len} 字 | 回复: {reply_len} 字{time_info}"
        )
    
    # ==================== 重试信息 ====================
    
    @classmethod
    def print_retry_info(
        cls,
        word_surface: str,
        retry_count: int,
        max_retry: int,
        reason: str,
    ) -> None:
        """打印重试信息"""
        LogHelper.warning(f"[重试 {retry_count}/{max_retry}] {word_surface} - {reason}")


# ==================== 便捷函数（向后兼容） ====================

def print_llm_task(*args, **kwargs):
    """便捷函数：打印 LLM 任务日志"""
    LogTable.print_llm_task(*args, **kwargs)

def print_stage_header(*args, **kwargs):
    """便捷函数：打印阶段标题"""
    LogTable.print_stage_header(*args, **kwargs)

def print_batch_summary(*args, **kwargs):
    """便捷函数：打印批量汇总"""
    LogTable.print_batch_summary(*args, **kwargs)

def print_log_table(*args, **kwargs):
    """便捷函数：打印日志表格"""
    LogTable.print_log_table(*args, **kwargs)
