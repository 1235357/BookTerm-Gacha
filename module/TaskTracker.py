"""
BookTerm Gacha - Task Tracker Module
=====================================

全局任务状态追踪器，用于显示常驻的底部进度面板。

【Windows 兼容性修复】
- 在模块加载时启用 Windows VT100 转义序列支持
- 使用 Console(force_terminal=True, legacy_windows=False)
- 适当的刷新频率避免闪烁

功能：
1. 追踪并发任务的状态（等待中、思考中、接收回复、完成）
2. 统计成功/失败/重试次数
3. 显示实时进度条和详细统计信息
4. 原地更新，不刷屏

使用方式：
    tracker = TaskTracker(total=100, task_name="词义分析")
    with tracker:
        tracker.start_task(task_id, word_surface)
        tracker.update_task(task_id, "thinking")
        tracker.complete_task(task_id, success=True)
"""

import os
import sys
import time
import threading
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from rich import box
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.console import Console, Group
from rich.progress import (
    Progress,
    ProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
    SpinnerColumn,
)

from module.LogTable import LogTable


# ==================== 日志抑制控制 ====================
_suppress_logging: bool = False
_gui_hook: Optional[callable] = None


def is_logging_suppressed() -> bool:
    """检查是否应该抑制日志输出"""
    return _suppress_logging


def set_logging_suppressed(value: bool) -> None:
    """设置日志抑制状态"""
    global _suppress_logging
    _suppress_logging = value


def set_gui_hook(hook: Optional[callable]) -> None:
    global _gui_hook
    _gui_hook = hook


class TaskStatus(Enum):
    """任务状态枚举"""
    WAITING = "waiting"
    SENDING = "sending"
    THINKING = "thinking"
    RECEIVING = "receiving"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskState:
    """单个任务的状态"""
    task_id: str
    word_surface: str = ""
    status: TaskStatus = TaskStatus.WAITING
    start_time: float = 0
    end_time: float = 0
    think_chars: int = 0
    reply_chars: int = 0
    chunks: int = 0
    error: Optional[str] = None
    retry_count: int = 0


class PhaseBarColumn(ProgressColumn):
    def __init__(self, tracker: "TaskTracker", bar_width: int = 40):
        super().__init__()
        self.tracker = tracker
        self.bar_width = int(bar_width or 0) if isinstance(bar_width, int) else 40
        if self.bar_width <= 0:
            self.bar_width = 40

    def render(self, task) -> Text:
        width = self.bar_width
        with self.tracker._lock:
            t_total = int(getattr(self.tracker, "translate_total", 0) or 0)
            r_expected = int(getattr(self.tracker, "review_expected_total", 0) or 0)
            t_done = int(getattr(self.tracker, "translate_completed", 0) or 0)
            r_done = int(getattr(self.tracker, "review_completed", 0) or 0)

        if t_total > 0:
            r_total = min(r_expected, max(0, t_done))
        else:
            r_total = r_expected

        total = max(1, int(t_total) + int(r_total))
        left = int(round(width * (int(t_total) / total))) if total > 0 else 0
        left = max(0, min(width, left))
        right = max(0, width - left)

        t_ratio = max(0.0, min(1.0, t_done / max(1, t_total))) if t_total > 0 else 0.0
        r_ratio = max(0.0, min(1.0, r_done / max(1, r_total))) if r_total > 0 else 0.0

        t_filled = int(round(left * t_ratio))
        r_filled = int(round(right * r_ratio))

        bar = Text()
        if left > 0:
            if t_filled > 0:
                bar.append("━" * t_filled, style="cyan")
            if left - t_filled > 0:
                bar.append("─" * (left - t_filled), style="grey37")
        if right > 0:
            if r_filled > 0:
                bar.append("━" * r_filled, style="magenta")
            if right - r_filled > 0:
                bar.append("─" * (right - r_filled), style="grey37")
        return bar


class TaskTracker:
    """
    全局任务追踪器
    
    【Windows 兼容性】
    - Console(force_terminal=True, legacy_windows=False)
    - 启用 VT100 转义序列支持
    - 不要过于频繁地调用 update()
    """
    
    def __init__(
        self,
        total: int,
        task_name: str = "任务",
        max_concurrent: int = 5,
        translate_total: int = 0,
        review_total: int = 0,
    ):
        self.total = total
        self.task_name = task_name
        self.max_concurrent = max_concurrent

        self.translate_total = int(translate_total or 0)
        self.review_expected_total = int(review_total or 0)
        self.translate_completed = 0
        self.review_completed = 0
        self._stable_phase: Dict[str, int] = {}
        
        # 核心计数
        self.success_count = 0
        self.failed_in_round = 0
        self.retry_round = 0
        
        # 任务状态映射
        self._tasks: Dict[str, TaskState] = {}
        self._lock = threading.Lock()
        
        # 响应时间统计
        self._response_times: List[float] = []
        self._failed_reasons: Dict[str, int] = defaultdict(int)
        self._finalized_task_ids: set[str] = set()
        
        # 时间追踪
        self.start_time = time.time()
        
        # 【关键】使用全局统一的 Console 实例（来自 LogTable）
        # 这样 LogTable 的输出才能正确被 Live 下文管理器捕获和处理
        self._console = LogTable.get_console()
        self._live: Optional[Live] = None
        
        # 创建内部进度条
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            PhaseBarColumn(self, bar_width=40),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("/"),
            TimeRemainingColumn(),
            console=self._console,
            expand=False,
        )
        self._progress_task = None

    def _get_stable_task_id(self, task_id: str) -> str:
        tid = str(task_id or "")
        if "." in tid:
            return tid.split(".", 1)[0]
        return tid

    def mark_translated(self, task_id: str) -> None:
        stable = self._get_stable_task_id(task_id)
        with self._lock:
            stage = int(self._stable_phase.get(stable, 0) or 0)
            if stage >= 1:
                return
            self._stable_phase[stable] = 1
            if self.translate_total > 0:
                self.translate_completed = min(self.translate_total, self.translate_completed + 1)
        self._refresh()

    def mark_reviewed(self, task_id: str) -> None:
        stable = self._get_stable_task_id(task_id)
        with self._lock:
            stage = int(self._stable_phase.get(stable, 0) or 0)
            if stage >= 2:
                return
            if self.translate_total > 0 and stage < 1:
                self._stable_phase[stable] = 1
                self.translate_completed = min(self.translate_total, self.translate_completed + 1)
            self._stable_phase[stable] = 2
            if self.review_expected_total > 0:
                self.review_completed = min(self.review_expected_total, self.review_completed + 1)
        self._refresh()
    
    def __enter__(self):
        """进入上下文：启动 Live 显示"""
        self._progress_task = self._progress.add_task(
            f"[cyan]{self.task_name}",
            total=self.total
        )
        
        # 【关键】Live 配置
        # - refresh_per_second=2: 降低刷新频率减少闪烁
        # - screen=False: 不使用全屏模式
        # - transient=False: 完成后保留
        # - redirect_stdout=True: 重定向标准输出，让 print 正常工作
        # - redirect_stderr=True: 重定向标准错误
        self._live = Live(
            self._build_panel(),
            console=self._console,
            refresh_per_second=2,  # 降低刷新频率
            transient=False,
            screen=False,
            redirect_stdout=True,
            redirect_stderr=True,
        )
        self._live.__enter__()
        
        # 【关键修复】将 LogHelper 的 Console 输出重定向到 Live 的代理流
        # 当 Live(redirect_stdout=True) 激活时，sys.stdout 会被替换为 FileProxy。
        # 但 LogHelper._console 仍持有原始的 stdout 文件句柄，导致日志通过 Console 打印时绕过了 Live 的控制，
        # 从而破坏了 Live 的光标位置管理（出现刷屏）。
        # 这里我们将 Console 的内部文件句柄临时指向当前的 sys.stdout (即 Live 代理)，确保同步。
        if hasattr(self._console, "file"):
            self._original_console_file = self._console.file
            self._console.file = sys.stdout
            
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文：停止 Live 显示"""
        # 恢复 Console 的原始文件句柄
        if hasattr(self, "_original_console_file") and self._console:
            self._console.file = self._original_console_file

        if self._live:
            # 最终更新一次
            self._live.update(self._build_panel())
            self._live.__exit__(exc_type, exc_val, exc_tb)
        return False
    
    def _build_panel(self) -> Group:
        """
        构建紧凑版面板（移除 Panel 边框，改为 Group 组合）
        
        【Windows 兼容性修复】
        将所有信息压缩到 1-2 行，移除 Panel 边框，减少垂直高度，
        从而大幅降低控制台光标回退的难度，避免刷屏。
        """
        # 统计各状态数量
        status_counts = {status: 0 for status in TaskStatus}
        total_think_chars = 0
        total_reply_chars = 0
        total_chunks = 0
        
        with self._lock:
            for task in self._tasks.values():
                status_counts[task.status] += 1
                total_think_chars += task.think_chars
                total_reply_chars += task.reply_chars
                total_chunks += task.chunks
        
        # 计算活跃任务数
        active_count = (
            status_counts[TaskStatus.SENDING] +
            status_counts[TaskStatus.THINKING] +
            status_counts[TaskStatus.RECEIVING]
        )
        
        # 计算待处理数
        pending_count = max(0, self.total - self.success_count)
        
        # 计算平均响应时间
        avg_time = 0.0
        if self._response_times:
            avg_time = sum(self._response_times) / len(self._response_times)
        
        # === 紧凑行：统计信息合并 ===
        # 格式: 📊 3/5 (发:1 思:2) │ 📈 17/30 (待:13 败:0) │ ⏱️ 1.2s │ 块: 123 (思:45 复:78)
        
        line_info = Text()
        
        # 1. 并发部分
        line_info.append("📊 ", style="bold")
        line_info.append(f"{active_count}/{self.max_concurrent}", style="bold cyan")
        
        details = []
        if status_counts[TaskStatus.SENDING] > 0:
            details.append(f"发:{status_counts[TaskStatus.SENDING]}")
        if status_counts[TaskStatus.THINKING] > 0:
            details.append(f"思:{status_counts[TaskStatus.THINKING]}")
        if status_counts[TaskStatus.RECEIVING] > 0:
            details.append(f"收:{status_counts[TaskStatus.RECEIVING]}")
            
        if details:
            line_info.append(f" ({' '.join(details)})", style="dim")
            
        line_info.append(" │ ", style="dim")
        
        # 2. 进度部分
        line_info.append("📈 ", style="bold")
        display_success = min(self.success_count, self.total)
        line_info.append(f"{display_success}/{self.total}", style="bold green")
        
        prog_details = []
        if pending_count > 0:
            prog_details.append(f"待:{pending_count}")
        if self.failed_in_round > 0:
            prog_details.append(f"败:{self.failed_in_round}")
        if self.retry_round > 0:
            prog_details.append(f"轮:{self.retry_round}")
            
        if prog_details:
            line_info.append(f" ({' '.join(prog_details)})", style="dim")

        if self.translate_total > 0 or self.review_expected_total > 0:
            with self._lock:
                t_total = int(self.translate_total or 0)
                r_expected = int(self.review_expected_total or 0)
                t_done = int(self.translate_completed or 0)
                r_done = int(self.review_completed or 0)
            if t_total > 0:
                line_info.append(" │ ", style="dim")
                line_info.append(f"译:{min(t_done, t_total)}/{t_total}", style="cyan")
                line_info.append(" ", style="dim")
                line_info.append(f"校:{min(r_done, r_expected)}/{r_expected}", style="magenta")
            
        line_info.append(" │ ", style="dim")
        
        # 3. 耗时部分
        line_info.append("⏱️ ", style="bold")
        if avg_time > 0:
            color = "green" if avg_time < 60 else "yellow"
            line_info.append(f"{avg_time:.1f}s", style=f"bold {color}")
        else:
            line_info.append("--", style="dim")
            
        # 4. 失败原因（如果有）- 放到同一行末尾或第二行
        # 为了极度紧凑，我们尽量放在同一行，如果太长再换行
        # 这里先只显示流式统计
        if total_chunks > 0:
            line_info.append(" │ ", style="dim")
            line_info.append(f"块:{total_chunks}", style="dim")
        
        # 组合：只有两部分 [进度条, 统计行]
        # 移除 Panel 包装，直接返回 Group
        
        items = [self._progress, line_info]
        
        if self._failed_reasons:
             # 如果有失败原因，简要显示在第三行
            line_err = Text("❌ ", style="bold red")
            reasons = sorted(self._failed_reasons.items(), key=lambda x: -x[1])[:1] # 只显示 top 1
            for r, c in reasons:
                line_err.append(f"{r}({c}) ", style="red")
            items.append(line_err)
            
        return Group(*items)
    
    def start_task(self, task_id: str, word_surface: str = "") -> None:
        """开始一个任务"""
        with self._lock:
            self._tasks[task_id] = TaskState(
                task_id=task_id,
                word_surface=word_surface,
                status=TaskStatus.SENDING,
                start_time=time.time(),
            )
        self._refresh()
    
    def update_task(
        self,
        task_id: str,
        status: str,
        think_chars: int = 0,
        reply_chars: int = 0,
        chunks: int = 0,
    ) -> None:
        """更新任务状态"""
        status_map = {
            "waiting": TaskStatus.WAITING,
            "sending": TaskStatus.SENDING,
            "thinking": TaskStatus.THINKING,
            "receiving": TaskStatus.RECEIVING,
        }
        
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                if status in status_map:
                    task.status = status_map[status]
                task.think_chars = think_chars
                task.reply_chars = reply_chars
                task.chunks = chunks
        self._refresh()
    
    def complete_task(self, task_id: str, success: bool = True, error: Optional[str] = None) -> None:
        """完成一个任务"""
        with self._lock:
            if task_id in self._finalized_task_ids:
                return

            task = self._tasks.get(task_id)
            if task is None:
                task = TaskState(
                    task_id=task_id,
                    status=TaskStatus.WAITING,
                    start_time=time.time(),
                )
                self._tasks[task_id] = task

            elapsed = 0
            if task:
                task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
                task.error = error
                task.end_time = time.time()
                elapsed = task.end_time - task.start_time
            
            if success:
                self.success_count += 1
                if elapsed > 0:
                    self._response_times.append(elapsed)
                if self.review_expected_total > 0:
                    stable = self._get_stable_task_id(task_id)
                    stage = int(self._stable_phase.get(stable, 0) or 0)
                    if stage < 2:
                        if self.translate_total > 0 and stage < 1:
                            self._stable_phase[stable] = 1
                            self.translate_completed = min(self.translate_total, self.translate_completed + 1)
                        self._stable_phase[stable] = 2
                        self.review_completed = min(self.review_expected_total, self.review_completed + 1)
                elif self.translate_total > 0:
                    stable = self._get_stable_task_id(task_id)
                    stage = int(self._stable_phase.get(stable, 0) or 0)
                    if stage < 1:
                        self._stable_phase[stable] = 1
                        self.translate_completed = min(self.translate_total, self.translate_completed + 1)
            else:
                self.failed_in_round += 1
                if error:
                    short_error = self._simplify_error(error)
                    self._failed_reasons[short_error] += 1

            self._finalized_task_ids.add(task_id)
        
        # 更新进度条
        if success and self._progress_task is not None:
            self._progress.update(self._progress_task, completed=self.success_count)
        self._refresh()

    def reopen_task(self, task_id: str) -> None:
        """允许同一 task_id 在失败后被再次执行并再次 complete（用于滚动重试）"""
        with self._lock:
            self._finalized_task_ids.discard(task_id)
    
    def _simplify_error(self, error: str) -> str:
        """简化错误信息"""
        error = str(error)
        
        if "超时" in error or "timeout" in error.lower():
            return "超时"
        if "流式响应超时" in error or "stalled" in error.lower():
            return "流式卡住"
        if "假名残留" in error:
            return "假名残留"
        if "韩文残留" in error:
            return "韩文残留"
        if "模型退化" in error:
            return "模型退化"
        if "翻译失效" in error or "相似度" in error:
            return "翻译失效"
        if "行数不一致" in error or "行数错误" in error:
            return "行数错误"
        if "JSON" in error or "解析失败" in error:
            return "解析失败"
        if "实体类型" in error:
            return "类型不匹配"
        if "敏感内容" in error or "contentFilter" in error:
            return "敏感内容"
        if "数据结构" in error:
            return "数据结构错误"
        if "429" in error:
            return "并发限制(429)"
        if "403" in error or "401" in error or "PermissionDenied" in error:
            return "权限错误"
        if "连接" in error or "connect" in error.lower():
            return "网络连接"
        
        return error[:15] if len(error) > 15 else error
    
    def start_retry_round(self) -> None:
        """开始新的重试轮次"""
        with self._lock:
            self.retry_round += 1
            self.failed_in_round = 0
            self._failed_reasons.clear()
            self._tasks = {k: v for k, v in self._tasks.items() 
                          if v.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED)}
        self._refresh()
    
    def add_retry(self) -> None:
        """增加重试计数（兼容旧接口）"""
        self.start_retry_round()
    
    def set_description(self, description: str) -> None:
        """设置进度条描述"""
        if self._progress_task is not None:
            self._progress.update(self._progress_task, description=description)
        self._refresh()
    
    def _refresh(self) -> None:
        """刷新显示"""
        hook = _gui_hook
        if callable(hook):
            try:
                with self._lock:
                    snapshot = {
                        "task_name": self.task_name,
                        "total": int(self.total or 0),
                        "max_concurrent": int(self.max_concurrent or 0),
                        "translate_total": int(getattr(self, "translate_total", 0) or 0),
                        "review_total": int(getattr(self, "review_expected_total", 0) or 0),
                        "translate_completed": int(getattr(self, "translate_completed", 0) or 0),
                        "review_completed": int(getattr(self, "review_completed", 0) or 0),
                        "success": int(getattr(self, "success_count", 0) or 0),
                        "failed_in_round": int(getattr(self, "failed_in_round", 0) or 0),
                        "retry_round": int(getattr(self, "retry_round", 0) or 0),
                        "elapsed_seconds": max(0.0, float(time.time() - float(getattr(self, "start_time", time.time()) or time.time()))),
                    }
                hook(snapshot)
            except Exception:
                pass
        if self._live:
            self._live.update(self._build_panel())
    
    def remove_task(self, task_id: str) -> None:
        """移除任务"""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            avg_time = sum(self._response_times) / len(self._response_times) if self._response_times else 0
            return {
                "total": self.total,
                "success": self.success_count,
                "pending": self.total - self.success_count,
                "failed_in_round": self.failed_in_round,
                "retry_round": self.retry_round,
                "avg_response_time": avg_time,
                "failed_reasons": dict(self._failed_reasons),
            }


# ==================== 全局 Tracker 管理 ====================
_current_tracker: Optional[TaskTracker] = None


def get_current_tracker() -> Optional[TaskTracker]:
    """获取当前活跃的 tracker"""
    return _current_tracker


def set_current_tracker(tracker: Optional[TaskTracker]) -> None:
    """设置当前活跃的 tracker"""
    global _current_tracker
    _current_tracker = tracker
