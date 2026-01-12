"""
BookTerm Gacha - Task Tracker Module
=====================================

全局任务状态追踪器，用于显示常驻的底部进度面板。

功能：
1. 追踪并发任务的状态（等待中、思考中、接收回复、完成）
2. 统计成功/失败/重试次数
3. 显示实时进度条和统计信息
4. 不闪烁的常驻底部区域

使用方式：
    tracker = TaskTracker(total=100, task_name="词义分析")
    with tracker:
        # 执行任务...
        tracker.update_task(task_id, "thinking")
        tracker.complete_task(task_id, success=True)

Based on Rich Live + Table for persistent bottom panel
"""

import time
import threading
from typing import Optional, Dict, Set
from dataclasses import dataclass, field
from enum import Enum

from rich import box
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.console import Console, Group
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn, SpinnerColumn


class TaskStatus(Enum):
    """任务状态枚举"""
    WAITING = "waiting"      # 等待中
    SENDING = "sending"      # 发送请求中
    THINKING = "thinking"    # 模型思考中
    RECEIVING = "receiving"  # 接收回复中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 已失败


@dataclass
class TaskState:
    """单个任务的状态"""
    task_id: str
    status: TaskStatus = TaskStatus.WAITING
    start_time: float = 0
    think_chars: int = 0
    reply_chars: int = 0
    chunks: int = 0
    error: Optional[str] = None


class TaskTracker:
    """
    全局任务追踪器
    
    显示常驻的底部区域，包含：
    - 总进度条
    - 当前并发状态统计
    - 成功/失败/重试计数
    """
    
    def __init__(
        self,
        total: int,
        task_name: str = "任务",
        max_concurrent: int = 5,
    ):
        self.total = total
        self.task_name = task_name
        self.max_concurrent = max_concurrent
        
        # 状态计数
        self.completed = 0
        self.success_count = 0
        self.failed_count = 0
        self.retry_count = 0
        
        # 任务状态映射
        self._tasks: Dict[str, TaskState] = {}
        self._lock = threading.Lock()
        
        # 时间追踪
        self.start_time = time.time()
        
        # Rich 组件
        self._console = Console()
        self._live: Optional[Live] = None
        
        # 创建内部进度条
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("/"),
            TimeRemainingColumn(),
            console=self._console,
            expand=False,
        )
        self._progress_task = None
    
    def __enter__(self):
        """进入上下文：启动 Live 显示"""
        self._progress_task = self._progress.add_task(
            f"[cyan]{self.task_name}",
            total=self.total
        )
        self._live = Live(
            self._build_panel(),
            console=self._console,
            refresh_per_second=4,
            transient=False,  # 完成后保留
        )
        self._live.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文：停止 Live 显示"""
        if self._live:
            # 最终更新一次
            self._live.update(self._build_panel())
            self._live.__exit__(exc_type, exc_val, exc_tb)
        return False
    
    def _build_panel(self) -> Panel:
        """构建常驻的底部面板"""
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
        
        # 构建状态行
        status_line = Text()
        status_line.append("  📊 ", style="bold")
        status_line.append("并发: ", style="dim")
        status_line.append(f"{active_count}/{self.max_concurrent}", style="bold cyan")
        status_line.append(" │ ", style="dim")
        
        # 各阶段统计
        if status_counts[TaskStatus.SENDING] > 0:
            status_line.append("🚀", style="yellow")
            status_line.append(f"{status_counts[TaskStatus.SENDING]} ", style="yellow")
        if status_counts[TaskStatus.THINKING] > 0:
            status_line.append("🧠", style="magenta")
            status_line.append(f"{status_counts[TaskStatus.THINKING]} ", style="magenta")
        if status_counts[TaskStatus.RECEIVING] > 0:
            status_line.append("📝", style="cyan")
            status_line.append(f"{status_counts[TaskStatus.RECEIVING]} ", style="cyan")
        
        status_line.append(" │ ", style="dim")
        status_line.append("✓", style="green")
        status_line.append(f"{self.success_count} ", style="green")
        status_line.append("✗", style="red")
        status_line.append(f"{self.failed_count} ", style="red")
        
        if self.retry_count > 0:
            status_line.append("↻", style="yellow")
            status_line.append(f"{self.retry_count} ", style="yellow")
        
        # 流式统计
        if total_chunks > 0:
            status_line.append(" │ ", style="dim")
            status_line.append("块:", style="dim")
            status_line.append(f"{total_chunks} ", style="white")
            if total_think_chars > 0:
                status_line.append("思:", style="dim")
                status_line.append(f"{total_think_chars} ", style="magenta")
            if total_reply_chars > 0:
                status_line.append("复:", style="dim")
                status_line.append(f"{total_reply_chars} ", style="cyan")
        
        # 组合进度条和状态行
        content = Group(
            self._progress,
            status_line,
        )
        
        return Panel(
            content,
            title=f"[bold]{self.task_name}[/]",
            border_style="blue",
            padding=(0, 1),
        )
    
    def start_task(self, task_id: str) -> None:
        """开始一个任务"""
        with self._lock:
            self._tasks[task_id] = TaskState(
                task_id=task_id,
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
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
                task.error = error
            
            self.completed += 1
            if success:
                self.success_count += 1
            else:
                self.failed_count += 1
        
        # 更新进度条
        if self._progress_task is not None:
            self._progress.update(self._progress_task, completed=self.completed)
        self._refresh()
    
    def add_retry(self) -> None:
        """增加重试计数"""
        with self._lock:
            self.retry_count += 1
        self._refresh()
    
    def set_description(self, description: str) -> None:
        """设置进度条描述"""
        if self._progress_task is not None:
            self._progress.update(self._progress_task, description=description)
        self._refresh()
    
    def _refresh(self) -> None:
        """刷新显示"""
        if self._live:
            self._live.update(self._build_panel())
    
    def remove_task(self, task_id: str) -> None:
        """移除任务（用于清理已完成的任务）"""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]


# 全局 tracker 实例（用于流式请求更新）
_current_tracker: Optional[TaskTracker] = None


def get_current_tracker() -> Optional[TaskTracker]:
    """获取当前活跃的 tracker"""
    return _current_tracker


def set_current_tracker(tracker: Optional[TaskTracker]) -> None:
    """设置当前活跃的 tracker"""
    global _current_tracker
    _current_tracker = tracker
