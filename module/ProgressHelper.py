"""
BookTerm Gacha - Progress Helper Module
========================================

This module provides a standardized progress bar configuration for
displaying task progress throughout the application.

Features:
    - Consistent progress bar styling across all operations
    - Shows completed/total items, elapsed time, and remaining time
    - Uses Rich library for beautiful terminal output

Usage:
    from module.ProgressHelper import ProgressHelper
    
    with ProgressHelper.get_progress() as progress:
        task = progress.add_task("Processing...", total=100)
        for i in range(100):
            # do work
            progress.update(task, advance=1)

Based on KeywordGacha v0.13.1 by neavo
https://github.com/neavo/KeywordGacha
"""

from rich.progress import Progress
from rich.progress import BarColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn

class ProgressHelper:

    # 获取一个进度条实例
    @staticmethod
    def get_progress(**kwargs) -> Progress:
        return Progress(
            TextColumn("{task.description}", justify = "right"),
            "•",
            BarColumn(bar_width = None),
            "•",
            TextColumn("{task.completed}/{task.total}", justify = "right"),
            "•",
            TimeElapsedColumn(),
            "/",
            TimeRemainingColumn(),
            **kwargs
        )