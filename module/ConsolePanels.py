from typing import Any

from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.live import Live

from module.LogHelper import LogHelper
from module.ConfigStore import load_raw, save_raw, get_value, set_value, platform_summary


def _build_status_panel(llm) -> Panel:
    status = llm.get_runtime_status() if hasattr(llm, "get_runtime_status") else {}
    stats = status.get("runtime_stats", {}) if isinstance(status, dict) else {}

    t = Table(show_header=False, box=None)
    t.add_column("k", style="cyan", ratio=2)
    t.add_column("v", style="white", ratio=6)
    t.add_row("限流模式", str(status.get("mode", "")))
    t.add_row("Key 数量", f"{status.get('available_keys', 0)}/{status.get('key_count', 0)} (可用/总)")
    t.add_row("黑名单", str(status.get("blacklisted_keys", 0)))
    if status.get("mode") == "rpm_default":
        t.add_row("每Key新增", f"{status.get('per_key_rpm', 0):.2f} 次/分钟")
        t.add_row("全局新增", f"{status.get('total_rpm', 0):.2f} 次/分钟")
    t.add_row("请求统计", f"发起 {stats.get('requests_started', 0)} / 成功 {stats.get('requests_succeeded', 0)} / 失败 {stats.get('requests_failed', 0)}")
    t.add_row("等待(并发)", f"{float(stats.get('semaphore_wait_seconds', 0.0)):.2f}s")
    t.add_row("等待(全局限流)", f"{float(stats.get('global_limiter_wait_seconds', 0.0)):.2f}s")
    t.add_row("等待(每Key限流)", f"{float(stats.get('per_key_limiter_wait_seconds', 0.0)):.2f}s")
    t.add_row("最近错误", f"{stats.get('last_error_kind', '')} | {stats.get('last_error', '')[:120]}")
    return Panel.fit(t, title="运行状态", border_style="bright_blue")


async def show_status_panel(llm) -> None:
    LogHelper.print(_build_status_panel(llm))


def show_status_live(llm, refresh_seconds: float = 0.8) -> None:
    try:
        refresh = max(0.2, float(refresh_seconds))
    except Exception:
        refresh = 0.8
    import time as _t
    with Live(_build_status_panel(llm), refresh_per_second=max(1, int(1.0 / refresh)), transient=True) as live:
        try:
            while True:
                live.update(_build_status_panel(llm))
                _t.sleep(refresh)
        except KeyboardInterrupt:
            return


def show_config_panel(raw: dict[str, Any], path: str) -> None:
    name, pid, key_count = platform_summary(raw)
    t = Table(show_header=False, box=None)
    t.add_column("k", style="cyan", ratio=2)
    t.add_column("v", style="white", ratio=6)
    t.add_row("配置文件", path)
    t.add_row("当前平台", f"{pid} | {name}")
    t.add_row("平台Key数", str(key_count))
    t.add_row("default启用", str(bool(get_value(raw, "multi_key_default_enable", True))))
    t.add_row("每Key每分钟", str(get_value(raw, "multi_key_default_per_key_rpm", 1)))
    t.add_row("黑名单TTL(s)", str(get_value(raw, "api_key_blacklist_ttl_seconds", 3600)))
    t.add_row("并发上限", str(get_value(raw, "max_concurrent_requests", 0)))
    t.add_row("RPS阈值", str(get_value(raw, "request_frequency_threshold", 1)))
    t.add_row("退避基准(s)", str(get_value(raw, "task_retry_backoff_base_seconds", 2)))
    t.add_row("退避上限(s)", str(get_value(raw, "task_retry_backoff_max_seconds", 120)))
    LogHelper.print(Panel.fit(t, title="配置面板", border_style="bright_blue"))


def interactive_config_edit() -> bool:
    path, raw = load_raw()
    show_config_panel(raw, path)

    if not Confirm.ask("进入交互式编辑？", default=False):
        return False

    platforms = raw.get("platforms", [])
    if isinstance(platforms, list) and platforms:
        options = []
        for p in platforms:
            options.append(f"{p.get('id')}|{p.get('name', 'Unknown')}")
        LogHelper.print(Panel("\n".join(options), title="可选平台 (id|name)", border_style="cyan"))
        pid = Prompt.ask("设置 activate_platform（留空不改）", default="")
        if pid.strip():
            set_value(raw, "activate_platform", int(pid))

    v = Prompt.ask("multi_key_default_enable (true/false, 留空不改)", default="")
    if v.strip().lower() in ("true", "false"):
        set_value(raw, "multi_key_default_enable", v.strip().lower() == "true")

    v = Prompt.ask("multi_key_default_per_key_rpm (留空不改)", default="")
    if v.strip():
        set_value(raw, "multi_key_default_per_key_rpm", float(v))

    v = Prompt.ask("api_key_blacklist_ttl_seconds (留空不改)", default="")
    if v.strip():
        set_value(raw, "api_key_blacklist_ttl_seconds", int(float(v)))

    v = Prompt.ask("max_concurrent_requests (<=0 表示不设上限, 留空不改)", default="")
    if v.strip():
        set_value(raw, "max_concurrent_requests", int(float(v)))

    v = Prompt.ask("request_frequency_threshold (RPS, 留空不改)", default="")
    if v.strip():
        set_value(raw, "request_frequency_threshold", float(v))

    v = Prompt.ask("task_retry_backoff_base_seconds (留空不改)", default="")
    if v.strip():
        set_value(raw, "task_retry_backoff_base_seconds", float(v))

    v = Prompt.ask("task_retry_backoff_max_seconds (留空不改)", default="")
    if v.strip():
        set_value(raw, "task_retry_backoff_max_seconds", float(v))

    show_config_panel(raw, path)
    if Confirm.ask("保存并写回配置文件？", default=True):
        save_raw(path, raw)
        LogHelper.print(Panel("已保存。返回主菜单后会自动重新加载配置。", border_style="green"))
        return True
    return False
