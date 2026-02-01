"""
BookTerm Gacha - LLM Agent Module (重构版)
==========================================

参照兄弟项目 Dev-Experimental/module/Engine/TaskRequester.py 完全重写。

核心改动：
1. 多平台检测和优化（智谱、阿里云百炼 DeepSeek、OpenAI O系列等）
2. 【强制启用深度思考模式】- 不区分专家模式
3. 智能请求头构建（User-Agent、extra_body 等）
4. 流式输出 + 深度思考的完整支持
5. 完整的日志输出（打开黑盒子）

支持的平台：
    - 智谱 API (bigmodel.cn) - 深度思考 + 流式输出
    - 阿里云百炼 DeepSeek (api-inference.modelscope.cn) - 流式输出
    - OpenAI O系列 (o1, o3-mini, o4-mini) - max_completion_tokens
    - QWEN3 - /no_think 控制
    - 标准 OpenAI 兼容 API

Based on LinguaGacha's TaskRequester.py
"""

import re
import asyncio
import threading
import urllib.request
import time
import random
from functools import lru_cache
from types import SimpleNamespace
from typing import Optional

import httpx
import pykakasi
import json_repair as repair
from openai import AsyncOpenAI
from aiolimiter import AsyncLimiter
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn

from base.BaseData import BaseData
from model.NER import NER
from model.Word import Word
from module.Text.TextHelper import TextHelper
from module.LogHelper import LogHelper
from module.LogTable import LogTable
from module.TaskTracker import TaskTracker, get_current_tracker, set_current_tracker
from module.ErrorLogger import ErrorLogger


class LLM:
    """
    LLM Agent 模块 - BookTerm Gacha 的核心 AI 组件
    
    参照 Dev-Experimental/TaskRequester.py 重构
    """

    # ==================== 平台检测正则（借鉴 TaskRequester.py） ====================
    
    # 智谱 API 检测
    RE_ZHIPU: re.Pattern = re.compile(r"bigmodel\.cn|zhipu", flags=re.IGNORECASE)
    
    # 阿里云百炼 DeepSeek 模型检测
    RE_DASHSCOPE_DEEPSEEK_URL: re.Pattern = re.compile(r"api-inference\.modelscope\.cn", flags=re.IGNORECASE)
    RE_DASHSCOPE_DEEPSEEK_MODEL: re.Pattern = re.compile(r"deepseek-ai", flags=re.IGNORECASE)
    
    # NVIDIA Build DeepSeek 模型检测
    RE_NVIDIA_DEEPSEEK_URL: re.Pattern = re.compile(r"integrate\.api\.nvidia\.com", flags=re.IGNORECASE)
    RE_NVIDIA_DEEPSEEK_MODEL: re.Pattern = re.compile(r"deepseek-ai", flags=re.IGNORECASE)
    
    # OpenAI O系列模型 (o1, o3-mini, o4-mini-20240406 等)
    RE_O_SERIES: re.Pattern = re.compile(r"o\d$|o\d-", flags=re.IGNORECASE)
    
    # QWEN3 系列
    RE_QWEN3: re.Pattern = re.compile(r"qwen3", flags=re.IGNORECASE)
    
    # Gemini 2.5 Flash
    RE_GEMINI_2_5_FLASH: re.Pattern = re.compile(r"gemini-2\.5-flash", flags=re.IGNORECASE)
    
    # 多行压缩
    RE_LINE_BREAK: re.Pattern = re.compile(r"\n+")
    
    # 黑名单错误关键词（API Key 被封禁）
    RE_BLACKLIST_ERROR: re.Pattern = re.compile(r"blacklist|banned|suspended|prohibited", flags=re.IGNORECASE)

    # ==================== 任务类型 ====================

    class Type(BaseData):
        API_TEST: int = 100                  # 接口测试
        SURFACE_ANALYSIS: int = 200          # 语义分析
        TRANSLATE_CONTEXT: int = 300         # 翻译参考文本

    # ==================== LLM 配置参数 ====================

    class LLMConfig(BaseData):
        TEMPERATURE: float = 0.05
        TOP_P: float = 0.95
        MAX_TOKENS: float = 1024
        FREQUENCY_PENALTY: float = 0.0

    # 最大重试次数
    MAX_RETRY: int = 16
    
    # 强制音译阈值
    FORCE_TRANSLITERATE_THRESHOLD: int = 16
    
    # 单任务超时阈值（秒）- 超过此时间触发上下文限缩
    TASK_TIMEOUT_THRESHOLD: int = 430

    # ==================== 多 API Key 轮询管理（借鉴 TaskRequester.py） ====================
    
    # 类级别变量 - API Key 轮询索引
    API_KEY_INDEX: int = 0
    
    # 类级别变量 - API Key 黑名单（存储被封禁的 Key）
    BLACKLISTED_KEYS: dict[str, float] = {}
    
    # 类线程锁
    KEY_LOCK: threading.Lock = threading.Lock()

    # 类型映射表
    GROUP_MAPPING = {
        "角色" : ["姓氏", "名字"],
        "地点" : ["地点", "建筑", "设施"],
    }
    GROUP_MAPPING_BANNED = {
        "黑名单" : [
            "行为", "活动", "其他", "无法判断",
            "组织", "群体", "家族", "种族", "团体", "部门", "公司", "学校", "部活", "社团",
            "物品", "食品", "工具", "道具", "食物", "饮品", "饮料", "衣物", "服装", "家具", "电器", "器具",
            "生物", "植物", "动物", "怪物", "魔物", "妖怪", "精灵", "魔兽",
        ],
    }
    GROUP_MAPPING_ADDITIONAL = {
        "角色" : ["角色", "人", "人物", "人名"],
        "地点" : [],
    }

    # ==================== API Key 轮询管理（借鉴 TaskRequester.py）====================
    
    @classmethod
    def reset_api_state(cls) -> None:
        """重置 API 状态（每次任务开始前调用）"""
        cls.API_KEY_INDEX = 0
        cls.BLACKLISTED_KEYS.clear()
        LogHelper.debug("[API状态] 已重置 Key 索引和黑名单")
    
    @classmethod
    def add_to_blacklist(cls, key: str, ttl_seconds: float | None = None) -> None:
        with cls.KEY_LOCK:
            ttl = 3600.0 if ttl_seconds is None else float(ttl_seconds)
            ttl = max(1.0, ttl)
            cls.BLACKLISTED_KEYS[key] = time.time() + ttl
            LogHelper.warning(f"[API黑名单] Key 已被加入黑名单: {key[:20]}... (ttl={int(ttl)}s)")
    
    @classmethod
    def is_blacklisted(cls, key: str) -> bool:
        """检查 Key 是否在黑名单中"""
        with cls.KEY_LOCK:
            exp = cls.BLACKLISTED_KEYS.get(key)
            if exp is None:
                return False
            if time.time() >= exp:
                try:
                    del cls.BLACKLISTED_KEYS[key]
                except Exception:
                    pass
                return False
            return True
    
    @classmethod
    def get_available_key_count(cls, keys: list[str]) -> int:
        """获取可用 Key 数量"""
        with cls.KEY_LOCK:
            now = time.time()
            expired = [k for k, exp in cls.BLACKLISTED_KEYS.items() if now >= exp]
            for k in expired:
                try:
                    del cls.BLACKLISTED_KEYS[k]
                except Exception:
                    pass
            return sum(1 for k in keys if k not in cls.BLACKLISTED_KEYS)

    @classmethod
    def get_blacklisted_key_count(cls) -> int:
        with cls.KEY_LOCK:
            now = time.time()
            expired = [k for k, exp in cls.BLACKLISTED_KEYS.items() if now >= exp]
            for k in expired:
                try:
                    del cls.BLACKLISTED_KEYS[k]
                except Exception:
                    pass
            return len(cls.BLACKLISTED_KEYS)
    
    @classmethod
    def get_key(cls, keys: list[str]) -> str:
        """
        获取下一个可用的 API Key（轮询机制）
        
        参照 TaskRequester.py 的 get_key 方法
        - 支持单个 Key 或多个 Key 列表
        - 自动跳过被封禁的 Key
        - 轮询可用 Key
        """
        key: str = ""
        
        # 兼容：如果传入的是字符串而非列表
        if isinstance(keys, str):
            keys = [keys]
        
        if len(keys) == 0:
            key = "no_key_required"
        elif len(keys) == 1:
            key = keys[0]
            # 检查单个 Key 是否被封禁
            if cls.is_blacklisted(key):
                raise RuntimeError(f"所有 API Key 均已被封禁，无法继续执行任务。请检查 API Key 状态或联系服务提供商。")
        else:
            # 尝试找到一个未被封禁的 Key
            available_keys = [k for k in keys if not cls.is_blacklisted(k)]
            if len(available_keys) == 0:
                raise RuntimeError(f"所有 API Key 均已被封禁，无法继续执行任务。请检查 API Key 状态或联系服务提供商。")
            
            # 在可用 Key 中轮询
            with cls.KEY_LOCK:
                key = available_keys[cls.API_KEY_INDEX % len(available_keys)]
                cls.API_KEY_INDEX = (cls.API_KEY_INDEX + 1) % len(available_keys)
        
        return key

    # ==================== 初始化 ====================

    def __init__(self, config: SimpleNamespace) -> None:
        self.config = config
        # 支持多 API Key（兼容旧配置）
        api_key = config.api_key
        if isinstance(api_key, list):
            self.api_keys = api_key  # 多 Key 列表
            self.api_key = api_key[0] if api_key else ""  # 向后兼容
        else:
            self.api_keys = [api_key] if api_key else []  # 单 Key 转列表
            self.api_key = api_key
        
        self.base_url = config.base_url
        self.model_name = config.model_name
        self.request_timeout = config.request_timeout
        self.request_frequency_threshold = config.request_frequency_threshold
        # 最大并发请求数（用于控制并发上限，解决 429 问题）
        self.max_concurrent_requests = getattr(config, 'max_concurrent_requests', 5)
        first_chunk_timeout_seconds = int(getattr(config, "stream_first_chunk_timeout_seconds", 600) or 600)
        stall_timeout_seconds = int(getattr(config, "stream_stall_timeout_seconds", 120) or 120)
        self.stream_first_chunk_timeout_seconds = max(1, first_chunk_timeout_seconds)
        self.stream_stall_timeout_seconds = max(1, stall_timeout_seconds)
        self.stream_retry_attempts = max(1, int(getattr(config, "stream_retry_attempts", 3) or 3))
        self.stream_retry_backoff_seconds = max(0, int(getattr(config, "stream_retry_backoff_seconds", 2) or 2))

        # 初始化工具
        self.kakasi = pykakasi.kakasi()
        
        # 客户端缓存（key -> client）
        self._clients: dict[str, AsyncOpenAI] = {}
        self._stream_clients: dict[str, AsyncOpenAI] = {}
        
        # 创建初始客户端（向后兼容）
        self.client = self._create_client()
        self.stream_client = self._create_client_no_timeout()

        # 线程锁
        self.lock = threading.Lock()
        
        # 平台检测缓存
        self._platform_info: Optional[dict] = None
        
        # 打印多 Key 信息
        if len(self.api_keys) > 1:
            LogHelper.info(f"[多API轮询] 检测到 {len(self.api_keys)} 个 API Key，已启用轮询机制")
        
        self._multi_key_per_key_limiter_enable = False
        self._per_key_limiters: dict[str, AsyncLimiter] = {}
        self._per_key_limiter_rpm: float = 1.0
        self._key_last_used_at: dict[str, float] = {}
        self.runtime_stats = {
            "requests_started": 0,
            "requests_succeeded": 0,
            "requests_failed": 0,
            "semaphore_wait_seconds": 0.0,
            "global_limiter_wait_seconds": 0.0,
            "per_key_limiter_wait_seconds": 0.0,
            "errors_by_kind": {},
            "last_error": "",
            "last_error_kind": "",
        }

    def apply_runtime_config(self, config: SimpleNamespace) -> None:
        self.config = config
        api_key = config.api_key
        if isinstance(api_key, list):
            self.api_keys = api_key
            self.api_key = api_key[0] if api_key else ""
        else:
            self.api_keys = [api_key] if api_key else []
            self.api_key = api_key or ""

        self.base_url = config.base_url
        self.model_name = config.model_name
        self.request_timeout = config.request_timeout
        self.request_frequency_threshold = config.request_frequency_threshold
        self.max_concurrent_requests = getattr(config, "max_concurrent_requests", 0)

        first_chunk_timeout_seconds = int(getattr(config, "stream_first_chunk_timeout_seconds", 600) or 600)
        stall_timeout_seconds = int(getattr(config, "stream_stall_timeout_seconds", 120) or 120)
        self.stream_first_chunk_timeout_seconds = max(1, first_chunk_timeout_seconds)
        self.stream_stall_timeout_seconds = max(1, stall_timeout_seconds)
        self.stream_retry_attempts = max(1, int(getattr(config, "stream_retry_attempts", 3) or 3))
        self.stream_retry_backoff_seconds = max(0, int(getattr(config, "stream_retry_backoff_seconds", 2) or 2))

        self._clients.clear()
        self._stream_clients.clear()
        self.client = self._create_client()
        self.stream_client = self._create_client_no_timeout()
        self._platform_info = None

        self.set_request_limiter()

    # ==================== 客户端创建（借鉴 TaskRequester.py） ====================

    def _create_client(self, api_key: str = None) -> AsyncOpenAI:
        """
        创建标准 OpenAI 客户端
        
        超时设置参照 TaskRequester.py:
        - connect: 8秒 (TCP 连接)
        - read: request_timeout (等待响应)
        - write: 8秒 (发送请求)
        - pool: 8秒 (连接池)
        
        Args:
            api_key: 可选，指定使用的 API Key。如果不指定，使用默认的 self.api_key
        """
        key = api_key or self.api_key
        return AsyncOpenAI(
            api_key=key,
            base_url=self.base_url,
            timeout=httpx.Timeout(
                read=self.request_timeout,
                pool=8.0,
                write=8.0,
                connect=8.0,
            ),
            max_retries=1,
        )
    
    def _create_client_no_timeout(self, api_key: str = None) -> AsyncOpenAI:
        """
        创建无超时客户端（用于深度思考流式输出）
        
        智谱深度思考、阿里云百炼 DeepSeek、NVIDIA Build DeepSeek 等需要长时间思考的场景
        
        Args:
            api_key: 可选，指定使用的 API Key。如果不指定，使用默认的 self.api_key
        """
        key = api_key or self.api_key
        return AsyncOpenAI(
            api_key=key,
            base_url=self.base_url,
            timeout=httpx.Timeout(None),  # 完全禁用超时
            max_retries=1,
        )
    
    def _get_client_for_key(self, api_key: str) -> AsyncOpenAI:
        """获取指定 Key 的客户端（使用缓存）"""
        if api_key not in self._clients:
            self._clients[api_key] = self._create_client(api_key)
        return self._clients[api_key]
    
    def _get_stream_client_for_key(self, api_key: str) -> AsyncOpenAI:
        """获取指定 Key 的无超时客户端（使用缓存）"""
        if api_key not in self._stream_clients:
            self._stream_clients[api_key] = self._create_client_no_timeout(api_key)
        return self._stream_clients[api_key]

    # ==================== 平台检测（借鉴 TaskRequester.py） ====================

    def _detect_platform(self) -> dict:
        """
        检测当前 API 平台类型
        
        返回 dict 包含：
        - platform: 平台名称
        - is_zhipu: 是否智谱
        - is_dashscope_deepseek: 是否阿里云百炼 DeepSeek
        - is_nvidia_deepseek: 是否 NVIDIA Build DeepSeek
        - is_o_series: 是否 OpenAI O系列
        - is_qwen3: 是否 QWEN3
        - thinking_enabled: 是否启用深度思考
        - stream_required: 是否需要流式输出
        """
        if self._platform_info is not None:
            return self._platform_info
        
        info = {
            "platform": "openai_compatible",
            "is_zhipu": False,
            "is_dashscope_deepseek": False,
            "is_nvidia_deepseek": False,  # 新增 NVIDIA 支持
            "is_o_series": False,
            "is_qwen3": False,
            "is_gemini": False,
            "thinking_enabled": True,  # 【强制启用深度思考】
            "stream_required": False,
        }
        
        # 智谱 API 检测
        if self.RE_ZHIPU.search(self.base_url):
            info["platform"] = "zhipu"
            info["is_zhipu"] = True
            info["stream_required"] = True  # 智谱深度思考必须用流式
            LogHelper.info(f"[平台检测] 检测到 [bold green]智谱 API[/] - 启用深度思考 + 流式输出")
        
        # NVIDIA Build DeepSeek 检测（优先于阿里云百炼检测）
        elif self.RE_NVIDIA_DEEPSEEK_URL.search(self.base_url) and self.RE_NVIDIA_DEEPSEEK_MODEL.search(self.model_name):
            info["platform"] = "nvidia_deepseek"
            info["is_nvidia_deepseek"] = True
            info["stream_required"] = True  # NVIDIA DeepSeek 深度思考必须用流式
            LogHelper.info(f"[平台检测] 检测到 [bold magenta]NVIDIA Build DeepSeek[/] - 启用深度思考 + 流式输出")
            if len(self.api_keys) > 1:
                LogHelper.info(f"[多API轮询] 已配置 {len(self.api_keys)} 个 NVIDIA API Key")
        
        # 阿里云百炼 DeepSeek 检测
        elif self.RE_DASHSCOPE_DEEPSEEK_URL.search(self.base_url) and self.RE_DASHSCOPE_DEEPSEEK_MODEL.search(self.model_name):
            info["platform"] = "dashscope_deepseek"
            info["is_dashscope_deepseek"] = True
            info["stream_required"] = True  # DeepSeek 深度思考必须用流式
            LogHelper.info(f"[平台检测] 检测到 [bold green]阿里云百炼 DeepSeek[/] - 启用深度思考 + 流式输出")
            if len(self.api_keys) > 1:
                LogHelper.info(f"[多API轮询] 已配置 {len(self.api_keys)} 个 ModelScope API Key")
        
        # OpenAI O系列检测
        elif self.RE_O_SERIES.search(self.model_name):
            info["platform"] = "openai_o_series"
            info["is_o_series"] = True
            LogHelper.info(f"[平台检测] 检测到 [bold green]OpenAI O系列模型[/] - 使用 max_completion_tokens")
        
        # QWEN3 检测
        elif self.RE_QWEN3.search(self.model_name):
            info["platform"] = "qwen3"
            info["is_qwen3"] = True
            LogHelper.info(f"[平台检测] 检测到 [bold green]QWEN3 模型[/] - 思考模式自动控制")
        
        # Gemini 检测
        elif self.RE_GEMINI_2_5_FLASH.search(self.model_name):
            info["platform"] = "gemini"
            info["is_gemini"] = True
            LogHelper.info(f"[平台检测] 检测到 [bold green]Gemini 2.5 Flash[/]")
        
        else:
            LogHelper.info(f"[平台检测] 使用 [bold]标准 OpenAI 兼容 API[/]")
        
        self._platform_info = info
        return info

    # ============== 精确字符检测（借鉴 Dev-Experimental/TextBase） ==============
    
    # 拟声词/地名特殊字符集（借鉴 KanaFixer）
    # 这些假名在【孤立出现】时（前后都不是假名）应该被容差，不触发重试
    RULE_ONOMATOPOEIA = frozenset({
        "ッ", "っ",  # 促音
        "ぁ", "ぃ", "ぅ", "ぇ", "ぉ",  # 小写元音
        "ゃ", "ゅ", "ょ", "ゎ",  # 小写拗音
        "ァ", "ィ", "ゥ", "ェ", "ォ",  # 片假名小写元音
        "ャ", "ュ", "ョ", "ヮ",  # 片假名小写拗音
        "ー",  # 长音符（在非假名上下文中可能用于表示拉长）
        "ヶ", "ケ", "ヵ",  # 地名专用假名（前ヶ浜、蛇ヶ沢 等，表示"之"的意思）
        "の",  # 平假名"之"（地名中常见，如 見晴らしの丘）
    })
    
    # 检测文本是否包含日文假名（使用精确字符集）
    @staticmethod
    def contains_kana(text: str) -> bool:
        """
        检测文本是否包含日文假名（平假名或片假名）
        使用 TextHelper 的精确字符集，排除：
        - 濁点 ゛ (0x309B)、半濁点 ゜ (0x309C)
        - 中点 ・ (0x30FB)
        """
        if not text:
            return False
        return TextHelper.JA.any_hiragana(text) or TextHelper.JA.any_katakana(text)
    
    @staticmethod
    def contains_kana_strict(text: str) -> bool:
        """
        严格检测文本是否包含【有效的】日文假名残留
        排除孤立的拟声词字符（前后都不是假名的促音、小写假名等）
        
        规则（借鉴 KanaFixer）：
        - 如果假名字符前后都没有其他假名，且该字符属于拟声词字符集，则不算残留
        - 只要存在"真正的假名串"（连续假名或非拟声词假名），就算残留
        """
        if not text:
            return False
        
        # 快速检测：如果完全没有假名，直接返回 False
        if not LLM.contains_kana(text):
            return False
        
        # 逐字符分析
        length = len(text)
        for i, char in enumerate(text):
            # 判断是否为假名
            is_kana = TextHelper.JA.hiragana(char) or TextHelper.JA.katakana(char)
            if not is_kana:
                continue
            
            # 如果是拟声词字符，检查前后是否有假名
            if char in LLM.RULE_ONOMATOPOEIA:
                prev_char = text[i - 1] if i > 0 else None
                next_char = text[i + 1] if i < length - 1 else None
                
                is_prev_kana = prev_char is not None and (TextHelper.JA.hiragana(prev_char) or TextHelper.JA.katakana(prev_char))
                is_next_kana = next_char is not None and (TextHelper.JA.hiragana(next_char) or TextHelper.JA.katakana(next_char))
                
                # 前后都没有假名，这个拟声词字符是孤立的，不算残留
                if not is_prev_kana and not is_next_kana:
                    continue
            
            # 不是拟声词字符，或者拟声词字符前后有假名 → 算作残留
            return True
        
        return False

    @staticmethod
    def _calc_kana_ratio(text: str) -> float:
        if not text:
            return 0.0
        kana = 0
        total = 0
        for char in text:
            if char.isspace():
                continue
            total += 1
            if TextHelper.JA.hiragana(char) or TextHelper.JA.katakana(char):
                kana += 1
        return kana / total if total > 0 else 0.0

    # 检测文本是否包含韩文（使用精确字符集）
    @staticmethod
    def contains_korean(text: str) -> bool:
        """检测文本是否包含韩文"""
        if not text:
            return False
        return TextHelper.KO.any_hangeul(text)

    # 检测退化（重复字符模式）
    RE_DEGRADATION = re.compile(r"(.{1,3})\1{16,}", flags=re.IGNORECASE)
    
    @staticmethod
    def is_degraded(text: str) -> bool:
        """
        检测文本是否退化（模型输出重复内容）
        规则：1-3个字符重复16次以上视为退化
        """
        if not text:
            return False
        return bool(LLM.RE_DEGRADATION.search(text))

    # 计算 Jaccard 相似度
    @staticmethod
    def check_similarity(src: str, dst: str) -> float:
        """
        计算两个字符串的 Jaccard 相似度
        返回 0.0~1.0，值越大越相似
        """
        if not src or not dst:
            return 0.0
        set_src = set(src)
        set_dst = set(dst)
        union = len(set_src | set_dst)
        intersection = len(set_src & set_dst)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _split_expected_context_blocks(context_str: str) -> tuple[list[str], list[int]]:
        if not context_str:
            return [], []
        lines = [line.rstrip("\r\n") for line in context_str.splitlines()]
        headers: list[str] = []
        counts: list[int] = []
        current_count = 0
        for line in lines:
            s = line.strip()
            if not s:
                continue
            m = re.match(r"^【\s*上下文\s*(\d+)\s*】$", s)
            if m:
                if headers:
                    counts.append(current_count)
                headers.append(s)
                current_count = 0
            else:
                if headers:
                    current_count += 1
        if headers:
            counts.append(current_count)
        return headers, counts

    @staticmethod
    def _try_extract_fenced_block(text: str) -> str:
        if "```" not in (text or ""):
            return text or ""
        blocks = re.findall(r"```(?:\w+)?\s*([\s\S]*?)\s*```", text)
        for b in blocks:
            if "上下文" in b:
                return b
        return text or ""

    @staticmethod
    def _extract_context_blocks(text: str) -> list[tuple[int | None, str]]:
        if not text:
            return []
        pattern = re.compile(r"(?:【|\[|\(|（)\s*(?:参考\s*)?(?:上下文|上文|context)\s*(\d+)\s*(?:】|\]|\)|）)", flags=re.IGNORECASE)
        matches = list(pattern.finditer(text))
        if not matches:
            return []
        blocks: list[tuple[int | None, str]] = []
        for idx, m in enumerate(matches):
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            num: int | None = None
            try:
                num = int(m.group(1))
            except Exception:
                num = None
            blocks.append((num, text[start:end]))
        return blocks

    @staticmethod
    def _concat_text_lines(a: str, b: str) -> str:
        a = (a or "").strip()
        b = (b or "").strip()
        if not a:
            return b
        if not b:
            return a
        if a[-1].isascii() and a[-1].isalnum() and b[0].isascii() and b[0].isalnum():
            return f"{a} {b}"
        return f"{a}{b}"

    @staticmethod
    def _merge_excess_lines(lines: list[str], expected_count: int) -> list[str]:
        cleaned = [v.strip() for v in (lines or []) if isinstance(v, str) and v.strip()]
        if expected_count <= 0:
            return []
        if len(cleaned) <= expected_count:
            return cleaned
        if expected_count == 1:
            return ["".join(cleaned)]

        end_punct = set("。！？!?…」』”")
        start_punct = set("，,、.．:：;；)）]】}」』”")

        out: list[str] = []
        for i, cur in enumerate(cleaned):
            if not out:
                out.append(cur)
                continue

            remaining_after = len(cleaned) - (i + 1)
            slots_after = expected_count - (len(out) + 1)
            must_open_new = remaining_after == slots_after

            prev = out[-1]
            need_merge_somewhere = len(out) + remaining_after + 1 > expected_count
            looks_boundary = bool(prev) and (prev[-1] in end_punct) and (cur[:1] not in start_punct)

            if must_open_new:
                out.append(cur)
            elif need_merge_somewhere and not looks_boundary:
                out[-1] = LLM._concat_text_lines(out[-1], cur)
            else:
                out.append(cur)

        while len(out) > expected_count:
            out[-2] = LLM._concat_text_lines(out[-2], out[-1])
            out.pop()

        return out

    @staticmethod
    def parse_context_translate_output(context_str: str, response_result: str) -> tuple[list[str], dict]:
        original_lines = [line.strip() for line in (context_str or "").splitlines() if line.strip()]
        expected_headers, expected_counts = LLM._split_expected_context_blocks(context_str or "")
        expected_contexts = len(expected_headers)

        raw = (response_result or "").strip()
        raw = LLM._try_extract_fenced_block(raw).strip()

        if expected_contexts <= 0:
            return [line.strip() for line in raw.splitlines() if line.strip()], {"mode": "no_expected"}

        blocks = LLM._extract_context_blocks(raw)
        if not blocks:
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            mode = "fallback_lines_equal" if len(lines) == len(original_lines) else "fallback_no_blocks"
            return lines, {"mode": mode, "expected_lines": len(original_lines), "found_lines": len(lines)}

        blocks_by_num: dict[int, str] = {}
        blocks_in_order: list[str] = []
        for num, content in blocks:
            blocks_in_order.append(content)
            if isinstance(num, int) and num not in blocks_by_num:
                blocks_by_num[num] = content

        chosen_contents: list[str] = []
        chosen_from: list[str] = []
        missing_contexts: list[int] = []
        for i in range(expected_contexts):
            num = i + 1
            if num in blocks_by_num:
                chosen_contents.append(blocks_by_num[num])
                chosen_from.append("number")
                continue
            if len(blocks_in_order) > i:
                chosen_contents.append(blocks_in_order[i])
                chosen_from.append("order")
                continue
            chosen_contents.append("")
            chosen_from.append("missing")
            missing_contexts.append(num)

        if all(v == "number" for v in chosen_from):
            mode = "by_number"
        elif any(v == "number" for v in chosen_from):
            mode = "by_number_partial"
        elif any(v == "order" for v in chosen_from):
            mode = "by_order_partial"
        else:
            mode = "empty"

        out_lines: list[str] = []
        detail: dict = {
            "mode": mode,
            "expected_contexts": expected_contexts,
            "found_contexts": len(blocks_in_order),
            "missing_contexts": missing_contexts,
        }
        for idx, (header, expected_line_count, content) in enumerate(zip(expected_headers, expected_counts, chosen_contents), start=1):
            content_lines_raw = [line.strip() for line in (content or "").splitlines() if line.strip()]
            content_lines = content_lines_raw
            if len(content_lines) != expected_line_count and expected_line_count > 0 and len(content_lines) > expected_line_count:
                content_lines = LLM._merge_excess_lines(content_lines, expected_line_count)
                if len(content_lines) == expected_line_count:
                    fixed = detail.get("fixed") if isinstance(detail.get("fixed"), list) else []
                    fixed.append(
                        {
                            "context_index": idx,
                            "expected": expected_line_count,
                            "raw": len(content_lines_raw),
                            "merged": len(content_lines),
                        }
                    )
                    detail["fixed"] = fixed

            if expected_line_count > 0 and len(content_lines) != expected_line_count:
                mismatches = detail.get("mismatches") if isinstance(detail.get("mismatches"), list) else []
                mismatches.append(
                    {
                        "context_index": idx,
                        "expected": expected_line_count,
                        "actual": len(content_lines),
                    }
                )
                detail["mismatches"] = mismatches
            out_lines.append(header)
            out_lines.extend(content_lines)

        return out_lines, detail

    # ==================== 客户端方法（向后兼容） ====================

    def load_client(self) -> AsyncOpenAI:
        """向后兼容：返回标准客户端"""
        return self._create_client()
    
    def load_stream_client(self) -> AsyncOpenAI:
        """向后兼容：返回无超时客户端"""
        return self._create_client_no_timeout()

    # 设置语言
    def set_language(self, language: int) -> None:
        self.language = language

    # 加载指令
    def load_prompt(self) -> None:
        try:
            with open("prompt/prompt_context_translate.txt", "r", encoding = "utf-8-sig") as reader:
                self.prompt_context_translate = reader.read().strip()
        except Exception as e:
            LogHelper.error(f"加载配置文件时发生错误 - {LogHelper.get_trackback(e)}")


        try:
            with open("prompt/prompt_surface_analysis_without_translation.txt", "r", encoding = "utf-8-sig") as reader:
                self.prompt_surface_analysis_without_translation = reader.read().strip()
        except Exception as e:
            LogHelper.error(f"加载配置文件时发生错误 - {LogHelper.get_trackback(e)}")

        # 新增：加载基于上下文的词义分析 prompt
        try:
            with open("prompt/prompt_surface_analysis_with_context.txt", "r", encoding = "utf-8-sig") as reader:
                self.prompt_surface_analysis_with_context = reader.read().strip()
        except Exception as e:
            LogHelper.error(f"加载配置文件时发生错误 - {LogHelper.get_trackback(e)}")

    # 加载配置文件
    def load_llm_config(self) -> None:
        try:
            with open("resource/llm_config/api_test_config.json", "r", encoding = "utf-8-sig") as reader:
                LLM.API_TEST_CONFIG = LLM.LLMConfig()
                for key, value in repair.load(reader).items():
                    setattr(LLM.API_TEST_CONFIG, key.upper(), value)
        except Exception as e:
            LogHelper.error(f"加载配置文件时发生错误 - {LogHelper.get_trackback(e)}")

        try:
            with open("resource/llm_config/surface_analysis_config.json", "r", encoding = "utf-8-sig") as reader:
                LLM.SURFACE_ANALYSIS_CONFIG = LLM.LLMConfig()
                for key, value in repair.load(reader).items():
                    setattr(LLM.SURFACE_ANALYSIS_CONFIG, key.upper(), value)
        except Exception as e:
            LogHelper.error(f"加载配置文件时发生错误 - {LogHelper.get_trackback(e)}")

        try:
            with open("resource/llm_config/context_translate_config.json", "r", encoding = "utf-8-sig") as reader:
                LLM.CONTEXT_TRANSLATE_CONFIG = LLM.LLMConfig()
                for key, value in repair.load(reader).items():
                    setattr(LLM.CONTEXT_TRANSLATE_CONFIG, key.upper(), value)
        except Exception as e:
            LogHelper.error(f"加载配置文件时发生错误 - {LogHelper.get_trackback(e)}")

    # 设置请求限制器
    def set_request_limiter(self) -> None:
        auto_freq_downgrade_enable = bool(getattr(self.config, "request_frequency_auto_downgrade_enable", False))
        auto_freq_downgrade_threshold = float(getattr(self.config, "request_frequency_auto_downgrade_threshold", 20))
        auto_freq_downgrade_to = float(getattr(self.config, "request_frequency_auto_downgrade_to", 10))
        auto_llamacpp_detect_enable = bool(getattr(self.config, "llamacpp_auto_detect_enable", True))
        unlimited_concurrency_threshold = int(getattr(self.config, "unlimited_concurrency_threshold", 65536) or 65536)
        multi_key_default_enable = bool(getattr(self.config, "multi_key_default_enable", True))
        multi_key_default_per_key_rpm = float(getattr(self.config, "multi_key_default_per_key_rpm", 1) or 1)

        # 获取 llama.cpp 响应数据
        response_json = None
        if auto_llamacpp_detect_enable:
            try:
                with urllib.request.urlopen(f"{re.sub(r"/v1$", "", self.base_url)}/slots") as reader:
                    response_json = repair.load(reader)
            except Exception:
                LogHelper.debug("无法获取 [green]llama.cpp[/] 响应数据 ...")

        # 如果响应数据有效，则是 llama.cpp 接口
        if isinstance(response_json, list) and len(response_json) > 0:
            self.request_frequency_threshold = len(response_json)
            LogHelper.info("")
            LogHelper.info(f"检查到 [green]llama.cpp[/]，根据其配置，请求频率阈值自动设置为 [green]{len(response_json)}[/] 次/秒 ...")
            LogHelper.info("")
        # 否则，按在线接口设置（移除 MAX_TOKENS 限制，允许充足的上下文）
        else:
            LLM.API_TEST_CONFIG.MAX_TOKENS = 16 * 1024      # 16K，充足的输出空间
            LLM.SURFACE_ANALYSIS_CONFIG.MAX_TOKENS = 16 * 1024
            LLM.CONTEXT_TRANSLATE_CONFIG.MAX_TOKENS = 16 * 1024

        # 设置请求限制器
        # 【重要】两个参数是独立的概念：
        # - max_concurrent_requests: 控制同时在空中飞行的请求数量上限（并发槽位）
        # - request_frequency_threshold: 控制每秒发送新请求的频率（发送速度）
        # 
        # 例如：max_concurrent=90, frequency=10
        # 意味着：每秒发10个请求，但如果每个请求需要9秒返回，那么第9秒时就会有90个请求同时在飞行
        # 
        # 这对于多API Key轮询场景非常重要！
        if int(getattr(self, "max_concurrent_requests", 0) or 0) > 0:
            concurrent_limit = int(self.max_concurrent_requests)
        else:
            concurrent_limit = unlimited_concurrency_threshold

        key_count = len(self.api_keys)
        is_llamacpp = isinstance(response_json, list) and len(response_json) > 0
        if multi_key_default_enable and (not is_llamacpp) and key_count > 1:
            per_key_rpm = max(0.001, float(multi_key_default_per_key_rpm))
            total_rpm = per_key_rpm * key_count
            self._multi_key_per_key_limiter_enable = True
            self._per_key_limiters = {}
            self._per_key_limiter_rpm = per_key_rpm
            self.semaphore = asyncio.Semaphore(concurrent_limit)
            self.async_limiter = AsyncLimiter(max_rate=total_rpm, time_period=60)
            LogHelper.info(
                f"[并发控制] 最大并发: {concurrent_limit} | 频率限制: {total_rpm:.2f} 次/分钟（每Key {per_key_rpm:.2f} 次/分钟）"
            )
            return

        frequency_limit = max(1, int(self.request_frequency_threshold))
        
        # 【针对 429 错误的优化】
        # 如果 request_frequency_threshold 设置得很高（如 100），但实际 API 限制是 60 RPM
        # 那么 aiolimiter 会允许每秒发 100 个，导致瞬间触发 429
        # 因此，这里引入一个保守的 "每秒并发增量限制"
        # 默认情况下，对于非 llama.cpp 接口，我们将频率限制在合理范围内（例如 5-10 RPS）
        if auto_freq_downgrade_enable and self.request_frequency_threshold > auto_freq_downgrade_threshold and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
             LogHelper.warning(f"[并发控制] 高频请求 ({self.request_frequency_threshold} RPS)，自动降级至 {auto_freq_downgrade_to} RPS")
             self.request_frequency_threshold = float(auto_freq_downgrade_to)
             frequency_limit = max(1, int(self.request_frequency_threshold))

        LogHelper.info(f"[并发控制] 最大并发: {concurrent_limit} | 频率限制: {frequency_limit} 次/秒")
        
        if self.request_frequency_threshold > 1:
            self.semaphore = asyncio.Semaphore(concurrent_limit)
            self.async_limiter = AsyncLimiter(max_rate = self.request_frequency_threshold, time_period = 1)
        elif self.request_frequency_threshold > 0:
            self.semaphore = asyncio.Semaphore(concurrent_limit)
            self.async_limiter = AsyncLimiter(max_rate = 1, time_period = 1 / self.request_frequency_threshold)
        else:
            self.semaphore = asyncio.Semaphore(1)
            self.async_limiter = AsyncLimiter(max_rate = 1, time_period = 1)

    # ==================== 核心请求方法（完全重写，参照 TaskRequester.py） ====================

    def _get_per_key_limiter(self, api_key: str) -> AsyncLimiter | None:
        if not getattr(self, "_multi_key_per_key_limiter_enable", False):
            return None
        if not api_key or api_key == "no_key_required":
            return None
        limiter = getattr(self, "_per_key_limiters", None)
        if not isinstance(limiter, dict):
            self._per_key_limiters = {}
            limiter = self._per_key_limiters
        if api_key not in limiter:
            per_key_rpm = max(0.001, float(getattr(self, "_per_key_limiter_rpm", 1.0) or 1.0))
            limiter[api_key] = AsyncLimiter(max_rate=per_key_rpm, time_period=60)
        return limiter[api_key]

    def _get_worker_count(self, total: int) -> int:
        total = max(1, int(total))
        max_concurrent = int(getattr(self, "max_concurrent_requests", 0) or 0)
        if max_concurrent > 0:
            return min(max_concurrent, total)
        cap = int(getattr(self.config, "unlimited_worker_count_cap", 1000) or 1000)
        cap = max(1, cap)

        key_count = 0
        available_keys = 0
        try:
            key_count = len(self.api_keys) if isinstance(getattr(self, "api_keys", None), list) else 0
            available_keys = LLM.get_available_key_count(self.api_keys) if getattr(self, "api_keys", None) else 0
        except Exception:
            key_count = 0
            available_keys = 0

        if bool(getattr(self, "_multi_key_per_key_limiter_enable", False)) and key_count > 1:
            base = max(8, int(max(available_keys, 1) * 2))
        else:
            base = 32

        return min(total, cap, max(1, base))

    def _classify_exception(self, e: Exception) -> str:
        msg = str(e).lower()
        if "429" in msg or "rate limit" in msg or "ratelimit" in msg:
            return "rate_limit"
        if "timeout" in msg or "timed out" in msg or "首包超时" in msg or "流式响应超时" in msg or "请求超时" in msg or "任务超时" in msg:
            return "timeout"
        if "json" in msg or "解析" in msg or "数据结构" in msg:
            return "parse"
        if "相似度" in msg or "翻译失效" in msg:
            return "similarity"
        if "假名残留" in msg or "韩文残留" in msg or "korean" in msg or "kana" in msg:
            return "language_residue"
        if "permission" in msg or "denied" in msg or "banned" in msg or "blacklist" in msg or "prohibited" in msg:
            return "permission"
        return "other"

    def _record_error(self, kind: str, e: Exception) -> None:
        try:
            self.runtime_stats["last_error"] = str(e)
            self.runtime_stats["last_error_kind"] = kind
            d = self.runtime_stats.get("errors_by_kind", {})
            d[kind] = int(d.get(kind, 0)) + 1
            self.runtime_stats["errors_by_kind"] = d
        except Exception:
            return

    def _compute_retry_delay_seconds(self, e: Exception, attempt: int) -> float:
        base = float(getattr(self.config, "task_retry_backoff_base_seconds", 2) or 2)
        base_timeout = float(getattr(self.config, "task_retry_backoff_timeout_seconds", 8) or 8)
        base_rate = float(getattr(self.config, "task_retry_backoff_rate_limit_seconds", 15) or 15)
        max_delay = float(getattr(self.config, "task_retry_backoff_max_seconds", 120) or 120)
        jitter_ratio = float(getattr(self.config, "task_retry_jitter_ratio", 0.15) or 0.15)
        jitter_ratio = max(0.0, min(1.0, jitter_ratio))

        kind = self._classify_exception(e)
        if kind == "rate_limit":
            base_use = base_rate
        elif kind == "timeout":
            base_use = base_timeout
        else:
            base_use = base

        exp = max(0, min(int(attempt) - 1, 8))
        delay = base_use * (2 ** exp)
        delay = min(max_delay, max(0.0, delay))
        if delay <= 0:
            return 0.0
        jitter = delay * jitter_ratio
        if jitter > 0:
            delay = delay + random.uniform(-jitter, jitter)
        return max(0.0, min(max_delay, delay))

    def get_runtime_status(self) -> dict:
        key_count = len(self.api_keys)
        blacklisted = LLM.get_blacklisted_key_count()
        available = LLM.get_available_key_count(self.api_keys) if self.api_keys else 0
        mode = "rps"
        per_key_rpm = None
        total_rpm = None
        if getattr(self, "_multi_key_per_key_limiter_enable", False) and key_count > 1:
            mode = "rpm_default"
            per_key_rpm = float(getattr(self, "_per_key_limiter_rpm", 1.0) or 1.0)
            total_rpm = per_key_rpm * key_count
        blacklisted_keys = []
        try:
            now = time.time()
            with LLM.KEY_LOCK:
                for k, exp in LLM.BLACKLISTED_KEYS.items():
                    if now >= exp:
                        continue
                    blacklisted_keys.append(
                        {
                            "key": (k[:6] + "…" + k[-4:]) if isinstance(k, str) and len(k) > 12 else "***",
                            "expires_in_seconds": max(0, int(exp - now)),
                        }
                    )
        except Exception:
            blacklisted_keys = []

        recent_keys = []
        try:
            now = time.time()
            items = sorted(self._key_last_used_at.items(), key=lambda x: x[1], reverse=True)[:20]
            for k, ts in items:
                recent_keys.append(
                    {
                        "key": (k[:6] + "…" + k[-4:]) if isinstance(k, str) and len(k) > 12 else "***",
                        "last_used_seconds_ago": max(0, int(now - ts)),
                    }
                )
        except Exception:
            recent_keys = []

        return {
            "mode": mode,
            "key_count": key_count,
            "available_keys": available,
            "blacklisted_keys": blacklisted,
            "blacklisted_key_items": blacklisted_keys,
            "per_key_rpm": per_key_rpm,
            "total_rpm": total_rpm,
            "max_concurrent_requests": int(getattr(self, "max_concurrent_requests", 0) or 0),
            "recent_key_items": recent_keys,
            "runtime_stats": self.runtime_stats,
        }

    async def do_request(self, messages: list, llm_config: LLMConfig, retry: bool, enable_thinking: bool = True, task_id: str = "") -> tuple[Exception, dict, str, str, dict, dict]:
        """
        发送 LLM 请求（完全重写，参照 TaskRequester.py）
        
        【强制启用深度思考模式】- 不区分专家模式
        
        支持平台：
        - 智谱 API: 深度思考 + 流式输出
        - 阿里云百炼 DeepSeek: 流式输出 + enable_thinking
        - OpenAI O系列: max_completion_tokens
        - QWEN3: /no_think 控制
        - 标准 OpenAI 兼容 API
        
        Args:
            messages: 消息列表
            llm_config: LLM 配置
            retry: 是否为重试请求
            enable_thinking: 是否启用深度思考（默认 True，强制启用）
            task_id: 任务标识（用于 TaskTracker 更新）
            
        Returns:
            (error, usage, response_think, response_result, llm_request, llm_response)
        """
        import time as time_module
        start_time = time_module.time()
        
        error = None
        usage = None
        response_think = ""
        response_result = ""
        llm_request = None
        llm_response = None
        
        # 更新 tracker 状态为 sending
        tracker = get_current_tracker()
        if tracker and task_id:
            tracker.start_task(task_id)
        
        try:
            # 检测平台
            platform_info = self._detect_platform()
            
            # 构建基础请求参数
            llm_request = self._build_request_args(messages, llm_config, retry, platform_info)
            
            # 【已移除】API 请求详情打印（只保留最终结果，避免刷屏）
            # 如需调试，可取消下方注释：
            # LogTable.print_api_request(
            #     model=self.model_name,
            #     base_url=self.base_url,
            #     messages=messages,
            #     thinking_enabled=platform_info.get("thinking_enabled", True),
            #     stream_enabled=platform_info.get("stream_required", False),
            # )
            
            # 根据平台分发请求
            if platform_info.get("stream_required"):
                # 流式请求（智谱、阿里云百炼 DeepSeek）
                response_think, response_result, input_tokens, output_tokens = await self._do_stream_request(llm_request, task_id)
                
                # 构造 usage 代理对象
                class UsageProxy:
                    def __init__(self, pt, ct):
                        self.prompt_tokens = pt
                        self.completion_tokens = ct
                
                usage = UsageProxy(input_tokens, output_tokens)
                llm_response = {"stream": True, "thinking": response_think[:200] if response_think else ""}
            else:
                # 标准请求
                from openai import PermissionDeniedError
                last_exc: Exception | None = None
                attempt_limit = max(1, min(len(self.api_keys) if self.api_keys else 1, 16))
                response = None
                current_key = ""
                for _ in range(attempt_limit):
                    try:
                        current_key = LLM.get_key(self.api_keys)
                        if current_key:
                            try:
                                self._key_last_used_at[current_key] = time_module.time()
                            except Exception:
                                pass
                        client = self._get_client_for_key(current_key)
                        per_key_limiter = self._get_per_key_limiter(current_key)
                        self.runtime_stats["requests_started"] = int(self.runtime_stats.get("requests_started", 0)) + 1
                        t0 = time_module.monotonic()
                        async with self.semaphore:
                            t1 = time_module.monotonic()
                            self.runtime_stats["semaphore_wait_seconds"] = float(self.runtime_stats.get("semaphore_wait_seconds", 0.0)) + (t1 - t0)
                            t2 = time_module.monotonic()
                            async with self.async_limiter:
                                t3 = time_module.monotonic()
                                self.runtime_stats["global_limiter_wait_seconds"] = float(self.runtime_stats.get("global_limiter_wait_seconds", 0.0)) + (t3 - t2)
                                if per_key_limiter is not None:
                                    t4 = time_module.monotonic()
                                    async with per_key_limiter:
                                        t5 = time_module.monotonic()
                                        self.runtime_stats["per_key_limiter_wait_seconds"] = float(self.runtime_stats.get("per_key_limiter_wait_seconds", 0.0)) + (t5 - t4)
                                        response = await client.chat.completions.create(**llm_request)
                                else:
                                    response = await client.chat.completions.create(**llm_request)
                        self.runtime_stats["requests_succeeded"] = int(self.runtime_stats.get("requests_succeeded", 0)) + 1
                        break
                    except PermissionDeniedError as e:
                        error_msg = str(e)
                        if self.RE_BLACKLIST_ERROR.search(error_msg) and current_key:
                            try:
                                LLM.add_to_blacklist(current_key, ttl_seconds=float(getattr(self.config, "api_key_blacklist_ttl_seconds", 3600) or 3600))
                            except Exception:
                                pass
                            if LLM.get_available_key_count(self.api_keys) <= 0:
                                raise
                            last_exc = e
                            continue
                        raise
                    except Exception as e:
                        last_exc = e
                        raise

                if response is None and last_exc is not None:
                    raise last_exc

                llm_response = response.to_dict() if hasattr(response, "to_dict") else {"raw": str(response)}
                
                # 提取回复内容
                usage = response.usage
                message = response.choices[0].message
                
                # 尝试从多种格式中提取思考内容（借鉴 TaskRequester.py）
                if hasattr(message, "reasoning_content") and isinstance(message.reasoning_content, str):
                    response_think = self.RE_LINE_BREAK.sub("\n", message.reasoning_content.strip())
                    response_result = message.content.strip()
                elif "</think>" in message.content:
                    splited = message.content.split("</think>")
                    response_think = self.RE_LINE_BREAK.sub("\n", splited[0].removeprefix("<think>").strip())
                    response_result = splited[-1].strip()
                else:
                    response_think = ""
                    response_result = message.content.strip()
                
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0

            if response_result == "" and response_think:
                if "```" in response_think and re.search(r"\{[\s\S]*\}", response_think):
                    response_result = response_think.strip()
                else:
                    m = re.search(r"\{[\s\S]*\}", response_think)
                    if m:
                        response_result = m.group(0).strip()
                    elif "【上下文" in response_think:
                        response_result = response_think.strip()
            
            # 【已移除】API 响应详情打印（只保留最终任务结果表格，避免刷屏）
            # 如需调试，可取消下方注释：
            # LogTable.print_api_response(
            #     response_content=response_result,
            #     response_think=response_think,
            #     input_tokens=input_tokens if 'input_tokens' in dir() else (usage.prompt_tokens if usage else 0),
            #     output_tokens=output_tokens if 'output_tokens' in dir() else (usage.completion_tokens if usage else 0),
            #     elapsed_time=time_module.time() - start_time,
            # )
            
        except Exception as e:
            if isinstance(e, asyncio.TimeoutError) and not str(e):
                e = asyncio.TimeoutError("请求超时")
            error = e
            try:
                self.runtime_stats["requests_failed"] = int(self.runtime_stats.get("requests_failed", 0)) + 1
                kind = self._classify_exception(e)
                self._record_error(kind, e)
            except Exception:
                pass
            LogHelper.error(f"[LLM请求] 失败: {e}")
            try:
                ErrorLogger.log(
                    error_type="LLMRequestError",
                    message=str(e),
                    context={
                        "task_id": task_id,
                        "platform_name": getattr(self.config, "platform_name", ""),
                        "base_url": getattr(self, "base_url", ""),
                        "model": getattr(self, "model_name", ""),
                        "retry": bool(retry),
                        "llm_request": llm_request,
                        "llm_response": llm_response,
                        "response_think": response_think,
                        "response_result": response_result,
                    },
                )
            except Exception:
                pass
        
        return error, usage, response_think, response_result, llm_request, llm_response

    def _build_request_args(self, messages: list, llm_config: LLMConfig, retry: bool, platform_info: dict) -> dict:
        """
        构建请求参数（参照 TaskRequester.py 的 generate_openai_args）
        
        根据不同平台进行优化
        """
        # 基础参数
        args = {
            "model": self.model_name,
            "messages": messages,
            "temperature": max(llm_config.TEMPERATURE, 0.50) if retry else llm_config.TEMPERATURE,
            "top_p": llm_config.TOP_P,
            "max_tokens": max(4 * 1024, int(llm_config.MAX_TOKENS)),
            "frequency_penalty": max(llm_config.FREQUENCY_PENALTY, 0.2) if retry else llm_config.FREQUENCY_PENALTY,
            "extra_headers": {
                "User-Agent": "BookTermGacha/0.1.0 (https://github.com/neavo/KeywordGacha)"
            }
        }
        
        # ========== 智谱 API 优化 ==========
        if platform_info.get("is_zhipu"):
            args["stream"] = True
            args["stream_options"] = {"include_usage": True}
            args["extra_body"] = {
                "thinking": {"type": "enabled"}  # 【强制启用深度思考】
            }
        
        # ========== NVIDIA Build DeepSeek 优化 ==========
        elif platform_info.get("is_nvidia_deepseek"):
            args["stream"] = True
            args["stream_options"] = {"include_usage": True}
            # NVIDIA 专用思考模式启用方式
            args["extra_body"] = {"chat_template_kwargs": {"thinking": True}}
            # NVIDIA Build 限制 16384 max_tokens
            args["max_tokens"] = min(16384, max(4 * 1024, int(llm_config.MAX_TOKENS)))
        
        # ========== 阿里云百炼 DeepSeek 优化 ==========
        elif platform_info.get("is_dashscope_deepseek"):
            args["stream"] = True
            args["stream_options"] = {"include_usage": True}
            args["extra_body"] = {"enable_thinking": True}  # 强制启用思考模式
            # ModelScope 支持 32768 max_tokens
            args["max_tokens"] = min(32768, max(4 * 1024, int(llm_config.MAX_TOKENS)))
        
        # ========== OpenAI O系列优化 ==========
        elif platform_info.get("is_o_series"):
            # O系列不支持 max_tokens，使用 max_completion_tokens
            args.pop("max_tokens", None)
            args["max_completion_tokens"] = max(4 * 1024, int(llm_config.MAX_TOKENS))
        
        # ========== QWEN3 优化 ==========
        elif platform_info.get("is_qwen3"):
            # QWEN3 思考模式通过 /no_think 控制
            # 如果不需要思考，在消息末尾添加 /no_think
            pass  # 默认启用思考，不添加 /no_think
        
        return args

    @staticmethod
    def _delta_get(delta, key: str):
        if delta is None:
            return None
        if isinstance(delta, dict):
            return delta.get(key)
        return getattr(delta, key, None)

    @classmethod
    def _extract_stream_text(cls, delta) -> tuple[str, str]:
        think = ""
        content = ""

        for k in ("reasoning_content", "reasoning", "analysis", "thinking"):
            v = cls._delta_get(delta, k)
            if isinstance(v, str) and v:
                think = v
                break

        for k in ("content", "text", "answer", "output_text"):
            v = cls._delta_get(delta, k)
            if isinstance(v, str) and v:
                content = v
                break

        return think, content

    async def _do_stream_request(self, llm_request: dict, task_id: str = "") -> tuple[str, str, int, int]:
        """
        执行流式请求（用于深度思考模式）
        
        参照 TaskRequester.py 的 request_dashscope_deepseek_streaming / request_nvidia_deepseek_streaming
        
        【多 API Key 轮询支持】
        - 每次请求从 api_keys 列表中轮询选择一个可用的 Key
        - 被封禁的 Key 自动加入黑名单，不再使用
        
        【无刷屏版】- 更新全局 TaskTracker 而不是创建单独的 Live
        
        Args:
            llm_request: 请求参数
            task_id: 任务标识（用于更新 tracker）
        """
        import time
        from openai import PermissionDeniedError
        
        start_time = time.time()
        
        response_think = ""
        response_result = ""
        input_tokens = 0
        output_tokens = 0
        chunk_count = 0
        is_thinking = True
        
        # 获取全局 tracker
        tracker = get_current_tracker()
        
        def is_retryable(e: Exception) -> bool:
            if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
                return True
            if isinstance(e, httpx.HTTPError):
                return True
            try:
                import openai
                retryable_types = []
                for name in ("APITimeoutError", "InternalServerError", "RateLimitError", "APIConnectionError"):
                    t = getattr(openai, name, None)
                    if t:
                        retryable_types.append(t)
                if retryable_types and isinstance(e, tuple(retryable_types)):
                    return True
                api_status_error = getattr(openai, "APIStatusError", None)
                if api_status_error and isinstance(e, api_status_error):
                    status_code = getattr(e, "status_code", None)
                    if status_code is None and hasattr(e, "response") and e.response is not None:
                        status_code = getattr(e.response, "status_code", None)
                    if status_code in {408, 429, 500, 502, 503, 504, 522, 524}:
                        return True
            except Exception:
                pass
            msg = str(e).lower()
            markers = ("timeout", "timed out", "rate limit", "429", "connection", "connect", "reset", "broken pipe", "server error", "502", "503", "504")
            return any(m in msg for m in markers)

        last_exc: Exception | None = None
        for attempt in range(self.stream_retry_attempts):
            response_think = ""
            response_result = ""
            input_tokens = 0
            output_tokens = 0
            chunk_count = 0
            is_thinking = True

            try:
                current_key = LLM.get_key(self.api_keys)
                if current_key:
                    try:
                        self._key_last_used_at[current_key] = time.time()
                    except Exception:
                        pass
                stream_client = self._get_stream_client_for_key(current_key)
                per_key_limiter = self._get_per_key_limiter(current_key)
                create_timeout = int(max(1, self.stream_first_chunk_timeout_seconds))
                self.runtime_stats["requests_started"] = int(self.runtime_stats.get("requests_started", 0)) + 1
                t0 = time.monotonic()
                async with self.semaphore:
                    t1 = time.monotonic()
                    self.runtime_stats["semaphore_wait_seconds"] = float(self.runtime_stats.get("semaphore_wait_seconds", 0.0)) + (t1 - t0)
                    try:
                        t2 = time.monotonic()
                        async with self.async_limiter:
                            t3 = time.monotonic()
                            self.runtime_stats["global_limiter_wait_seconds"] = float(self.runtime_stats.get("global_limiter_wait_seconds", 0.0)) + (t3 - t2)
                            if per_key_limiter is not None:
                                t4 = time.monotonic()
                                async with per_key_limiter:
                                    t5 = time.monotonic()
                                    self.runtime_stats["per_key_limiter_wait_seconds"] = float(self.runtime_stats.get("per_key_limiter_wait_seconds", 0.0)) + (t5 - t4)
                                    stream = await asyncio.wait_for(
                                        stream_client.chat.completions.create(**llm_request),
                                        timeout=create_timeout,
                                    )
                            else:
                                stream = await asyncio.wait_for(
                                    stream_client.chat.completions.create(**llm_request),
                                    timeout=create_timeout,
                                )
                    except asyncio.TimeoutError as e:
                        raise asyncio.TimeoutError(f"首包超时: {create_timeout}s 无任何数据") from e

                    aiter = stream.__aiter__()
                    got_first_chunk = False

                    while True:
                        timeout_seconds = self.stream_first_chunk_timeout_seconds if not got_first_chunk else self.stream_stall_timeout_seconds
                        timeout_seconds = int(max(1, timeout_seconds))
                        try:
                            chunk = await asyncio.wait_for(aiter.__anext__(), timeout=timeout_seconds)
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError as e:
                            try:
                                close = getattr(stream, "close", None)
                                if close:
                                    r = close()
                                    if asyncio.iscoroutine(r):
                                        await r
                            except Exception:
                                pass
                            if not got_first_chunk:
                                raise asyncio.TimeoutError(f"首包超时: {timeout_seconds}s 无任何数据") from e
                            raise asyncio.TimeoutError(f"流式响应超时: {timeout_seconds}s 无新数据") from e

                        got_first_chunk = True
                        chunk_count += 1

                        if not chunk.choices:
                            if hasattr(chunk, "usage") and chunk.usage is not None:
                                try:
                                    input_tokens = int(chunk.usage.prompt_tokens)
                                except Exception:
                                    pass
                                try:
                                    output_tokens = int(chunk.usage.completion_tokens)
                                except Exception:
                                    pass
                            continue

                        delta = chunk.choices[0].delta
                        delta_think, delta_content = __class__._extract_stream_text(delta)

                        if delta_think:
                            response_think += delta_think
                            if tracker and task_id:
                                tracker.update_task(
                                    task_id=task_id,
                                    status="thinking",
                                    think_chars=len(response_think),
                                    reply_chars=len(response_result),
                                    chunks=chunk_count,
                                )

                        if delta_content:
                            if is_thinking:
                                is_thinking = False
                            response_result += delta_content
                            if tracker and task_id:
                                tracker.update_task(
                                    task_id=task_id,
                                    status="receiving",
                                    think_chars=len(response_think),
                                    reply_chars=len(response_result),
                                    chunks=chunk_count,
                                )

                response_think = self.RE_LINE_BREAK.sub("\n", response_think.strip())
                response_result = response_result.strip()
                if response_result == "" and response_think:
                    if "```" in response_think and re.search(r"\{[\s\S]*\}", response_think):
                        response_result = response_think.strip()
                    else:
                        m = re.search(r"\{[\s\S]*\}", response_think)
                        if m:
                            response_result = m.group(0).strip()
                        elif "【上下文" in response_think:
                            response_result = response_think.strip()
                self.runtime_stats["requests_succeeded"] = int(self.runtime_stats.get("requests_succeeded", 0)) + 1
                return response_think, response_result, input_tokens, output_tokens

            except PermissionDeniedError as e:
                error_msg = str(e)
                if self.RE_BLACKLIST_ERROR.search(error_msg):
                    try:
                        LLM.add_to_blacklist(current_key, ttl_seconds=float(getattr(self.config, "api_key_blacklist_ttl_seconds", 3600) or 3600))
                    except Exception:
                        pass
                    available_count = LLM.get_available_key_count(self.api_keys)
                    if available_count <= 0:
                        LogHelper.error(f"[致命错误] 所有 API Key 均已被封禁，任务无法继续！")
                        raise
                    last_exc = e
                    continue
                raise

            except RuntimeError as e:
                LogHelper.error(f"[致命错误] {e}")
                raise

            except Exception as e:
                last_exc = e
                try:
                    kind = self._classify_exception(e)
                    self._record_error(kind, e)
                except Exception:
                    pass
                if attempt + 1 >= self.stream_retry_attempts or not is_retryable(e):
                    try:
                        ErrorLogger.log(
                            error_type="StreamRequestError",
                            message=str(e),
                            context={
                                "task_id": task_id,
                                "attempt": attempt + 1,
                                "max_attempts": self.stream_retry_attempts,
                                "base_url": getattr(self, "base_url", ""),
                                "model": getattr(self, "model_name", ""),
                                "current_key": current_key if "current_key" in locals() else "",
                                "chunk_count": chunk_count,
                                "response_think": response_think,
                                "response_result": response_result,
                                "llm_request": llm_request,
                                "traceback": LogHelper.get_trackback(e),
                            },
                        )
                    except Exception:
                        pass
                    raise
                wait_time = self.stream_retry_backoff_seconds * (attempt + 1)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("流式请求失败：未知错误")

    # 接口测试任务
    async def api_test(self) -> bool:
        llm_request = {}
        llm_response = {}
        try:
            success = False

            error, usage, _, response_result, llm_request, llm_response = await self.do_request(
                [
                    {
                        "role": "system",
                        "content": self.prompt_surface_analysis_without_translation.replace("{PROMPT_GROUPS}", "、".join(("角色", "其他"))),
                    },
                    {
                        "role": "user",
                        "content": (
                            "目标词语：ダリヤ\n"
                            "参考文本原文：\n"
                            "魔導具師ダリヤはうつむかない"
                        ),
                    },
                ],
                LLM.API_TEST_CONFIG,
                True
            )

            if error != None:
                raise error

            if usage.completion_tokens >= LLM.SURFACE_ANALYSIS_CONFIG.MAX_TOKENS:
                raise Exception("返回结果错误（模型退化） ...")

            try:
                clean_json = (response_result or "").strip()
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_json:
                    clean_json = clean_json.split("```")[1].split("```")[0].strip()
                
                clean_json = clean_json.removesuffix("...").strip()
                clean_json = clean_json.replace("“", '"').replace("”", '"')
                
                result = repair.loads(clean_json)
            except Exception as e:
                LogHelper.warning(f"[JSON解析] 初次解析失败，尝试暴力修复: {e}")
                try:
                    json_match = re.search(r"\{.*\}", response_result, re.DOTALL)
                    if json_match:
                        clean_json = json_match.group(0).strip().removesuffix("...").strip()
                        clean_json = clean_json.replace("“", '"').replace("”", '"')
                        result = repair.loads(clean_json)
                    else:
                        raise Exception("未找到有效的 JSON 对象")
                except Exception as e2:
                    raise Exception(f"JSON 解析完全失败: {e2} | 原文: {response_result[:100]}...")

            if not isinstance(result, dict) or result == {}:
                raise Exception("返回结果错误（数据结构） ...")

            success = True
            LogHelper.info(f"{result}")

            return success
        except Exception as e:
            LogHelper.warning(f"{LogHelper.get_trackback(e)}")
            LogHelper.warning(f"llm_request - {llm_request}")
            LogHelper.warning(f"llm_response - {llm_response}")

    # 词义分析任务（支持使用已翻译的参考文本）
    async def surface_analysis(self, word: Word, words: list[Word], fake_name_mapping: dict[str, str], retry: bool, last_round: bool, task_id: str = "") -> Exception | None:
        import time as time_module
        start_time = time_module.time()
        response_think = ""
        response_result = ""
        usage = None
        error = None
        
        if True:
            try:
                if not hasattr(self, "prompt_groups"):
                    x = [v for group in LLM.GROUP_MAPPING.values() for v in group]
                    y = [v for group in LLM.GROUP_MAPPING_BANNED.values() for v in group]
                    self.prompt_groups = x + y

                # 获取参考文本原文（注意：这里也使用 context_shrink_level 来控制采样量）
                # 【新增】根据限缩级别动态调整采样数量
                if word.context_shrink_level > 0:
                    shrink_factors = [1.0, 0.5, 0.25]
                    factor = shrink_factors[min(word.context_shrink_level, len(shrink_factors) - 1)] if word.context_shrink_level < 3 else 0.1
                    actual_samples = max(1, int(Word.MAX_CONTEXT_SAMPLES * factor))
                    sampled_context, _ = word.sample_diverse_context(max_samples=actual_samples)
                    context_original = "\n\n".join([f"【上下文 {i+1}】\n{p}" for i, p in enumerate(sampled_context)])
                else:
                    context_original = word.get_context_str_for_surface_analysis(self.language)
                
                # 判断是否有已翻译的参考文本（新流程）
                has_translation = len(word.context_translation) > 0
                
                if self.language == NER.Language.ZH:
                    # 中文：直接基于原文分析
                    system_prompt = self.prompt_surface_analysis_without_translation.replace("{PROMPT_GROUPS}", "、".join(self.prompt_groups))
                    user_content = (
                        f"目标词语：{word.surface}"
                        + "\n" + f"参考文本原文：\n{context_original}"
                    )
                elif has_translation:
                    # 新流程：使用已翻译的参考文本进行分析
                    system_prompt = self.prompt_surface_analysis_with_context.replace("{PROMPT_GROUPS}", "、".join(self.prompt_groups))
                    context_translated = "\n".join(word.context_translation)
                    user_content = (
                        f"目标词语：{word.surface}"
                        + "\n" + f"参考文本原文：\n{context_original}"
                        + "\n" + f"参考文本译文：\n{context_translated}"
                    )
                else:
                    # 兼容：没有译文时，仍基于原文分析
                    system_prompt = self.prompt_surface_analysis_without_translation.replace("{PROMPT_GROUPS}", "、".join(self.prompt_groups))
                    user_content = (
                        f"目标词语：{word.surface}"
                        + "\n" + f"参考文本原文：\n{context_original}"
                    )

                error, usage, response_think, response_result, llm_request, llm_response = await self.do_request(
                    [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_content,
                        },
                    ],
                    LLM.SURFACE_ANALYSIS_CONFIG,
                    retry,
                    task_id=task_id,
                )

                if error is not None and ("首包超时" in str(error) or "流式响应超时" in str(error) or "请求超时" in str(error) or "任务超时" in str(error)):
                    elapsed = time_module.time() - start_time
                    word.context_shrink_level += 1
                    LogTable.print_llm_task(
                        task_name="词义分析",
                        word_surface=word.surface,
                        status="warning",
                        message=f"⏱️ 首包/流式超时 ({elapsed:.0f}s > {self.stream_first_chunk_timeout_seconds}s)，触发限缩: level {word.context_shrink_level}",
                        elapsed_time=elapsed,
                        extra_info={
                            "限缩级别": word.context_shrink_level,
                            "上下文长度": len(context_original),
                        },
                    )
                    raise Exception(f"任务超时 ({elapsed:.0f}s)，已触发上下文限缩 (level {word.context_shrink_level})")

                # 检查错误
                if error != None:
                    raise error

                # 检查是否超过最大 token 限制
                if usage.completion_tokens >= LLM.SURFACE_ANALYSIS_CONFIG.MAX_TOKENS:
                    raise Exception("返回结果错误（模型退化） ...")

                # 反序列化 JSON
                try:
                    # 尝试清理非 JSON 内容（有时模型会在 JSON 外输出废话）
                    clean_json = response_result
                    if "```json" in clean_json:
                        clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                    elif "```" in clean_json:
                        clean_json = clean_json.split("```")[1].split("```")[0].strip()
                    
                    # 尝试修复常见的 JSON 格式错误
                    # 1. 修复未转义的换行符
                    clean_json = clean_json.replace("\n", "\\n")
                    # 2. 修复中文引号
                    clean_json = clean_json.replace("“", '"').replace("”", '"')
                    
                    result = repair.loads(clean_json)
                except Exception as e:
                    LogHelper.warning(f"[JSON解析] 初次解析失败，尝试暴力修复: {e}")
                    # 暴力提取 JSON 对象
                    try:
                        json_match = re.search(r"\{.*\}", response_result, re.DOTALL)
                        if json_match:
                            clean_json = json_match.group(0)
                            result = repair.loads(clean_json)
                        else:
                            raise Exception("未找到有效的 JSON 对象")
                    except Exception as e2:
                        raise Exception(f"JSON 解析完全失败: {e2} | 原文: {response_result[:100]}...")

                if not isinstance(result, dict) or result == {}:
                    raise Exception("返回结果错误（数据结构） ...")

                # 清理一下格式
                for k, v in result.items():
                    result[k] = re.sub(r".*[:：]+", "", TextHelper.strip_punctuation(v))

                # 获取结果
                word.group = result.get("group", "")
                word.gender = result.get("gender", "")
                word.context_summary = result.get("summary", "")
                word.surface_translation = result.get("translation", "")
                word.llmrequest_surface_analysis = llm_request
                word.llmresponse_surface_analysis = llm_response

                # 【关键】假名残留检测：如果 translation 中包含有效的日文假名，说明翻译失败，需要重试
                # 使用 contains_kana_strict 允许孤立的拟声词字符（如单独的ッ、ヶ等）
                if LLM.contains_kana_strict(word.surface_translation):
                    raise Exception(f"假名残留: {word.surface_translation}")
                
                # 【关键】韩文残留检测
                if LLM.contains_korean(word.surface_translation):
                    raise Exception(f"韩文残留: {word.surface_translation}")
                
                # 【新增】退化检测：检查译名是否退化（重复字符）
                if LLM.is_degraded(word.surface_translation):
                    raise Exception(f"模型退化: {word.surface_translation}")
                
                # 【新增】相似度检测：译名与原名过于相似，说明未真正翻译
                SIMILARITY_THRESHOLD = 0.80
                similarity = LLM.check_similarity(word.surface, word.surface_translation)
                if similarity >= SIMILARITY_THRESHOLD:
                    raise Exception(f"翻译失效（相似度 {similarity:.2%}）")

                # 生成罗马音，汉字有时候会生成重复的罗马音，所以需要去重
                results = list(set([item.get("hepburn", "") for item in self.kakasi.convert(word.surface)]))
                word.surface_romaji = (" ".join(results)).strip()

                # 还原伪名
                fake_name_mapping_ex = {v: k for k, v in fake_name_mapping.items()}
                if word.surface in fake_name_mapping_ex:
                    word.surface = fake_name_mapping_ex.get(word.surface)
                    word.surface_romaji = ""
                    word.surface_translation = ""

                # 匹配实体类型
                matched = False
                for k, v in LLM.GROUP_MAPPING.items():
                    if word.group in set(v):
                        word.group = k
                        matched = True
                        break
                for k, v in LLM.GROUP_MAPPING_ADDITIONAL.items():
                    if word.group in set(v):
                        word.group = k
                        matched = True
                        break

                # 处理未命中目标类型的情况
                if matched == False:
                    # 检查是否属于黑名单类型（食品、物品、动物等）- 直接过滤掉，不重试
                    banned_types = set(v for vals in LLM.GROUP_MAPPING_BANNED.values() for v in vals)
                    if word.group in banned_types:
                        word.group = ""  # 设为空，后续会被过滤掉
                        # 打印详细日志：过滤非目标实体
                        LogTable.print_llm_task(
                            task_name="词义分析",
                            word_surface=word.surface,
                            status="info",
                            message=f"过滤非目标实体类型: {result.get('group', '')}",
                            input_tokens=usage.prompt_tokens if usage else 0,
                            output_tokens=usage.completion_tokens if usage else 0,
                            elapsed_time=time_module.time() - start_time,
                            extra_info={"原始分类": result.get("group", ""), "判定": "黑名单类型"},
                        )
                    elif last_round == True:
                        word.group = ""
                        LogTable.print_llm_task(
                            task_name="词义分析",
                            word_surface=word.surface,
                            status="warning",
                            message=f"无法匹配的实体类型: {result.get('group', '')}（最后一轮）",
                            response_content=response_result,
                            input_tokens=usage.prompt_tokens if usage else 0,
                            output_tokens=usage.completion_tokens if usage else 0,
                            elapsed_time=time_module.time() - start_time,
                        )
                    else:
                        error = Exception("无法匹配的实体类型 ...")
                        LogTable.print_llm_task(
                            task_name="词义分析",
                            word_surface=word.surface,
                            status="warning",
                            message=f"无法匹配的实体类型: {result.get('group', '')}，将重试",
                            response_content=response_result,
                            input_tokens=usage.prompt_tokens if usage else 0,
                            output_tokens=usage.completion_tokens if usage else 0,
                            elapsed_time=time_module.time() - start_time,
                        )
                else:
                    # 成功：打印详细日志
                    LogTable.print_llm_task(
                        task_name="词义分析",
                        word_surface=word.surface,
                        status="success",
                        message=f"{word.surface} → {word.surface_translation} [{word.group}]",
                        request_content=user_content,
                        response_content=response_result,
                        response_think=response_think,
                        input_tokens=usage.prompt_tokens if usage else 0,
                        output_tokens=usage.completion_tokens if usage else 0,
                        elapsed_time=time_module.time() - start_time,
                        extra_info={
                            "译名": word.surface_translation,
                            "类型": word.group,
                            "性别": word.gender,
                            "摘要": word.context_summary[:50] + "..." if len(word.context_summary) > 50 else word.context_summary,
                        },
                    )
            except Exception as e:
                # 失败：打印详细错误日志
                try:
                    ErrorLogger.log(
                        error_type="SurfaceAnalysisError",
                        message=str(e),
                        context={
                            "task_id": task_id,
                            "word_surface": getattr(word, "surface", ""),
                            "context_shrink_level": getattr(word, "context_shrink_level", 0),
                            "retry": bool(retry),
                            "last_round": bool(last_round),
                            "request_content": user_content if "user_content" in dir() else "",
                            "response_think": response_think if "response_think" in dir() else "",
                            "response_result": response_result if "response_result" in dir() else "",
                            "usage": {
                                "prompt_tokens": getattr(usage, "prompt_tokens", 0) if "usage" in dir() and usage else 0,
                                "completion_tokens": getattr(usage, "completion_tokens", 0) if "usage" in dir() and usage else 0,
                            },
                            "traceback": LogHelper.get_trackback(e),
                        },
                    )
                except Exception:
                    pass
                LogTable.print_llm_task(
                    task_name="词义分析",
                    word_surface=word.surface,
                    status="error",
                    message=f"任务失败: {str(e)}",
                    request_content=user_content if 'user_content' in dir() else None,
                    response_content=response_result if 'response_result' in dir() else None,
                    input_tokens=usage.prompt_tokens if usage and 'usage' in dir() else 0,
                    output_tokens=usage.completion_tokens if usage and 'usage' in dir() else 0,
                    elapsed_time=time_module.time() - start_time,
                )
                error = e
        return error

    # 批量执行词义分析任务
    async def translate_and_surface_analysis_batch(self, words: list[Word], fake_name_mapping: dict[str, str]) -> list[Word]:
        import time as time_module
        batch_start_time = time_module.time()

        failure: list[Word] = []
        success: list[Word] = []

        LogTable.print_stage_header("参考翻译 + 词义分析", stage_num=1)
        LogHelper.info(f"待处理词条数: {len(words)}")
        LogHelper.print("")

        translate_total = len(words) if self.language != NER.Language.ZH else 0
        with TaskTracker(
            total=len(words),
            task_name="翻译+校对",
            max_concurrent=self.max_concurrent_requests,
            translate_total=translate_total,
            review_total=len(words),
        ) as tracker:
            set_current_tracker(tracker)
            try:
                ct_attempts: dict[int, int] = {i: 0 for i in range(len(words))}
                sa_attempts: dict[int, int] = {i: 0 for i in range(len(words))}
                task_ids: dict[int, str] = {i: f"task_{i}_{words[i].surface}" for i in range(len(words))}

                queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
                for i in range(len(words)):
                    queue.put_nowait((0.0, i))

                worker_count = self._get_worker_count(len(words))

                async def worker() -> None:
                    while True:
                        ready_at, idx = await queue.get()
                        try:
                            if idx is None:
                                return
                            now = time_module.time()
                            if ready_at > now:
                                await asyncio.sleep(ready_at - now)

                            word = words[idx]
                            task_id = task_ids[idx]

                            if self.language != NER.Language.ZH:
                                err = await self.context_translate(word, words, retry=ct_attempts[idx] > 0, task_id=task_id)
                                if err is not None:
                                    tracker.complete_task(task_id, success=False, error=str(err))
                                    ct_attempts[idx] += 1
                                    if ct_attempts[idx] <= LLM.MAX_RETRY:
                                        tracker.reopen_task(task_id)
                                        delay = self._compute_retry_delay_seconds(err, ct_attempts[idx])
                                        queue.put_nowait((time_module.time() + delay, idx))
                                    else:
                                        failure.append(word)
                                    continue
                                tracker.mark_translated(task_id)

                            err = await self.surface_analysis(
                                word,
                                words,
                                fake_name_mapping,
                                retry=sa_attempts[idx] > 0,
                                last_round=sa_attempts[idx] >= LLM.MAX_RETRY,
                                task_id=task_id,
                            )
                            if err is None:
                                success.append(word)
                                tracker.complete_task(task_id, success=True)
                                continue

                            tracker.complete_task(task_id, success=False, error=str(err))
                            sa_attempts[idx] += 1

                            if sa_attempts[idx] >= LLM.FORCE_TRANSLITERATE_THRESHOLD:
                                if word.surface_translation and LLM.contains_kana_strict(word.surface_translation):
                                    LogTable.print_retry_info(
                                        word_surface=word.surface,
                                        retry_count=sa_attempts[idx],
                                        max_retry=LLM.FORCE_TRANSLITERATE_THRESHOLD,
                                        reason="触发强制音译",
                                    )
                                    word.surface_translation = self.force_transliterate(word.surface)
                                tracker.reopen_task(task_id)
                                success.append(word)
                                tracker.complete_task(task_id, success=True)
                                continue

                            if sa_attempts[idx] <= LLM.MAX_RETRY:
                                tracker.reopen_task(task_id)
                                delay = self._compute_retry_delay_seconds(err, sa_attempts[idx])
                                queue.put_nowait((time_module.time() + delay, idx))
                            else:
                                failure.append(word)
                        finally:
                            queue.task_done()

                workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
                await queue.join()
                for _ in range(worker_count):
                    queue.put_nowait((0.0, None))
                await asyncio.gather(*workers)
            finally:
                set_current_tracker(None)

        LogTable.print_batch_summary(
            task_name="翻译+校对",
            total=len(words),
            success=len(success),
            failed=len(failure),
            elapsed_time=time_module.time() - batch_start_time,
        )

        return words

    async def surface_analysis_batch(self, words: list[Word], fake_name_mapping: dict[str, str]) -> list[Word]:
        import time as time_module
        batch_start_time = time_module.time()
        
        failure: list[Word] = []
        success: list[Word] = []

        # 打印阶段标题
        LogTable.print_stage_header("词义分析", stage_num=2)
        LogHelper.info(f"待处理词条数: {len(words)}")
        LogHelper.print("")
        
        # 使用 TaskTracker 替代 Progress，实现常驻底部面板
        with TaskTracker(total=len(words), task_name="词义分析", max_concurrent=self.max_concurrent_requests) as tracker:
            # 设置全局 tracker，供流式请求更新
            set_current_tracker(tracker)
            
            try:
                attempts: dict[int, int] = {i: 0 for i in range(len(words))}
                task_ids: dict[int, str] = {i: f"sa_{i}_{words[i].surface}" for i in range(len(words))}

                queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
                for i in range(len(words)):
                    queue.put_nowait((0.0, i))

                worker_count = self._get_worker_count(len(words))

                async def worker() -> None:
                    while True:
                        ready_at, idx = await queue.get()
                        try:
                            if idx is None:
                                return
                            now = time_module.time()
                            if ready_at > now:
                                await asyncio.sleep(ready_at - now)

                            word = words[idx]
                            task_id = task_ids[idx]
                            retry = attempts[idx] > 0
                            last_round = attempts[idx] >= LLM.MAX_RETRY

                            err = await self.surface_analysis(word, words, fake_name_mapping, retry, last_round, task_id=task_id)
                            if err is None:
                                success.append(word)
                                tracker.complete_task(task_id, success=True)
                                continue

                            tracker.complete_task(task_id, success=False, error=str(err))
                            attempts[idx] += 1

                            if attempts[idx] >= LLM.FORCE_TRANSLITERATE_THRESHOLD:
                                if word.surface_translation and LLM.contains_kana_strict(word.surface_translation):
                                    LogTable.print_retry_info(
                                        word_surface=word.surface,
                                        retry_count=attempts[idx],
                                        max_retry=LLM.FORCE_TRANSLITERATE_THRESHOLD,
                                        reason="触发强制音译",
                                    )
                                    word.surface_translation = self.force_transliterate(word.surface)
                                tracker.reopen_task(task_id)
                                success.append(word)
                                tracker.complete_task(task_id, success=True)
                                continue

                            if attempts[idx] <= LLM.MAX_RETRY:
                                tracker.reopen_task(task_id)
                                delay = self._compute_retry_delay_seconds(err, attempts[idx])
                                queue.put_nowait((time_module.time() + delay, idx))
                            else:
                                failure.append(word)
                        finally:
                            queue.task_done()

                workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
                await queue.join()
                for _ in range(worker_count):
                    queue.put_nowait((0.0, None))
                await asyncio.gather(*workers)
            finally:
                # 清除全局 tracker
                set_current_tracker(None)
        
        # 【最终强制音译兜底】对于最终仍失败的词条，如果其 surface_translation 仍包含有效假名，强制音译
        forced_count = 0
        for word in words:
            if word.surface_translation and LLM.contains_kana_strict(word.surface_translation):
                LogTable.print_llm_task(
                    task_name="强制音译兜底",
                    word_surface=word.surface,
                    status="warning",
                    message=f"{word.surface_translation} → {self.force_transliterate(word.surface)}",
                )
                word.surface_translation = self.force_transliterate(word.surface)
                forced_count += 1

        # 打印批量汇总
        LogTable.print_batch_summary(
            task_name="词义分析",
            total=len(words),
            success=len(success),
            failed=len(failure),
            elapsed_time=time_module.time() - batch_start_time,
        )
        
        if forced_count > 0:
            LogHelper.info(f"[强制音译] 共处理 {forced_count} 个词条")

        return words
    
    # ============== 强制音译兜底机制 ==============
    
    # 罗马音→中文音节对照表
    ROMAJI_TO_CN = {
        # 元音
        "a": "阿", "i": "伊", "u": "乌", "e": "艾", "o": "欧",
        # 辅音 + 元音
        "ka": "卡", "ki": "奇", "ku": "库", "ke": "科", "ko": "科",
        "sa": "萨", "si": "西", "shi": "西", "su": "苏", "se": "塞", "so": "索",
        "ta": "塔", "ti": "提", "chi": "奇", "tu": "图", "tsu": "兹", "te": "特", "to": "托",
        "na": "娜", "ni": "尼", "nu": "努", "ne": "内", "no": "诺",
        "ha": "哈", "hi": "希", "hu": "胡", "fu": "福", "he": "黑", "ho": "霍",
        "ma": "玛", "mi": "米", "mu": "穆", "me": "梅", "mo": "莫",
        "ya": "亚", "yi": "伊", "yu": "尤", "ye": "耶", "yo": "约",
        "ra": "拉", "ri": "里", "ru": "鲁", "re": "雷", "ro": "罗",
        "wa": "瓦", "wi": "韦", "we": "维", "wo": "沃",
        "n": "恩", "nn": "恩",
        # 浊音
        "ga": "加", "gi": "基", "gu": "古", "ge": "格", "go": "戈",
        "za": "扎", "zi": "吉", "ji": "吉", "zu": "祖", "ze": "泽", "zo": "佐",
        "da": "达", "di": "迪", "du": "杜", "de": "德", "do": "多",
        "ba": "巴", "bi": "比", "bu": "布", "be": "贝", "bo": "博",
        "pa": "帕", "pi": "皮", "pu": "普", "pe": "培", "po": "波",
        # 拗音（常见）
        "kya": "基亚", "kyu": "基尤", "kyo": "基欧",
        "sha": "夏", "shu": "修", "sho": "肖",
        "cha": "查", "chu": "丘", "cho": "乔",
        "nya": "尼亚", "nyu": "纽", "nyo": "纽",
        "hya": "希亚", "hyu": "休", "hyo": "秀",
        "mya": "米亚", "myu": "缪", "myo": "妙",
        "rya": "利亚", "ryu": "琉", "ryo": "良",
        "gya": "贾", "gyu": "久", "gyo": "乔",
        "ja": "贾", "ju": "朱", "jo": "乔",
        "bya": "比亚", "byu": "标", "byo": "表",
        "pya": "皮亚", "pyu": "飘", "pyo": "票",
    }
    
    def force_transliterate(self, surface: str) -> str:
        """
        将日文词汇强制转换为中文音译
        使用 pykakasi 转罗马音，然后映射到中文音节
        """
        try:
            # 转换为罗马音
            results = self.kakasi.convert(surface)
            romaji = "".join([item.get("hepburn", "") for item in results])
            
            if not romaji:
                return surface  # 无法转换，返回原文
            
            # 将罗马音转换为中文音节
            return self._romaji_to_chinese(romaji.lower())
        except Exception as e:
            LogHelper.warning(f"[强制音译] 转换失败: {surface} - {e}")
            return surface
    
    def _romaji_to_chinese(self, romaji: str) -> str:
        """将罗马音字符串转换为中文音节"""
        result = []
        i = 0
        while i < len(romaji):
            matched = False
            # 优先尝试匹配较长的音节（3字符 → 2字符 → 1字符）
            for length in [3, 2, 1]:
                if i + length <= len(romaji):
                    segment = romaji[i:i+length]
                    if segment in LLM.ROMAJI_TO_CN:
                        result.append(LLM.ROMAJI_TO_CN[segment])
                        i += length
                        matched = True
                        break
            if not matched:
                # 跳过无法匹配的字符（如促音等）
                i += 1
        
        return "".join(result) if result else romaji

    # 参考文本翻译任务（新流程：在词义分析之前执行，为其提供上下文理解）
    async def context_translate(self, word: Word, words: list[Word], retry: bool, task_id: str = "") -> Exception | None:
        import time as time_module
        start_time = time_module.time()
        context_str = ""
        response_think = ""
        response_result = ""
        usage = None
        error = None
        
        if True:
            try:
                # 获取参考文本（使用增加后的 token 阈值）
                # 【关键】context_shrink_level 会影响采样数量，返回本次使用的索引
                context_str, sampled_indices = word.get_context_str_for_translate(self.language)
                
                # 如果没有可用的上下文了（所有索引都被排除），降级处理
                if not context_str or not sampled_indices:
                    LogTable.print_llm_task(
                        task_name="参考文本翻译",
                        word_surface=word.surface,
                        status="warning",
                        message="所有上下文都已被排除，跳过翻译阶段",
                        elapsed_time=time_module.time() - start_time,
                    )
                    word.context_translation = []
                    return
                
                error, usage, response_think, response_result, llm_request, llm_response = await self.do_request(
                    [
                        {
                            "role": "system",
                            "content": self.prompt_context_translate,
                        },
                        {
                            "role": "user",
                            "content": f"参考上下文：\n{context_str}",
                        },
                    ],
                    LLM.CONTEXT_TRANSLATE_CONFIG,
                    retry,
                    task_id=task_id,
                )

                if error is not None and ("首包超时" in str(error) or "流式响应超时" in str(error) or "请求超时" in str(error) or "任务超时" in str(error)):
                    elapsed = time_module.time() - start_time
                    for idx in sampled_indices:
                        word.failed_context_indices.add(idx)
                    word.context_shrink_level += 1
                    LogTable.print_llm_task(
                        task_name="参考文本翻译",
                        word_surface=word.surface,
                        status="warning",
                        message=f"⏱️ 首包/流式超时 ({elapsed:.0f}s > {self.stream_first_chunk_timeout_seconds}s)，触发限缩: level {word.context_shrink_level}",
                        request_content=context_str[:200] + "..." if len(context_str) > 200 else context_str,
                        elapsed_time=elapsed,
                        extra_info={
                            "限缩级别": word.context_shrink_level,
                            "已排除索引": len(word.failed_context_indices),
                            "采样索引": str(sampled_indices),
                        },
                    )
                    raise Exception(f"任务超时 ({elapsed:.0f}s)，已触发上下文限缩 (level {word.context_shrink_level})")

                if error != None:
                    # 【关键】检测敏感内容过滤错误，触发限缩策略
                    error_str = str(error)
                    if "contentFilter" in error_str or "'1301'" in error_str or "content_filter" in error_str:
                        # 记录本次失败的上下文索引
                        for idx in sampled_indices:
                            word.failed_context_indices.add(idx)
                        
                        # 提升限缩级别
                        if word.context_shrink_level < 3:
                            word.context_shrink_level += 1
                            LogTable.print_llm_task(
                                task_name="参考文本翻译",
                                word_surface=word.surface,
                                status="warning",
                                message=f"敏感内容过滤触发，限缩级别: {word.context_shrink_level}，已排除 {len(word.failed_context_indices)} 个索引",
                                request_content=context_str,
                                elapsed_time=time_module.time() - start_time,
                            )
                        else:
                            LogTable.print_llm_task(
                                task_name="参考文本翻译",
                                word_surface=word.surface,
                                status="warning",
                                message=f"敏感内容过滤触发，单条重采样模式，已排除 {len(word.failed_context_indices)} 个索引",
                                request_content=context_str,
                                elapsed_time=time_module.time() - start_time,
                            )
                    raise error

                # 检查是否超过最大 token 限制
                if usage.completion_tokens >= LLM.CONTEXT_TRANSLATE_CONFIG.MAX_TOKENS:
                    raise Exception("返回结果错误（模型退化） ...")

                try:
                    context_translation, parse_detail = LLM.parse_context_translate_output(context_str, response_result)
                except Exception as e:
                    try:
                        ErrorLogger.log(
                            error_type="ContextTranslateParseError",
                            message=str(e),
                            context={
                                "task_id": task_id,
                                "word_surface": getattr(word, "surface", ""),
                                "context_shrink_level": getattr(word, "context_shrink_level", 0),
                                "sampled_indices": str(sampled_indices) if "sampled_indices" in locals() else "",
                                "context_str": context_str,
                                "response_result": response_result,
                                "traceback": LogHelper.get_trackback(e),
                            },
                        )
                    except Exception:
                        pass
                    raise
                
                # 【新增】退化检测：检查整体输出是否存在重复模式
                if LLM.is_degraded(response_result):
                    raise Exception("模型退化（输出重复内容）")

                if not context_translation:
                    raise Exception("返回结果错误（空回复） ...")

                missing_contexts = parse_detail.get("missing_contexts") if isinstance(parse_detail, dict) else []
                expected_contexts = parse_detail.get("expected_contexts") if isinstance(parse_detail, dict) else 0
                if isinstance(missing_contexts, list) and isinstance(expected_contexts, int) and expected_contexts > 0:
                    if len(missing_contexts) >= max(3, expected_contexts // 3):
                        # C:\LLM\BookTerm Gacha - Experiment 不强制要求行数/上下文完全对应
                        # 仅记录警告，不抛出异常，以免浪费并发资源
                        LogTable.print_llm_task(
                            task_name="参考文本翻译",
                            word_surface=word.surface,
                            status="warning",
                            message=f"上下文缺失 {len(missing_contexts)}/{expected_contexts} (非致命)，继续处理",
                        )

                body_text = "\n".join([line for line in context_translation if not (line or "").strip().startswith("【")]).strip()
                kana_ratio = LLM._calc_kana_ratio(body_text)
                if kana_ratio >= 0.05:
                    raise Exception(f"翻译失效（假名残留过多，{kana_ratio:.1%}）")
                
                # 【新增】相似度检测：逐行检查，高相似度说明未真正翻译
                SIMILARITY_THRESHOLD = 0.80
                original_lines = [line.strip() for line in context_str.splitlines() if line.strip() != ""]
                header_re = re.compile(r"^【\s*上下文\s*\d+\s*】$")
                
                # 只在行数匹配时做逐行相似度检测
                if len(context_translation) == len(original_lines):
                    high_similarity_count = 0
                    for orig, trans in zip(original_lines, context_translation):
                        if header_re.match(orig) or header_re.match(trans):
                            continue
                        if re.search(r"[\u3040-\u30FFA-Za-z]", orig) is None:
                            continue
                        similarity = LLM.check_similarity(orig, trans)
                        if similarity >= SIMILARITY_THRESHOLD:
                            high_similarity_count += 1
                    
                    # 超过50%的行高相似度，认为翻译基本失效
                    if high_similarity_count > len(original_lines) * 0.5:
                        # 再次检查是否包含中文（避免误判）
                        # 计算汉字比例（排除标点等）
                        cn_char_count = len(re.findall(r"[\u4e00-\u9fa5]", body_text))
                        total_char_count = len(body_text)
                        cn_ratio = cn_char_count / total_char_count if total_char_count > 0 else 0
                        
                        if cn_ratio < 0.2: # 如果中文比例确实很低，才报错
                            raise Exception(f"翻译失效（相似度过高且中文比例低 {cn_ratio:.1%}，{high_similarity_count}/{len(original_lines)} 行）")
                        else:
                            LogTable.print_llm_task(
                                task_name="参考文本翻译",
                                word_surface=word.surface,
                                status="warning",
                                message=f"相似度较高但包含中文 ({cn_ratio:.1%})，保留结果",
                            )
                
                # 检测翻译是否失效：仅当译文与原文【完全相同】且【原文确实包含明显外文成分】时才报错。
                has_obvious_foreign = re.search(r"[\u3040-\u30FF\uAC00-\uD7AFA-Za-z]", context_str) is not None
                if len(context_translation) > 0 and context_translation == original_lines and has_obvious_foreign:
                    # 同样增加中文比例检查
                    cn_char_count = len(re.findall(r"[\u4e00-\u9fa5]", body_text))
                    total_char_count = len(body_text)
                    cn_ratio = cn_char_count / total_char_count if total_char_count > 0 else 0
                    
                    if cn_ratio < 0.2:
                        raise Exception("翻译失效（译文与原文相同） ...")

                word.context_translation = context_translation
                word.llmrequest_context_translate = llm_request
                word.llmresponse_context_translate = llm_response
                
                # 成功：打印详细日志
                LogTable.print_llm_task(
                    task_name="参考文本翻译",
                    word_surface=word.surface,
                    status="success",
                    message=f"上下文翻译完成，{len(context_translation)} 行",
                    request_content=context_str,
                    response_content=response_result,
                    response_think=response_think,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    elapsed_time=time_module.time() - start_time,
                    extra_info={
                        "原文行数": len(original_lines),
                        "译文行数": len(context_translation),
                        "采样索引": str(sampled_indices),
                    },
                )
            except Exception as e:
                # 失败：打印详细错误日志
                try:
                    ErrorLogger.log(
                        error_type="ContextTranslateError",
                        message=str(e),
                        context={
                            "task_id": task_id,
                            "word_surface": getattr(word, "surface", ""),
                            "context_shrink_level": getattr(word, "context_shrink_level", 0),
                            "failed_context_indices_count": len(getattr(word, "failed_context_indices", set()) or set()),
                            "sampled_indices": str(sampled_indices) if "sampled_indices" in locals() else "",
                            "context_str": context_str,
                            "response_think": response_think,
                            "response_result": response_result,
                            "usage": {
                                "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                            },
                            "traceback": LogHelper.get_trackback(e),
                        },
                    )
                except Exception:
                    pass
                LogTable.print_llm_task(
                    task_name="参考文本翻译",
                    word_surface=word.surface,
                    status="error",
                    message=f"任务失败: {str(e)}",
                    request_content=context_str if context_str else None,
                    response_content=response_result if response_result else None,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    elapsed_time=time_module.time() - start_time,
                )
                error = e
        return error

    # 批量执行参考文本翻译任务
    async def context_translate_batch(self, words: list[Word]) -> list[Word]:
        import time as time_module
        batch_start_time = time_module.time()
        
        failure: list[Word] = []
        success: list[Word] = []

        # 打印阶段标题
        LogTable.print_stage_header("参考文本翻译", stage_num=1)
        LogHelper.info(f"待处理词条数: {len(words)}")
        LogHelper.print("")
        
        # 使用 TaskTracker 替代 Progress，实现常驻底部面板
        with TaskTracker(total=len(words), task_name="参考文本翻译", max_concurrent=self.max_concurrent_requests) as tracker:
            # 设置全局 tracker，供流式请求更新
            set_current_tracker(tracker)
            
            try:
                attempts: dict[int, int] = {i: 0 for i in range(len(words))}
                task_ids: dict[int, str] = {i: f"ct_{i}_{words[i].surface}" for i in range(len(words))}

                queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
                for i in range(len(words)):
                    queue.put_nowait((0.0, i))

                worker_count = self._get_worker_count(len(words))

                async def worker() -> None:
                    while True:
                        ready_at, idx = await queue.get()
                        try:
                            if idx is None:
                                return
                            now = time_module.time()
                            if ready_at > now:
                                await asyncio.sleep(ready_at - now)

                            word = words[idx]
                            task_id = task_ids[idx]
                            retry = attempts[idx] > 0

                            err = await self.context_translate(word, words, retry, task_id=task_id)
                            if err is None:
                                success.append(word)
                                tracker.complete_task(task_id, success=True)
                                continue

                            tracker.complete_task(task_id, success=False, error=str(err))
                            attempts[idx] += 1

                            if attempts[idx] <= LLM.MAX_RETRY:
                                tracker.reopen_task(task_id)
                                delay = self._compute_retry_delay_seconds(err, attempts[idx])
                                queue.put_nowait((time_module.time() + delay, idx))
                            else:
                                failure.append(word)
                        finally:
                            queue.task_done()

                workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
                await queue.join()
                for _ in range(worker_count):
                    queue.put_nowait((0.0, None))
                await asyncio.gather(*workers)
            finally:
                # 清除全局 tracker
                set_current_tracker(None)
        
        # 打印批量汇总
        LogTable.print_batch_summary(
            task_name="参考文本翻译",
            total=len(words),
            success=len(success),
            failed=len(failure),
            elapsed_time=time_module.time() - batch_start_time,
        )

        return words

    # ============== 第三阶段：问题修复（全自动） ==============
    
    # 问题修复配置（类级别定义）
    class FixConfig:
        TEMPERATURE: float = 0.3    # 低温度，更确定性
        TOP_P: float = 0.95
        MAX_TOKENS: int = 256
        FREQUENCY_PENALTY: float = 0.0
    
    def load_fix_prompt(self) -> None:
        """加载问题修复 prompt"""
        try:
            with open("prompt/prompt_fix_translation.txt", "r", encoding="utf-8-sig") as f:
                self.prompt_fix_translation = f.read().strip()
        except:
            self.prompt_fix_translation = ""
            LogHelper.warning("[问题修复] 未找到 prompt_fix_translation.txt")
    
    async def fix_translation(self, word: Word, success: list[Word], total: int) -> None:
        """
        修复单个词条的翻译问题
        
        Args:
            word: 待修复的词条
            success: 成功列表
            total: 总数
        """
        import time as time_module
        start_time = time_module.time()
        
        if True:
            error = None
            llm_request = {}
            llm_response = {}
            response_result = ""
            usage = None
            
            try:
                # 构建修复请求
                context_original = word.get_context_str_for_surface_analysis(self.language)
                context_translation = "\n".join(word.context_translation) if word.context_translation else "(无)"
                
                user_content = (
                    f"目标词语：{word.surface}"
                    f"\n当前译名：{word.surface_translation}"
                    f"\n罗马音：{word.surface_romaji}"
                    f"\n实体类型：{word.group}"
                    f"\n词义摘要：{word.context_summary}"
                    f"\n\n参考上下文原文：\n{context_original}"
                    f"\n\n参考上下文译文：\n{context_translation}"
                )
                
                # 创建修复配置对象
                fix_config = LLM.LLMConfig()
                fix_config.TEMPERATURE = LLM.FixConfig.TEMPERATURE
                fix_config.TOP_P = LLM.FixConfig.TOP_P
                fix_config.MAX_TOKENS = LLM.FixConfig.MAX_TOKENS
                fix_config.FREQUENCY_PENALTY = LLM.FixConfig.FREQUENCY_PENALTY
                
                error, usage, response_think, response_result, llm_request, llm_response = await self.do_request(
                    [
                        {
                            "role": "system",
                            "content": self.prompt_fix_translation,
                        },
                        {
                            "role": "user",
                            "content": user_content,
                        },
                    ],
                    fix_config,
                    True  # retry mode
                )
                
                if error is not None:
                    raise error
                
                # 反序列化 JSON
                try:
                    # 尝试清理非 JSON 内容（有时模型会在 JSON 外输出废话）
                    clean_json = response_result
                    if "```json" in clean_json:
                        clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                    elif "```" in clean_json:
                        clean_json = clean_json.split("```")[1].split("```")[0].strip()
                    
                    # 尝试修复常见的 JSON 格式错误
                    # 1. 修复未转义的换行符
                    clean_json = clean_json.replace("\n", "\\n")
                    # 2. 修复中文引号
                    clean_json = clean_json.replace("“", '"').replace("”", '"')
                    
                    result = repair.loads(clean_json)
                except Exception as e:
                    LogHelper.warning(f"[JSON解析] 初次解析失败，尝试暴力修复: {e}")
                    # 暴力提取 JSON 对象
                    try:
                        json_match = re.search(r"\{.*\}", response_result, re.DOTALL)
                        if json_match:
                            clean_json = json_match.group(0)
                            result = repair.loads(clean_json)
                        else:
                            raise Exception("未找到有效的 JSON 对象")
                    except Exception as e2:
                        raise Exception(f"JSON 解析完全失败: {e2} | 原文: {response_result[:100]}...")

                if not isinstance(result, dict):
                    raise Exception("返回结果错误（数据结构）")
                
                new_translation = result.get("translation", "")
                confidence = result.get("confidence", "low")
                
                # 验证新译名是否仍有假名
                if LLM.contains_kana_strict(new_translation):
                    new_translation = self.force_transliterate(word.surface)
                    confidence = "forced"
                
                # 更新译名
                old_translation = word.surface_translation
                word.surface_translation = new_translation
                word.llmrequest_fix = llm_request
                word.llmresponse_fix = llm_response
                
                # 打印成功日志
                LogTable.print_llm_task(
                    task_name="问题修复",
                    word_surface=word.surface,
                    status="success",
                    message=f"{old_translation} → {new_translation} [{confidence}]",
                    request_content=user_content,
                    response_content=response_result,
                    response_think=response_think,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    elapsed_time=time_module.time() - start_time,
                    extra_info={
                        "原译名": old_translation,
                        "新译名": new_translation,
                        "置信度": confidence,
                    },
                )
                
            except Exception as e:
                # 修复失败，使用强制音译
                forced_translation = self.force_transliterate(word.surface)
                LogTable.print_llm_task(
                    task_name="问题修复",
                    word_surface=word.surface,
                    status="warning",
                    message=f"LLM 失败，强制音译: {word.surface_translation} → {forced_translation}",
                    request_content=user_content if 'user_content' in dir() else None,
                    response_content=response_result if response_result else None,
                    elapsed_time=time_module.time() - start_time,
                    extra_info={"错误": str(e)},
                )
                word.surface_translation = forced_translation
                error = None  # 强制音译成功，不算错误
            finally:
                if error is None:
                    with self.lock:
                        success.append(word)
    
    async def fix_translation_batch(self, words: list[Word]) -> list[Word]:
        """
        批量执行问题修复
        
        筛选条件：
        1. 译名包含有效假名（使用严格检测）
        2. 译名与原名相似度过高（>= 80%）
        3. 译名为空
        
        Args:
            words: 所有词条
            
        Returns:
            修复后的词条列表
        """
        import time as time_module
        batch_start_time = time_module.time()
        
        # 加载 prompt
        self.load_fix_prompt()
        if not self.prompt_fix_translation:
            LogHelper.warning("[问题修复] 跳过（无 prompt）")
            return words
        
        # 打印阶段标题
        LogTable.print_stage_header("问题修复", stage_num=3)
        
        # 筛选问题词条
        problem_words = []
        problem_reasons = {}
        for word in words:
            translation = word.surface_translation or ""
            
            # 条件1: 假名残留
            if LLM.contains_kana_strict(translation):
                problem_words.append(word)
                problem_reasons[word.surface] = "假名残留"
                continue
            
            # 条件2: 相似度过高
            if translation and LLM.check_similarity(word.surface, translation) >= 0.80:
                problem_words.append(word)
                problem_reasons[word.surface] = "相似度过高"
                continue
            
            # 条件3: 空翻译
            if not translation.strip():
                problem_words.append(word)
                problem_reasons[word.surface] = "空翻译"
                continue
        
        if not problem_words:
            LogHelper.info("[问题修复] 无需修复，所有词条正常")
            return words
        
        # 打印问题词条列表
        LogHelper.info(f"发现 {len(problem_words)} 个问题词条:")
        for word in problem_words:
            reason = problem_reasons.get(word.surface, "未知")
            LogHelper.print(f"  • {word.surface} → {word.surface_translation or '(空)'} [dim]({reason})[/dim]")
        LogHelper.print("")
        
        success = []
        tasks = [
            asyncio.create_task(self.fix_translation(word, success, len(problem_words)))
            for word in problem_words
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # 最终兜底：对仍有假名的词条强制音译
        forced_count = 0
        for word in words:
            if word.surface_translation and LLM.contains_kana_strict(word.surface_translation):
                old_trans = word.surface_translation
                word.surface_translation = self.force_transliterate(word.surface)
                LogTable.print_llm_task(
                    task_name="最终兜底",
                    word_surface=word.surface,
                    status="warning",
                    message=f"强制音译: {old_trans} → {word.surface_translation}",
                )
                forced_count += 1
        
        # 打印批量汇总
        LogTable.print_batch_summary(
            task_name="问题修复",
            total=len(problem_words),
            success=len(success),
            failed=len(problem_words) - len(success),
            elapsed_time=time_module.time() - batch_start_time,
        )
        
        if forced_count > 0:
            LogHelper.info(f"[最终兜底] 强制音译处理了 {forced_count} 个词条")
        
        return words
