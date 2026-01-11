"""
BookTerm Gacha - LLM Agent Module
=================================

This module implements the core LLM (Large Language Model) agent for semantic analysis
and translation of extracted terms. It handles:

- API communication with OpenAI-compatible LLM services
- Surface form analysis (determining term types: character names, locations, etc.)
- Context translation for better understanding of terms
- Kana residue detection and handling (Japanese text processing)
- Retry logic with forced transliteration fallback
- Rate limiting and concurrent request management

Key Classes:
    LLM: Main agent class that orchestrates all LLM-related operations

Dependencies:
    - openai: Async OpenAI client for API communication
    - pykakasi: Japanese text romanization
    - aiolimiter: Async rate limiting
    - rich: Progress bar display

Based on KeywordGacha v0.13.1 by neavo
https://github.com/neavo/KeywordGacha
"""

import re
import asyncio
import threading
import urllib.request
from types import SimpleNamespace

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

class LLM:

    # 任务类型
    class Type(BaseData):

        API_TEST: int = 100                  # 语义分析
        SURFACE_ANALYSIS: int = 200          # 语义分析
        TRANSLATE_CONTEXT: int = 300         # 翻译参考文本

    # LLM 配置参数
    class LLMConfig(BaseData):

        TEMPERATURE: float = 0.05
        TOP_P: float = 0.95
        MAX_TOKENS: float = 1024
        FREQUENCY_PENALTY: float = 0.0

    # 最大重试次数（减少到 8 次，超过后使用强制音译兜底）
    MAX_RETRY: int = 16
    
    # 强制音译阈值（超过此重试次数后，对问题词条直接使用强制音译）
    FORCE_TRANSLITERATE_THRESHOLD: int = 16

    # 类型映射表 - 只保留 "角色" 和 "地点" 两种类型
    GROUP_MAPPING = {
        "角色" : ["姓氏", "名字"],
        "地点" : ["地点", "建筑", "设施"],
    }
    GROUP_MAPPING_BANNED = {
        "黑名单" : [
            # 一般性排除
            "行为", "活动", "其他", "无法判断",
            # 组织/群体类
            "组织", "群体", "家族", "种族", "团体", "部门", "公司", "学校", "部活", "社团",
            # 物品类
            "物品", "食品", "工具", "道具", "食物", "饮品", "饮料", "衣物", "服装", "家具", "电器", "器具",
            # 生物类（非人类角色）
            "生物", "植物", "动物", "怪物", "魔物", "妖怪", "精灵", "魔兽",
        ],
    }
    GROUP_MAPPING_ADDITIONAL = {
        "角色" : ["角色", "人", "人物", "人名"],
        "地点" : [],
    }

    def __init__(self, config: SimpleNamespace) -> None:
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.model_name = config.model_name
        self.request_timeout = config.request_timeout
        self.request_frequency_threshold = config.request_frequency_threshold

        # 初始化
        self.kakasi = pykakasi.kakasi()
        self.client = self.load_client()

        # 线程锁
        self.lock = threading.Lock()

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

    # 初始化 OpenAI 客户端
    def load_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            timeout = self.request_timeout,
            api_key = self.api_key,
            base_url = self.base_url,
            max_retries = 0
        )

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
            with open("prompt/prompt_surface_analysis_with_translation.txt", "r", encoding = "utf-8-sig") as reader:
                self.prompt_surface_analysis_with_translation = reader.read().strip()
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
        # 获取 llama.cpp 响应数据
        try:
            response_json = None
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
        if self.request_frequency_threshold > 1:
            self.semaphore = asyncio.Semaphore(self.request_frequency_threshold)
            self.async_limiter = AsyncLimiter(max_rate = self.request_frequency_threshold, time_period = 1)
        elif self.request_frequency_threshold > 0:
            self.semaphore = asyncio.Semaphore(1)
            self.async_limiter = AsyncLimiter(max_rate = 1, time_period = 1 / self.request_frequency_threshold)
        else:
            self.semaphore = asyncio.Semaphore(1)
            self.async_limiter = AsyncLimiter(max_rate = 1, time_period = 1)

    # 异步发送请求到 OpenAI 获取模型回复
    async def do_request(self, messages: list, llm_config: LLMConfig, retry: bool) -> tuple[Exception, dict, str, str, dict, dict]:
        try:
            error, usage, response_think, response_result, llm_request, llm_response = None, None, None, None, None, None

            llm_request = {
                "model" : self.model_name,
                "stream" : False,
                "temperature" : max(llm_config.TEMPERATURE, 0.50) if retry == True else llm_config.TEMPERATURE,
                "top_p" : llm_config.TOP_P,
                "max_tokens" : llm_config.MAX_TOKENS,
                # 同时设置 max_tokens 和 max_completion_tokens 时 OpenAI 接口会报错
                # "max_completion_tokens" : llm_config.MAX_TOKENS,
                "frequency_penalty" : max(llm_config.FREQUENCY_PENALTY, 0.2) if retry == True else llm_config.FREQUENCY_PENALTY,
                "messages" : messages,
            }

            response = await self.client.chat.completions.create(**llm_request)

            # OpenAI 的 API 返回的对象通常是 OpenAIObject 类型
            # 该类有一个内置方法可以将其转换为字典
            llm_response = response.to_dict()

            # 提取回复内容
            usage = response.usage
            message = response.choices[0].message
            if hasattr(message, "reasoning_content") and isinstance(message.reasoning_content, str):
                response_think = message.reasoning_content.replace("\n\n", "\n").strip()
                response_result = message.content.strip()
            elif "</think>" in message.content:
                splited = message.content.split("</think>")
                response_think = splited[0].removeprefix("<think>").replace("\n\n", "\n").strip()
                response_result = splited[-1].strip()
            else:
                response_think = ""
                response_result = message.content.strip()
        except Exception as e:
            error = e
        finally:
            return error, usage, response_think, response_result, llm_request, llm_response

    # 接口测试任务
    async def api_test(self) -> bool:
        async with self.semaphore, self.async_limiter:
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

                # 检查错误
                if error != None:
                    raise error

                # 检查是否超过最大 token 限制
                if usage.completion_tokens >= LLM.SURFACE_ANALYSIS_CONFIG.MAX_TOKENS:
                    raise Exception("返回结果错误（模型退化） ...")

                # 反序列化 JSON
                result = repair.loads(response_result)
                if not isinstance(result, dict) or result == {}:
                    raise Exception("返回结果错误（数据结构） ...")

                # 输出结果
                success = True
                LogHelper.info(f"{result}")

                return success
            except Exception as e:
                LogHelper.warning(f"{LogHelper.get_trackback(e)}")
                LogHelper.warning(f"llm_request - {llm_request}")
                LogHelper.warning(f"llm_response - {llm_response}")

    # 词义分析任务（支持使用已翻译的参考文本）
    async def surface_analysis(self, word: Word, words: list[Word], fake_name_mapping: dict[str, str], success: list[Word], retry: bool, last_round: bool) -> None:
        async with self.semaphore, self.async_limiter:
            try:
                if not hasattr(self, "prompt_groups"):
                    x = [v for group in LLM.GROUP_MAPPING.values() for v in group]
                    y = [v for group in LLM.GROUP_MAPPING_BANNED.values() for v in group]
                    self.prompt_groups = x + y

                # 获取参考文本原文
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

                error, usage, _, response_result, llm_request, llm_response = await self.do_request(
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
                    retry
                )

                # 检查错误
                if error != None:
                    raise error

                # 检查是否超过最大 token 限制
                if usage.completion_tokens >= LLM.SURFACE_ANALYSIS_CONFIG.MAX_TOKENS:
                    raise Exception("返回结果错误（模型退化） ...")

                # 反序列化 JSON
                result = repair.loads(response_result)
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
                        LogHelper.debug(f"[词义分析] 命中额外类型 - {word.surface} [green]->[/] {word.group} ...")
                        word.group = k
                        matched = True
                        break

                # 处理未命中目标类型的情况
                if matched == False:
                    # 检查是否属于黑名单类型（食品、物品、动物等）- 直接过滤掉，不重试
                    banned_types = set(v for vals in LLM.GROUP_MAPPING_BANNED.values() for v in vals)
                    if word.group in banned_types:
                        LogHelper.debug(f"[词义分析] 过滤非目标实体 - {word.surface} [green]->[/] {word.group} ...")
                        word.group = ""  # 设为空，后续会被过滤掉
                    elif last_round == True:
                        LogHelper.warning(f"[词义分析] 无法匹配的实体类型 - {word.surface} [green]->[/] {word.group} ...")
                        word.group = ""
                    else:
                        LogHelper.warning(f"[词义分析] 无法匹配的实体类型 - {word.surface} [green]->[/] {word.group} ...")
                        error = Exception("无法匹配的实体类型 ...")
            except Exception as e:
                LogHelper.debug(f"[词义分析] 子任务执行失败: {word.surface} - {e}")
                LogHelper.debug(f"llm_request - {llm_request}")
                LogHelper.debug(f"llm_response - {llm_response}")
                error = e
            finally:
                if error == None:
                    with self.lock:
                        success.append(word)

    # 批量执行词义分析任务
    async def surface_analysis_batch(self, words: list[Word], fake_name_mapping: dict[str, str]) -> list[Word]:
        failure: list[Word] = []
        success: list[Word] = []
        
        # 记录每个词条的重试次数
        retry_counts: dict[str, int] = {}

        LogHelper.print("")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("/"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[cyan]词义分析", total=len(words))
            
            for i in range(LLM.MAX_RETRY + 1):
                if i == 0:
                    retry = False
                    words_this_round = words
                elif len(failure) > 0:
                    retry = True
                    words_this_round = failure
                    
                    # 达到强制音译阈值的词条，直接使用强制音译，不再重试
                    still_retry = []
                    for word in words_this_round:
                        key = f"{word.surface}|{word.group}"
                        retry_counts[key] = retry_counts.get(key, 0) + 1
                        
                        if retry_counts[key] >= LLM.FORCE_TRANSLITERATE_THRESHOLD:
                            # 直接使用强制音译
                            if word.surface_translation and LLM.contains_kana_strict(word.surface_translation):
                                LogHelper.warning(f"[词义分析] 重试{retry_counts[key]}次仍失败，触发强制音译 - {word.surface}")
                                word.surface_translation = self.force_transliterate(word.surface)
                            with self.lock:
                                success.append(word)
                            progress.update(task, completed=len(success))
                        else:
                            still_retry.append(word)
                    
                    words_this_round = still_retry
                    if not words_this_round:
                        break
                    
                    progress.update(task, description=f"[yellow]词义分析 (重试 {i}/{LLM.MAX_RETRY})")
                else:
                    break

                # 执行异步任务
                async def run_with_progress(word: Word):
                    await self.surface_analysis(word, words, fake_name_mapping, success, retry, i == LLM.MAX_RETRY)
                    progress.update(task, completed=len(success))
                
                tasks = [asyncio.create_task(run_with_progress(word)) for word in words_this_round]
                await asyncio.gather(*tasks, return_exceptions=True)

                # 获得失败任务的列表
                success_pairs = {(word.surface, word.group) for word in success}
                failure = [word for word in words if (word.surface, word.group) not in success_pairs]
        
        LogHelper.print("")

        # 【最终强制音译兜底】对于最终仍失败的词条，如果其 surface_translation 仍包含有效假名，强制音译
        # 使用 contains_kana_strict 允许孤立的拟声词字符
        for word in words:
            if word.surface_translation and LLM.contains_kana_strict(word.surface_translation):
                LogHelper.warning(f"[词义分析] 触发强制音译兜底 - {word.surface} 原译名: {word.surface_translation}")
                word.surface_translation = self.force_transliterate(word.surface)
                LogHelper.info(f"[词义分析] 强制音译结果 - {word.surface} → {word.surface_translation}")

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
    async def context_translate(self, word: Word, words: list[Word], success: list[Word], retry: bool) -> None:
        async with self.semaphore, self.async_limiter:
            try:
                # 获取参考文本（使用增加后的 token 阈值）
                # 【关键】context_shrink_level 会影响采样数量，返回本次使用的索引
                context_str, sampled_indices = word.get_context_str_for_translate(self.language)
                
                # 如果没有可用的上下文了（所有索引都被排除），降级处理
                if not context_str or not sampled_indices:
                    LogHelper.warning(f"[参考文本翻译] 所有上下文都已被排除，跳过翻译阶段（{word.surface}）")
                    word.context_translation = []
                    with self.lock:
                        success.append(word)
                    return
                
                error, usage, _, response_result, llm_request, llm_response = await self.do_request(
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
                    retry
                )

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
                            LogHelper.warning(f"[参考文本翻译] 敏感内容过滤触发，限缩级别提升至 {word.context_shrink_level}，已排除 {len(word.failed_context_indices)} 个索引（{word.surface}）")
                        else:
                            # 已经是单条采样模式，继续重采样（不跳过！）
                            LogHelper.warning(f"[参考文本翻译] 敏感内容过滤触发，单条重采样模式，已排除 {len(word.failed_context_indices)} 个索引（{word.surface}）")
                    raise error

                # 检查是否超过最大 token 限制
                if usage.completion_tokens >= LLM.CONTEXT_TRANSLATE_CONFIG.MAX_TOKENS:
                    raise Exception("返回结果错误（模型退化） ...")

                context_translation = [line.strip() for line in response_result.splitlines() if line.strip() != ""]
                
                # 【新增】退化检测：检查整体输出是否存在重复模式
                if LLM.is_degraded(response_result):
                    LogHelper.warning(f"[参考文本翻译] 模型退化检测触发（重复输出），将重试 ...（{word.surface}）")
                    raise Exception("模型退化（输出重复内容）")
                
                # 【新增】相似度检测：逐行检查，高相似度说明未真正翻译
                SIMILARITY_THRESHOLD = 0.80
                original_lines = [line.strip() for line in context_str.splitlines() if line.strip() != ""]
                
                # 只在行数匹配时做逐行相似度检测
                if len(context_translation) == len(original_lines):
                    high_similarity_count = 0
                    for orig, trans in zip(original_lines, context_translation):
                        similarity = LLM.check_similarity(orig, trans)
                        if similarity >= SIMILARITY_THRESHOLD:
                            high_similarity_count += 1
                    
                    # 超过50%的行高相似度，认为翻译基本失效
                    if high_similarity_count > len(original_lines) * 0.5:
                        LogHelper.warning(f"[参考文本翻译] 相似度检测触发（{high_similarity_count}/{len(original_lines)} 行高相似度），将重试 ...（{word.surface}）")
                        raise Exception(f"翻译失效（相似度过高，{high_similarity_count}/{len(original_lines)} 行）")
                
                # 检测翻译是否失效：仅当译文与原文【完全相同】且【原文确实包含明显外文成分】时才报错。
                # 这样可以避免"纯汉字/符号行本来就无需翻译"的场景被误判，从而减少无意义重试。
                has_obvious_foreign = re.search(r"[\u3040-\u30FF\uAC00-\uD7AFA-Za-z]", context_str) is not None
                if len(context_translation) > 0 and context_translation == original_lines and has_obvious_foreign:
                    raise Exception("翻译失效（译文与原文相同） ...")

                word.context_translation = context_translation
                word.llmrequest_context_translate = llm_request
                word.llmresponse_context_translate = llm_response
            except Exception as e:
                LogHelper.debug(f"[参考文本翻译] 子任务执行失败: {word.surface} - {e}")
                LogHelper.debug(f"llm_request - {llm_request}")
                LogHelper.debug(f"llm_response - {llm_response}")
                error = e
            finally:
                if error == None:
                    with self.lock:
                        success.append(word)
                    LogHelper.info(f"[参考文本翻译] 已完成 {len(success)} / {len(words)} ...")

    # 批量执行参考文本翻译任务
    async def context_translate_batch(self, words: list[Word]) -> list[Word]:
        failure: list[Word] = []
        success: list[Word] = []

        LogHelper.print("")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("/"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[cyan]参考文本翻译", total=len(words))
            
            for i in range(LLM.MAX_RETRY + 1):
                if i == 0:
                    retry = False
                    words_this_round = words
                elif len(failure) > 0:
                    retry = True
                    words_this_round = failure
                    progress.update(task, description=f"[yellow]参考文本翻译 (重试 {i}/{LLM.MAX_RETRY})")
                else:
                    break

                # 执行异步任务
                async def run_with_progress(word: Word):
                    await self.context_translate(word, words, success, retry)
                    progress.update(task, completed=len(success))
                
                tasks = [asyncio.create_task(run_with_progress(word)) for word in words_this_round]
                await asyncio.gather(*tasks, return_exceptions=True)

                # 获得失败任务的列表
                success_pairs = {(word.surface, word.group) for word in success}
                failure = [word for word in words if (word.surface, word.group) not in success_pairs]
        
        LogHelper.print("")

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
        async with self.semaphore, self.async_limiter:
            error = None
            llm_request = {}
            llm_response = {}
            
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
                
                error, usage, _, response_result, llm_request, llm_response = await self.do_request(
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
                
                # 解析结果
                result = repair.loads(response_result)
                if not isinstance(result, dict):
                    raise Exception("返回结果错误（数据结构）")
                
                new_translation = result.get("translation", "")
                confidence = result.get("confidence", "low")
                
                # 验证新译名是否仍有假名
                if LLM.contains_kana_strict(new_translation):
                    LogHelper.warning(f"[问题修复] 修复失败（仍有假名）: {word.surface} → {new_translation}，使用强制音译")
                    new_translation = self.force_transliterate(word.surface)
                    confidence = "forced"
                
                # 更新译名
                old_translation = word.surface_translation
                word.surface_translation = new_translation
                word.llmrequest_fix = llm_request
                word.llmresponse_fix = llm_response
                
                LogHelper.info(f"[问题修复] {word.surface}: {old_translation} → {new_translation} [{confidence}]")
                
            except Exception as e:
                LogHelper.warning(f"[问题修复] 失败，使用强制音译: {word.surface} - {LogHelper.get_trackback(e)}")
                # 修复失败，使用强制音译
                word.surface_translation = self.force_transliterate(word.surface)
                error = None  # 强制音译成功，不算错误
            finally:
                if error is None:
                    with self.lock:
                        success.append(word)
                    LogHelper.info(f"[问题修复] 已完成 {len(success)} / {total} ...")
    
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
        # 加载 prompt
        self.load_fix_prompt()
        if not self.prompt_fix_translation:
            LogHelper.warning("[问题修复] 跳过（无 prompt）")
            return words
        
        # 筛选问题词条
        problem_words = []
        for word in words:
            translation = word.surface_translation or ""
            
            # 条件1: 假名残留
            if LLM.contains_kana_strict(translation):
                problem_words.append(word)
                continue
            
            # 条件2: 相似度过高
            if translation and LLM.check_similarity(word.surface, translation) >= 0.80:
                problem_words.append(word)
                continue
            
            # 条件3: 空翻译
            if not translation.strip():
                problem_words.append(word)
                continue
        
        if not problem_words:
            LogHelper.info("[问题修复] 无需修复，所有词条正常")
            return words
        
        LogHelper.info(f"[问题修复] 发现 {len(problem_words)} 个问题词条，开始修复...")
        
        success = []
        tasks = [
            asyncio.create_task(self.fix_translation(word, success, len(problem_words)))
            for word in problem_words
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # 最终兜底：对仍有假名的词条强制音译
        for word in words:
            if word.surface_translation and LLM.contains_kana_strict(word.surface_translation):
                LogHelper.warning(f"[最终兜底] 强制音译: {word.surface}")
                word.surface_translation = self.force_transliterate(word.surface)
        
        return words