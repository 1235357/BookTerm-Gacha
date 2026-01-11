"""
BookTerm Gacha - Word Entity Module
====================================

This module defines the Word class, which represents a single extracted term/entity
with all its associated metadata and context. Each Word object tracks:

- Surface form: The original text as it appears in the source
- Translation: The translated/transliterated form
- Context: Surrounding text samples for semantic analysis
- Grouping: Category (character name, location, etc.)
- Scoring: NER confidence and frequency count

Key Features:
    - Token counting with tiktoken (OpenAI tokenizer)
    - Context sampling with configurable sample sizes
    - Sensitive content handling with context shrinking
    - Thread-safe caching for performance

Context Sampling Strategy:
    The Word class implements intelligent context sampling to provide
    LLM with representative text snippets while staying within token limits.
    This is crucial for accurate semantic analysis of terms.

Sensitive Content Handling:
    When LLM refuses translation due to sensitive content, the Word
    automatically shrinks its context (50% -> 25% -> single sample)
    and tracks failed context indices to avoid re-sampling problematic content.

Based on KeywordGacha v0.13.1 by neavo
https://github.com/neavo/KeywordGacha
"""

import re
import random
import threading

import tiktoken
import tiktoken_ext
from tiktoken_ext import openai_public

from base.BaseData import BaseData

class Word(BaseData):

    # 必须显式的引用这两个库，否则打包后会报错
    tiktoken_ext
    openai_public

    # 去重
    RE_DUPLICATE = re.compile(r"[\r\n]+", flags = re.IGNORECASE)

    # 缓存
    CACHE = {}
    CACHE_LOCK = threading.Lock()

    # ============== 上下文采样配置（从 config.json 加载，这里是默认值） ==============
    # 最大采样位置数：从多少个不同位置采样上下文
    MAX_CONTEXT_SAMPLES = 10
    # 每个位置的最大 token 数
    TOKENS_PER_SAMPLE = 512

    @classmethod
    def set_config(cls, max_context_samples: int = None, tokens_per_sample: int = None) -> None:
        """从配置文件设置采样参数"""
        if max_context_samples is not None:
            cls.MAX_CONTEXT_SAMPLES = max_context_samples
        if tokens_per_sample is not None:
            cls.TOKENS_PER_SAMPLE = tokens_per_sample

    def __init__(self) -> None:
        super().__init__()

        # 默认值
        self.score: float = 0.0
        self.count: int = 0
        self.context: list[str] = []
        self.context_summary: str = ""
        self.context_translation: list[str] = []
        self.surface: str = ""
        self.surface_romaji: str = ""
        self.surface_translation: str = ""
        self.group: str = ""
        self.gender: str = ""
        self.input_lines: list[str] = []

        # 调试信息
        self.llmrequest_surface_analysis: dict = {}
        self.llmrequest_context_translate: dict = {}
        self.llmrequest_fix: dict = {}  # 第三阶段：问题修复
        self.llmresponse_surface_analysis: dict = {}
        self.llmresponse_context_translate: dict = {}
        self.llmresponse_fix: dict = {}  # 第三阶段：问题修复

        # ============== 敏感内容限缩状态 ==============
        # 当翻译因敏感内容被拒绝时，逐步减少上下文采样数量
        self.context_shrink_level: int = 0  # 限缩级别：0=完整, 1=50%, 2=25%, 3+=单条（随机重采样）
        self.failed_context_indices: set[int] = set()  # 记录失败的上下文起始索引，重采样时避开

    # 获取token数量，优先从缓存中获取
    def get_token_count(self, line: str) -> int:
        count = 0

        # 优先从缓存中取数据
        with Word.CACHE_LOCK:
            if line in Word.CACHE:
                count = Word.CACHE[line]
            else:
                count = len(tiktoken.get_encoding("o200k_base").encode(line))
                Word.CACHE[line] = count

        return count

    # 按阈值截取文本，如果句子长度全部超过阈值，则取最接近阈值的一条
    def clip_lines(self, lines: list[str], line_threshold: int, token_threshold: int) -> tuple[list[str], int]:
        context = []
        context_token_count = 0

        for line in lines:
            # 行数阈值有效，且超过行数阈值，则跳出循环
            if line_threshold > 0 and len(context) > line_threshold:
                break

            line_token_count = self.get_token_count(line)

            # 跳过超出阈值的句子
            if line_token_count > token_threshold:
                continue

            # 更新参考文本与计数
            context.append(line)
            context_token_count = context_token_count + line_token_count

            # 如果计数超过 Token 阈值，则跳出循环
            if context_token_count > token_threshold:
                break

        # 如果句子长度全部超过 Token 阈值，则取最接近阈值的一条
        if len(lines) > 0 and len(context) == 0:
            line = min(lines, key = lambda line: abs(self.get_token_count(line) - token_threshold))

            context.append(line)
            context_token_count = self.get_token_count(line)

        return context, context_token_count

    # 按长度截取参考文本并返回，
    def clip_context(self, line_threshold: int, token_threshold: int) -> list[str]:
        # 先从参考文本中截取
        context, context_token_count = self.clip_lines(self.context, line_threshold, token_threshold)

        # 如果句子长度不足 75%，则尝试全文匹配中补充
        if context_token_count < token_threshold * 0.75:
            context_set = set(self.context)
            context_ex, _ = self.clip_lines(
                sorted(
                    # 筛选出未包含在当前参考文本中且包含关键词的文本以避免重复
                    [line for line in self.input_lines if self.surface in line and line not in context_set],
                    key = lambda line: self.get_token_count(line),
                    reverse = True
                ),
                line_threshold - len(context),
                token_threshold - context_token_count,
            )

            # 追加参考文本
            context.extend(context_ex)

        return context

    # ============== 新增：多样化上下文采样（核心改进） ==============
    def sample_diverse_context(self, max_samples: int = None, tokens_per_sample: int = None, exclude_indices: set[int] = None) -> tuple[list[str], list[int]]:
        """
        从词语出现的不同位置采样上下文，确保语境多样性
        
        核心思想：
        - 同一个词可能出现100次，但只翻译一次
        - 不同位置的上下文提供不同的语境信息
        - 每个采样点聚合多条连续句子，直到达到 token 限制
        
        Args:
            max_samples: 最大采样数量（默认 MAX_CONTEXT_SAMPLES）
            tokens_per_sample: 每个样本的最大token数（默认 TOKENS_PER_SAMPLE）
            exclude_indices: 需要排除的上下文起始索引集合（用于敏感内容重采样）
            
        Returns:
            (多样化采样的上下文列表, 本次采样使用的起始索引列表)
        """
        if max_samples is None:
            max_samples = Word.MAX_CONTEXT_SAMPLES
        if tokens_per_sample is None:
            tokens_per_sample = Word.TOKENS_PER_SAMPLE
        if exclude_indices is None:
            exclude_indices = set()
            
        if not self.context:
            return [], []
        
        # 如果上下文数量很少，直接返回全部
        if len(self.context) <= max_samples:
            return self.context, list(range(len(self.context)))
        
        # 策略：均匀分布采样 + 聚合连续句子
        # 1. 将上下文分成 max_samples 组（按出现顺序）
        # 2. 从每组的起始位置开始，聚合连续句子直到达到 token 限制
        
        sampled = []
        sampled_indices = []  # 记录本次采样使用的起始索引
        group_size = len(self.context) // max_samples
        
        for i in range(max_samples):
            # 计算当前组的起始位置
            start_idx = i * group_size
            
            # 如果该起始索引在排除列表中，尝试在组内找其他位置
            if start_idx in exclude_indices:
                found_alternative = False
                for offset in range(1, group_size):
                    alt_idx = start_idx + offset
                    if alt_idx < len(self.context) and alt_idx not in exclude_indices:
                        start_idx = alt_idx
                        found_alternative = True
                        break
                if not found_alternative:
                    continue  # 整个组都被排除，跳过
            
            # 从起始位置开始聚合句子，直到达到 token 限制
            aggregated_lines = []
            aggregated_tokens = 0
            
            for j in range(start_idx, len(self.context)):
                line = self.context[j]
                line_tokens = self.get_token_count(line)
                
                # 如果加上这一行会超出限制，且已经有内容了，就停止
                if aggregated_tokens + line_tokens > tokens_per_sample and aggregated_lines:
                    break
                
                aggregated_lines.append(line)
                aggregated_tokens += line_tokens
                
                # 如果已经达到目标的80%以上，可以停止了
                if aggregated_tokens >= tokens_per_sample * 0.8:
                    break
            
            # 将聚合的句子合并为一个段落
            if aggregated_lines:
                paragraph = "\n".join(aggregated_lines)
                sampled.append(paragraph)
                sampled_indices.append(start_idx)  # 记录本次采样的起始索引
        
        return sampled, sampled_indices

    # ============== 重写：获取用于翻译的上下文（无 token 限制，多样化采样） ==============
    def get_context_str_for_translate(self, language: int) -> tuple[str, list[int]]:
        """
        获取用于参考文本翻译的上下文
        
        设计理念：
        - 移除硬性 token 限制
        - 从不同位置采样，确保语境多样性
        - 每个位置约 512 token，最多 10 个位置
        - 使用明确的编号分隔每个上下文段落
        - 【新增】支持敏感内容限缩：根据 context_shrink_level 减少采样量
        - 【新增】限缩级别>=3时，随机单条重采样，避开之前失败的索引
        
        Returns:
            (上下文字符串, 本次采样使用的起始索引列表)
        """
        # 根据限缩级别计算实际采样数量
        # level 0: 100% (max_samples)
        # level 1: 50%
        # level 2: 25%
        # level >= 3: 单条（随机重采样，避开失败索引）
        shrink_factors = [1.0, 0.5, 0.25]
        
        if self.context_shrink_level >= 3:
            # 限缩级别>=3：单条随机重采样模式
            actual_samples = 1
            # 使用 failed_context_indices 排除之前失败的位置
            exclude_indices = self.failed_context_indices
        elif self.context_shrink_level < len(shrink_factors):
            factor = shrink_factors[self.context_shrink_level]
            actual_samples = max(1, int(Word.MAX_CONTEXT_SAMPLES * factor))
            exclude_indices = set()
        else:
            actual_samples = 1
            exclude_indices = set()
        
        # 限缩级别>=3时，使用随机采样而非固定位置采样
        if self.context_shrink_level >= 3 and len(self.context) > 1:
            sampled_context, sampled_indices = self._random_single_sample(exclude_indices)
        else:
            # 使用多样化采样获取上下文
            sampled_context, sampled_indices = self.sample_diverse_context(
                max_samples=actual_samples,
                tokens_per_sample=Word.TOKENS_PER_SAMPLE,
                exclude_indices=exclude_indices
            )
        
        # 为每个上下文段落添加编号标记
        numbered_paragraphs = []
        for i, paragraph in enumerate(sampled_context, 1):
            # 清理段落内的多余换行
            clean_paragraph = Word.RE_DUPLICATE.sub("\n", paragraph)
            numbered_paragraphs.append(f"【上下文 {i}】\n{clean_paragraph}")
        
        return "\n\n".join(numbered_paragraphs), sampled_indices
    
    def _random_single_sample(self, exclude_indices: set[int]) -> tuple[list[str], list[int]]:
        """
        随机采样单条上下文，避开已失败的索引
        
        Args:
            exclude_indices: 需要排除的索引集合
            
        Returns:
            (采样的上下文列表, 采样使用的索引列表)
        """
        if not self.context:
            return [], []
        
        # 获取可用的索引
        available_indices = [i for i in range(len(self.context)) if i not in exclude_indices]
        
        if not available_indices:
            # 所有索引都被排除了，返回空（这种情况应该在调用处处理）
            return [], []
        
        # 随机选择一个索引
        chosen_idx = random.choice(available_indices)
        
        # 从该位置聚合句子
        aggregated_lines = []
        aggregated_tokens = 0
        
        for j in range(chosen_idx, len(self.context)):
            line = self.context[j]
            line_tokens = self.get_token_count(line)
            
            if aggregated_tokens + line_tokens > Word.TOKENS_PER_SAMPLE and aggregated_lines:
                break
            
            aggregated_lines.append(line)
            aggregated_tokens += line_tokens
            
            if aggregated_tokens >= Word.TOKENS_PER_SAMPLE * 0.8:
                break
        
        if aggregated_lines:
            paragraph = "\n".join(aggregated_lines)
            return [paragraph], [chosen_idx]
        
        return [], []

    # ============== 重写：获取用于词义分析的上下文（无 token 限制，多样化采样） ==============
    def get_context_str_for_surface_analysis(self, language: int) -> str:
        """
        获取用于词义分析的上下文
        
        设计理念：
        - 与翻译阶段使用相同的多样化采样策略
        - 保持上下文的一致性
        - 使用明确的编号分隔每个上下文段落
        """
        # 使用多样化采样获取上下文（忽略返回的索引列表）
        sampled_context, _ = self.sample_diverse_context(
            max_samples=Word.MAX_CONTEXT_SAMPLES,
            tokens_per_sample=Word.TOKENS_PER_SAMPLE
        )
        
        # 为每个上下文段落添加编号标记
        numbered_paragraphs = []
        for i, paragraph in enumerate(sampled_context, 1):
            # 清理段落内的多余换行
            clean_paragraph = Word.RE_DUPLICATE.sub("\n", paragraph)
            numbered_paragraphs.append(f"【上下文 {i}】\n{clean_paragraph}")
        
        return "\n\n".join(numbered_paragraphs)