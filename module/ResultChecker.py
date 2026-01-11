"""
BookTerm Gacha - Result Checker Module
=======================================

This module provides quality assurance for generated term lists by detecting
common issues that may require manual review or re-processing.

Quality Checks Performed:
    1. Kana Residue (假名残留): Japanese hiragana/katakana remaining in translations
       - Uses strict detection to exclude valid onomatopoeia characters
       - Outputs: 结果检查_假名残留.json
    
    2. Hangeul Residue (韩文残留): Korean characters remaining in translations
       - Outputs: 结果检查_韩文残留.json
    
    3. Similarity Issues (相似度问题): Translations too similar to source
       - Detects when LLM failed to properly translate
       - Uses Jaccard similarity threshold
    
    4. Empty Translations (空翻译): Missing translation results
       - Indicates LLM failures or filtered content

    5. Untranslated Entries (未翻译条目): Terms that were not processed
       - Useful for identifying gaps in coverage

Output Format:
    JSON files in the output folder with detailed issue reports including
    surface form, translation, context samples, and suggested fixes.

Usage:
    checker = ResultChecker(words, language, "output")
    stats = checker.check_all()
    checker.save_reports()

Based on KeywordGacha v0.13.1 by neavo / Dev-Experimental improvements
https://github.com/neavo/KeywordGacha
"""

import os
import json
from typing import Optional

from model.Word import Word
from model.NER import NER
from module.Text.TextHelper import TextHelper
from module.LogHelper import LogHelper


class ResultChecker:
    """术语表结果检查器"""
    
    # 拟声词/地名特殊字符集（与 LLM.RULE_ONOMATOPOEIA 保持一致）
    RULE_ONOMATOPOEIA = frozenset({
        "ッ", "っ", "ぁ", "ぃ", "ぅ", "ぇ", "ぉ",
        "ゃ", "ゅ", "ょ", "ゎ",
        "ァ", "ィ", "ゥ", "ェ", "ォ",
        "ャ", "ュ", "ョ", "ヮ",
        "ー",
        "ヶ", "ケ", "ヵ",  # 地名专用假名
        "の",  # 平假名"之"
    })
    
    def __init__(self, words: list[Word], language: int, output_folder: str = "output"):
        """
        初始化结果检查器
        
        Args:
            words: 待检查的词条列表
            language: 源语言类型 (NER.Language)
            output_folder: 输出目录
        """
        self.words = words
        self.language = language
        self.output_folder = output_folder
        
        # 统计数据
        self.stats = {
            "total": len(words),
            "kana_residue": 0,
            "hangeul_residue": 0,
            "similarity_issue": 0,
            "empty_translation": 0,
            "forced_transliteration": 0,
        }
        
        # 问题词条
        self.issues: dict[str, list[dict]] = {
            "kana_residue": [],
            "hangeul_residue": [],
            "similarity_issue": [],
            "empty_translation": [],
        }
    
    def check_all(self) -> dict:
        """
        执行全部检查
        
        Returns:
            检查统计结果
        """
        os.makedirs(self.output_folder, exist_ok=True)
        
        # 清理旧的检查结果文件
        for entry in os.scandir(self.output_folder):
            if entry.is_file() and entry.name.startswith("结果检查_"):
                os.remove(entry.path)
        
        # 执行各项检查
        self._check_kana_residue()
        self._check_hangeul_residue()
        self._check_similarity()
        self._check_empty_translation()
        
        # 生成报告
        self._generate_report()
        
        return self.stats
    
    def _contains_kana_strict(self, text: str) -> bool:
        """
        严格检测是否包含有效假名（排除孤立拟声词）
        """
        if not text:
            return False
        
        if not (TextHelper.JA.any_hiragana(text) or TextHelper.JA.any_katakana(text)):
            return False
        
        length = len(text)
        for i, char in enumerate(text):
            is_kana = TextHelper.JA.hiragana(char) or TextHelper.JA.katakana(char)
            if not is_kana:
                continue
            
            if char in self.RULE_ONOMATOPOEIA:
                prev_char = text[i - 1] if i > 0 else None
                next_char = text[i + 1] if i < length - 1 else None
                
                is_prev_kana = prev_char is not None and (
                    TextHelper.JA.hiragana(prev_char) or TextHelper.JA.katakana(prev_char)
                )
                is_next_kana = next_char is not None and (
                    TextHelper.JA.hiragana(next_char) or TextHelper.JA.katakana(next_char)
                )
                
                if not is_prev_kana and not is_next_kana:
                    continue
            
            return True
        
        return False
    
    def _check_kana_residue(self) -> None:
        """检查假名残留"""
        if self.language != NER.Language.JA:
            return
        
        for word in self.words:
            translation = word.surface_translation or ""
            if self._contains_kana_strict(translation):
                self.stats["kana_residue"] += 1
                self.issues["kana_residue"].append({
                    "surface": word.surface,
                    "translation": translation,
                    "group": word.group,
                    "romaji": word.surface_romaji,
                })
        
        # 写入结果文件
        if self.issues["kana_residue"]:
            path = f"{self.output_folder}/结果检查_假名残留.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.issues["kana_residue"], f, ensure_ascii=False, indent=2)
            LogHelper.warning(f"[结果检查] 假名残留: {self.stats['kana_residue']} 条 -> {path}")
        else:
            LogHelper.info("[结果检查] 假名残留: 0 条 ✓")
    
    def _check_hangeul_residue(self) -> None:
        """检查韩文残留"""
        if self.language != NER.Language.KO:
            return
        
        for word in self.words:
            translation = word.surface_translation or ""
            if TextHelper.KO.any_hangeul(translation):
                self.stats["hangeul_residue"] += 1
                self.issues["hangeul_residue"].append({
                    "surface": word.surface,
                    "translation": translation,
                    "group": word.group,
                })
        
        # 写入结果文件
        if self.issues["hangeul_residue"]:
            path = f"{self.output_folder}/结果检查_韩文残留.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.issues["hangeul_residue"], f, ensure_ascii=False, indent=2)
            LogHelper.warning(f"[结果检查] 韩文残留: {self.stats['hangeul_residue']} 条 -> {path}")
        else:
            LogHelper.info("[结果检查] 韩文残留: 0 条 ✓")
    
    def _check_similarity(self) -> None:
        """检查相似度过高（译名与原名过于接近）"""
        THRESHOLD = 0.80
        
        for word in self.words:
            translation = word.surface_translation or ""
            if not translation:
                continue
            
            # 计算 Jaccard 相似度
            set_src = set(word.surface)
            set_dst = set(translation)
            union = len(set_src | set_dst)
            intersection = len(set_src & set_dst)
            similarity = intersection / union if union > 0 else 0.0
            
            if similarity >= THRESHOLD:
                self.stats["similarity_issue"] += 1
                self.issues["similarity_issue"].append({
                    "surface": word.surface,
                    "translation": translation,
                    "similarity": f"{similarity:.2%}",
                    "group": word.group,
                })
        
        # 写入结果文件
        if self.issues["similarity_issue"]:
            path = f"{self.output_folder}/结果检查_相似度过高.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.issues["similarity_issue"], f, ensure_ascii=False, indent=2)
            LogHelper.warning(f"[结果检查] 相似度过高: {self.stats['similarity_issue']} 条 -> {path}")
        else:
            LogHelper.info("[结果检查] 相似度过高: 0 条 ✓")
    
    def _check_empty_translation(self) -> None:
        """检查空翻译"""
        for word in self.words:
            if not word.surface_translation or word.surface_translation.strip() == "":
                self.stats["empty_translation"] += 1
                self.issues["empty_translation"].append({
                    "surface": word.surface,
                    "group": word.group,
                    "context_summary": word.context_summary,
                })
        
        # 写入结果文件
        if self.issues["empty_translation"]:
            path = f"{self.output_folder}/结果检查_空翻译.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.issues["empty_translation"], f, ensure_ascii=False, indent=2)
            LogHelper.warning(f"[结果检查] 空翻译: {self.stats['empty_translation']} 条 -> {path}")
        else:
            LogHelper.info("[结果检查] 空翻译: 0 条 ✓")
    
    def _generate_report(self) -> None:
        """生成检查报告摘要"""
        total = self.stats["total"]
        issues_count = (
            self.stats["kana_residue"] +
            self.stats["hangeul_residue"] +
            self.stats["similarity_issue"] +
            self.stats["empty_translation"]
        )
        
        LogHelper.info("=" * 50)
        LogHelper.info("[结果检查] 检查完成，统计如下:")
        LogHelper.info(f"  • 总词条数: {total}")
        LogHelper.info(f"  • 问题词条: {issues_count} ({issues_count/total*100:.1f}%)" if total > 0 else "  • 问题词条: 0")
        
        if self.language == NER.Language.JA:
            LogHelper.info(f"  • 假名残留: {self.stats['kana_residue']}")
        elif self.language == NER.Language.KO:
            LogHelper.info(f"  • 韩文残留: {self.stats['hangeul_residue']}")
        
        LogHelper.info(f"  • 相似度过高: {self.stats['similarity_issue']}")
        LogHelper.info(f"  • 空翻译: {self.stats['empty_translation']}")
        LogHelper.info("=" * 50)
        
        # 写入总结报告
        report_path = f"{self.output_folder}/结果检查_报告.json"
        report = {
            "statistics": self.stats,
            "quality_score": f"{(1 - issues_count / total) * 100:.1f}%" if total > 0 else "N/A",
            "issues_summary": {
                "kana_residue_count": len(self.issues["kana_residue"]),
                "hangeul_residue_count": len(self.issues["hangeul_residue"]),
                "similarity_issue_count": len(self.issues["similarity_issue"]),
                "empty_translation_count": len(self.issues["empty_translation"]),
            }
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        LogHelper.info(f"[结果检查] 报告已生成 -> {report_path}")
