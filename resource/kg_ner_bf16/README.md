---
language:
- zh
- en
- ja
- ko
pipeline_tag: token-classification
license: apache-2.0
---

### 前言

- 一个对多种语言的小说与游戏脚本进行专门优化的语言模型系列，在最开始是为了 [KeywordGacha](https://github.com/neavo/KeywordGacha) 而创造的
- [KeywordGacha](https://github.com/neavo/KeywordGacha) 是一个使用 OpenAI 兼容接口自动生成小说、漫画、字幕、游戏脚本等内容文本中实体词语表的翻译辅助工具
- 在 [KeywordGacha](https://github.com/neavo/KeywordGacha) 的开发过程中，我们发现社区中并没有满足需求的语言模型模型，所以自己动手创造了一个 ：）

### 综述

- 通过针对性的预训练，本系列模型：
  - 对 `轻小说`、`游戏脚本`、`漫画脚本` 等故事性文本内容具有极好的理解能力
  - 特别是 `剑与魔法`、`超能力战斗`、`异世界冒险` 等常见的 ACG 题材的故事内容
- AND NSFW IS OK
- 支持多种语言 
  - 目前已针对以下语言进行了预训练：`中文`、`英文`、`日文`、`韩文`
  - 未来计划针对以下语言进行预训练：`俄文`
- 目前我们提供以下预训练模型：

| 模型 | 版本 | 说明 |
| :--: | :--: | :--:|
| [keyword_gacha_multilingual_base](https://huggingface.co/neavo/keyword_gacha_multilingual_base) | 20250128 | 基础模型 |
| [keyword_gacha_multilingual_ner](https://huggingface.co/neavo/keyword_gacha_multilingual_ner)  | 20250131 | 预训练实体识别模型 |

### 基础模型 Base

- 在 [modern_bert_multilingual_nodecay](https://huggingface.co/neavo/modern_bert_multilingual_nodecay) 进行退火得到的模型
- 训练量大约 1B Token，包含 4 种不同语言的语料
- 主要训练参数
  - Batch Size : 1792
  - Learing Rate : 5e-04
  - Maximum Sequence Length : 512
  - Optimizer : adamw_torch
  - LR Scheduler: warmup_stable_decay
  - Train Precision : bf16 mix
  
- 使用说明
  - 暂无，基础模型一般不直接使用，需针对具体下游任务进行微调后使用

### 实体识别模型 NER

- 在 Base 模型的基础上，使用了大约 100,000 条合成语料进行 NER 任务的微调
- 与人工校对的实体词语表进行对比，可以达到 `90%-95%` 的实际准确率
  - 与 [KeywordGacha](https://github.com/neavo/KeywordGacha) 搭配使用时
  - 实际任务环境中的实测数据，并非预设测试集上的 F1 Score 这类理论上的指标
- 训练参数如下：
  - Batch Size : 32
  - Learing Rate : 6e-06
  - Optimizer : adamw_torch
  - LR Scheduler: cosine
  - Warnup Ratio : 0.1
  - Train Precision : bf16 mix

- 使用说明
  - 待补充

### 其他
- 训练脚本 [Github](https://github.com/neavo/KeywordGachaModel)