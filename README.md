# BookTerm Gacha

An LLM-powered terminology extraction agent for books and story scripts.

**Version:** `v0.2.0-Multi-Platform-API-Rotation`

BookTerm Gacha reads your source text (EPUB/TXT/MD and more), extracts named entities with a local BERT NER model, then uses an OpenAI-compatible LLM API to **classify** and **translate / transliterate** the terms into a glossary you can feed into translation tools.

This fork is heavily optimized for **Japanese → Chinese** name/place consistency (katakana handling, kana-residue checks, forced transliteration fallback), while still supporting **ZH/EN/JA/KO** input modes.

---

## What you get

For each detected term group, the tool generates:

- `output/<input>_<group>_词典.json` — simple dictionary mapping (`src -> dst`)
- `output/<input>_<group>_术语表.json` — glossary list with metadata (`src/dst/info`)
- `output/<input>_<group>_galtransl.txt` — TSV for GalTransl-style import
- `output/<input>_<group>_日志.txt` — human-readable audit log (context, translation, summary)

And a post-run QA pass produces reports like:

- `output/结果检查_假名残留.json`
- `output/结果检查_相似度过高.json`
- `output/结果检查_空翻译.json`
- `output/结果检查_报告.json`

> Note: Output filenames currently use Chinese group names (`角色`, `地点`) because the pipeline’s classification taxonomy is Chinese-first.

---

## Project layout

Key folders/files in this workspace:

```
BookTerm Gacha/
  app.py                     # main entry (menu + pipeline)
  config.json                 # runtime settings (multi-platform, thresholds)
  version.txt

  input/                      # put your books/scripts here
  output/                     # generated glossary + QA reports
  log/                        # runtime logs

  model/
    NER.py                    # BERT NER extraction
    LLM.py                    # LLM agent (streaming, retry, rotation)
    Word.py                   # term object + context sampling

  module/
    FileManager.py            # reads many formats; writes outputs
    EnvChecker.py             # auto environment check/repair (source run)
    ResultChecker.py          # QA checks
    RubyCleaner.py            # removes Japanese ruby/furigana
    TaskTracker.py            # persistent bottom progress panel

  prompt/                     # LLM prompt templates (Chinese)
  blacklist/                  # language-specific stopword lists

  resource/
    kg_ner_bf16/              # local NER model (token-classification)
    llm_config/               # per-stage LLM generation configs
```

---

## Quick start (Windows)

### Option A: Use the packaged EXE

If you have a built release (`app.exe`), you typically only need to:

1. Edit `config.json` (set platform + API keys).
2. Put your files into `input/`.
3. Run `app.exe`.

> In PyInstaller builds, environment checks are skipped because dependencies are bundled.

### Option B: Run from source (recommended for developers / GPU)

Requirements:

- Python **3.10+**
- (Optional) NVIDIA GPU for faster NER

Run:

- `00_开发.bat` (Windows helper), or
- `python app.py`

On first run, `app.py` calls `module/EnvChecker.py` to:

- validate Python version
- detect GPU/CUDA
- install missing packages using common mirrors
- reinstall PyTorch with CUDA wheels when a GPU is present but PyTorch is CPU-only

---

## Configuration (config.json)

`config.json` is **the main runtime configuration**.

This project uses a “value + comment” style for many fields:

```json
"count_threshold": [2, "...comment..." ]
```

Only the **first** element is read as the real value.

### 1) Multi-platform LLM configuration

BookTerm Gacha talks to LLMs via **OpenAI-compatible Chat Completions** (`openai.AsyncOpenAI`), and can switch between multiple platforms.

Key fields:

- `activate_platform`: the `id` of the active platform entry.
- `platforms[]`: list of platform configs.
  - `api_url`: base URL (OpenAI-compatible)
  - `api_key`: **array** of keys (supports rotation)
  - `model`: model name string
  - `thinking`: hint flag (platform-specific; see below)

Example (trimmed):

```json
{
  "activate_platform": 1,
  "platforms": [
    {
      "id": 1,
      "name": "NVIDIA-DeepSeek-V3.2",
      "api_url": "https://integrate.api.nvidia.com/v1",
      "api_key": ["key1", "key2"],
      "model": "deepseek-ai/deepseek-v3.2",
      "thinking": true
    }
  ]
}
```

### 2) API key rotation & blacklist

If `api_key` contains multiple keys, the tool enables **round-robin rotation**:

- Each request picks the next available key.
- If a key hits a **provider ban / blacklist** (detected from `PermissionDenied` + keywords like `banned/suspended/blacklist`), the key is added to an in-memory blacklist and skipped.
- If all keys are blacklisted, the run stops with a fatal error.

This is designed for high-throughput batch runs where you want to:

- avoid per-key rate ceilings
- reduce total wall-clock time by distributing load

### 3) Rate limiting & concurrency

Two knobs control request pressure:

- `request_frequency_threshold`: “how many requests per second we *send*”
- `max_concurrent_requests`: “how many requests can be *in-flight* simultaneously”

Implementation details (see `model/LLM.py`):

- A semaphore caps concurrent in-flight requests.
- An async rate limiter (`aiolimiter.AsyncLimiter`) caps the send rate.

Why two limits? Because your average request latency matters:

If average latency is $L$ seconds, and you send $R$ requests/sec, you’ll have about $R \cdot L$ requests in flight. A high $L$ with a high $R$ will explode concurrency unless capped.

Practical tuning tips:

- Getting `429 Too Many Requests`:
  - lower `request_frequency_threshold`
  - and/or lower `max_concurrent_requests`
- You have many keys and want speed:
  - keep `request_frequency_threshold` around `5–10`
  - set `max_concurrent_requests` higher (e.g. `30–90`) *only if your provider and network can handle it*

### 4) NER and filtering knobs

These parameters are crucial to keep LLM calls affordable and improve quality.

- `score_threshold` (default `0.60`)
  - After NER, terms below this confidence are dropped.
- `count_threshold` (default `2`)
  - Terms with fewer occurrences are dropped.
  - This saves a lot of LLM tokens on one-off noise.
- `max_display_length` (default `32`)
  - Filters out abnormal long “terms”.
- `ner_target_types` (default `['PER', 'LOC']`)
  - NER stage keeps only selected entity types.
  - Including `ORG/PRD` increases coverage but can multiply LLM cost.

### 5) Context sampling controls

LLMs are only as good as the context you give them. This project implements **diverse context sampling** in `model/Word.py`:

- `max_context_samples` (default `5` in config; loaded into `Word.MAX_CONTEXT_SAMPLES`)
- `tokens_per_sample` (default `512` in config; loaded into `Word.TOKENS_PER_SAMPLE`)

How it works:

1. For each term, all lines containing the term are collected.
2. The occurrences are split into $N$ evenly-spaced groups.
3. From each group’s starting position, adjacent lines are aggregated until ~80% of the token budget.
4. Each aggregated paragraph is emitted with a stable header `【上下文 i】`.

This yields contexts from different parts of the book (first appearance, later usage, dialogue vs narration, etc.).

### 6) Traditional Chinese output

- `traditional_chinese_enable`: if `true`, output Chinese strings are converted using OpenCC (`s2tw`).

---

## How the pipeline works (v0.2.0)

The runtime workflow lives in `app.py`.

### Stage 0 — Text ingestion & normalization

`module/FileManager.py` reads files from `input/` (or a user-provided path) and normalizes lines:

- Unicode normalization + fullwidth/halfwidth fixes (`module/Normalizer.py`)
- whitespace cleanup
- language filtering (keeps only lines containing characters of the selected language)

Supported input formats include:

- books: `.epub`, `.txt`, `.md`, `.xlsx`
- scripts/subtitles: `.ass`, `.srt`, `.rpy`, `.trans`
- json formats: KV-style JSON, message JSON

### Stage 1 — NER (local BERT, GPU-friendly)

`model/NER.py` uses Hugging Face Transformers `pipeline('token-classification')` with aggregation.

Key techniques:

- **Chunking to 512 tokens** so long books can be processed safely.
- **bf16/fp16** on GPU when available (`is_torch_bf16_gpu_available`).
- **Ruby/Furigana removal** (`module/RubyCleaner.py`) for Japanese sources.
- **Stopword blacklist** (`blacklist/*.json`) to remove pronouns/particles/etc.
- **Japanese-specific rule**: keep only candidates that contain kana (hiragana/katakana), because the main use case is katakana-style names that must be transliterated.

The result is a list of candidate entities with confidence scores.

### Stage 1.5 — Heuristic filtering

In `app.py`:

- merge duplicates by surface, keep max score
- drop low-confidence by `score_threshold`
- search per-term contexts; compute occurrence count
- drop low-frequency by `count_threshold`
- drop overly long “terms” by `max_display_length`

These filters reduce LLM workload dramatically.

### Stage 2 — LLM context translation (NEW workflow)

For non-Chinese source languages, v0.2.0 forces a two-step strategy:

1. **Translate sampled contexts into Simplified Chinese** (`context_translate_batch`).
2. Use the translated context to improve the next stage’s semantic decisions.

Hard rules in the translation prompt (see `prompt/prompt_context_translate.txt`):

- keep `【上下文 N】` headers exactly
- keep line alignment (no merging/splitting lines)
- forbid copying kana/latin/hangeul into the Chinese output

Robustness tactics:

- per-task timeout with `asyncio.wait_for`
- automatic context shrinking when timeouts or content filters happen

### Stage 3 — LLM surface analysis (classification + final term translation)

For each term, the LLM returns a JSON object with:

- `summary` (Chinese)
- `group` (fine-grained taxonomy)
- `gender` (`男/女/无法判断`)
- `translation` (final Chinese term)

Then the code maps fine-grained types into output groups:

- `角色` (Character) — includes `姓氏/名字/...`
- `地点` (Location)

Everything else is filtered out as “not target”.

Quality gates (automatic retries):

- strict kana residue detection (allows isolated onomatopoeia markers like `ッ`, `ヶ`)
- hangeul residue detection for KO
- degraded output detection (repetitive patterns)
- “translation failed” similarity detection (Jaccard similarity ≥ 0.80)

### Stage 4 — Automatic fix pass (problem terms only)

After surface analysis, a third-stage fixer (`fix_translation_batch`) runs only on problematic entries:

- empty translations
- kana residue
- similarity-too-high

If fixing still fails, the tool uses a **forced transliteration fallback**:

1. Convert to romaji via `pykakasi`
2. Map romaji syllables to common Chinese phonetic chunks

This ensures you always get a usable Chinese output, even under bad model behavior.

### Stage 5 — Output & QA reports

Finally, outputs are written per group, then `module/ResultChecker.py` generates QA reports.

---

## LLM platform compatibility notes

`model/LLM.py` automatically detects certain platforms and adjusts request shape:

- Zhipu (bigmodel.cn): stream + thinking
- NVIDIA Build DeepSeek: stream + `chat_template_kwargs.thinking`
- ModelScope DeepSeek: stream + `enable_thinking`
- OpenAI O-series: uses `max_completion_tokens` instead of `max_tokens`

Everything else uses the standard OpenAI-compatible format.

LLM generation parameters are primarily controlled by:

- `resource/llm_config/context_translate_config.json`
- `resource/llm_config/surface_analysis_config.json`
- `resource/llm_config/api_test_config.json`

> The per-platform `temperature/top_p/...` fields in `config.json` are currently not the primary source of sampling parameters; they are reserved / informational.

---

## Troubleshooting

### “No input found”

- Put files into `input/`, or paste a path when prompted.

### GPU is not used

- NER uses GPU only if `torch.cuda.is_available()` is `True`.
- If you have an NVIDIA GPU but PyTorch is CPU-only, the environment checker will try to reinstall CUDA wheels.

### Too many requests / 429

- Lower `request_frequency_threshold`.
- Lower `max_concurrent_requests`.
- Add more API keys and keep rotation enabled.

### Timeouts

- Increase `request_timeout`.
- Increase `task_timeout_threshold` (per term; triggers context shrinking on timeout).

### Content filter errors

The tool automatically:

- excludes the sampled context indices that triggered filtering
- shrinks context sampling and retries

If it still fails, consider lowering:

- `max_context_samples`
- `tokens_per_sample`

### “All API keys are blacklisted”

- Your provider rejected the keys (ban/suspension) and the tool stopped using them.
- Replace keys, or switch `activate_platform` to another provider.

### Output looks untranslated / too similar

- Check `output/结果检查_相似度过高.json`.
- Try a stronger model, or lower temperature in `resource/llm_config/*`.

---

## Developer notes (build)

Windows helpers:

- `00_开发.bat` — run from source
- `01_打包.bat` — PyInstaller build

PyInstaller spec:

- `BookTermGacha.spec` bundles Python deps into `dist/BookTermGacha/_internal`
- `build_finish.py` copies runtime resources (prompts, blacklist, NER model, LLM configs)

---

## Credits

- Based on **KeywordGacha v0.13.1** (neavo) design and ecosystem.
- Bundled NER model under `resource/kg_ner_bf16/` declares **Apache-2.0** in its model card.

---

## About this README

This README is generated from the current workspace code in `BookTerm Gacha/` (v0.2.0) and focuses on the actual implemented behavior:

- three-stage LLM flow (translate → analyze → fix)
- multi-platform OpenAI-compatible APIs
- API key rotation + blacklist
- diversified context sampling
- strict QA checks and forced transliteration fallback
<h1><p align="center">📚 BookTerm Gacha</p></h1>
<p align="center"><strong>An LLM-Powered Agent for Automated Book Terminology Extraction</strong></p>
<p align="center"><em>Multi-Platform LLM Support with API Key Rotation - Extract Character & Location Names from Japanese Literatures</em></p>

<p align="center">
  <img src="https://img.shields.io/badge/Version-0.2.0-brightgreen.svg" alt="Version"/>
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
  <img src="https://img.shields.io/badge/LLM-Multi_Platform-orange.svg" alt="Multi Platform"/>
  <img src="https://img.shields.io/badge/NVIDIA-DeepSeek_V3.2-76B900.svg" alt="NVIDIA"/>
  <img src="https://img.shields.io/badge/ModelScope-阿里云百炼-blue.svg" alt="ModelScope"/>
  <img src="https://img.shields.io/badge/Zhipu-GLM_4.6v-red.svg" alt="Zhipu GLM"/>
  <img src="https://img.shields.io/badge/NER-BERT-purple.svg" alt="BERT NER"/>
  <img src="https://img.shields.io/badge/GPU-CUDA_Supported-76B900.svg" alt="CUDA"/>
</p>

---

## 📋 Table of Contents

- [What is BookTerm Gacha?](#-what-is-bookterm-gacha)
- [Key Features](#-key-features)
- [Quick Start Guide](#-quick-start-guide)
- [Installation Methods](#-installation-methods)
  - [Method 1: Download Release (Recommended for Most Users)](#method-1-download-release-recommended-for-most-users)
  - [Method 2: Clone Repository (For Developers & GPU Users)](#method-2-clone-repository-for-developers--gpu-users)
- [Configuration Guide](#-configuration-guide)
- [How to Use](#-how-to-use)
- [Understanding the Output](#-understanding-the-output)
- [Troubleshooting](#-troubleshooting)
- [Technical Details](#-technical-details)
- [Release Notes](#-release-notes)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🎯 What is BookTerm Gacha?

**BookTerm Gacha** is an intelligent tool that automatically extracts character names, location names, and other important terminology from books and generates translation glossaries.

### The Problem It Solves

When translating Japanese Literatures to Chinese, translators face a common challenge:

> **How do you consistently translate character names like `アリス`, `トリシューラ`, or `ティナーシャ`?**

These katakana names need to be transliterated into Chinese, and keeping them consistent throughout a book (or book series) is tedious and error-prone.

### The Solution

BookTerm Gacha automates this process:

1. **Reads** your EPUB/TXT/MD book files
2. **Extracts** all character and location names using AI (BERT NER model)
3. **Analyzes** each name with context using LLM (Zhipu GLM)
4. **Generates** a terminology glossary ready for use with translation tools

### Who Is This For?

- 📖 **Japanese Literatures Translators** - Create consistent terminology tables for your translation projects
- 🎮 **Fan Translation Groups** - Standardize character names across team members
- 📚 **Translation Tool Users** - Generate glossaries for [LinguaGacha](https://github.com/neavo/LinguaGacha), [GalTransl](https://github.com/xd2333/GalTransl), and similar tools
- 🤖 **AI/LLM Enthusiasts** - Learn how to build practical LLM Agent applications

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📖 **Book-Focused** | Optimized specifically for EPUB, TXT, and MD book formats |
| 🇯🇵 **Japanese Optimized** | Fine-tuned for Japanese → Chinese translation with zero kana residue |
| � **Multi-Platform LLM** | Support for NVIDIA Build, ModelScope (阿里云百炼), Zhipu GLM, and more |
| 🔄 **API Key Rotation** | Multiple API keys with automatic round-robin polling for massive throughput |
| 🚫 **Smart Blacklist** | Automatic detection and blacklisting of banned/expired API keys |
| 🆓 **Free Tier Options** | Zhipu GLM-4-Flash FREE, NVIDIA/ModelScope generous free quotas |
| 🚀 **GPU Acceleration** | Automatic CUDA detection for fast NER processing (optional) |
| 💡 **Deep Thinking** | Support for reasoning models (DeepSeek R1, GLM with thinking) |
| 📊 **Rich Progress** | Beautiful progress bars showing exactly what's happening |
| ✅ **Quality Validation** | Automatic result checking for kana residue and issues |
| 📝 **Multiple Formats** | Outputs in JSON dictionary, LinguaGacha, and GalTransl formats |

---

## 🚀 Quick Start Guide

**For users who just want to get started quickly:**

1. Download the latest release from [GitHub Releases](https://github.com/1235357/BookTerm-Gacha/releases)
2. Extract the ZIP file to any folder
3. Choose your LLM platform and get API key(s):
   - **NVIDIA Build** (Recommended): [build.nvidia.com](https://build.nvidia.com/) - DeepSeek V3.2 免费额度
   - **ModelScope 阿里云百炼**: [modelscope.cn](https://www.modelscope.cn/) - 免费额度
   - **Zhipu AI 智谱**: [bigmodel.cn](https://bigmodel.cn/) - GLM-4-Flash 完全免费
4. Edit `config.json` - add your API keys and select platform
5. Put your EPUB/TXT files in the `input/` folder
6. Run `app.exe`
7. Find your terminology glossary in the `output/` folder

**That's it!** For detailed instructions, continue reading below.

---

## 📦 Installation Methods

There are **two ways** to use BookTerm Gacha. Choose the one that fits your needs:

| Method | Best For | GPU Support | Difficulty |
|--------|----------|-------------|------------|
| **[Method 1: Download Release](#method-1-download-release-recommended-for-most-users)** | Most users, quick setup | CPU only (bundled) | ⭐ Easy |
| **[Method 2: Clone Repository](#method-2-clone-repository-for-developers--gpu-users)** | Developers, GPU users | Full CUDA support | ⭐⭐⭐ Advanced |

---

### Method 1: Download Release (Recommended for Most Users)

This is the **easiest way** to get started. The release package includes everything you need.

#### Step 1: Download the Release

1. Go to [GitHub Releases](https://github.com/1235357/BookTermGacha/releases)
2. Download the latest `BookTermGacha-v0.1.0.zip` file
3. Extract the ZIP to any folder (e.g., `C:\BookTermGacha\` or `D:\Tools\BookTermGacha\`)

#### Step 2: Understand the Folder Structure

After extraction, you'll see:

```
BookTermGacha/
├── app.exe                 # Main executable - double-click to run
├── config.json             # Configuration file - YOU NEED TO EDIT THIS
├── version.txt             # Version information
│
├── blacklist/              # Filter lists (pre-configured, don't modify)
│   ├── jp_语气助词.json
│   ├── jp_人称代词.json
│   └── ...
│
├── prompt/                 # LLM prompts (pre-configured, don't modify)
│   └── ...
│
├── resource/               # Required resources
│   ├── kg_ner_bf16/        # BERT NER model (DO NOT DELETE)
│   └── llm_config/         # LLM configuration presets
│
├── input/                  # PUT YOUR BOOKS HERE
│   └── (empty - add your EPUB/TXT files)
│
├── output/                 # RESULTS APPEAR HERE
│   └── (empty - generated files will be here)
│
└── log/                    # Log files for troubleshooting
    └── (auto-generated)
```

#### Step 3: Get Your API Keys

BookTerm Gacha 支持多个 LLM 平台，可以配置多个 API Key 进行轮询以获得更高吞吐量：

**🏆 推荐平台对比:**

| 平台 | 模型 | 免费额度 | 速度 | 质量 | 推荐度 |
|------|------|----------|------|------|--------|
| **NVIDIA Build** | DeepSeek V3.2 | 1000 次/天 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🥇 首选 |
| **ModelScope 阿里云百炼** | DeepSeek V3.2/R1 | 充足 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🥈 备选 |
| **Zhipu AI 智谱** | GLM-4.6v-Flash | 无限制 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 🥉 免费首选 |

**获取 API Key:**

**NVIDIA Build (推荐):**
1. 访问 [https://build.nvidia.com/](https://build.nvidia.com/)
2. 注册/登录 NVIDIA 账号
3. 搜索 "DeepSeek V3.2" 模型
4. 点击 "Get API Key" 获取密钥
5. 可注册多个账号获取多个 Key 用于轮询

**ModelScope 阿里云百炼:**
1. 访问 [https://www.modelscope.cn/](https://www.modelscope.cn/)
2. 使用支付宝/阿里云账号登录
3. 进入模型推理页面
4. 获取 API Key (格式: `ms-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)

**Zhipu AI 智谱 (完全免费):**
1. 访问 [https://bigmodel.cn/](https://bigmodel.cn/)
2. 注册账号 (需手机验证)
3. 进入控制台创建 API Key

> 💡 **性能提示**: 使用多个 API Key 轮询可以大幅提升处理速度！建议每个平台准备 3-5 个 Key。

#### Step 4: Configure Your API Keys

1. Open `config.json` with any text editor (Notepad, VS Code, etc.)
2. The new multi-platform configuration format:

```json
{
    "activate_platform": 1,
    "platforms": [
        {
            "id": 0,
            "name": "智谱GLM-4.6v-flash(免费)",
            "api_url": "https://open.bigmodel.cn/api/paas/v4",
            "api_key": ["your-zhipu-api-key"],
            "model": "glm-4.6v-flash",
            "thinking": true,
            "description": "智谱免费模型，支持深度思考"
        },
        {
            "id": 1,
            "name": "NVIDIA-DeepSeek-V3.2",
            "api_url": "https://integrate.api.nvidia.com/v1",
            "api_key": [
                "nvapi-key1",
                "nvapi-key2",
                "nvapi-key3"
            ],
            "model": "deepseek-ai/deepseek-v3.2",
            "thinking": true,
            "description": "NVIDIA Build DeepSeek V3.2，多Key轮询"
        },
        {
            "id": 2,
            "name": "魔塔-DeepSeek-V3.2",
            "api_url": "https://api-inference.modelscope.cn/v1/",
            "api_key": [
                "ms-key1",
                "ms-key2"
            ],
            "model": "deepseek-ai/DeepSeek-V3.2",
            "thinking": true,
            "description": "阿里云百炼 ModelScope"
        }
    ]
}
```

3. **设置 `activate_platform`** 为你想使用的平台 ID:
   - `0` = 智谱 GLM (免费)
   - `1` = NVIDIA DeepSeek V3.2 (推荐)
   - `2` = ModelScope DeepSeek V3.2
   - `3` = ModelScope DeepSeek R1

4. **添加你的 API Keys** 到对应平台的 `api_key` 数组中

5. **Save** the file

#### Step 5: Add Your Books

1. Copy your Japanese book files into the `input/` folder
2. Supported formats:
   - `.epub` - E-book format (recommended)
   - `.txt` - Plain text (must be UTF-8 encoded)
   - `.md` - Markdown files

#### Step 6: Run the Program

1. **Double-click** `app.exe` to start
2. A console window will open showing progress
3. Wait for processing to complete (time depends on book size)
4. Check the `output/` folder for your results

#### What If It Doesn't Work?

If you encounter errors:
- See the [Troubleshooting](#-troubleshooting) section
- Check the `log/` folder for detailed error messages
- If GPU-related issues occur, consider [Method 2](#method-2-clone-repository-for-developers--gpu-users)

---

### Method 2: Clone Repository (For Developers & GPU Users)

Remember to clone https://huggingface.co/neavo/keyword_gacha_multilingual_ner to "\BookTerm Gacha\resource\kg_ner_bf16" (Since "model.safetensors" is too large for GitHub)

Choose this method if:
- ✅ You have an NVIDIA GPU and want **faster processing** (3-10x speedup)
- ✅ The release version **doesn't recognize your GPU**
- ✅ You want to **modify the code** or contribute to development
- ✅ You want to use a **different Python version** or environment
- ✅ You're experiencing **compatibility issues** with the release version

#### Prerequisites

Before starting, make sure you have:

| Requirement | How to Check | How to Install |
|-------------|--------------|----------------|
| **Python 3.10+** | `python --version` | [python.org](https://www.python.org/downloads/) |
| **Git** (optional) | `git --version` | [git-scm.com](https://git-scm.com/) |
| **NVIDIA GPU** (optional) | `nvidia-smi` | Driver from [nvidia.com](https://www.nvidia.com/drivers/) |
| **CUDA Toolkit** (for GPU) | `nvcc --version` | [CUDA Downloads](https://developer.nvidia.com/cuda-downloads) |

#### Step 1: Clone or Download the Repository

**Option A: Using Git (Recommended)**
```bash
# Open Command Prompt or PowerShell
git clone https://github.com/1235357/BookTermGacha.git
cd BookTermGacha
```

**Option B: Download ZIP**
1. Go to the repository page
2. Click "Code" → "Download ZIP"
3. Extract to your preferred location
4. Open Command Prompt/PowerShell and navigate to the folder:
```bash
cd C:\path\to\BookTermGacha
```

#### Step 2: Create a Virtual Environment (Highly Recommended)

A virtual environment keeps this project's dependencies separate from other Python projects.

**Windows (Command Prompt):**
```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# You should see (venv) at the start of your command line
```

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate it (you may need to allow script execution first)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

> ⚠️ **Important**: Always activate the virtual environment before running commands!

#### Step 3: Install PyTorch (CRITICAL for GPU Users)

This is the **most important step** for GPU acceleration. You must install PyTorch with the correct CUDA version **BEFORE** installing other dependencies.

**First, check your CUDA version:**
```bash
nvidia-smi
```

Look for "CUDA Version" in the output (e.g., "CUDA Version: 12.4").

**Then install PyTorch with matching CUDA:**

| Your CUDA Version | Installation Command |
|-------------------|---------------------|
| **CUDA 12.6** (RTX 40 series) | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126` |
| **CUDA 12.4** | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` |
| **CUDA 12.1** | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| **CUDA 11.8** (GTX 10/16/20 series) | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` |
| **No GPU / CPU Only** | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` |

**Example for CUDA 12.4:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### Step 4: Install Other Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages:
- `transformers` - For BERT NER model
- `openai` - For LLM API calls
- `ebooklib` - For EPUB reading
- `rich` - For beautiful console output
- `pykakasi` - For Japanese text processing
- And more...

#### Step 5: Verify Your Installation

Run these commands to make sure everything is working:

```bash
# Check Python version
python --version

# Check if PyTorch sees your GPU
python -c "import torch; print(f'PyTorch Version: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"

# Check transformers
python -c "from transformers import AutoModel; print('Transformers: OK')"

# Check other dependencies
python -c "import openai, ebooklib, rich; print('All dependencies: OK')"
```

**Expected output (with GPU):**
```
PyTorch Version: 2.5.1+cu124
CUDA Available: True
GPU Device: NVIDIA GeForce RTX 4090
Transformers: OK
All dependencies: OK
```

**Expected output (CPU only):**
```
PyTorch Version: 2.5.1+cpu
CUDA Available: False
GPU Device: CPU only
Transformers: OK
All dependencies: OK
```

#### Step 6: Configure Your API Key

Same as Method 1 - edit `config.json` with your Zhipu AI API key.

#### Step 7: Run the Program

```bash
# Make sure virtual environment is activated
# (venv) should appear in your prompt

python app.py
```

#### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| `CUDA Available: False` even with GPU | Reinstall PyTorch with correct CUDA version |
| `ModuleNotFoundError` | Make sure venv is activated, run `pip install -r requirements.txt` |
| Permission errors on Windows | Run Command Prompt as Administrator |
| Python not found | Add Python to PATH or use full path |

---

## ⚙️ Configuration Guide

The `config.json` file controls all settings. Here's a detailed explanation:

### Platform Configuration (核心配置)

```json
{
    "activate_platform": 1,
    "platforms": [...]
}
```

| Setting | Description | Values |
|---------|-------------|--------|
| `activate_platform` | 当前激活的平台 ID | `0`, `1`, `2`, `3` |
| `platforms` | 平台配置数组 | 见下方详细说明 |

### Platform Object Structure (平台配置结构)

```json
{
    "id": 1,
    "name": "NVIDIA-DeepSeek-V3.2",
    "api_url": "https://integrate.api.nvidia.com/v1",
    "api_key": ["key1", "key2", "key3"],
    "model": "deepseek-ai/deepseek-v3.2",
    "thinking": true,
    "top_p": 0.95,
    "temperature": 0.95,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "description": "描述文字"
}
```

| Field | Description | Example |
|-------|-------------|---------|
| `id` | 平台唯一标识 | `0`, `1`, `2`, `3` |
| `name` | 平台显示名称 | `"NVIDIA-DeepSeek-V3.2"` |
| `api_url` | API 端点 URL | `"https://integrate.api.nvidia.com/v1"` |
| `api_key` | API Key 数组 (支持多 Key 轮询) | `["key1", "key2"]` |
| `model` | 模型名称 | `"deepseek-ai/deepseek-v3.2"` |
| `thinking` | 是否启用深度思考模式 | `true` / `false` |

### Pre-configured Platforms (预配置平台)

| ID | Platform | API URL | Model | Free Tier |
|----|----------|---------|-------|-----------|
| `0` | **智谱 GLM** | `https://open.bigmodel.cn/api/paas/v4` | `glm-4.6v-flash` | ✅ 完全免费 |
| `1` | **NVIDIA Build** | `https://integrate.api.nvidia.com/v1` | `deepseek-ai/deepseek-v3.2` | ✅ 1000次/天 |
| `2` | **ModelScope V3.2** | `https://api-inference.modelscope.cn/v1/` | `deepseek-ai/DeepSeek-V3.2` | ✅ 充足额度 |
| `3` | **ModelScope R1** | `https://api-inference.modelscope.cn/v1/` | `deepseek-ai/DeepSeek-R1-0528` | ✅ 充足额度 |

### Multi-API Key Rotation (多 Key 轮询)

```json
"api_key": [
    "nvapi-key1-xxxxx",
    "nvapi-key2-xxxxx",
    "nvapi-key3-xxxxx",
    "nvapi-key4-xxxxx",
    "nvapi-key5-xxxxx"
]
```

**特性：**
- 🔄 **自动轮询**: 请求自动分配到不同的 API Key
- 🚫 **智能黑名单**: 被封禁的 Key 自动加入黑名单，不影响其他 Key
- ⚡ **并发提升**: 5 个 Key = 5 倍吞吐量
- 📊 **状态显示**: 启动时显示可用 Key 数量

### Optional Settings (可选配置)

```json
{
    "count_threshold": [2, "出现次数阈值"],
    "score_threshold": [0.60, "NER 置信度阈值 (0.0-1.0)"],
    "max_context_samples": [5, "上下文采样段落数"],
    "tokens_per_sample": [512, "每段最大 token 数"],
    "ner_target_types": [["PER", "LOC"], "提取的实体类型"],
    "request_timeout": [1800, "API 超时时间(秒)"],
    "request_frequency_threshold": [10, "每秒最大请求数"],
    "max_concurrent_requests": [90, "最大并发请求数"],
    "traditional_chinese_enable": [false, "繁体中文输出"]
}
```

| Setting | Description | Default | 推荐值 |
|---------|-------------|---------|--------|
| `count_threshold` | 词语最少出现次数 | `2` | 保持默认 |
| `score_threshold` | NER 置信度阈值 | `0.60` | `0.50-0.70` |
| `request_frequency_threshold` | 每秒请求数上限 | `10` | 多 Key 时设为 `5-10` |
| `max_concurrent_requests` | 最大并发数 | `90` | 多 Key 时可增加 |
| `traditional_chinese_enable` | 繁体中文 | `false` | 台湾/香港用户设为 `true` |

### Adding Custom LLM Providers (添加自定义平台)

可以在 `platforms` 数组中添加任何 OpenAI 兼容的 API：

```json
{
    "id": 4,
    "name": "My-Custom-Provider",
    "api_url": "https://api.example.com/v1",
    "api_key": ["your-api-key"],
    "model": "model-name",
    "thinking": false,
    "top_p": 0.95,
    "temperature": 0.7,
    "description": "自定义平台"
}
```

**支持的平台类型：**

| Provider | Base URL | Model | Notes |
|----------|----------|-------|-------|
| **NVIDIA Build** | `https://integrate.api.nvidia.com/v1` | `deepseek-ai/deepseek-v3.2` | 🥇 推荐，支持思考模式 |
| **ModelScope** | `https://api-inference.modelscope.cn/v1/` | `deepseek-ai/DeepSeek-V3.2` | 🥈 阿里云百炼 |
| **Zhipu AI** | `https://open.bigmodel.cn/api/paas/v4` | `glm-4.6v-flash` | 🆓 完全免费 |
| **DeepSeek** | `https://api.deepseek.com/v1` | `deepseek-chat` | 官方 API |
| **OpenAI** | `https://api.openai.com/v1` | `gpt-4o-mini` | 高质量，高成本 |
| **Local LLM** | `http://localhost:11434/v1` | varies | Ollama 等本地模型 |

---

## 📖 How to Use

### Step-by-Step Workflow

#### 1. Prepare Your Input Files

- Place your Japanese book files in the `input/` folder
- Supported formats: `.epub`, `.txt`, `.md`
- Multiple files can be processed in one run
- File names can be in any language (Japanese, English, Chinese, etc.)

**Example:**
```
input/
├── 転生したらスライムだった件 1.epub
├── Unnamed Memory.epub
├── my_novel.txt
└── another_book.md
```

#### 2. Run the Program

**Release version:** Double-click `app.exe`

**Development version:** 
```bash
python app.py
```

#### 3. Monitor Progress

The program shows real-time progress:

```
╭─ BookTerm Gacha v0.1.0 ─╮
│ Processing: 転生したらスライムだった件 1.epub
╰──────────────────────────╯

[1/4] Loading book...
████████████████████████████████████████ 100% Reading EPUB

[2/4] NER Entity Extraction (BERT)...
████████████████████████████████████████ 100% Found 127 entities

[3/4] Context Translation (LLM)...
████████████████░░░░░░░░░░░░░░░░░░░░░░░░  42% Processing entity 54/127

[4/4] Semantic Analysis (LLM)...
████████████████████████████████████████ 100% Analysis complete

✓ Processing complete! Check output/ folder.
```

#### 4. Collect Your Results

After processing, check the `output/` folder:

```
output/
├── 転生したらスライムだった件 1_角色_词典.json      # Character dictionary
├── 転生したらスライムだった件 1_角色_术语表.json    # LinguaGacha glossary
├── 転生したらスライムだった件 1_角色_galtransl.txt  # GalTransl format
├── 転生したらスライムだった件 1_角色_日志.txt       # Detailed log
├── 転生したらスライムだった件 1_地点_词典.json      # Location dictionary
├── 転生したらスライムだった件 1_地点_术语表.json    # Location glossary
└── 結果検査_报告.json                              # Quality report
```

---

## 📊 Understanding the Output

### Dictionary Format (`_词典.json`)

The main output - a list of terms with translations:

```json
[
    {
        "src": "リムル",
        "dst": "利姆鲁",
        "info": "主角，转生成史莱姆的日本人，后成为魔王。"
    },
    {
        "src": "シズ",
        "dst": "静",
        "info": "女性冒险者，被召唤到异世界的日本人。"
    }
]
```

| Field | Description |
|-------|-------------|
| `src` | Original Japanese name (source) |
| `dst` | Chinese translation (destination) |
| `info` | Character description/summary |

### LinguaGacha Format (`_术语表.json`)

Ready to import into [LinguaGacha](https://github.com/neavo/LinguaGacha):

```json
[
    {
        "src": "リムル",
        "dst": "利姆鲁",
        "info": "角色 - 男 - 主角，转生成史莱姆的日本人..."
    }
]
```

### GalTransl Format (`_galtransl.txt`)

For use with [GalTransl](https://github.com/xd2333/GalTransl):

```
リムル | 利姆鲁
シズ | 静
ヴェルドラ | 维鲁多拉
```

### Quality Report (`結果検査_报告.json`)

Automatically checks for issues:

```json
{
    "假名残留": ["エルフの里 → 精灵の里"],
    "未翻译条目": ["アルビス"],
    "相似度问题": []
}
```

---

## 🔧 Troubleshooting

### Common Problems and Solutions

#### "API key invalid" Error

**Problem:** The program says your API key is invalid.

**Solutions:**
1. Double-check your API key in `config.json`
2. Make sure there are no extra spaces
3. Verify the key is active on [bigmodel.cn](https://bigmodel.cn/)
4. Check if you've exceeded the free tier limits

#### GPU Not Detected (Release Version)

**Problem:** The release version runs on CPU even though you have an NVIDIA GPU.

**Why:** The release is bundled with CPU-only PyTorch for maximum compatibility.

**Solution:** Use [Method 2](#method-2-clone-repository-for-developers--gpu-users) to install with proper CUDA support.

#### "CUDA out of memory" Error

**Problem:** GPU runs out of memory during NER processing.

**Solutions:**
1. Close other GPU-intensive applications
2. Process smaller files or split large books
3. The program will automatically fall back to CPU if needed

#### "Module not found" Error

**Problem:** Python can't find required packages.

**Solutions:**
```bash
# Make sure venv is activated
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### Slow Processing

**Tips to speed up:**
1. Use GPU (Method 2 installation)
2. Reduce `max_context_samples` in config
3. Increase `count_threshold` to process fewer entities
4. Process books one at a time

#### Japanese Text Displays as Garbled Characters

**Problem:** Output shows `???` or garbled text.

**Solutions:**
1. Make sure your terminal supports UTF-8
2. For Windows: Run `chcp 65001` before starting
3. Open output files with UTF-8 encoding (use VS Code or Notepad++)

---

## 🔬 Technical Details

### How It Works

BookTerm Gacha uses a **4-stage pipeline**:

```
┌─────────────────────────────────────────────────────────────┐
│                    BookTerm Gacha Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Stage 1: NER Extraction (BERT)                             │
│  ─────────────────────────────                              │
│  • Reads book text from EPUB/TXT/MD                         │
│  • BERT model identifies named entities                     │
│  • Filters: Only keeps names WITH kana characters           │
│  • Output: List of potential character/location names       │
│                                                             │
│                         ↓                                   │
│                                                             │
│  Stage 2: Context Sampling                                  │
│  ────────────────────────                                   │
│  • For each entity, finds paragraphs where it appears       │
│  • Samples N context paragraphs (configurable)              │
│  • Prepares context for LLM analysis                        │
│                                                             │
│                         ↓                                   │
│                                                             │
│  Stage 3: LLM Analysis (Zhipu GLM)                          │
│  ─────────────────────────────────                          │
│  • Sends entity + context to LLM                            │
│  • LLM returns: translation, gender, category, summary      │
│  • Validation: Checks for kana residue, degradation         │
│  • Retry logic: Up to 8 retries with fallback               │
│                                                             │
│                         ↓                                   │
│                                                             │
│  Stage 4: Output Generation                                 │
│  ─────────────────────────                                  │
│  • ResultChecker validates all entries                      │
│  • Generates multiple output formats                        │
│  • Creates quality report                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Why BERT + LLM?

| Component | Task | Why This Choice |
|-----------|------|-----------------|
| **BERT NER** | Find names in text | Fast, accurate, runs locally |
| **Zhipu GLM** | Translate & analyze | Better context understanding, creative translation |

Using both gives the **best of both worlds** - fast extraction with intelligent analysis.

### Smart Kana Handling

The system handles Japanese kana intelligently:

- **Strict Mode**: Flags any kana remaining in translations
- **Tolerance**: Allows certain kana (っ, ー, ヶ) that appear in place names
- **Fallback**: After 5 failed retries, force-transliterates using romaji → Chinese mapping

---

## 📋 Release Notes

### Version 0.2.0 - Multi-Platform & API Key Rotation

**Release Date:** January 2026

**🎉 Major Version Upgrade!**

This release brings massive performance improvements with multi-platform LLM support and API key rotation.

#### ✨ New Features

- **🌐 Multi-Platform Support**: NVIDIA Build, ModelScope (阿里云百炼), Zhipu GLM 全面支持
- **🔄 API Key Rotation**: 多 API Key 自动轮询，大幅提升并发吞吐量
- **🚫 Smart Blacklist**: 自动检测被封禁的 Key 并加入黑名单
- **💡 Deep Thinking Mode**: 支持 DeepSeek V3.2/R1 的推理模式
- **⚡ NVIDIA DeepSeek**: 首选推荐平台，速度与质量兼具
- **📊 Enhanced Status**: 启动时显示平台信息和可用 Key 数量

#### 🔧 Technical Improvements

- 全新的多平台配置格式 (`platforms` 数组)
- 智能平台检测 (NVIDIA/ModelScope/Zhipu 自动识别)
- 流式响应优化，支持 `reasoning_content` 提取
- API Key 黑名单机制 (自动处理 403 错误)
- 并发控制优化 (`max_concurrent_requests` 配置)

#### 📦 Pre-configured Platforms

| Platform | Model | Features |
|----------|-------|----------|
| **智谱 GLM-4.6v-flash** | 免费无限制 | 深度思考 |
| **NVIDIA DeepSeek V3.2** | 5 Key 轮询 | 高速推理 |
| **ModelScope DeepSeek V3.2** | 5 Key 轮询 | 阿里云百炼 |
| **ModelScope DeepSeek R1** | 推理模型 | 深度思考 |

#### ⚠️ Breaking Changes

- `config.json` 格式已更新为多平台格式
- 旧配置需要迁移到新的 `platforms` 数组格式
- 新增 `activate_platform` 字段指定活动平台

---

### Version 0.1.0-Zhipu_GLM-Optimize

**Release Date:** January 2026

**🎉 First Major Release!**

This is the first stable release of BookTerm Gacha, specifically optimized for the Zhipu GLM API.

#### ✨ New Features

- **Zhipu GLM Optimization**: Fine-tuned prompts and settings for best results with GLM-4-Flash
- **Complete Workflow**: Full pipeline from book input to glossary output
- **Multiple Output Formats**: JSON dictionary, LinguaGacha glossary, GalTransl format
- **Rich Progress Display**: Beautiful console output with progress bars
- **Quality Validation**: Automatic checking for kana residue and translation issues
- **Smart Retry Logic**: Intelligent retry with forced transliteration fallback

#### 🔧 Technical Improvements

- Optimized NER filtering (only processes kana-containing entities)
- Place name particle handling (ヶ, の, etc.)
- Context-aware kana detection with tolerance rules
- Comprehensive error handling and logging

#### 📦 What's Included

- Pre-configured for Zhipu GLM API (free tier available)
- BERT NER model for Japanese entity extraction
- Blacklist filters for common words (particles, pronouns, etc.)
- LLM prompts optimized for terminology extraction

#### ⚠️ Known Limitations

- Release version uses CPU-only PyTorch (use dev setup for GPU)
- Optimized for Japanese → Chinese (other languages may work but untested)
- Large books (500k+ characters) may take 30+ minutes

#### 🙏 Based On

- [KeywordGacha v0.13.1](https://github.com/neavo/KeywordGacha) - Core workflow and NER model
- [LinguaGacha](https://github.com/neavo/LinguaGacha) - Validation patterns and text utilities

---

## 🙏 Acknowledgments

This project wouldn't be possible without:

- **[KeywordGacha](https://github.com/neavo/KeywordGacha)** by neavo - The original project that inspired this fork
- **[LinguaGacha](https://github.com/neavo/LinguaGacha)** by neavo - Design patterns and validation logic
- **[NVIDIA Build](https://build.nvidia.com/)** - High-performance DeepSeek API
- **[ModelScope 阿里云百炼](https://www.modelscope.cn/)** - Generous free LLM API
- **[Zhipu AI / BigModel](https://bigmodel.cn/)** - Free LLM API that makes this accessible to everyone
- **[Hugging Face Transformers](https://huggingface.co/)** - The BERT NER pipeline
- **The Japanese Literatures Translation Community** - For feedback and testing

---

## 📄 License

This project is released under the **MIT License**.

You are free to:
- ✅ Use commercially
- ✅ Modify
- ✅ Distribute
- ✅ Use privately

With the condition that you include the original license and copyright notice.

**If you use this tool in your translation work, please credit appropriately.**

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report Bugs**: Open an issue describing the problem
2. **Suggest Features**: Share your ideas for improvements
3. **Submit Code**: Fork, make changes, submit a pull request
4. **Improve Docs**: Help make this README even better
5. **Share**: Tell others about this tool!

### Development Setup

```bash
# Clone the repo
git clone https://github.com/1235357/BookTermGacha.git
cd BookTermGacha

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install with GPU support (see Method 2)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Run tests
python -m pytest -v

# Make your changes and submit a PR!
```

---

## ❓ FAQ

**Q: 哪个 LLM 平台最推荐？**
A: NVIDIA Build 的 DeepSeek V3.2 是首选，速度快质量高。ModelScope（阿里云百炼）是备选，智谱 GLM 完全免费适合入门。

**Q: 多 API Key 轮询有什么好处？**
A: 5 个 Key = 5 倍吞吐量！可以大幅缩短处理时间。建议每个平台准备 3-5 个 Key。

**Q: API Key 被封禁了怎么办？**
A: 程序会自动检测并将被封的 Key 加入黑名单，不影响其他 Key 继续工作。

**Q: Is the Zhipu API really free?**
A: Yes! The GLM-4.6v-Flash model has a free tier with no limits. Perfect for getting started.

**Q: Can I use this for Korean/English books?**
A: Currently optimized for Japanese → Chinese. Other languages may work but are untested.

**Q: How long does processing take?**
A: Depends on book size and API Key count. With 5 Keys, a typical 100k character book takes 3-8 minutes.

**Q: 如何添加新的 LLM 平台？**
A: 在 `config.json` 的 `platforms` 数组中添加新平台配置，设置 `activate_platform` 为新平台的 ID。

**Q: Why are some names not extracted?**
A: The NER model focuses on names with kana. Pure kanji names (like 田中) are skipped as they don't need transliteration.

**Q: 深度思考模式（thinking）有什么作用？**
A: 启用后模型会进行更深入的推理分析，翻译质量更高，但速度略慢。推荐保持开启。

---

<p align="center">
  <strong>📚 BookTerm Gacha v0.2.0</strong>
  <br>
  <em>Multi-Platform LLM Support with API Key Rotation</em>
  <br>
  <em>Transforming Japanese Literatures into Translation-Ready Glossaries</em>
  <br><br>
  Made for the Japanese Literatures Translation Community
</p>
