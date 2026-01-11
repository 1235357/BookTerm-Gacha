<h1><p align="center">📚 BookTerm Gacha</p></h1>
<p align="center"><strong>An LLM-Powered Agent for Automated Book Terminology Extraction</strong></p>


<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
  <img src="https://img.shields.io/badge/LLM-Agent-orange.svg" alt="LLM Agent"/>
  <img src="https://img.shields.io/badge/NER-BERT-purple.svg" alt="BERT NER"/>
</p>

---

## 🎯 Project Overview

**BookTerm Gacha** is a specialized fork of [KeywordGacha v0.13.1](https://github.com/neavo/KeywordGacha), redesigned as an **LLM Agent** specifically optimized for:

- 📖 **Book-focused terminology extraction** (EPUB, TXT, MD formats)
- 🇯🇵 **Japanese light novel optimization** with zero-tolerance for kana residue
- 🤖 **LLM Agent development practice** - a real-world AI agent implementation
- 🔧 **Customizable workflow** with transparent, debuggable stages

### What Makes This Different?

| Feature | Original KG v0.13.1 | BookTerm Gacha (This Project) |
|---------|---------------------|-------------------------------|
| **Focus** | General (games, subtitles, books) | Books only (EPUB, TXT, MD) |
| **Target Language** | Multi-language | Optimized for Japanese → Chinese |
| **Kana Handling** | Basic detection | Strict detection + smart tolerance |
| **Retry Logic** | Simple retry | Staged retry + forced transliteration |
| **Progress Display** | Basic logging | Rich progress bars |
| **Result Validation** | None | Comprehensive result checker |
| **Agent Design** | Monolithic | Modular LLM Agent architecture |

---

## 🧠 Core Philosophy: LLM Agent Development

This project is a **practical LLM Agent development exercise**. The core insight is:

> **A terminology table is essentially a mapping from source language entities to target language translations.**
> 
> For Japanese books, this means: `日文假名/汉字 → 中文译名`

### The Agent Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        BookTerm Gacha Workflow                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  STAGE 1: NER Entity Extraction (BERT Model)                     │  │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │  │
│  │  • Input: Raw book text (EPUB/TXT/MD)                            │  │
│  │  • Model: Fine-tuned BERT for Japanese NER                       │  │
│  │  • Output: Entity list (PER: persons, LOC: locations)            │  │
│  │  • Filter: Only keep entities WITH kana (pure kanji filtered)    │  │
│  │  • GPU: Auto-detect CUDA for acceleration                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  STAGE 2: Context Translation (LLM)                              │  │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │  │
│  │  • For each entity: Sample N context paragraphs                  │  │
│  │  • LLM translates context to Chinese                             │  │
│  │  • Validation: Check for degradation, kana residue               │  │
│  │  • Retry: Up to 8 times with context reduction strategy          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  STAGE 3: Semantic Analysis (LLM)                                │  │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │  │
│  │  • Input: Entity + Original context + Translated context         │  │
│  │  • LLM outputs: { summary, group, gender, translation }          │  │
│  │  • Strict validation: Zero kana in translation                   │  │
│  │  • Fallback: Forced transliteration after 5 retries              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  STAGE 4: Result Validation & Output                             │  │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │  │
│  │  • ResultChecker: Scan for kana residue, similarity issues       │  │
│  │  • Output: JSON glossary, log files, GalTransl format            │  │
│  │  • Report: Detailed statistics and issue tracking                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Technical Deep Dive

### Key Innovations

#### 1. Smart Kana Detection with Tolerance

```python
# Problem: Some kana should be tolerated (onomatopoeia, place name particles)
# Solution: Context-aware detection

RULE_ONOMATOPOEIA = frozenset({
    "ッ", "っ",      # Sokuon (gemination)
    "ぁ", "ぃ", "ぅ", "ぇ", "ぉ",  # Small vowels
    "ゃ", "ゅ", "ょ", "ゎ",        # Small ya/yu/yo
    "ー",            # Long vowel mark
    "ヶ", "ケ", "ヵ", # Place name particles (前ヶ浜 → 前之滨)
    "の",            # Possessive particle in place names
})

# Only flag as "kana residue" if the kana is NOT isolated
# e.g., "咖ッ啡" → tolerate (isolated ッ)
# e.g., "カッコいい" → flag (ッ surrounded by kana)
```

#### 2. Staged Retry with Forced Transliteration

```python
MAX_RETRY = 8
FORCE_TRANSLITERATE_THRESHOLD = 5

# After 5 failed retries:
# 1. Use pykakasi to convert to romaji
# 2. Map romaji to Chinese phonetic equivalents
# 3. Guarantee a Chinese output (no kana residue)
```

#### 3. NER Filtering Strategy

```python
# Key insight: We only need entities WITH kana
# Pure kanji entities (田中, 東京) don't need terminology tables
# They can be directly preserved or simply converted

def verify_by_language(text: str, language: int) -> bool:
    if language == Language.JA:
        # Must contain at least one kana character
        if not (any_hiragana(text) or any_katakana(text)):
            return False  # Filter out pure kanji
    return True
```

#### 4. Prompt Engineering for Particle Handling

```
【group Selection Rules (Important)】
- If it's a particle/auxiliary word (の、は、が、を、です、ます, etc.)
  → Must select "无法判断" (Cannot Determine) or "其他" (Other)

【Special Kana Handling in Place Names】
- ヶ / ケ / ヵ: Means "of/no", e.g., 「前ヶ浜」→「前之滨」
- の: Means "of", e.g., 「見晴らしの丘」→「瞭望之丘」
```

---

## 📦 Installation & Setup

### Prerequisites

- **Python 3.10+** (3.11 or 3.12 recommended)
- **NVIDIA GPU** (optional but recommended for NER acceleration)
- **LLM API access** (free option available, see below)

### Step-by-Step Installation

#### Step 1: Clone or Download

```bash
# Option A: Clone with git
git clone https://github.com/YOUR_USERNAME/BookTermGacha.git
cd BookTermGacha

# Option B: Download ZIP and extract
```

#### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

#### Step 3: Install PyTorch with CUDA (GPU Users)

**⚠️ IMPORTANT**: Install PyTorch FIRST with the correct CUDA version for GPU acceleration.

```bash
# Check your CUDA version first
nvidia-smi

# Then install PyTorch with matching CUDA version:

# CUDA 12.6 (Latest GPUs - RTX 40 series, etc.)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (Older GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU Only (No NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Step 4: Install Other Dependencies

```bash
pip install -r requirements.txt
```

#### Step 5: Verify Installation

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"

# Check transformers
python -c "from transformers import AutoModel; print('Transformers OK')"
```

#### Step 6: Configure API

Edit `config.json` with your LLM API credentials (see below).

#### Step 7: Run

```bash
python app.py
```

### LLM API Configuration

Edit `config.json`:

```json
{
    "api_key": ["YOUR_API_KEY", "API key from your LLM provider"],
    "base_url": ["https://open.bigmodel.cn/api/paas/v4", "API endpoint URL"],
    "model_name": ["glm-4-flash", "Model name to use"]
}
```

#### 🆓 Free API Option: Zhipu AI (智谱AI / BigModel)

You can use **FREE** models from [bigmodel.cn](https://bigmodel.cn/):

1. Register at https://bigmodel.cn/
2. Get your API key from the console
3. Use these settings:
   ```json
   {
       "api_key": ["your-api-key-here"],
       "base_url": ["https://open.bigmodel.cn/api/paas/v4"],
       "model_name": ["glm-4-flash"]
   }
   ```

**Note**: `glm-4-flash` and `glm-4v-flash` are FREE models with generous rate limits!

#### Other Supported Providers

| Provider | Base URL | Recommended Model |
|----------|----------|-------------------|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Volcano Engine | See [wiki](https://github.com/neavo/KeywordGacha/wiki/VolcEngine) | `doubao-pro-32k` |

---

## 📁 Input & Output

### Supported Input Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| EPUB | `.epub` | E-book format (recommended for light novels) |
| Plain Text | `.txt` | UTF-8 encoded text files |
| Markdown | `.md` | Markdown documents |

Place your files in the `input/` folder before running.

### Output Files

After processing, you'll find these files in `output/`:

| File | Description |
|------|-------------|
| `input_角色_词典.json` | Character terminology dictionary |
| `input_角色_术语表.json` | LinguaGacha glossary format |
| `input_角色_galtransl.txt` | GalTransl GPT dictionary format |
| `input_角色_日志.txt` | Detailed analysis log with summaries |
| `input_地点_*.json/txt` | Same formats for locations |
| `结果检查_报告.json` | Quality check report |

### JSON Dictionary Format

```json
[
    {
        "src": "アリス",
        "dst": "爱丽丝",
        "info": "女主角，金发碧眼的少女，拥有治愈魔法的能力。"
    },
    {
        "src": "トリシューラ",
        "dst": "特里修拉",
        "info": "神秘的剑士，真实身份不明。"
    }
]
```

### LinguaGacha Glossary Format

```json
[
    {
        "src": "アリス",
        "dst": "爱丽丝",
        "info": "角色 - 女 - 女主角，金发碧眼的少女..."
    }
]
```

---

## 🔧 Configuration Options

Edit `config.json` to customize behavior:

| Option | Default | Description |
|--------|---------|-------------|
| `count_threshold` | `2` | Minimum occurrence count to include entity |
| `score_threshold` | `0.60` | NER confidence threshold (0.0-1.0) |
| `max_display_length` | `32` | Maximum entity display length |
| `max_context_samples` | `5` | Number of context paragraphs to sample |
| `tokens_per_sample` | `512` | Max tokens per context sample |
| `ner_target_types` | `["PER", "LOC"]` | Entity types to extract |
| `request_timeout` | `1800` | API request timeout (seconds) |
| `request_frequency_threshold` | `5` | Max requests per second |
| `traditional_chinese_enable` | `false` | Output Traditional Chinese |

---

## 🏗️ Project Structure

```
BookTermGacha/
├── app.py                  # Main entry point
├── config.json             # Configuration file
├── requirements.txt        # Python dependencies
├── version.txt             # Version info (v0.13.1-Refactor)
│
├── model/                  # Core models
│   ├── LLM.py             # LLM agent with retry logic, validation
│   ├── NER.py             # BERT-based NER extraction
│   └── Word.py            # Word data structure
│
├── module/                 # Utility modules
│   ├── FileManager.py     # File I/O handling
│   ├── LogHelper.py       # Logging utilities (Rich-based)
│   ├── ResultChecker.py   # Quality validation & reporting
│   ├── RubyCleaner.py     # Ruby/furigana annotation removal
│   ├── Normalizer.py      # Text normalization
│   ├── File/              # Format-specific readers
│   │   ├── EPUB.py        # EPUB reader
│   │   ├── TXT.py         # TXT reader
│   │   └── MD.py          # Markdown reader
│   └── Text/              # Text processing utilities
│       ├── TextHelper.py  # Character detection, manipulation
│       └── TextBase.py    # Base text utilities
│
├── prompt/                 # LLM prompts (customizable)
│   ├── prompt_context_translate.txt
│   ├── prompt_surface_analysis_with_context.txt
│   ├── prompt_surface_analysis_with_translation.txt
│   └── prompt_surface_analysis_without_translation.txt
│
├── blacklist/             # Filter lists
│   ├── jp_语气助词.json   # Japanese particles blacklist
│   ├── jp_人称代词.json   # Japanese pronouns blacklist
│   ├── jp_亲属关系.json   # Japanese family terms blacklist
│   └── custom.json        # Custom blacklist (add your own)
│
├── resource/              # Resources
│   ├── kg_ner_bf16/       # BERT NER model (required)
│   └── llm_config/        # LLM configuration presets
│
├── input/                 # Place your books here
├── output/                # Generated terminology tables
└── docs/                  # Documentation
    └── IMPROVEMENT_ANALYSIS.md  # Technical analysis
```

---

## 🔄 Comparison with Original KeywordGacha

### Architecture Comparison

| Aspect | KG v0.13.1 (Original) | KG v0.20.2 (New) | BookTerm Gacha (This) |
|--------|----------------------|------------------|----------------------|
| **UI** | CLI | GUI (PyQt) | CLI (Rich) |
| **NER** | BERT | Native LLM | BERT + Smart Filter |
| **Focus** | General | General | Books only |
| **Workflow** | 2-stage | AI-native | 4-stage Agent |
| **Validation** | None | Basic | Comprehensive |
| **Kana Handling** | Basic | Basic | Strict + Tolerance |
| **Fallback** | None | None | Forced Transliteration |

### What We Borrowed from KG v0.13.1

- ✅ BERT NER model and tokenization pipeline
- ✅ Basic workflow structure (NER → Context → Analysis)
- ✅ File format readers (EPUB, TXT, MD)
- ✅ Blacklist filtering system

### What We Borrowed from LinguaGacha (Dev-Experimental)

- ✅ `ResponseChecker` patterns (degradation detection)
- ✅ `TextHelper` precise character set definitions
- ✅ `KanaFixer` onomatopoeia handling logic
- ✅ Kana tolerance ratio concept (10%)

### Our Innovations

1. **Smart NER Filtering**: Only keep kana-containing entities (pure kanji filtered)
2. **Staged Retry with Fallback**: Guaranteed Chinese output via forced transliteration
3. **Place Name Particle Handling**: ヶ, の, etc. treated as "之"
4. **Prompt Engineering**: Guide LLM to handle particles and edge cases
5. **Rich Progress Display**: Clear visibility into agent operations
6. **Comprehensive Validation**: Detect and report all quality issues
7. **Result Checker**: Post-processing quality assurance

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest test_improvements.py -v

# Test specific functionality
python -m pytest test_improvements.py::test_contains_kana_strict -v
python -m pytest test_improvements.py::test_result_checker -v
```

### Test Coverage

- ✅ `test_contains_kana_strict` - Kana detection with tolerance
- ✅ `test_is_degraded` - Degradation detection (repeated characters)
- ✅ `test_check_similarity` - Jaccard similarity checking
- ✅ `test_force_transliterate` - Forced transliteration fallback
- ✅ `test_verify_kana_only` - NER filtering logic
- ✅ `test_result_checker` - Result validation module
- ✅ `test_blacklist_particles` - Particle blacklist filtering

---

## 🚀 Building for Release

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pyinstaller pytest

# Run tests
python -m pytest test_improvements.py -v

# Check syntax
python -m py_compile app.py model/LLM.py model/NER.py
```

### Creating Executable

```bash
# Build with PyInstaller
pyinstaller --onefile --name BookTermGacha app.py

# The executable will be in dist/BookTermGacha.exe
```

### Release Package Structure

```
BookTermGacha-v0.13.1-Refactor/
├── BookTermGacha.exe      # Main executable (or app.py for source)
├── config.json            # Configuration (user edits this)
├── requirements.txt       # For source installations
├── README.md              # This file
│
├── prompt/                # LLM prompts
│   └── *.txt
│
├── blacklist/             # Filter lists
│   └── *.json
│
├── resource/
│   └── kg_ner_bf16/       # BERT model (REQUIRED - ~500MB)
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── ...
│
├── input/                 # User places books here
│   └── (empty)
│
└── output/                # Results appear here
    └── (empty)
```

### GPU Support

The NER stage automatically detects CUDA:
- **With GPU**: Uses bf16 precision for fast inference
- **Without GPU**: Falls back to CPU (slower but works)

No configuration needed - it's automatic!

---

## 📋 Changelog

### v0.13.1-Refactor (Current)

**New Features:**
- Smart kana detection with onomatopoeia tolerance
- Place name particle handling (ヶ, ケ, ヵ, の)
- Forced transliteration fallback (romaji → Chinese)
- Rich progress bars for all stages
- Comprehensive ResultChecker module
- Particle handling in prompts

**Improvements:**
- Reduced max retry: 32 → 8
- Earlier forced transliteration: after 5 retries
- Simplified logging output
- Better error messages

**Based on:**
- KeywordGacha v0.13.1 (core workflow)
- LinguaGacha Dev-Experimental (validation patterns)

---

## 🙏 Acknowledgments

- [KeywordGacha](https://github.com/neavo/KeywordGacha) by neavo - Original project and inspiration
- [LinguaGacha](https://github.com/neavo/LinguaGacha) by neavo - Design patterns and utilities
- [Zhipu AI / BigModel](https://bigmodel.cn/) - Free LLM API for development and testing

---

## 📄 License

This project is based on KeywordGacha and follows the same licensing terms.

**Important**: If you use this tool in your translation work, please credit appropriately.

---

## 🤝 Contributing

This project serves as an **LLM Agent development learning exercise**. Contributions are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `python -m pytest test_improvements.py -v`
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Areas for Contribution

- 🌐 Multi-language support (Korean, English sources)
- 📊 Better progress visualization
- 🔧 Additional output formats
- 📝 Documentation improvements
- 🧪 More test cases

---

## ❓ FAQ

**Q: Why focus only on books?**
A: Games and subtitles have different terminology patterns. Books (especially light novels) have consistent character/location naming that benefits most from terminology tables.

**Q: Why filter out pure kanji entities?**
A: Pure kanji entities (田中, 東京) don't need terminology tables - they can be preserved as-is or trivially converted. The real challenge is kana (アリス, トリシューラ) which need proper transliteration.

**Q: Why use BERT + LLM instead of pure LLM?**
A: BERT NER is faster and more reliable for entity extraction. LLM is better for semantic analysis and translation. Combining both gives the best results.

**Q: Can I use other LLM providers?**
A: Yes! Any OpenAI-compatible API works. Just update `config.json` with your provider's URL and API key.

---

<p align="center">
  <strong>Built with ❤️ as an LLM Agent Development Exercise</strong>
  <br>
  <em>Transforming Japanese Literatures into Translation-Ready Glossaries</em>
</p>
