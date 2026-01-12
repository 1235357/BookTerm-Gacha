<h1><p align="center">📚 BookTerm Gacha</p></h1>
<p align="center"><strong>An LLM-Powered Agent for Automated Book Terminology Extraction</strong></p>
<p align="center"><em>Specifically Optimized for Zhipu GLM API - Extract Character & Location Names from Japanese Literatures</em></p>

<p align="center">
  <img src="https://img.shields.io/badge/Version-0.1.0-brightgreen.svg" alt="Version"/>
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
  <img src="https://img.shields.io/badge/LLM-Zhipu_GLM-orange.svg" alt="Zhipu GLM"/>
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
| 🆓 **Free LLM API** | Uses Zhipu GLM-4-Flash API which offers FREE tier with generous limits |
| 🚀 **GPU Acceleration** | Automatic CUDA detection for fast NER processing (optional) |
| 🔄 **Smart Retry** | Intelligent retry logic with forced transliteration fallback |
| 📊 **Rich Progress** | Beautiful progress bars showing exactly what's happening |
| ✅ **Quality Validation** | Automatic result checking for kana residue and issues |
| 📝 **Multiple Formats** | Outputs in JSON dictionary, LinguaGacha, and GalTransl formats |

---

## 🚀 Quick Start Guide

**For users who just want to get started quickly:**

1. Download the latest release from [GitHub Releases](https://github.com/1235357/BookTerm-Gacha/releases)
2. Extract the ZIP file to any folder
3. Get a FREE API key from [Zhipu AI](https://bigmodel.cn/)
4. Edit `config.json` and add your API key
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

#### Step 3: Get Your FREE API Key

BookTerm Gacha uses **Zhipu AI's GLM-4-Flash** model, which offers a **FREE tier**:

1. **Go to** [https://bigmodel.cn/](https://bigmodel.cn/)
2. **Register** for a free account (you'll need a phone number for verification)
3. **Log in** and go to the console/dashboard
4. **Navigate to** API Keys section
5. **Create** a new API key and **copy** it

> 💡 **Tip**: The GLM-4-Flash model is completely FREE with generous rate limits (typically 1000+ requests/day). Perfect for processing entire book series!

#### Step 4: Configure Your API Key

1. Open `config.json` with any text editor (Notepad, VS Code, etc.)
2. Find the `api_key` line and replace `YOUR_API_KEY_HERE` with your actual key:

```json
{
    "api_key": ["your-actual-api-key-here", "API密钥"],
    "base_url": ["https://open.bigmodel.cn/api/paas/v4", "API地址"],
    "model_name": ["glm-4-flash", "模型名称"]
}
```

3. **Save** the file

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

Remember to clone https://huggingface.co/neavo/keyword_gacha_multilingual_ner to "\BookTerm Gacha\resource\kg_ner_bf16"

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

### Essential Settings (Must Configure)

```json
{
    "api_key": ["YOUR_API_KEY", "Your Zhipu AI API key"],
    "base_url": ["https://open.bigmodel.cn/api/paas/v4", "API endpoint"],
    "model_name": ["glm-4-flash", "Model to use"]
}
```

| Setting | Description | Default |
|---------|-------------|---------|
| `api_key` | Your Zhipu AI API key (required) | - |
| `base_url` | API endpoint URL | `https://open.bigmodel.cn/api/paas/v4` |
| `model_name` | LLM model name | `glm-4-flash` (free) |

### Optional Settings

```json
{
    "count_threshold": [2, "Minimum occurrences to include entity"],
    "score_threshold": [0.60, "NER confidence threshold (0.0-1.0)"],
    "max_context_samples": [5, "Context paragraphs to sample"],
    "tokens_per_sample": [512, "Max tokens per context"],
    "ner_target_types": [["PER", "LOC"], "Entity types to extract"],
    "request_timeout": [1800, "API timeout in seconds"],
    "request_frequency_threshold": [5, "Max requests per second"],
    "traditional_chinese_enable": [false, "Output Traditional Chinese"]
}
```

| Setting | Description | Default | Recommended |
|---------|-------------|---------|-------------|
| `count_threshold` | Minimum times a name must appear | `2` | Keep default |
| `score_threshold` | How confident NER must be (0-1) | `0.60` | `0.50-0.70` |
| `max_context_samples` | Context paragraphs for LLM | `5` | `3-7` |
| `request_frequency_threshold` | Rate limit (requests/sec) | `5` | Lower if hitting limits |
| `traditional_chinese_enable` | Use Traditional Chinese | `false` | `true` for TW/HK |

### Using Different LLM Providers

While optimized for Zhipu GLM, you can use other OpenAI-compatible APIs:

| Provider | Base URL | Model | Notes |
|----------|----------|-------|-------|
| **Zhipu AI** | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` | FREE tier available |
| **DeepSeek** | `https://api.deepseek.com/v1` | `deepseek-chat` | Very affordable |
| **OpenAI** | `https://api.openai.com/v1` | `gpt-4o-mini` | Best quality, higher cost |
| **Local LLM** | `http://localhost:11434/v1` | varies | Via Ollama, etc. |

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

**Q: Is the Zhipu API really free?**
A: Yes! The GLM-4-Flash model has a free tier with generous limits. Perfect for personal use.

**Q: Can I use this for Korean/English books?**
A: Currently optimized for Japanese → Chinese. Other languages may work but are untested.

**Q: How long does processing take?**
A: Depends on book size. A typical Japanese Literatures (100k characters) takes 10-25 minutes.

**Q: Can I use my own LLM?**
A: Yes! Any OpenAI-compatible API works. Just update the config.

**Q: Why are some names not extracted?**
A: The NER model focuses on names with kana. Pure kanji names (like 田中) are skipped as they don't need transliteration.

---

<p align="center">
  <strong>📚 BookTerm Gacha v0.1.0</strong>
  <br>
  <em>Transforming Japanese Literatures into Translation-Ready Glossaries</em>
  <br><br>
  Made for the Japanese Literatures Translation Community
</p>
