# -*- mode: python ; coding: utf-8 -*-
"""
BookTerm Gacha - PyInstaller Spec File
======================================

打包配置文件，模仿 KeywordGacha - Stable 的结构

打包命令：
    pyinstaller BookTermGacha.spec --noconfirm

输出结构：
    dist/BookTermGacha/
    ├── app.exe                 # 主程序
    ├── config.json             # 配置文件（需手动复制）
    ├── version.txt             # 版本文件（需手动复制）
    ├── blacklist/              # 黑名单（需手动复制）
    ├── input/                  # 输入目录（需手动复制）
    ├── output/                 # 输出目录（需手动复制）
    ├── prompt/                 # Prompt 目录（需手动复制）
    ├── resource/               # 资源目录（需手动复制）
    └── _internal/              # Python 依赖
"""

import os
import sys
from pathlib import Path

# 获取 site-packages 路径
def get_site_packages():
    """获取当前 Python 环境的 site-packages 路径"""
    import site
    paths = site.getsitepackages()
    # 找到包含 Lib/site-packages 的路径
    for p in paths:
        if 'site-packages' in p:
            return p
    return paths[0]

site_packages = get_site_packages()
print(f"Site-packages path: {site_packages}")

# ============== 分析配置 ==============
a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        # pykakasi 数据文件（假名转换必需）
        (os.path.join(site_packages, 'pykakasi'), 'pykakasi'),
        # pecab 数据文件（韩文处理必需）
        (os.path.join(site_packages, 'pecab'), 'pecab'),
        # sudachidict_core（日文分词必需）
        (os.path.join(site_packages, 'sudachidict_core'), 'sudachidict_core'),
        # tiktoken 数据文件
        (os.path.join(site_packages, 'tiktoken'), 'tiktoken'),
        # transformers 配置（NER 模型必需）
        (os.path.join(site_packages, 'transformers'), 'transformers'),
    ],
    hiddenimports=[
        # PyTorch 相关
        'torch',
        'torch.cuda',
        'torch.backends',
        'torch.backends.cudnn',
        'torchvision',
        'torchaudio',
        
        # Transformers 相关
        'transformers',
        'transformers.models',
        'transformers.models.bert',
        'transformers.models.bert.modeling_bert',
        'transformers.models.bert.tokenization_bert',
        'transformers.pipelines',
        'transformers.pipelines.token_classification',
        
        # OpenAI / API 相关
        'openai',
        'openai.resources',
        'openai.resources.chat',
        'openai.resources.chat.completions',
        'aiolimiter',
        'httpx',
        'httpcore',
        'anyio',
        'sniffio',
        'h11',
        'certifi',
        'httpx._transports',
        'httpx._transports.default',
        
        # 文本处理
        'tiktoken',
        'tiktoken_ext',
        'tiktoken_ext.openai_public',
        'pykakasi',
        'pykakasi.kakasi',
        'sudachipy',
        'sudachidict_core',
        'pecab',
        'opencc',
        
        # 文件格式
        'ebooklib',
        'ebooklib.epub',
        'openpyxl',
        'lxml',
        'lxml.etree',
        'lxml.html',
        'bs4',
        'bs4.builder',
        'bs4.builder._lxml',
        
        # Rich CLI
        'rich',
        'rich.console',
        'rich.table',
        'rich.progress',
        'rich.panel',
        'rich.live',
        'rich.logging',
        
        # 日志
        'loguru',
        'loguru._logger',
        
        # JSON 修复
        'json_repair',
        
        # 异步
        'asyncio',
        'concurrent.futures',
        
        # 标准库
        'typing_extensions',
        'packaging',
        'packaging.version',
        'packaging.specifiers',
        'packaging.requirements',
        'filelock',
        'tqdm',
        'yaml',
        'PIL',
        'numpy',
        'regex',
        'safetensors',
        'huggingface_hub',
        'requests',
        'urllib3',
        'charset_normalizer',
        'idna',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除 Qt 相关（避免冲突）
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'tkinter',
        '_tkinter',
        
        # 排除不需要的大型包
        'matplotlib',
        'scipy',
        'pandas',
        'sklearn',
        'tensorflow',
        'keras',
        'cv2',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'black',
        'flake8',
        'mypy',
        
        # 排除其他不需要的
        'zmq',
        'psutil',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# ============== PYZ 压缩 ==============
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=None,
)

# ============== EXE 配置 ==============
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='app',  # 直接命名为 app.exe
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # 不使用 UPX 压缩（避免问题）
    console=True,  # 控制台应用
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 可以添加图标：icon='resource/icon.ico'
)

# ============== COLLECT 收集所有文件 ==============
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='BookTermGacha',
)
