"""
BookTerm Gacha - 打包后处理脚本
================================

在 PyInstaller 打包完成后运行此脚本，复制必要的资源文件到 dist 目录。

用法：
    python build_finish.py
"""

import os
import shutil
from pathlib import Path

def main():
    # 源目录和目标目录
    src_dir = Path(".")
    dist_dir = Path("dist/BookTermGacha")
    
    if not dist_dir.exists():
        print("错误：dist/BookTermGacha 目录不存在！请先运行 pyinstaller 打包。")
        return False
    
    # 需要复制的文件和目录
    items_to_copy = [
        # 配置文件
        ("config.json", "config.json"),
        ("version.txt", "version.txt"),
        
        # 目录
        ("blacklist", "blacklist"),
        ("prompt", "prompt"),
        ("resource/kg_ner_bf16", "resource/kg_ner_bf16"),
        ("resource/llm_config", "resource/llm_config"),
    ]
    
    # 需要创建的空目录
    dirs_to_create = [
        "input",
        "output",
        "log",
    ]
    
    print("=" * 60)
    print("BookTerm Gacha - 打包后处理")
    print("=" * 60)
    
    # 复制文件和目录
    for src_name, dst_name in items_to_copy:
        src_path = src_dir / src_name
        dst_path = dist_dir / dst_name
        
        if src_path.exists():
            # 确保目标父目录存在
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            if src_path.is_dir():
                if dst_path.exists():
                    shutil.rmtree(dst_path)
                shutil.copytree(src_path, dst_path)
                print(f"✓ 复制目录: {src_name} -> {dst_name}")
            else:
                shutil.copy2(src_path, dst_path)
                print(f"✓ 复制文件: {src_name} -> {dst_name}")
        else:
            print(f"⚠ 跳过（不存在）: {src_name}")
    
    # 创建空目录
    for dir_name in dirs_to_create:
        dir_path = dist_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {dir_name}")
    
    # 重命名 _internal 目录（如果需要）
    internal_dir = dist_dir / "_internal"
    if not internal_dir.exists():
        # PyInstaller 可能创建了其他名称的内部目录
        for item in dist_dir.iterdir():
            if item.is_dir() and item.name not in ["blacklist", "prompt", "resource", "input", "output", "log"]:
                if item.name != "_internal":
                    # 这可能是内部依赖目录
                    pass
    
    print("=" * 60)
    print("打包后处理完成！")
    print(f"输出目录: {dist_dir.absolute()}")
    print("=" * 60)
    
    # 检查必要文件
    required_files = ["app.exe", "config.json", "version.txt"]
    missing = [f for f in required_files if not (dist_dir / f).exists()]
    
    if missing:
        print(f"\n⚠ 警告：缺少必要文件: {', '.join(missing)}")
    else:
        print("\n✓ 所有必要文件已就位，可以分发！")
    
    return True


if __name__ == "__main__":
    main()
