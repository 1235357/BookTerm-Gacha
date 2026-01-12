"""
BookTerm Gacha - Environment Checker Module
============================================

This module provides automatic environment detection, validation, and repair
for the BookTerm Gacha application. It ensures all required dependencies
are installed with proper CUDA/GPU support.

Features:
    - Python version validation
    - Package dependency checking
    - CUDA/GPU detection and PyTorch CUDA support verification
    - Automatic package installation with fallback mirrors
    - Smart PyTorch reinstallation for CUDA support

Usage:
    from module.EnvChecker import EnvChecker
    
    checker = EnvChecker()
    if not checker.check_and_repair():
        sys.exit(1)

Based on KeywordGacha v0.13.1 by neavo
https://github.com/neavo/KeywordGacha
"""

import os
import sys
import subprocess
import importlib
import platform
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field

# Rich 可能还没安装，所以先尝试导入，失败则用简单打印
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    RICH_AVAILABLE = True
    _console = Console()
except ImportError:
    RICH_AVAILABLE = False
    _console = None


@dataclass
class PackageInfo:
    """包信息"""
    name: str                      # pip 包名
    import_name: str = ""          # import 时的模块名（如果不同）
    required: bool = True          # 是否必需
    min_version: str = ""          # 最低版本要求
    
    def __post_init__(self):
        if not self.import_name:
            self.import_name = self.name.replace("-", "_")


class EnvChecker:
    """环境检查和自动修复器"""
    
    # Python 最低版本要求
    MIN_PYTHON_VERSION = (3, 10)
    
    # 国内镜像源列表（按优先级排序）
    MIRROR_SOURCES = [
        ("清华大学", "https://pypi.tuna.tsinghua.edu.cn/simple"),
        ("阿里云", "https://mirrors.aliyun.com/pypi/simple"),
        ("华为云", "https://repo.huaweicloud.com/repository/pypi/simple"),
        ("豆瓣", "https://pypi.douban.com/simple"),
    ]
    
    # PyTorch CUDA 镜像源
    PYTORCH_CUDA_SOURCES = [
        ("清华镜像", "https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels"),
        ("官方源", "https://download.pytorch.org/whl"),
    ]
    
    # 核心依赖包列表
    CORE_PACKAGES: List[PackageInfo] = [
        # Rich 必须最先安装（用于美化输出）
        PackageInfo("rich", "rich", True),
        PackageInfo("loguru", "loguru", True),
        
        # Transformers / NER
        PackageInfo("transformers", "transformers", True),
        
        # LLM API
        PackageInfo("openai", "openai", True),
        PackageInfo("aiolimiter", "aiolimiter", True),
        
        # 文本处理
        PackageInfo("tiktoken", "tiktoken", True),
        PackageInfo("pykakasi", "pykakasi", True),
        PackageInfo("sudachipy", "sudachipy", True),
        PackageInfo("sudachidict-core", "sudachidict_core", True),
        PackageInfo("pecab", "pecab", False),  # 韩语，非必需
        PackageInfo("opencc-python-reimplemented", "opencc", True),
        
        # 文件格式
        PackageInfo("ebooklib", "ebooklib", True),
        PackageInfo("openpyxl", "openpyxl", True),
        PackageInfo("lxml", "lxml", True),
        PackageInfo("beautifulsoup4", "bs4", True),
        
        # 工具
        PackageInfo("json-repair", "json_repair", True),
    ]
    
    def __init__(self):
        self.issues: List[str] = []
        self.fixes_applied: List[str] = []
        self.cuda_version: Optional[str] = None
        self.gpu_name: Optional[str] = None
        self.pytorch_cuda_available: bool = False
        self.working_mirror: Optional[str] = None
    
    # ==================== 日志输出 ====================
    
    def _print(self, message: str, style: str = "") -> None:
        """打印消息（兼容 Rich 不可用的情况）"""
        if RICH_AVAILABLE and _console:
            if style:
                _console.print(f"[{style}]{message}[/{style}]")
            else:
                _console.print(message)
        else:
            # 移除 Rich 标记
            import re
            clean_msg = re.sub(r'\[/?[^\]]+\]', '', message)
            print(clean_msg)
    
    def _print_header(self, title: str) -> None:
        """打印标题"""
        if RICH_AVAILABLE and _console:
            _console.print()
            _console.rule(f"[bold cyan]{title}[/bold cyan]", style="cyan")
            _console.print()
        else:
            print(f"\n{'='*60}")
            print(f"  {title}")
            print(f"{'='*60}\n")
    
    def _print_status(self, item: str, status: str, ok: bool) -> None:
        """打印状态行"""
        if ok:
            icon = "✓"
            color = "green"
        else:
            icon = "✗"
            color = "red"
        
        if RICH_AVAILABLE and _console:
            _console.print(f"  [{color}]{icon}[/{color}] {item}: [{color}]{status}[/{color}]")
        else:
            print(f"  {icon} {item}: {status}")
    
    def _print_info(self, message: str) -> None:
        """打印信息"""
        if RICH_AVAILABLE and _console:
            _console.print(f"  [dim]ℹ[/dim] {message}")
        else:
            print(f"  ℹ {message}")
    
    def _print_warning(self, message: str) -> None:
        """打印警告"""
        if RICH_AVAILABLE and _console:
            _console.print(f"  [yellow]⚠[/yellow] {message}")
        else:
            print(f"  ⚠ {message}")
    
    def _print_error(self, message: str) -> None:
        """打印错误"""
        if RICH_AVAILABLE and _console:
            _console.print(f"  [red]✗[/red] {message}")
        else:
            print(f"  ✗ {message}")
    
    def _print_success(self, message: str) -> None:
        """打印成功"""
        if RICH_AVAILABLE and _console:
            _console.print(f"  [green]✓[/green] {message}")
        else:
            print(f"  ✓ {message}")
    
    # ==================== 环境检测 ====================
    
    def check_python_version(self) -> bool:
        """检查 Python 版本"""
        current = sys.version_info[:2]
        required = self.MIN_PYTHON_VERSION
        
        ok = current >= required
        status = f"{current[0]}.{current[1]}" + (f" (需要 >= {required[0]}.{required[1]})" if not ok else "")
        self._print_status("Python 版本", status, ok)
        
        if not ok:
            self.issues.append(f"Python 版本过低: {current[0]}.{current[1]}，需要 >= {required[0]}.{required[1]}")
        
        return ok
    
    def check_cuda_environment(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        检测 CUDA 环境
        
        Returns:
            (cuda_available, cuda_version, gpu_name)
        """
        cuda_available = False
        cuda_version = None
        gpu_name = None
        
        # 方法1: 尝试运行 nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(",")
                if len(parts) >= 1:
                    gpu_name = parts[0].strip()
                cuda_available = True
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass
        
        # 方法2: 获取 CUDA 版本
        if cuda_available:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
                )
                if result.returncode == 0:
                    # 从驱动版本推断 CUDA 版本
                    driver_version = result.stdout.strip()
                    cuda_version = self._driver_to_cuda_version(driver_version)
            except Exception:
                pass
        
        # 存储结果
        self.cuda_version = cuda_version
        self.gpu_name = gpu_name
        
        # 打印状态
        if cuda_available:
            self._print_status("NVIDIA GPU", f"{gpu_name}", True)
            self._print_status("CUDA 支持", f"驱动支持 CUDA {cuda_version or '未知版本'}", True)
        else:
            self._print_status("NVIDIA GPU", "未检测到", False)
            self._print_warning("╔════════════════════════════════════════════════════════════╗")
            self._print_warning("║  未检测到 NVIDIA GPU，程序将使用 CPU 模式运行               ║")
            self._print_warning("║  NER 实体识别速度会显著降低（约 10-50 倍）                  ║")
            self._print_warning("║                                                            ║")
            self._print_warning("║  如果您有 NVIDIA 显卡，请检查：                            ║")
            self._print_warning("║  1. 是否已安装 NVIDIA 显卡驱动                             ║")
            self._print_warning("║  2. 驱动版本是否过旧（建议 >= 470）                        ║")
            self._print_warning("║  3. 下载驱动: https://www.nvidia.cn/drivers/               ║")
            self._print_warning("╚════════════════════════════════════════════════════════════╝")
        
        return cuda_available, cuda_version, gpu_name
    
    def _driver_to_cuda_version(self, driver_version: str) -> Optional[str]:
        """根据驱动版本推断支持的 CUDA 版本"""
        try:
            major = int(driver_version.split(".")[0])
            # NVIDIA 驱动版本与 CUDA 版本对应关系（近似）
            if major >= 560:
                return "12.6"
            elif major >= 550:
                return "12.4"
            elif major >= 530:
                return "12.1"
            elif major >= 520:
                return "11.8"
            elif major >= 510:
                return "11.6"
            elif major >= 470:
                return "11.4"
            else:
                return "11.0"
        except Exception:
            return None
    
    def check_pytorch(self) -> Tuple[bool, bool]:
        """
        检查 PyTorch 安装状态
        
        Returns:
            (installed, cuda_enabled)
        """
        installed = False
        cuda_enabled = False
        
        try:
            import torch
            installed = True
            cuda_enabled = torch.cuda.is_available()
            
            version = torch.__version__
            if cuda_enabled:
                device_name = torch.cuda.get_device_name(0)
                self._print_status("PyTorch", f"{version} (CUDA 已启用)", True)
                self._print_info(f"GPU 设备: {device_name}")
                self.pytorch_cuda_available = True
            else:
                self._print_status("PyTorch", f"{version} (仅 CPU)", False)
                if self.cuda_version:
                    self._print_warning("检测到 GPU 但 PyTorch 未启用 CUDA，建议重新安装")
        except ImportError:
            self._print_status("PyTorch", "未安装", False)
            self.issues.append("PyTorch 未安装")
        
        return installed, cuda_enabled
    
    def check_package(self, pkg: PackageInfo) -> bool:
        """检查单个包是否已安装"""
        try:
            importlib.import_module(pkg.import_name)
            return True
        except ImportError:
            return False
    
    def check_all_packages(self) -> Dict[str, bool]:
        """检查所有依赖包"""
        results = {}
        missing_required = []
        missing_optional = []
        
        for pkg in self.CORE_PACKAGES:
            installed = self.check_package(pkg)
            results[pkg.name] = installed
            
            if not installed:
                if pkg.required:
                    missing_required.append(pkg.name)
                else:
                    missing_optional.append(pkg.name)
        
        # 打印摘要
        total = len(self.CORE_PACKAGES)
        installed_count = sum(1 for v in results.values() if v)
        
        if installed_count == total:
            self._print_status("依赖包", f"全部已安装 ({installed_count}/{total})", True)
        else:
            self._print_status("依赖包", f"已安装 {installed_count}/{total}", False)
            if missing_required:
                self._print_warning(f"缺少必需包: {', '.join(missing_required)}")
                self.issues.append(f"缺少必需包: {', '.join(missing_required)}")
            if missing_optional:
                self._print_info(f"缺少可选包: {', '.join(missing_optional)}")
        
        return results
    
    # ==================== 自动修复 ====================
    
    def _find_working_mirror(self) -> Optional[str]:
        """测试并找到可用的镜像源"""
        if self.working_mirror:
            return self.working_mirror
        
        self._print_info("正在测试镜像源连接...")
        
        for name, url in self.MIRROR_SOURCES:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--dry-run", "-i", url, "pip"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
                )
                if result.returncode == 0 or "already satisfied" in result.stdout.lower():
                    self._print_success(f"使用镜像源: {name} ({url})")
                    self.working_mirror = url
                    return url
            except Exception:
                continue
        
        self._print_warning("所有镜像源均不可用，尝试使用默认源")
        return None
    
    def _pip_install(self, packages: List[str], extra_args: List[str] = None) -> bool:
        """使用 pip 安装包"""
        if not packages:
            return True
        
        mirror = self._find_working_mirror()
        
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
        if mirror:
            cmd.extend(["-i", mirror, "--trusted-host", mirror.split("//")[1].split("/")[0]])
        if extra_args:
            cmd.extend(extra_args)
        cmd.extend(packages)
        
        self._print_info(f"正在安装: {', '.join(packages)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10分钟超时
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
            )
            
            if result.returncode == 0:
                self._print_success(f"安装成功: {', '.join(packages)}")
                return True
            else:
                self._print_error(f"安装失败: {result.stderr[:200] if result.stderr else '未知错误'}")
                return False
        except subprocess.TimeoutExpired:
            self._print_error("安装超时，请检查网络连接")
            return False
        except Exception as e:
            self._print_error(f"安装出错: {e}")
            return False
    
    def install_pytorch_cuda(self) -> bool:
        """安装支持 CUDA 的 PyTorch"""
        if not self.cuda_version:
            self._print_warning("未检测到 CUDA，将安装 CPU 版本 PyTorch")
            return self._pip_install(["torch", "torchvision", "torchaudio"])
        
        # 确定 CUDA 版本对应的 PyTorch wheel URL
        cuda_map = {
            "12.6": "cu126",
            "12.4": "cu124", 
            "12.1": "cu121",
            "11.8": "cu118",
        }
        
        # 找到最接近的 CUDA 版本
        cuda_tag = None
        try:
            cuda_major_minor = ".".join(self.cuda_version.split(".")[:2])
            if cuda_major_minor in cuda_map:
                cuda_tag = cuda_map[cuda_major_minor]
            else:
                # 尝试找到最接近的版本
                cuda_float = float(cuda_major_minor)
                for ver, tag in sorted(cuda_map.items(), key=lambda x: float(x[0]), reverse=True):
                    if cuda_float >= float(ver):
                        cuda_tag = tag
                        break
        except Exception:
            cuda_tag = "cu121"  # 默认使用 CUDA 12.1
        
        if not cuda_tag:
            cuda_tag = "cu121"
        
        self._print_info(f"将安装 PyTorch with CUDA {cuda_tag}")
        
        # 先卸载现有的 PyTorch
        self._print_info("正在卸载现有 PyTorch...")
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        )
        
        # 尝试从不同源安装
        for source_name, source_url in self.PYTORCH_CUDA_SOURCES:
            self._print_info(f"尝试从 {source_name} 安装 PyTorch CUDA...")
            
            index_url = f"{source_url}/{cuda_tag}"
            
            cmd = [
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", index_url
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1200,  # 20分钟超时（PyTorch 很大）
                    creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
                )
                
                if result.returncode == 0:
                    # 验证安装
                    try:
                        # 强制重新加载 torch
                        if "torch" in sys.modules:
                            del sys.modules["torch"]
                        import torch
                        if torch.cuda.is_available():
                            self._print_success(f"PyTorch CUDA 安装成功！GPU: {torch.cuda.get_device_name(0)}")
                            self.pytorch_cuda_available = True
                            return True
                        else:
                            self._print_warning("PyTorch 已安装但 CUDA 不可用")
                    except Exception as e:
                        self._print_warning(f"验证失败: {e}")
                else:
                    self._print_warning(f"从 {source_name} 安装失败，尝试下一个源...")
            except subprocess.TimeoutExpired:
                self._print_warning(f"从 {source_name} 安装超时，尝试下一个源...")
            except Exception as e:
                self._print_warning(f"安装出错: {e}")
        
        # 所有源都失败，安装 CPU 版本作为后备
        self._print_warning("无法安装 CUDA 版本，将安装 CPU 版本")
        return self._pip_install(["torch", "torchvision", "torchaudio"])
    
    def install_missing_packages(self, missing: List[str]) -> bool:
        """安装缺失的包"""
        if not missing:
            return True
        
        # 排除 PyTorch 相关包（单独处理）
        pytorch_packages = {"torch", "torchvision", "torchaudio"}
        other_packages = [p for p in missing if p not in pytorch_packages]
        
        success = True
        
        # 安装其他包
        if other_packages:
            if not self._pip_install(other_packages):
                success = False
        
        return success
    
    # ==================== 主入口 ====================
    
    def check_and_repair(self, auto_repair: bool = True) -> bool:
        """
        检查环境并自动修复
        
        Args:
            auto_repair: 是否自动修复问题
            
        Returns:
            环境是否就绪
        """
        self._print_header("🔍 环境检测")
        
        # 1. 检查 Python 版本
        python_ok = self.check_python_version()
        if not python_ok:
            self._print_error("Python 版本过低，请升级到 3.10 或更高版本")
            return False
        
        # 2. 检查 CUDA 环境
        cuda_available, cuda_version, gpu_name = self.check_cuda_environment()
        
        # 3. 检查 PyTorch
        pytorch_installed, pytorch_cuda = self.check_pytorch()
        
        # 4. 检查其他依赖包
        package_status = self.check_all_packages()
        
        # 收集需要修复的问题
        need_pytorch_reinstall = cuda_available and pytorch_installed and not pytorch_cuda
        need_pytorch_install = not pytorch_installed
        missing_packages = [pkg.name for pkg in self.CORE_PACKAGES if not package_status.get(pkg.name, False)]
        
        # 如果一切正常
        if not need_pytorch_reinstall and not need_pytorch_install and not missing_packages:
            self._print_header("✅ 环境检测完成")
            self._print_success("所有依赖已就绪，环境正常！")
            if cuda_available and pytorch_cuda:
                self._print_success(f"GPU 加速已启用: {gpu_name}")
            return True
        
        # 需要修复
        if not auto_repair:
            self._print_header("⚠️ 环境问题")
            if need_pytorch_install:
                self._print_error("PyTorch 未安装")
            if need_pytorch_reinstall:
                self._print_warning("PyTorch 未启用 CUDA，建议重新安装")
            if missing_packages:
                self._print_error(f"缺少依赖包: {', '.join(missing_packages)}")
            return False
        
        # 自动修复
        self._print_header("🔧 自动修复")
        
        # 修复 PyTorch
        if need_pytorch_install or need_pytorch_reinstall:
            self._print_info("正在修复 PyTorch 安装...")
            if not self.install_pytorch_cuda():
                self._print_warning("PyTorch 安装可能不完整，但程序仍可运行（使用 CPU）")
        
        # 修复其他包
        if missing_packages:
            self._print_info("正在安装缺失的依赖包...")
            if not self.install_missing_packages(missing_packages):
                self._print_error("部分依赖包安装失败")
                return False
        
        # 最终验证
        self._print_header("🔄 重新验证")
        
        # 重新检查 PyTorch
        pytorch_installed, pytorch_cuda = self.check_pytorch()
        
        # 重新检查包
        final_missing = []
        for pkg in self.CORE_PACKAGES:
            if pkg.required and not self.check_package(pkg):
                final_missing.append(pkg.name)
        
        if final_missing:
            self._print_status("依赖包", f"仍缺少: {', '.join(final_missing)}", False)
            return False
        
        self._print_header("✅ 环境修复完成")
        self._print_success("所有依赖已就绪！")
        
        if pytorch_cuda:
            self._print_success(f"GPU 加速已启用: {self.gpu_name or 'NVIDIA GPU'}")
        elif cuda_available:
            self._print_warning("GPU 可用但 PyTorch CUDA 未启用，将使用 CPU 模式")
        else:
            self._print_info("将使用 CPU 模式运行")
        
        return True
    
    def print_environment_summary(self) -> None:
        """打印环境摘要（用于启动时显示）"""
        if not RICH_AVAILABLE:
            return
        
        try:
            import torch
            pytorch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            device = torch.cuda.get_device_name(0) if cuda_available else "CPU"
        except ImportError:
            pytorch_version = "未安装"
            cuda_available = False
            device = "N/A"
        
        table = Table(
            box=box.ROUNDED,
            title="[bold]运行环境[/bold]",
            title_style="cyan",
            expand=False,
        )
        table.add_column("项目", style="dim")
        table.add_column("状态", justify="right")
        
        table.add_row("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        table.add_row("PyTorch", pytorch_version)
        table.add_row("CUDA", "[green]已启用[/green]" if cuda_available else "[yellow]未启用[/yellow]")
        table.add_row("设备", device)
        
        _console.print()
        _console.print(table)
        _console.print()


# 便捷函数
def check_environment(auto_repair: bool = True) -> bool:
    """
    检查并修复运行环境
    
    Args:
        auto_repair: 是否自动修复问题
        
    Returns:
        环境是否就绪
    """
    # 检测是否在 PyInstaller 打包环境中运行
    # 如果是，跳过环境检查（依赖已经打包）
    if getattr(sys, 'frozen', False):
        # 在打包环境中，简单打印启动信息
        print("=" * 60)
        print("  BookTerm Gacha - Starting...")
        print("=" * 60)
        return True
    
    checker = EnvChecker()
    return checker.check_and_repair(auto_repair)


if __name__ == "__main__":
    # 直接运行时进行环境检查
    check_environment(auto_repair=True)
