@echo off
chcp 65001 >nul
title BookTerm Gacha 打包工具

echo ============================================================
echo BookTerm Gacha 一键打包工具
echo ============================================================
echo.

:: 检查 PyInstaller
echo [1/4] 检查 PyInstaller...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo PyInstaller 未安装，正在安装...
    pip install pyinstaller
    if errorlevel 1 (
        echo 安装 PyInstaller 失败！
        pause
        exit /b 1
    )
)
echo ✓ PyInstaller 已就绪

:: 清理旧的打包文件
echo.
echo [2/4] 清理旧的打包文件...
if exist "dist\BookTermGacha" rmdir /s /q "dist\BookTermGacha"
if exist "build" rmdir /s /q "build"
echo ✓ 清理完成

:: 执行打包
echo.
echo [3/4] 执行 PyInstaller 打包...
echo （这可能需要几分钟，请耐心等待）
echo.
pyinstaller BookTermGacha.spec --noconfirm
if errorlevel 1 (
    echo.
    echo ✗ 打包失败！请检查错误信息。
    pause
    exit /b 1
)
echo.
echo ✓ PyInstaller 打包完成

:: 复制资源文件
echo.
echo [4/4] 复制资源文件...
python build_finish.py
if errorlevel 1 (
    echo.
    echo ✗ 资源文件复制失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 打包完成！
echo 输出目录: dist\BookTermGacha
echo 可执行文件: dist\BookTermGacha\app.exe
echo ============================================================
echo.
echo 按任意键打开输出目录...
pause >nul
start "" "dist\BookTermGacha"
