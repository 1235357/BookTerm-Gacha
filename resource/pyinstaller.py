import os
import sys
import PyInstaller.__main__

# 获取 pip 安装的 root 路径（site-packages）
def get_pip_root() -> str:
    return os.path.join(sys.prefix, "lib", "site-packages")

cmd = [
    "./app.py",
    "--clean",      # Clean PyInstaller cache
    "--onedir",     # Create a one-folder bundle (recommended)
    # "--onefile", # Create a single executable (slower startup)
    "--noconfirm",  # Auto replace output directory
    "--distpath=./dist",
    "--name=BookTermGacha"
]

if os.path.exists("./requirements.txt"):
    # 生成配置
    with open("./requirements.txt", "r", encoding = "utf-8-sig") as reader:
        for line in reader:
            if not line.strip().startswith(("#", "--")):
                cmd.append("--hidden-import=" + line.strip())

            if line.strip() == "pecab":
                cmd.append(f"--add-data={get_pip_root()}/pecab:pecab")

            if line.strip() == "pykakasi":
                cmd.append(f"--add-data={get_pip_root()}/pykakasi:pykakasi")

    # 执行打包
    PyInstaller.__main__.run(cmd)

    # 更名为 app.exe
    if os.path.isfile("./dist/BookTermGacha/BookTermGacha.exe"):
        os.rename("./dist/BookTermGacha/BookTermGacha.exe", "./dist/BookTermGacha/app.exe")