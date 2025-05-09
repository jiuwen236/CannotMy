import subprocess
import sys
import os
import shutil
import importlib
from pathlib import Path

# 配置区（用户可根据需要修改这些参数）
CONFIG = {
    "use_venv": True,                    # 是否使用虚拟环境
    "venv_dir": "packagingenv",                  # 虚拟环境目录
    "python_version": (3, 11),           # 指定Python版本
    "python_search_paths": [    # Python可能的安装路径
    r"C:\Python*",          # 默认安装路径
    r"C:\Program Files\Python*",
    r"C:\Users\*\AppData\Local\Programs\Python\Python*",
    r"C:\msys64\mingw64\bin"  # 适用于MSYS2环境
    ],
    "pypi_mirror": "https://pypi.tuna.tsinghua.edu.cn/simple",  # 镜像源
    "requirements": "packaging_requirements.txt",  # 依赖文件路径
    "source_script": "main.py",          # 主程序文件路径
    "icon_file": r"ico/icon_64x64.ico",             # 图标文件路径
    "output_dir": "output",              # 输出目录
    "console": True,
    "add_data": [                        # 需要打包的附加数据
        (r"packagingenv/Lib/site-packages/rapidocr", "rapidocr")
    ],
    "copy_files": [                      # 需要复制的额外文件/目录
        r"C:\Windows\System32\msvcp140.dll",
        r"C:\Windows\System32\vcruntime140.dll",
        r"C:\Windows\System32\vcruntime140_1.dll",
        "arknights.csv",
        "models",
        "images",
        "platform-tools",
        "ico",
    ]
}

def find_python_executable():
    """在系统中查找符合版本要求的Python"""
    required_major, required_minor = CONFIG["python_version"]
    
    # 1. 首先检查环境变量中的Python
    print("正在搜索环境变量中的Python...")
    for path in os.get_exec_path():
        if "python" in path.lower():
            python_exe = Path(path) / "python.exe"
            if python_exe.exists():
                version = get_python_version(python_exe)
                if version and version[:2] == (required_major, required_minor):
                    return python_exe

    # 2. 搜索常见安装路径
    print("扫描系统安装路径...")
    search_patterns = [
        *CONFIG["python_search_paths"],
        str(Path.home() / "AppData/Local/Microsoft/WindowsApps/python*.exe")  # Windows应用商店安装
    ]

    for pattern in search_patterns:
        for path in Path().glob(pattern):
            if path.is_dir():
                python_candidate = path / "python.exe"
            else:
                python_candidate = path
            
            if python_candidate.exists():
                version = get_python_version(python_candidate)
                if version and version[:2] == (required_major, required_minor):
                    return python_candidate.resolve()

    # 3. 如果都找不到，尝试py命令
    print("尝试使用py launcher...")
    try:
        result = subprocess.check_output(["py", f"-{required_major}.{required_minor}", "-c", "import sys; print(sys.executable)"])
        return Path(result.decode().strip())
    except Exception as e:
        print(f"py命令查找失败: {e}")

    return None

def get_python_version(python_exe):
    """获取指定Python解释器的版本"""
    try:
        result = subprocess.check_output([str(python_exe), "--version"], stderr=subprocess.STDOUT)
        version_str = result.decode().split()[1]
        return tuple(map(int, version_str.split('.')[:2]))
    except Exception as e:
        print(f"获取版本失败 {python_exe}: {e}")
        return None

def validate_python_version():
    """验证Python版本是否符合要求"""
    required = CONFIG["python_version"]
    if sys.version_info.major != required[0] or sys.version_info.minor != required[1]:
        print(f"需要Python {required[0]}.{required[1]}，当前版本：{sys.version_info.major}.{sys.version_info.minor}")
        return False
    return True

def create_venv_with_specified_python(python_exe):
    """使用指定Python创建虚拟环境"""
    venv_path = Path(CONFIG["venv_dir"])
    
    if venv_path.exists():
        print(f"虚拟环境已存在: {venv_path}")
        return True

    try:
        # 使用找到的Python创建虚拟环境
        subprocess.check_call([str(python_exe), "-m", "venv", str(venv_path)])
        print(f"成功使用 {python_exe} 创建虚拟环境")
        return True
    except subprocess.CalledProcessError as e:
        print(f"创建虚拟环境失败: {e}")
        return False

def install_dependencies():
    """安装项目依赖"""
    venv_pip = str(Path(CONFIG["venv_dir"]) / "Scripts" / "pip.exe")
    req_file = Path(CONFIG["requirements"])
    
    if not req_file.exists():
        print(f"依赖文件 {req_file} 不存在，跳过安装")
        return True

    try:
        subprocess.check_call([
            venv_pip, "install", "-r", str(req_file),
            "-i", CONFIG["pypi_mirror"]
        ])
        return True
    except subprocess.CalledProcessError as e:
        print(f"依赖安装失败：{e}")
        return False

def build_exe():
    """使用PyInstaller打包"""
    venv_python = str(Path(CONFIG["venv_dir"]) / "Scripts" / "python.exe")
    script_path = Path(CONFIG["source_script"])
    icon_path = Path(CONFIG["icon_file"])
    build_cmd = [
        venv_python, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--console" if CONFIG["console"] else "--windowed",  # 动态选择参数
        "--name", script_path.stem,
        "--distpath", CONFIG["output_dir"],
        "--workpath", "build",
        "--exclude-module", "train",
        "--exclude-module", "predict"
    ]

    if icon_path.exists():
        build_cmd.extend(["--icon", str(icon_path)])

    # 添加附加数据
    for src, dest in CONFIG["add_data"]:
        build_cmd.extend(["--add-data", f"{Path(src).resolve()};{dest}"])

    build_cmd.append(str(script_path))

    try:
        subprocess.check_call(build_cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"打包失败：{e}")
        return False

def copy_additional_files():
    """复制额外文件到输出目录"""
    exe_dir = Path(CONFIG["output_dir"]) / Path(CONFIG["source_script"]).stem
    if not exe_dir.exists():
        print(f"输出目录不存在: {exe_dir}")
        return False

    for item in CONFIG["copy_files"]:
        src = Path(item)
        dest = exe_dir / src.name

        if not src.exists():
            print(f"警告：{src} 不存在，跳过复制")
            continue

        try:
            if src.is_dir():
                # 特殊处理images目录，排除tmp和nums子目录
                if src.name == "images":
                    def ignore_func(dir, names):
                        """忽略tmp和nums子目录"""
                        dir_path = Path(dir)
                        # 仅在images根目录下排除特定子目录
                        if dir_path.resolve() == src.resolve():
                            return [n for n in names if n in {'tmp', 'nums'}]
                        return []
                    shutil.copytree(src, dest, ignore=ignore_func, dirs_exist_ok=True)
                else:
                    shutil.copytree(src, dest, dirs_exist_ok=True)
                print(f"已复制目录：{src} -> {dest}")
            else:
                shutil.copy2(src, dest)
                print(f"已复制文件：{src} -> {dest}")
        except Exception as e:
            print(f"复制失败：{e}")

    return True

def main():
    if CONFIG["use_venv"]:
        if not validate_python_version():
            print("\n正在尝试自动查找合适的Python版本...")
            target_python = find_python_executable()
            
            if not target_python:
                print(f"未找到Python {CONFIG['python_version']}，请执行以下操作之一：")
                print("1. 安装Python {0}.{1}".format(*CONFIG['python_version']))
                print("2. 修改CONFIG中的python_version配置")
                print("3. 指定Python路径（例如：CONFIG['venv_dir'] = r'C:\\path\\to\\python.exe'）")
                return
                
            print(f"找到符合条件的Python: {target_python}")
            if not create_venv_with_specified_python(target_python):
                return

        # 后续使用虚拟环境中的Python
        venv_python = Path(CONFIG["venv_dir"]) / "Scripts" / "python.exe"
        if not venv_python.exists():
            print(f"虚拟环境不完整，缺少 {venv_python}")
            return
    if not install_dependencies():
        return

    if not build_exe():
        return

    if not copy_additional_files():
        return

    print(f"\n打包成功！输出目录：{Path(CONFIG['output_dir']).resolve()}")

if __name__ == "__main__":
    main()