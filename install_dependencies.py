

"""
中文车牌识别系统依赖安装脚本
此脚本将自动安装所有必要的依赖，包括PyTorch（支持GPU检测）
"""

import os
import sys
import platform
import subprocess
import importlib.util
import argparse


def is_windows():
    """检查是否为Windows系统"""
    return platform.system() == 'Windows'


def is_macos():
    """检查是否为macOS系统"""
    return platform.system() == 'Darwin'


def is_linux():
    """检查是否为Linux系统"""
    return platform.system() == 'Linux'


def check_gpu():
    """检查系统是否有可用的GPU"""
    try:
        # 检查NVIDIA GPU
        if is_windows():
            # 在Windows上使用wmic命令检查GPU
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True, text=True
            )
            if "NVIDIA" in result.stdout:
                return "cuda"
        else:
            # 在Linux/macOS上检查NVIDIA GPU
            try:
                result = subprocess.run(
                    ["nvidia-smi"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    return "cuda"
            except FileNotFoundError:
                pass
        
        # 检查Apple Silicon GPU (macOS)
        if is_macos() and platform.processor() == 'arm':
            return "mps"
        
        return "cpu"
    except Exception as e:
        print(f"检查GPU时出错: {e}")
        return "cpu"


def check_package(package_name):
    """检查Python包是否已安装"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None


def run_command(command, description):
    """执行命令并显示进度条"""
    print(f"{description}...")
    try:
        # 在Windows上使用shell=True以支持命令解析
        process = subprocess.Popen(
            command, 
            shell=is_windows(), 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        # 尝试导入tqdm以显示进度条
        try:
            from tqdm import tqdm
            import time
            
            # 创建一个简单的进度条，由于无法知道确切的总进度，使用循环更新
            with tqdm(total=100, desc="安装进度", unit="%") as pbar:
                last_update = time.time()
                while process.poll() is None:
                    # 每0.5秒更新一次进度条
                    if time.time() - last_update > 0.5:
                        # 随机增加一点进度，直到安装完成
                        pbar.update(1)
                        if pbar.n >= 90:  # 保留最后10%进度
                            pbar.n = 90
                        last_update = time.time()
                    
                    # 读取输出但不显示，避免进度条混乱
                    output = process.stdout.readline()
                    error = process.stderr.readline()
                
                # 完成时设置进度为100%
                pbar.n = 100
                pbar.refresh()
                
        except ImportError:
            # 如果tqdm不可用，使用简单的输出方式
            print("安装tqdm库以显示进度条...")
            while process.poll() is None:
                output = process.stdout.readline()
                if output:
                    print(output.strip())
                
                error = process.stderr.readline()
                if error:
                    print(f"错误: {error.strip()}", file=sys.stderr)
        
        # 显示最后剩余的输出
        remaining_stdout = process.stdout.read()
        remaining_stderr = process.stderr.read()
        if remaining_stdout:
            print(remaining_stdout.strip())
        if remaining_stderr:
            print(f"错误: {remaining_stderr.strip()}", file=sys.stderr)
        
        return process.returncode == 0
    except Exception as e:
        print(f"执行命令时出错: {e}")
        return False


def show_pytorch_install_commands(device_type):
    """显示PyTorch和torchvision的安装命令"""
    print(f"\n检测到设备类型: {device_type}")
    print("请手动安装PyTorch和torchvision，这是本项目的核心依赖。")
    print("根据您的设备类型，推荐使用以下命令安装:")
    
    if device_type == "cuda":
        # NVIDIA GPU用户
        print("\n对于NVIDIA GPU (CUDA):")
        print("Windows系统:")
        print("pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html")
        print("Linux/macOS系统:")
        print("pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html")
    elif device_type == "mps" and is_macos():
        # Apple Silicon用户
        print("\n对于Apple Silicon (M1/M2/M3) 芯片:")
        print("pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2")
    else:
        # CPU用户
        print("\n对于CPU版本:")
        print("pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -f https://download.pytorch.org/whl/cpu")
    
    print("\n注意: 以上命令可能需要根据您的系统和CUDA版本进行调整。")
    print("建议访问官方网站获取最新的安装命令: https://pytorch.org/get-started/locally/")
    print("安装完成后，请重新运行此脚本以安装其他依赖。")
    return True


def install_dependencies():
    """安装除PyTorch外的所有依赖项并显示安装状态"""
    # 检查PyTorch是否已安装
    if not check_package("torch") or not check_package("torchvision"):
        print("错误: PyTorch或torchvision尚未安装")
        device_type = check_gpu()
        show_pytorch_install_commands(device_type)
        return False
    
    print("\n==== 依赖状态检查 ====")
    print("✅ PyTorch 已安装")
    print("✅ torchvision 已安装")
    
    # 获取需要安装的依赖列表
    dependencies_to_install = []
    
    # 从requirements.txt安装其他依赖
    if os.path.exists("requirements.txt"):
        # 读取requirements.txt并过滤出非PyTorch依赖
        with open("requirements.txt", "r") as f:
            lines = f.readlines()
        
        # 过滤掉PyTorch相关依赖
        for line in lines:
            line = line.strip()
            if line and not line.startswith(("torch", "torchvision")):
                dependencies_to_install.append(line)
    else:
        print("未找到requirements.txt文件，使用默认依赖列表")
        # 使用默认的常用依赖
        dependencies_to_install = [
            "opencv-python>=4.5.0",
            "numpy>=1.20.0",
            "pillow>=8.0.0",
            "ultralytics>=8.0.0",
            "matplotlib>=3.3.0",
            "pandas>=1.1.0",
            "imutils>=0.5.4",
            "tqdm>=4.60.0",
            "PySide6>=6.3.0",
        ]
    
    # 检查并显示每个依赖的状态
    installed_packages = []
    packages_to_install = []
    
    for dep in dependencies_to_install:
        # 提取包名（去掉版本限制）
        # 处理各种版本限制符号: ==, >=, <=, ~=
        if "==" in dep:
            package_name = dep.split("==")[0].strip()
        elif ">=" in dep:
            package_name = dep.split(">=")[0].strip()
        elif "<=" in dep:
            package_name = dep.split("<=")[0].strip()
        elif "~=" in dep:
            package_name = dep.split("~=")[0].strip()
        else:
            package_name = dep.strip()
        
        # 特殊处理一些包名映射
        if package_name == "opencv-python":
            check_name = "cv2"
        elif package_name == "pillow":
            check_name = "PIL"
        else:
            check_name = package_name.lower()
        
        if check_package(check_name):
            installed_packages.append(dep)
            print(f"✅ {dep} 已安装")
        else:
            packages_to_install.append(dep)
            print(f"⚠️ {dep} 未安装，将进行安装")
    
    # 如果有需要安装的依赖
    if packages_to_install:
        print(f"\n需要安装 {len(packages_to_install)} 个依赖包...")
        command = [sys.executable, "-m", "pip", "install"]
        # 添加参数避免重新下载已安装的包
        command.extend(["--no-deps", "--upgrade-strategy", "only-if-needed"])
        # 如果启用了清华源，添加镜像源
        if USE_TUNA:
            command.extend(["-i", "https://pypi.tuna.tsinghua.edu.cn/simple", "--trusted-host", "pypi.tuna.tsinghua.edu.cn"])
        command.extend(packages_to_install)
        if not run_command(command, "安装缺失的依赖项"):
            print("依赖项安装失败，退出")
            return False
    else:
        print("\n所有依赖包都已安装，无需额外安装")
    
    # 验证安装（跳过PyTorch和torchvision，因为已经在前面检查过）
    required_packages = ["cv2", "numpy", "PIL", "ultralytics", "PySide6", "tqdm", "imutils"]
    missing_packages = []
    
    for pkg in required_packages:
        if pkg == "cv2":
            pkg_name = "cv2"
        elif pkg == "PIL":
            pkg_name = "PIL"
        elif pkg == "PySide6":
            pkg_name = "PySide6"
        elif pkg == "tqdm":
            pkg_name = "tqdm"
        elif pkg == "imutils":
            pkg_name = "imutils"
        else:
            pkg_name = pkg
        
        if not check_package(pkg_name):
            missing_packages.append(pkg)
    
    if missing_packages:
        print(f"以下包安装失败: {', '.join(missing_packages)}")
        return False
    
    print("\n==== 依赖安装总结 ====")
    print(f"✅ 已安装的依赖: {len(installed_packages)} 个")
    print(f"✅ 新安装的依赖: {len(packages_to_install)} 个")
    print("所有依赖项安装成功！")
    return True


def main():
    parser = argparse.ArgumentParser(description='安装中文车牌识别系统依赖')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu', 'mps'], default='auto',
                        help='选择设备类型 (默认: 自动检测)')
    parser.add_argument('--force', action='store_true',
                        help='强制重新安装所有依赖')
    parser.add_argument('--use-tuna', action='store_true', default=True,
                        help='使用清华源镜像加速下载 (默认: True)')
    args = parser.parse_args()
    
    # setup_environment函数会调用main()来获取参数，然后在那里执行主要逻辑
    return args



def setup_environment():
    """设置环境和依赖安装选项"""
    args = main()
    
    print("中文车牌识别系统依赖安装脚本")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {platform.python_version()}")
    print(f"使用清华源: {args.use_tuna}")
    
    # 检测设备是否支持CUDA
    gpu_type = check_gpu()
    if gpu_type == "cuda":
        print("✅ 检测到NVIDIA GPU，支持CUDA加速")
    elif gpu_type == "mps" and is_macos():
        print("✅ 检测到Apple Silicon芯片，支持MPS加速")
    else:
        print("⚠️ 未检测到可用的GPU加速，将使用CPU模式")
    
    if args.force:
        print("强制重新安装所有依赖")
    else:
        print("仅安装缺失的依赖")
    
    # 全局变量，存储清华源设置
    global USE_TUNA
    USE_TUNA = args.use_tuna
    
    # 检查PyTorch和torchvision是否已安装
    if not check_package("torch") or not check_package("torchvision"):
        device_type = args.device if args.device != 'auto' else check_gpu()
        show_pytorch_install_commands(device_type)
        return 1
    
    # 安装其他依赖
    if install_dependencies():
        print("\n依赖安装完成！")
        print("现在您可以运行项目中的训练、测试脚本或GUI界面了。")
        print("\n使用指南:")
        print("- 运行GUI界面: python main.py")
        print("- 训练LPRNet: python train_LPRNet.py")
        print("- 测试LPRNet: python test_LPRNet.py")
        print("- 训练YOLO: python train_yolo.py")
        print("- 测试YOLO: python test_yolo.py")
        return 0
    else:
        print("依赖安装失败，请检查错误信息并重试。")
        return 1

# 全局变量
USE_TUNA = True

if __name__ == '__main__':
    sys.exit(setup_environment())