

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
    """执行命令并显示进度"""
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
        
        # 实时显示输出
        while process.poll() is None:
            output = process.stdout.readline()
            if output:
                print(output.strip())
            
            error = process.stderr.readline()
            if error:
                print(f"错误: {error.strip()}", file=sys.stderr)
        
        return process.returncode == 0
    except Exception as e:
        print(f"执行命令时出错: {e}")
        return False


def install_pytorch(device_type):
    """安装PyTorch和torchvision"""
    print(f"检测到设备类型: {device_type}")
    
    # 检查PyTorch是否已安装
    if check_package("torch"):
        print("PyTorch 已安装，跳过安装")
        return True
    
    if device_type == "cuda":
        # 安装支持CUDA的PyTorch版本
        if is_windows():
            command = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"]
        else:
            command = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"]
    elif device_type == "mps" and is_macos():
        # 安装支持MPS的PyTorch版本 (Apple Silicon)
        command = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
    else:
        # 安装CPU版本的PyTorch
        if is_windows():
            command = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"]
        else:
            command = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"]
    
    return run_command(command, "安装PyTorch和torchvision")


def install_dependencies():
    """安装所有依赖项"""
    # 首先安装PyTorch（特殊处理）
    device_type = check_gpu()
    if not install_pytorch(device_type):
        print("PyTorch安装失败，尝试使用CPU版本")
        if not install_pytorch("cpu"):
            print("PyTorch安装失败，退出")
            return False
    
    # 从requirements.txt安装其他依赖
    if os.path.exists("requirements.txt"):
        # 先安装基础依赖，跳过已安装的PyTorch和torchvision
        # 读取requirements.txt并过滤出非PyTorch依赖
        with open("requirements.txt", "r") as f:
            lines = f.readlines()
        
        # 过滤掉已安装的PyTorch相关依赖
        non_pytorch_deps = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith(("torch", "torchvision")):
                non_pytorch_deps.append(line)
        
        # 安装剩余的依赖
        if non_pytorch_deps:
            command = [sys.executable, "-m", "pip", "install"] + non_pytorch_deps
            if not run_command(command, "安装其他依赖项"):
                print("依赖项安装失败，退出")
                return False
    else:
        print("未找到requirements.txt文件，尝试直接安装常用依赖")
        # 直接安装常用依赖
        common_deps = [
            "opencv-python>=4.5.0",
            "numpy>=1.20.0",
            "pillow>=8.0.0",
            "ultralytics>=8.0.0",
            "matplotlib>=3.3.0",
            "pandas>=1.1.0"
        ]
        command = [sys.executable, "-m", "pip", "install"] + common_deps
        if not run_command(command, "安装常用依赖项"):
            print("依赖项安装失败，退出")
            return False
    
    # 验证安装
    required_packages = ["torch", "torchvision", "cv2", "numpy", "PIL", "ultralytics", "PySide6"]
    missing_packages = []
    
    for pkg in required_packages:
        if pkg == "cv2":
            pkg_name = "cv2"
        elif pkg == "PIL":
            pkg_name = "PIL"
        elif pkg == "PySide6":
            pkg_name = "PySide6"
        else:
            pkg_name = pkg
        
        if not check_package(pkg_name):
            missing_packages.append(pkg)
    
    if missing_packages:
        print(f"以下包安装失败: {', '.join(missing_packages)}")
        return False
    
    print("所有依赖项安装成功！")
    return True


def main():
    parser = argparse.ArgumentParser(description='安装中文车牌识别系统依赖')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu', 'mps'], default='auto',
                        help='选择设备类型 (默认: 自动检测)')
    parser.add_argument('--force', action='store_true',
                        help='强制重新安装所有依赖')
    args = parser.parse_args()
    
    print("中文车牌识别系统依赖安装脚本")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {platform.python_version()}")
    
    if args.force:
        print("强制重新安装所有依赖")
    else:
        print("仅安装缺失的依赖")
    
    # 如果指定了设备类型，直接使用
    if args.device != 'auto':
        print(f"使用指定的设备类型: {args.device}")
        if not install_pytorch(args.device):
            print("PyTorch安装失败，退出")
            return False
    
    # 安装其他依赖
    if install_dependencies():
        print("\n依赖安装完成！")
        print("现在您可以运行项目中的训练、测试脚本或GUI界面了。")
        print("\n使用指南:")
        print("- 运行GUI界面: python plate_recognition_gui.py")
        print("- 训练LPRNet: python train_LPRNet.py")
        print("- 测试LPRNet: python test_LPRNet.py")
        print("- 训练YOLO: python train_yolo.py")
        return 0
    else:
        print("依赖安装失败，请检查错误信息并重试。")
        return 1


if __name__ == '__main__':
    sys.exit(main())