#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
车牌识别系统的图形用户界面
使用PySide6构建，支持图片、视频和摄像头输入
"""

import os
import sys
import cv2
import torch
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QTextEdit, QTabWidget, QFrame, QSplitter, QMessageBox,
    QProgressBar, QSizePolicy
)
from PySide6.QtGui import QPixmap, QImage, QFont, QIcon
from PySide6.QtCore import Qt, QThread, Signal, Slot

# 导入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from yolo_utils import YOLOPlateDetector
from model.LPRNet import build_lprnet
from data.load_data import CHARS, CHARS_DICT


class VideoProcessingThread(QThread):
    """视频处理线程，用于在后台处理视频或摄像头流，避免UI卡顿"""
    frame_updated = Signal(np.ndarray, list)  # 信号：发送更新后的帧和车牌结果
    processing_completed = Signal()  # 信号：处理完成
    error_occurred = Signal(str)  # 信号：发生错误
    progress_updated = Signal(int)  # 信号：进度更新

    def __init__(self, detector, recognizer, input_source, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化视频处理线程
        
        Args:
            detector: YOLOPlateDetector实例
            recognizer: LPRNet实例
            input_source: 视频文件路径或摄像头ID
            device: 运行设备
        """
        super().__init__()
        self.detector = detector
        self.recognizer = recognizer
        self.input_source = input_source
        self.device = device
        self.running = False
        self.paused = False

    def run(self):
        """线程运行函数"""
        try:
            self.running = True
            # 打开视频文件或摄像头
            if isinstance(self.input_source, int) or self.input_source.isdigit():
                cap = cv2.VideoCapture(int(self.input_source))
            else:
                cap = cv2.VideoCapture(self.input_source)
                # 获取视频总帧数
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                current_frame = 0

            if not cap.isOpened():
                self.error_occurred.emit(f"无法打开输入源: {self.input_source}")
                return

            while self.running:
                if self.paused:
                    self.msleep(50)
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                # 处理当前帧
                result_frame, plate_results = self.process_frame(frame)
                self.frame_updated.emit(result_frame, plate_results)

                # 更新进度（如果是视频文件）
                if not isinstance(self.input_source, int) and not self.input_source.isdigit():
                    current_frame += 1
                    progress = int((current_frame / total_frames) * 100)
                    self.progress_updated.emit(progress)

                # 控制帧率
                self.msleep(30)  # 约33fps

            cap.release()
            self.processing_completed.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))

    def process_frame(self, frame):
        """处理单帧图像，检测并识别车牌"""
        # 创建结果图像的副本
        result_frame = frame.copy()

        # 检测车牌坐标
        plate_coords = self.detector.detect_plates(frame)

        # 识别每个车牌
        plate_results = []
        for i, plate_info in enumerate(plate_coords):
            x1, y1, x2, y2, conf = plate_info

            try:
                # 裁剪车牌区域
                plate_image = frame[y1:y2, x1:x2]

                # 识别车牌字符
                plate_text = self.recognize_plate(plate_image)
                plate_results.append({
                    'coords': (x1, y1, x2, y2),
                    'confidence': conf,
                    'text': plate_text
                })

                # 绘制车牌边框
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 绘制文本 - 使用PIL支持中文显示
                label = f"{plate_text} ({conf:.2f})"
                try:
                    from PIL import Image, ImageDraw, ImageFont
                    # 将OpenCV图像转换为PIL图像
                    result_frame_pil = Image.fromarray(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(result_frame_pil)
                    # 尝试加载中文字体
                    try:
                        # Windows系统默认字体
                        font = ImageFont.truetype("simhei.ttf", 16)
                    except:
                        # 如果找不到指定字体，使用默认字体
                        font = ImageFont.load_default()
                    # 绘制文本
                    draw.text((x1, y1 - 20), label, font=font, fill=(255, 0, 0))
                    # 将PIL图像转换回OpenCV图像
                    result_frame = cv2.cvtColor(np.array(result_frame_pil), cv2.COLOR_RGB2BGR)
                except:
                    # 如果PIL不可用，回退到OpenCV绘制
                    # 尝试使用支持中文的字体（需要系统中安装）
                    cv2.putText(result_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

            except Exception as e:
                continue

        return result_frame, plate_results

    def recognize_plate(self, plate_image):
        """使用LPRNet识别车牌字符"""
        try:
            # 检查输入图像是否有效
            if plate_image is None or len(plate_image.shape) != 3:
                return "识别失败"

            # 图像预处理
            img = cv2.resize(plate_image, (94, 24))
            img = img.astype('float32')
            img -= 127.5
            img *= 0.0078125
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, axis=0)

            # 转换为PyTorch张量并移动到设备
            img_tensor = torch.from_numpy(img)
            if self.device == 'cuda' and torch.cuda.is_available():
                img_tensor = img_tensor.cuda()

            # 前向传播
            with torch.no_grad():
                prebs = self.recognizer(img_tensor)

            # 使用贪婪解码函数进行解码
            plate_text = self.greedy_decode(prebs, CHARS)
            return plate_text
        except Exception:
            return "识别失败"

    def greedy_decode(self, prebs, chars):
        """贪婪解码函数，用于解码LPRNet模型的输出"""
        # 将张量转换为numpy数组
        if isinstance(prebs, torch.Tensor):
            if prebs.is_cuda:
                prebs = prebs.cpu().detach().numpy()
            else:
                prebs = prebs.detach().numpy()

        # 确保prebs的维度正确
        if prebs.ndim == 3:
            prebs = prebs[0]  # 去除批次维度

        # 提取预测标签
        preb_label = list()
        for j in range(prebs.shape[1]):
            preb_label.append(np.argmax(prebs[:, j], axis=0))

        # 去除重复字符和空白字符
        no_repeat_blank_label = list()
        if len(preb_label) > 0:
            pre_c = preb_label[0]
            if pre_c != len(chars) - 1:
                no_repeat_blank_label.append(pre_c)

            for c in preb_label[1:]:
                if (pre_c == c) or (c == len(chars) - 1):
                    if c == len(chars) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c

        # 将数字标签转换为字符
        plate_str = ""
        for idx in no_repeat_blank_label:
            try:
                if 0 <= idx < len(chars):
                    plate_str += chars[idx]
                else:
                    plate_str += '?'
            except:
                plate_str += '?'

        # 如果识别结果为空，返回默认文本
        if not plate_str:
            plate_str = "无法识别"

        return plate_str

    def stop(self):
        """停止处理"""
        self.running = False
        self.wait()

    def pause(self):
        """暂停处理"""
        self.paused = True

    def resume(self):
        """恢复处理"""
        self.paused = False


class PlateRecognitionGUI(QMainWindow):
    """车牌识别系统的主窗口"""
    def __init__(self):
        """初始化主窗口"""
        super().__init__()
        # 设置窗口标题和大小
        self.setWindowTitle("中文车牌识别系统")
        self.resize(1200, 800)

        # 初始化变量
        self.detector = None  # YOLO检测器
        self.recognizer = None  # LPRNet识别器
        self.processing_thread = None  # 视频处理线程
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 运行设备
        self.current_video_path = None  # 当前选择的视频文件路径

        # 初始化UI
        self.init_ui()

        # 加载模型
        self.load_models()

    def init_ui(self):
        """初始化用户界面"""
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QVBoxLayout(central_widget)

        # 创建菜单栏
        self.create_menu_bar()

        # 创建按钮区域
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(10, 10, 10, 10)
        button_layout.setSpacing(10)

        # 添加按钮
        self.btn_select_image = QPushButton("选择图片")
        self.btn_select_image.clicked.connect(self.select_image)
        button_layout.addWidget(self.btn_select_image)

        self.btn_select_video = QPushButton("选择视频")
        self.btn_select_video.clicked.connect(self.select_video)
        button_layout.addWidget(self.btn_select_video)

        self.btn_open_camera = QPushButton("打开摄像头")
        self.btn_open_camera.clicked.connect(self.open_camera)
        button_layout.addWidget(self.btn_open_camera)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_stop.setEnabled(False)
        button_layout.addWidget(self.btn_stop)

        self.btn_pause_resume = QPushButton("暂停")
        self.btn_pause_resume.clicked.connect(self.toggle_pause_resume)
        self.btn_pause_resume.setEnabled(False)
        button_layout.addWidget(self.btn_pause_resume)

        # 添加到主布局
        main_layout.addLayout(button_layout)

        # 创建进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 创建分割器，用于分隔图像显示和结果区域
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 创建图像显示区域
        image_frame = QFrame()
        image_frame.setFrameShape(QFrame.StyledPanel)
        image_layout = QVBoxLayout(image_frame)

        self.image_label = QLabel("请选择图片、视频或打开摄像头")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_layout.addWidget(self.image_label)

        splitter.addWidget(image_frame)

        # 创建结果显示区域
        result_frame = QFrame()
        result_frame.setFrameShape(QFrame.StyledPanel)
        result_layout = QVBoxLayout(result_frame)

        result_layout.addWidget(QLabel("识别结果:"))
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)

        splitter.addWidget(result_frame)

        # 设置分割器的初始大小
        splitter.setSizes([800, 400])

    def create_menu_bar(self):
        """创建菜单栏"""
        menu_bar = self.menuBar()

        # 文件菜单
        file_menu = menu_bar.addMenu("文件")

        # 选择图片动作
        select_image_action = file_menu.addAction("选择图片")
        select_image_action.triggered.connect(self.select_image)

        # 选择视频动作
        select_video_action = file_menu.addAction("选择视频")
        select_video_action.triggered.connect(self.select_video)

        # 打开摄像头动作
        open_camera_action = file_menu.addAction("打开摄像头")
        open_camera_action.triggered.connect(self.open_camera)

        # 退出动作
        exit_action = file_menu.addAction("退出")
        exit_action.triggered.connect(self.close)

        # 帮助菜单
        help_menu = menu_bar.addMenu("帮助")

        # 关于动作
        about_action = help_menu.addAction("关于")
        about_action.triggered.connect(self.show_about_dialog)

    def load_models(self):
        """加载YOLO和LPRNet模型"""
        try:
            # 尝试自动查找模型文件
            yolo_model_path = self.find_model_file(['./runs/train/yolo_lpr/weights/best.pt', './yolo11n.pt', './yolov8n.pt'])
            lpr_model_path = self.find_model_file(['./weights/Final_LPRNet_model_20251012_150100.pth'])

            # 初始化YOLO检测器
            self.detector = YOLOPlateDetector(yolo_model_path)

            # 加载LPRNet模型
            self.recognizer = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0.5)
            self.recognizer.to(self.device)
            self.recognizer.load_state_dict(torch.load(lpr_model_path, weights_only=True))
            self.recognizer.eval()

            self.log_message(f"成功加载YOLO模型: {yolo_model_path}")
            self.log_message(f"成功加载LPRNet模型: {lpr_model_path}")
            self.log_message(f"使用设备: {self.device}")

        except Exception as e:
            self.log_message(f"加载模型失败: {str(e)}")
            QMessageBox.warning(self, "模型加载失败", f"无法加载模型文件: {str(e)}")

    def find_model_file(self, possible_paths):
        """查找模型文件，返回第一个存在的文件路径"""
        for path in possible_paths:
            if os.path.exists(path):
                return path
        # 如果没有找到，返回第一个路径作为默认路径
        return possible_paths[0]

    def select_image(self):
        """选择并处理图像"""
        # 停止正在进行的处理
        self.stop_processing()

        # 打开文件对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", ".", "图像文件 (*.jpg *.jpeg *.png *.bmp)")

        if file_path:
            try:
                # 显示进度条
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(20)

                # 读取图像
                image = cv2.imread(file_path)
                if image is None:
                    raise FileNotFoundError(f"无法读取图像: {file_path}")

                self.progress_bar.setValue(40)

                # 处理图像
                result_image, plate_results = self.process_image(image)

                self.progress_bar.setValue(80)

                # 显示结果
                self.display_image(result_image)
                self.display_results(plate_results)

                self.progress_bar.setValue(100)
                # 延迟隐藏进度条
                QApplication.processEvents()
                QThread.msleep(500)
                self.progress_bar.setVisible(False)

            except Exception as e:
                self.log_message(f"处理图像时出错: {str(e)}")
                QMessageBox.warning(self, "处理失败", f"处理图像时出错: {str(e)}")
                self.progress_bar.setVisible(False)

    def select_video(self):
        """选择并处理视频文件"""
        # 停止正在进行的处理
        self.stop_processing()

        # 打开文件对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", ".", "视频文件 (*.mp4 *.avi *.mov *.mkv)")

        if file_path:
            # 保存当前选择的视频文件路径
            self.current_video_path = file_path
            try:
                # 显示进度条
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)

                # 创建视频处理线程
                self.processing_thread = VideoProcessingThread(
                    self.detector, self.recognizer, file_path, self.device
                )
                self.processing_thread.frame_updated.connect(self.on_frame_updated)
                self.processing_thread.processing_completed.connect(self.on_processing_completed)
                self.processing_thread.error_occurred.connect(self.on_error_occurred)
                self.processing_thread.progress_updated.connect(self.progress_bar.setValue)

                # 启动线程
                self.processing_thread.start()

                # 更新按钮状态
                self.update_button_states(True)

            except Exception as e:
                self.log_message(f"打开视频时出错: {str(e)}")
                QMessageBox.warning(self, "打开失败", f"打开视频时出错: {str(e)}")
                self.progress_bar.setVisible(False)

    def open_camera(self):
        """打开摄像头进行实时车牌识别"""
        # 停止正在进行的处理
        self.stop_processing()

        try:
            # 隐藏进度条（摄像头模式不需要进度条）
            self.progress_bar.setVisible(False)

            # 创建视频处理线程（使用摄像头ID 0）
            self.processing_thread = VideoProcessingThread(
                self.detector, self.recognizer, 0, self.device
            )
            self.processing_thread.frame_updated.connect(self.on_frame_updated)
            self.processing_thread.processing_completed.connect(self.on_processing_completed)
            self.processing_thread.error_occurred.connect(self.on_error_occurred)

            # 启动线程
            self.processing_thread.start()

            # 更新按钮状态
            self.update_button_states(True)

        except Exception as e:
            self.log_message(f"打开摄像头时出错: {str(e)}")
            QMessageBox.warning(self, "打开失败", f"打开摄像头时出错: {str(e)}")

    def stop_processing(self):
        """停止正在进行的处理"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread = None

        # 清空当前选择的视频文件路径
        self.current_video_path = None

        # 重置图像标签显示初始提示文本
        self.image_label.setText("请选择图片、视频或打开摄像头")

        # 清空识别结果
        self.result_text.clear()
        
        # 更新按钮状态
        self.update_button_states(False)
        self.progress_bar.setVisible(False)

    def toggle_pause_resume(self):
        """切换暂停/恢复状态"""
        if self.processing_thread:
            if self.processing_thread.paused:
                self.processing_thread.resume()
                self.btn_pause_resume.setText("暂停")
            else:
                self.processing_thread.pause()
                self.btn_pause_resume.setText("恢复")

    def update_button_states(self, processing):
        """更新按钮状态"""
        self.btn_select_image.setEnabled(not processing)
        self.btn_select_video.setEnabled(not processing)
        self.btn_open_camera.setEnabled(not processing)
        self.btn_stop.setEnabled(processing)
        self.btn_pause_resume.setEnabled(processing)
        if processing:
            self.btn_pause_resume.setText("暂停")

    def process_image(self, image):
        """处理单张图像，检测并识别车牌"""
        # 创建结果图像的副本
        result_image = image.copy()

        # 检测车牌坐标
        plate_coords = self.detector.detect_plates(image)

        # 识别每个车牌
        plate_results = []
        for i, plate_info in enumerate(plate_coords):
            x1, y1, x2, y2, conf = plate_info

            try:
                # 裁剪车牌区域
                plate_image = image[y1:y2, x1:x2]

                # 识别车牌字符
                plate_text = self.recognize_plate(plate_image)
                plate_results.append({
                    'coords': (x1, y1, x2, y2),
                    'confidence': conf,
                    'text': plate_text
                })

                # 绘制车牌边框
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 绘制文本 - 使用PIL支持中文显示
                label = f"{plate_text} ({conf:.2f})"
                try:
                    from PIL import Image, ImageDraw, ImageFont
                    # 将OpenCV图像转换为PIL图像
                    result_image_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(result_image_pil)
                    # 尝试加载中文字体
                    try:
                        # Windows系统默认字体
                        font = ImageFont.truetype("simhei.ttf", 16)
                    except:
                        # 如果找不到指定字体，使用默认字体
                        font = ImageFont.load_default()
                    # 绘制文本
                    draw.text((x1, y1 - 20), label, font=font, fill=(255, 0, 0))
                    # 将PIL图像转换回OpenCV图像
                    result_image = cv2.cvtColor(np.array(result_image_pil), cv2.COLOR_RGB2BGR)
                except:
                    # 如果PIL不可用，回退到OpenCV绘制
                    # 尝试使用支持中文的字体（需要系统中安装）
                    cv2.putText(result_image, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

            except Exception as e:
                continue

        return result_image, plate_results

    def recognize_plate(self, plate_image):
        """使用LPRNet识别车牌字符"""
        try:
            # 检查输入图像是否有效
            if plate_image is None or len(plate_image.shape) != 3:
                return "识别失败"

            # 图像预处理
            img = cv2.resize(plate_image, (94, 24))
            img = img.astype('float32')
            img -= 127.5
            img *= 0.0078125
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, axis=0)

            # 转换为PyTorch张量并移动到设备
            img_tensor = torch.from_numpy(img)
            if self.device == 'cuda' and torch.cuda.is_available():
                img_tensor = img_tensor.cuda()

            # 前向传播
            with torch.no_grad():
                prebs = self.recognizer(img_tensor)

            # 使用贪婪解码函数进行解码
            plate_text = self.greedy_decode(prebs, CHARS)
            return plate_text
        except Exception:
            return "识别失败"

    def greedy_decode(self, prebs, chars):
        """贪婪解码函数，用于解码LPRNet模型的输出"""
        # 将张量转换为numpy数组
        if isinstance(prebs, torch.Tensor):
            if prebs.is_cuda:
                prebs = prebs.cpu().detach().numpy()
            else:
                prebs = prebs.detach().numpy()

        # 确保prebs的维度正确
        if prebs.ndim == 3:
            prebs = prebs[0]  # 去除批次维度

        # 提取预测标签
        preb_label = list()
        for j in range(prebs.shape[1]):
            preb_label.append(np.argmax(prebs[:, j], axis=0))

        # 去除重复字符和空白字符
        no_repeat_blank_label = list()
        if len(preb_label) > 0:
            pre_c = preb_label[0]
            if pre_c != len(chars) - 1:
                no_repeat_blank_label.append(pre_c)

            for c in preb_label[1:]:
                if (pre_c == c) or (c == len(chars) - 1):
                    if c == len(chars) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c

        # 将数字标签转换为字符
        plate_str = ""
        for idx in no_repeat_blank_label:
            try:
                if 0 <= idx < len(chars):
                    plate_str += chars[idx]
                else:
                    plate_str += '?'
            except:
                plate_str += '?'

        # 如果识别结果为空，返回默认文本
        if not plate_str:
            plate_str = "无法识别"

        return plate_str

    def display_image(self, image):
        """在界面上显示图像"""
        # 将OpenCV图像转换为QPixmap
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # 调整图像大小以适应标签
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # 显示图像
        self.image_label.setPixmap(scaled_pixmap)

    def display_results(self, plate_results):
        """在界面上显示识别结果"""
        if not plate_results:
            self.result_text.setPlainText("未检测到车牌")
            return

        # 构建结果文本
        result_text = f"检测到 {len(plate_results)} 个车牌:\n\n"
        for i, result in enumerate(plate_results):
            x1, y1, x2, y2 = result['coords']
            result_text += f"车牌 {i+1}: {result['text']}\n"
            result_text += f"  置信度: {result['confidence']:.2f}\n"
            result_text += f"  位置: ({x1}, {y1})-({x2}, {y2})\n\n"

        # 显示结果
        self.result_text.setPlainText(result_text)

    def log_message(self, message):
        """记录消息到控制台"""
        print(message)

    @Slot(np.ndarray, list)
    def on_frame_updated(self, frame, plate_results):
        """处理帧更新信号"""
        # 显示帧
        self.display_image(frame)
        # 显示结果
        self.display_results(plate_results)

    @Slot()
    def on_processing_completed(self):
        """处理处理完成信号"""
        self.log_message("处理完成")
        self.update_button_states(False)
        self.progress_bar.setVisible(False)

    @Slot(str)
    def on_error_occurred(self, error_message):
        """处理错误信号"""
        self.log_message(f"错误: {error_message}")
        QMessageBox.warning(self, "处理错误", error_message)
        self.update_button_states(False)
        self.progress_bar.setVisible(False)

    def show_about_dialog(self):
        """显示关于对话框"""
        QMessageBox.about(
            self, "关于中文车牌识别系统",
            "中文车牌识别系统\n\n"
            "版本: 1.0.0\n"
            "基于YOLO和LPRNet实现\n\n"
            "功能: 支持图片、视频和摄像头输入的车牌检测与识别"
        )

    def resizeEvent(self, event):
        """处理窗口大小变化事件"""
        super().resizeEvent(event)
        # 如果图像标签中有图像，则调整图像大小以适应新的窗口大小
        if not self.image_label.pixmap().isNull():
            scaled_pixmap = self.image_label.pixmap().scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        """处理窗口关闭事件"""
        # 停止正在进行的处理
        self.stop_processing()
        event.accept()


def main():
    """主函数"""
    # 创建应用程序
    app = QApplication(sys.argv)
    # 创建主窗口
    window = PlateRecognitionGUI()
    # 显示主窗口
    window.show()
    # 运行应用程序
    sys.exit(app.exec())


if __name__ == '__main__':
    main()