#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO工具函数
用于加载YOLO模型和检测车牌区域
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO


class YOLOPlateDetector:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.45):
        """初始化YOLO车牌检测器
        
        Args:
            model_path: YOLO模型权重文件路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到YOLO模型文件: {model_path}")
        
        # 加载YOLO模型
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def detect_plates(self, image, return_image=False):
        """检测图像中的车牌区域
        
        Args:
            image: 输入图像 (numpy数组或图像路径)
            return_image: 是否返回绘制了检测结果的图像
            
        Returns:
            plates: 检测到的车牌区域列表，每个元素包含: [x1, y1, x2, y2, confidence]
            result_image: 如果return_image=True，返回绘制了检测结果的图像
        """
        # 读取图像（如果输入是路径）
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"无法读取图像: {image}")
        else:
            img = image.copy()
            
        # 执行检测
        results = self.model(img, conf=self.conf_threshold, iou=self.iou_threshold)
        
        # 提取检测结果
        plates = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # 获取置信度
                conf = box.conf[0].item()
                # 获取类别ID
                cls = box.cls[0].item()
                
                # 只保留车牌类别（类别ID为0）
                if cls == 0:
                    plates.append([int(x1), int(y1), int(x2), int(y2), conf])
                    
                    # 在图像上绘制边界框
                    if return_image:
                        # 绘制边界框
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        # 绘制置信度，使用PIL支持中文显示
                        label = f"车牌: {conf:.2f}"
                        try:
                            from PIL import Image, ImageDraw, ImageFont
                            # 将OpenCV图像转换为PIL图像
                            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(img_pil)
                            # 尝试加载中文字体
                            try:
                                # Windows系统默认字体
                                font = ImageFont.truetype("simhei.ttf", 16)
                            except:
                                # 如果找不到指定字体，使用默认字体
                                font = ImageFont.load_default()
                            # 绘制文本
                            draw.text((int(x1), int(y1) - 20), label, font=font, fill=(0, 255, 0))
                            # 将PIL图像转换回OpenCV图像
                            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                        except:
                            # 如果PIL不可用，回退到OpenCV绘制
                            cv2.putText(img, label, (int(x1), int(y1) - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if return_image:
            return plates, img
        else:
            return plates
    
    def crop_plate_images(self, image, plates):
        """从图像中裁剪出车牌区域
        
        Args:
            image: 输入图像 (numpy数组)
            plates: 检测到的车牌区域列表
            
        Returns:
            plate_images: 裁剪出的车牌图像列表
        """
        plate_images = []
        
        for plate in plates:
            x1, y1, x2, y2, _ = plate
            
            # 确保坐标在图像范围内
            h, w = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # 裁剪车牌区域
            plate_img = image[y1:y2, x1:x2]
            
            # 检查裁剪结果是否有效
            if plate_img.size > 0:
                plate_images.append(plate_img)
        
        return plate_images
    
    def detect_and_crop(self, image, return_image=False):
        """一站式检测并裁剪车牌
        
        Args:
            image: 输入图像 (numpy数组或图像路径)
            return_image: 是否返回绘制了检测结果的图像
            
        Returns:
            plate_images: 裁剪出的车牌图像列表
            result_image: 如果return_image=True，返回绘制了检测结果的图像
        """
        # 读取图像（如果输入是路径）
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"无法读取图像: {image}")
        else:
            img = image.copy()
        
        # 检测车牌
        plates, result_img = self.detect_plates(img, return_image=True)
        
        # 裁剪车牌图像
        plate_images = self.crop_plate_images(img, plates)
        
        if return_image:
            return plate_images, result_img
        else:
            return plate_images


# 测试函数
def test_yolo_detector(model_path, image_path):
    """测试YOLO车牌检测器
    
    Args:
        model_path: YOLO模型路径
        image_path: 测试图像路径
    """
    try:
        # 初始化检测器
        detector = YOLOPlateDetector(model_path)
        
        # 检测并显示结果
        plate_images, result_img = detector.detect_and_crop(image_path, return_image=True)
        
        print(f"检测到 {len(plate_images)} 个车牌")
        
        # 显示结果
        cv2.imshow("检测结果", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 显示裁剪出的车牌
        for i, plate_img in enumerate(plate_images):
            cv2.imshow(f"车牌 {i+1}", plate_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"测试过程中出错: {str(e)}")


if __name__ == '__main__':
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description='测试YOLO车牌检测器')
    parser.add_argument('--model', default='./runs/train/yolo_lpr/weights/best.pt', help='YOLO模型路径')
    parser.add_argument('--image', required=True, help='测试图像路径')
    
    args = parser.parse_args()
    test_yolo_detector(args.model, args.image)