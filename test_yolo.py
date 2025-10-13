#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO车牌检测模型测试脚本
用于测试YOLO模型是否可以准确检测车牌位置
"""

import os
import cv2
import argparse
import numpy as np
from yolo_utils import YOLOPlateDetector


def process_single_image(detector, image_path, save_dir=None):
    """处理单张图像并检测车牌
    
    Args:
        detector: YOLOPlateDetector实例
        image_path: 图像路径
        save_dir: 保存结果的目录路径，如果为None则不保存
        
    Returns:
        dict: 包含检测结果的字典
    """
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告: 无法读取图像 {image_path}")
            return None
        
        # 检测车牌
        plates, result_image = detector.detect_plates(image, return_image=True)
        
        # 记录结果
        result = {
            'image_path': image_path,
            'plate_count': len(plates),
            'plates': plates,
            'result_image': result_image
        }
        
        print(f"在 {os.path.basename(image_path)} 中检测到 {len(plates)} 个车牌")
        
        # 保存结果图像
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"detected_{os.path.basename(image_path)}")
            cv2.imwrite(save_path, result_image)
            print(f"结果已保存至 {save_path}")
            
        return result
        
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {str(e)}")
        return None


def process_directory(detector, dir_path, save_dir=None):
    """处理目录中的所有图像
    
    Args:
        detector: YOLOPlateDetector实例
        dir_path: 图像目录路径
        save_dir: 保存结果的目录路径，如果为None则不保存
        
    Returns:
        list: 包含所有图像检测结果的列表
    """
    results = []
    
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # 遍历目录中的所有文件
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        
        # 只处理图像文件
        if os.path.isfile(file_path) and \
           any(filename.lower().endswith(ext) for ext in image_extensions):
            
            result = process_single_image(detector, file_path, save_dir)
            if result:
                results.append(result)
    
    return results


def visualize_results(results, display=True):
    """可视化检测结果
    
    Args:
        results: 检测结果列表
        display: 是否显示结果图像
    """
    if not results:
        print("没有检测结果可供可视化")
        return
    
    # 统计信息
    total_images = len(results)
    total_plates = sum(result['plate_count'] for result in results)
    
    print(f"\n=== 检测统计信息 ===")
    print(f"总处理图像数: {total_images}")
    print(f"总检测车牌数: {total_plates}")
    print(f"平均每图像车牌数: {total_plates/total_images:.2f} 个")
    
    # 显示结果图像
    if display:
        for result in results:
            # 调整图像大小以便显示
            h, w = result['result_image'].shape[:2]
            max_size = 800
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                display_image = cv2.resize(result['result_image'], new_size)
            else:
                display_image = result['result_image']
            
            # 显示图像
            window_name = f"检测结果: {os.path.basename(result['image_path'])} ({result['plate_count']}个车牌)"
            cv2.imshow(window_name, display_image)
        
        print("按任意键关闭所有窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='YOLO车牌检测模型测试脚本')
    parser.add_argument('--model', default='./weights/best.pt', 
                      help='YOLO模型权重文件路径')
    parser.add_argument('--input', required=True, 
                      help='输入图像文件或目录路径')
    parser.add_argument('--conf', type=float, default=0.5, 
                      help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, 
                      help='IoU阈值')
    parser.add_argument('--save', type=str, default=None, 
                      help='保存结果图像的目录路径')
    parser.add_argument('--no-display', action='store_true', 
                      help='不显示结果图像')
    
    args = parser.parse_args()
    
    try:
        # 初始化YOLO检测器
        print(f"正在加载YOLO模型: {args.model}")
        detector = YOLOPlateDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        print("YOLO模型加载成功")
        
        # 处理输入
        if os.path.isfile(args.input):
            # 处理单张图像
            print(f"正在处理图像: {args.input}")
            result = process_single_image(detector, args.input, args.save)
            results = [result] if result else []
            
        elif os.path.isdir(args.input):
            # 处理目录中的所有图像
            print(f"正在处理目录: {args.input}")
            results = process_directory(detector, args.input, args.save)
            
        else:
            print(f"错误: 输入路径 {args.input} 不存在")
            return
        
        # 可视化结果
        visualize_results(results, not args.no_display)
        
    except Exception as e:
        print(f"测试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()