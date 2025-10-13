#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用YOLOv8训练车牌检测模型
基于ultralytics库实现
"""

import os
import argparse
from ultralytics import YOLO


def get_parser():
    parser = argparse.ArgumentParser(description='训练YOLO车牌检测模型')
    parser.add_argument('--model', default='yolov8n.pt', help='预训练模型路径或名称')
    parser.add_argument('--config', default='./yolo_config.yaml', help='YOLO配置文件路径')
    parser.add_argument('--epochs', default=100, type=int, help='训练轮数')
    parser.add_argument('--batch_size', default=16, type=int, help='批次大小')
    parser.add_argument('--img_size', default=640, type=int, help='输入图像大小')
    parser.add_argument('--lr0', default=0.01, type=float, help='初始学习率')
    parser.add_argument('--device', default='', help='训练设备，留空自动选择')
    parser.add_argument('--name', default='yolo_lpr', help='训练结果保存名称')
    parser.add_argument('--project', default='runs/train', help='训练结果保存路径')
    parser.add_argument('--resume', default=False, action='store_true', help='是否从上次训练结果继续')
    
    return parser


def main():
    # 解析命令行参数
    args = get_parser().parse_args()
    
    print(f"开始训练YOLO车牌检测模型...")
    print(f"配置参数:\n" \
          f"  模型: {args.model}\n" \
          f"  配置文件: {args.config}\n" \
          f"  训练轮数: {args.epochs}\n" \
          f"  批次大小: {args.batch_size}\n" \
          f"  图像大小: {args.img_size}\n" \
          f"  初始学习率: {args.lr0}\n" \
          f"  保存名称: {args.name}\n" \
          f"  保存路径: {args.project}")
    
    # 初始化YOLO模型
    model = YOLO(args.model)
    
    # 训练模型
    try:
        results = model.train(
            data=args.config,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.img_size,
            lr0=args.lr0,
            device=args.device,
            name=args.name,
            project=args.project,
            resume=args.resume,
            exist_ok=True  # 如果保存目录已存在，继续训练
        )
        
        # 验证模型
        metrics = model.val()
        print(f"模型验证结果:\n" \
              f"  mAP50: {metrics.box.map50}\n" \
              f"  mAP50-95: {metrics.box.map}")
        
        # 导出模型为torchscript格式（PyTorch推荐的部署格式）
        model.export(format='torchscript')
        print(f"模型训练完成并已导出到 {args.project}/{args.name}/weights/")
        print(f"注意：PyTorch原生格式的模型已经保存在 {args.project}/{args.name}/weights/best.pt")
        
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        return False
    
    return True


if __name__ == '__main__':
    main()