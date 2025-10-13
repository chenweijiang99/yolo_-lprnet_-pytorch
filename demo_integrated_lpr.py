#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
集成YOLO车牌检测和LPRNet字符识别的演示脚本
"""

import os
import cv2
import argparse
import torch
import numpy as np
from yolo_utils import YOLOPlateDetector

# 导入LPRNet相关模块
from model.LPRNet import build_lprnet
from data.load_data import CHARS, CHARS_DICT


def greedy_decode(prebs, CHARS):
    """贪婪解码函数，用于解码LPRNet模型的输出，参考test_LPRNet.py实现
    
    Args:
        prebs: 模型的输出预测结果 (PyTorch张量)
        CHARS: 字符集
        
    Returns:
        解码后的车牌字符串
    """
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
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        
        for c in preb_label[1:]:
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
    
    # 将数字标签转换为字符
    plate_str = ""
    for idx in no_repeat_blank_label:
        try:
            if 0 <= idx < len(CHARS):
                plate_str += CHARS[idx]
            else:
                plate_str += '?'
        except:
            plate_str += '?'
    
    # 如果识别结果为空，返回空字符串
    if not plate_str:
        plate_str = "无法识别"
        
    return plate_str

def load_lprnet_model(pretrained_model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """加载LPRNet模型
    
    Args:
        pretrained_model_path: 预训练模型路径
        device: 运行设备
        
    Returns:
        lprnet: 加载好的LPRNet模型
    """
    # 构建LPRNet模型
    lprnet = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0.5)
    lprnet.to(device)
    
    # 加载预训练权重
    if os.path.exists(pretrained_model_path):
        lprnet.load_state_dict(torch.load(pretrained_model_path, weights_only=True))
        print(f"成功加载LPRNet预训练模型: {pretrained_model_path}")
        lprnet.eval()  # 设置为评估模式
        return lprnet
    else:
        raise FileNotFoundError(f"未找到LPRNet预训练模型: {pretrained_model_path}")


def recognize_plate(lprnet, plate_image, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """使用LPRNet识别车牌字符，完全参考test_LPRNet.py中的实现方式
    
    Args:
        lprnet: LPRNet模型
        plate_image: 车牌图像
        device: 运行设备
        
    Returns:
        plate_text: 识别出的车牌文本
    """
    try:
        # 检查输入图像是否有效
        if plate_image is None or len(plate_image.shape) != 3:
            raise ValueError(f"无效的车牌图像，形状: {plate_image.shape if plate_image is not None else None}")
        
        # 图像预处理 - 完全按照test_LPRNet.py的数据加载方式，添加归一化步骤
        img = cv2.resize(plate_image, (94, 24))  # 调整大小为LPRNet输入尺寸
        img = img.astype('float32')
        img -= 127.5  # 归一化步骤1
        img *= 0.0078125  # 归一化步骤2
        img = img.transpose(2, 0, 1)  # 转换为[C, H, W]
        img = np.expand_dims(img, axis=0)  # 添加批次维度
        
        # 转换为PyTorch张量并移动到设备
        img_tensor = torch.from_numpy(img)
        if device == 'cuda' and torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        
        # 前向传播
        with torch.no_grad():
            prebs = lprnet(img_tensor)
        
        # 使用本地实现的贪婪解码函数进行解码
        plate_text = greedy_decode(prebs, CHARS)
        
        return plate_text
    except Exception as e:
        print(f"识别函数内部错误: {type(e).__name__}: {str(e)}")
        # 为了调试，打印更多信息
        print(f"  图像形状: {plate_image.shape if plate_image is not None else None}")
        print(f"  设备: {device}")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        # 发生错误时返回默认值，避免中断流程
        return "识别失败"


def process_image(image_path, yolo_detector, lprnet, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """处理单个图像，检测并识别车牌
    
    Args:
        image_path: 图像路径
        yolo_detector: YOLO检测器
        lprnet: LPRNet识别器
        device: 运行设备
        
    Returns:
        result_image: 绘制了检测和识别结果的图像
        plate_results: 车牌检测和识别结果列表
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    
    # 创建结果图像的副本
    result_image = image.copy()
    
    # 检测车牌坐标（不绘制任何内容）
    plate_coords = yolo_detector.detect_plates(image)
    
    # 识别每个车牌
    plate_results = []
    for i, plate_info in enumerate(plate_coords):
        x1, y1, x2, y2, conf = plate_info
        
        try:
            # 裁剪车牌区域
            plate_image = image[y1:y2, x1:x2]
            
            # 识别车牌字符
            plate_text = recognize_plate(lprnet, plate_image, device)
            plate_results.append({
                'coords': (x1, y1, x2, y2),
                'confidence': conf,
                'text': plate_text
            })
            
            # 使用PIL库在图像上统一绘制中文标签和边界框
            try:
                # 尝试使用PIL库绘制中文
                from PIL import Image, ImageDraw, ImageFont
                import numpy as np
                
                # 将OpenCV图像转换为PIL图像
                pil_image = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)
                
                # 尝试使用系统字体，如果失败则使用默认字体
                try:
                    # 对于Windows系统，使用中文字体
                    font = ImageFont.truetype("simhei.ttf", 16)  # 尝试加载黑体
                except:
                    try:
                        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 16)  # 尝试Windows字体路径
                    except:
                        font = ImageFont.load_default()  # 使用默认字体
                
                # 绘制车牌边框（使用绿色）
                draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)
                
                # 绘制文本（使用红色）
                label = f"车牌: {plate_text} ({conf:.2f})"
                draw.text((x1, y1 - 20), label, font=font, fill=(255, 0, 0))  # BGR转RGB，红色
                
                # 将PIL图像转换回OpenCV图像
                result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                # 如果PIL方法失败，回退到OpenCV方法
                print(f"PIL绘制中文失败，回退到OpenCV: {str(e)}")
                # 绘制边界框
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 绘制文本
                label = f"车牌: {plate_text} ({conf:.2f})"
                cv2.putText(result_image, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
            
        except Exception as e:
            print(f"识别车牌时出错: {str(e)}")
            continue
    
    return result_image, plate_results


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='集成YOLO车牌检测和LPRNet字符识别')
    parser.add_argument('--yolo_model', default='./runs/train/yolo_lpr/weights/best.pt', help='YOLO模型路径')
    parser.add_argument('--lpr_model', default='./weights/Final_LPRNet_model.pth', help='LPRNet模型路径')
    parser.add_argument('--image', required=True, help='测试图像路径')
    parser.add_argument('--save', default=False, action='store_true', help='是否保存结果图像')
    parser.add_argument('--conf_threshold', default=0.5, type=float, help='YOLO检测置信度阈值')
    parser.add_argument('--iou_threshold', default=0.45, type=float, help='YOLO检测IoU阈值')
    
    args = parser.parse_args()
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    try:
        # 初始化YOLO检测器
        yolo_detector = YOLOPlateDetector(
            args.yolo_model,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold
        )
        
        # 加载LPRNet模型
        lprnet = load_lprnet_model(args.lpr_model, device)
        
        # 处理图像
        print(f"处理图像: {args.image}")
        result_image, plate_results = process_image(args.image, yolo_detector, lprnet, device)
        
        # 显示结果
        print(f"检测到 {len(plate_results)} 个车牌")
        for i, result in enumerate(plate_results):
            x1, y1, x2, y2 = result['coords']
            print(f"车牌 {i+1}: {result['text']} (置信度: {result['confidence']:.2f}) 位置: ({x1}, {y1})-({x2}, {y2})")
        
        # 显示处理后的图像
        cv2.imshow("车牌检测与识别结果", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存结果图像
        if args.save:
            save_path = os.path.splitext(args.image)[0] + '_result.jpg'
            cv2.imwrite(save_path, result_image)
            print(f"结果图像已保存到: {save_path}")
            
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")


if __name__ == '__main__':
    main()