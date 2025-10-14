#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCPD车牌提取工具
使用YOLO模型检测CCPD数据集中的车牌并裁剪保存为单独的图片，用于LPRNet训练

代码需要用到pybaseutils工具，请使用pip安装即可：pip install pybaseutils
"""

import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from pybaseutils import file_utils, image_utils
from yolo_utils import YOLOPlateDetector


class CCPDPlateExtractor:
    # 类属性：字符映射字典
    # 省份列表，索引从0开始
    PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", 
                "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
                "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
    
    # 地市列表，索引从0开始
    CITIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 
             'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
             'W', 'X', 'Y', 'Z', 'O']
    
    # 车牌号字典，索引从0开始
    PLATE_CHARS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 
                 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5',
                 '6', '7', '8', '9', 'O']
    
    def __init__(self, ccpd_root, output_dir, model_path, conf_threshold=0.5):
        """初始化CCPD车牌提取器
        
        Args:
            ccpd_root: CCPD数据集根目录
            output_dir: 裁剪后车牌图像的保存目录
            model_path: YOLO模型权重文件路径
            conf_threshold: 置信度阈值
        """
        self.ccpd_root = ccpd_root
        self.output_dir = output_dir
        self.detector = YOLOPlateDetector(model_path, conf_threshold=conf_threshold)
        
        # 创建输出目录
        os.makedirs(os.path.join(self.output_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'test'), exist_ok=True)
    
    def extract(self, train_ratio=0.8, vis=False):
        """从CCPD数据集中提取车牌
        
        Args:
            train_ratio: 训练集比例
            vis: 是否可视化处理结果
        """
        # 收集所有图像文件
        all_images = file_utils.get_images_list(self.ccpd_root)
        
        if not all_images:
            print(f"未找到任何图像文件，路径: {self.ccpd_root}")
            return
        
        # 划分训练集和测试集
        np.random.shuffle(all_images)
        split_idx = int(len(all_images) * train_ratio)
        train_images = all_images[:split_idx]
        test_images = all_images[split_idx:]
        
        print(f"找到 {len(all_images)} 张图像\n" \
              f"训练集: {len(train_images)} 张图像\n" \
              f"测试集: {len(test_images)} 张图像")
        
        # 提取训练集车牌
        print("提取训练集车牌...")
        self._extract_plates(train_images, 'train', vis)
        
        # 提取测试集车牌
        print("提取测试集车牌...")
        self._extract_plates(test_images, 'test', vis)
        
        print(f"车牌提取完成！\n" \
              f"车牌图像保存在: {self.output_dir}")
    
    def _extract_plates(self, image_paths, split, vis=False):
        """从指定的图像列表中提取车牌
        
        Args:
            image_paths: 图像路径列表
            split: 数据集类型 (train或test)
            vis: 是否可视化处理结果
        """
        for img_path in tqdm(image_paths, desc=f'提取{split}集车牌'):
            try:
                # 读取图像
                img = cv2.imread(img_path)
                if img is None:
                    print(f"无法读取图像: {img_path}")
                    continue
                
                # 检测车牌
                plates = self.detector.detect_plates(img)
                
                # 处理检测到的车牌
                if plates:
                    # 只取置信度最高的车牌
                    best_plate = max(plates, key=lambda x: x[4])
                    x1, y1, x2, y2, _ = best_plate
                    
                    # 裁剪车牌区域
                    # 稍微扩大裁剪区域，确保包含完整车牌
                    h, w = img.shape[:2]
                    expand_ratio = 0.05  # 扩大5%
                    expand_w = int((x2 - x1) * expand_ratio)
                    expand_h = int((y2 - y1) * expand_ratio)
                    
                    x1 = max(0, x1 - expand_w)
                    y1 = max(0, y1 - expand_h)
                    x2 = min(w, x2 + expand_w)
                    y2 = min(h, y2 + expand_h)
                    
                    plate_img = img[y1:y2, x1:x2]
                    
                    # 调整图像尺寸为94*24，这是LPRNet模型要求的输入尺寸
                    target_size = (94, 24)
                    plate_img = cv2.resize(plate_img, target_size)
                    
                    # 从CCPD文件名中提取车牌字符
                    img_name = os.path.basename(img_path)
                    
                    # 创建保存路径
                    save_dir = os.path.join(self.output_dir, split)
                    
                    # 解析车牌字符
                    plate_chars = self._get_plate_licenses(img_path)
                    
                    if plate_chars:
                        # 直接使用车牌字符作为文件名，重复则覆盖
                        save_filename = f"{plate_chars}.jpg"
                        save_path = os.path.join(save_dir, save_filename)
                        cv2.imwrite(save_path, plate_img)
                        
                        if vis:
                            print(f"保存车牌: {save_filename}")
                    else:
                        # 如果无法从文件名提取车牌信息，使用原始文件名的一部分
                        base_name = os.path.splitext(img_name)[0]
                        temp_name = base_name[:8]  # 取前8个字符
                        random_suffix = np.random.randint(0, 1000)
                        save_filename = f"temp_{temp_name}_{random_suffix}.jpg"
                        save_path = os.path.join(save_dir, save_filename)
                        cv2.imwrite(save_path, plate_img)
                    
                    # 可视化
                    if vis:
                        # 在原图上绘制检测结果
                        image_with_plate = img.copy()
                        cv2.rectangle(image_with_plate, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        if plate_chars:
                            cv2.putText(image_with_plate, plate_chars, (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.imshow(f"Detected Plate - {split}", image_with_plate)
                        cv2.waitKey(1)  # 短暂显示
            except Exception as e:
                print(f"处理文件 {img_path} 时出错: {str(e)}")
                continue
        
        if vis:
            cv2.destroyAllWindows()
    
    def _get_plate_licenses(self, image_file):
        """
        从CCPD图像文件名解析车牌字符
        
        Args:
            image_file: 图像文件路径
        
        Returns:
            str: 解析出的车牌字符，如果解析失败则返回None
        """
        try:
            # 解析文件名获取标注信息
            info = self._parser_annotations(image_file)
            if not info or not info.get("plates"):
                return None
            
            # 返回车牌字符
            return info["plates"][0]  # 假设每个图像只有一个车牌
        except Exception as e:
            print(f"解析车牌字符时出错: {str(e)}")
            return None
    
    def _parser_annotations(self, image_file):
        """
        解析CCPD图像文件名，提取标注信息
        
        Args:
            image_file: 图像文件路径
        
        Returns:
            dict: 包含标注信息的字典
        """
        filename = os.path.basename(image_file)
        try:
            annotations = filename.split("-")
            if len(annotations) < 5:
                return None
            
            # 从文件名提取车牌字符编码（第5部分）
            plate_code_part = annotations[4]
            plate_codes = plate_code_part.split("_")
            
            # 确保有足够的编码部分来组成车牌
            if len(plate_codes) < 7:
                return None
            
            try:
                # 解析省份索引（第一个数字）
                province_idx = int(plate_codes[0])
                # 解析地市索引（第二个数字）
                city_idx = int(plate_codes[1])
                # 解析后面的车牌字符索引
                plate_char_idxs = []
                for code in plate_codes[2:]:
                    try:
                        idx = int(code)
                        plate_char_idxs.append(idx)
                    except ValueError:
                        continue
                
                # 验证索引是否有效
                if (0 <= province_idx < len(self.PROVINCES) and 
                    0 <= city_idx < len(self.CITIES) and 
                    len(plate_char_idxs) >= 3):  # 至少需要3个字符以确保有效的车牌
                    
                    # 构建完整车牌号
                    plate_chars = (self.PROVINCES[province_idx] + 
                                  self.CITIES[city_idx])
                    
                    # 处理剩余字符，支持7位普通车牌和8位新能源车牌
                    for idx in plate_char_idxs:
                        if 0 <= idx < len(self.PLATE_CHARS):
                            plate_chars += self.PLATE_CHARS[idx]
                        
                    # 新能源车牌有8位字符，普通蓝牌有7位字符
                    # 这里不做特殊处理，直接返回所有解析到的字符
                    return {"filename": filename, "plates": [plate_chars]}
                else:
                    # 索引无效
                    return None
            except ValueError:
                # 编码转换失败
                return None
        except Exception as e:
            print(f"解析文件名时出错: {str(e)}")
            return None


def get_parser():
    parser = argparse.ArgumentParser(description='CCPD车牌提取工具')
    parser.add_argument('--ccpd_root', default='./data/CCPD/CCPD2020/ccpd_green/val', help='CCPD数据集根目录')
    parser.add_argument('--output_dir', default='./data/lprnet', help='裁剪后车牌图像的保存目录')
    parser.add_argument('--model_path', default='./weights/best.pt', help='YOLO模型权重文件路径')
    parser.add_argument('--train_ratio', default=0.8, type=float, help='训练集比例')
    parser.add_argument('--conf_threshold', default=0.5, type=float, help='置信度阈值')
    parser.add_argument('--vis', default=False, action='store_true', help='是否可视化处理结果')
    
    return parser


def main():
    args = get_parser().parse_args()
    
    # 确保目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    extractor = CCPDPlateExtractor(
        ccpd_root=args.ccpd_root,
        output_dir=args.output_dir,
        model_path=args.model_path,
        conf_threshold=args.conf_threshold
    )
    
    extractor.extract(train_ratio=args.train_ratio, vis=args.vis)


if __name__ == '__main__':
    main()