

"""
CCPD数据集处理工具
用于将CCPD数据集转换为YOLO训练所需的格式
"""

import os
import argparse
import cv2
import numpy as np
import shutil
from tqdm import tqdm


class CCPDProcessor:
    def __init__(self, ccpd_root, output_dir):
        self.ccpd_root = ccpd_root
        self.output_dir = output_dir
        self.image_dir = os.path.join(output_dir, 'images')
        self.label_dir = os.path.join(output_dir, 'labels')
        
        # 创建输出目录
        os.makedirs(os.path.join(self.image_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.image_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.label_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.label_dir, 'val'), exist_ok=True)
        
    def process(self, train_ratio=0.8):
        """处理CCPD数据集，生成YOLO格式的标注文件"""
        # 收集所有图像文件
        all_images = []
        for root, _, files in os.walk(self.ccpd_root):
            for file in files:
                if file.endswith('.jpg'):
                    all_images.append(os.path.join(root, file))
        
        # 划分训练集和验证集
        np.random.shuffle(all_images)
        split_idx = int(len(all_images) * train_ratio)
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]
        
        print(f"找到 {len(all_images)} 张图像\n" \
              f"训练集: {len(train_images)} 张图像\n" \
              f"验证集: {len(val_images)} 张图像")
        
        # 处理训练集
        print("处理训练集...")
        self._process_images(train_images, 'train')
        
        # 处理验证集
        print("处理验证集...")
        self._process_images(val_images, 'val')
        
        print(f"数据集处理完成！\n" \
              f"图像保存在: {self.image_dir}\n" \
              f"标注文件保存在: {self.label_dir}")
        
    def _process_images(self, image_paths, split):
        """处理指定的图像列表"""
        for img_path in tqdm(image_paths, desc=f'处理{split}集'):
            try:
                # 解析文件名获取车牌信息
                img_name = os.path.basename(img_path)
                coords = self._parse_coordinates(img_name)
                
                if coords is None:
                    continue
                
                # 读取图像获取尺寸
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                
                # 计算YOLO格式的标注
                yolo_coords = self._convert_to_yolo_format(coords, w, h)
                
                # 复制图像到目标目录
                dest_img_path = os.path.join(self.image_dir, split, img_name)
                shutil.copy(img_path, dest_img_path)
                
                # 创建标注文件
                label_name = os.path.splitext(img_name)[0] + '.txt'
                label_path = os.path.join(self.label_dir, split, label_name)
                
                with open(label_path, 'w', encoding='utf-8') as f:
                    f.write(yolo_coords)
                    
            except Exception as e:
                print(f"处理文件 {img_path} 时出错: {str(e)}")
                continue
    
    def _parse_coordinates(self, img_name):
        """从CCPD文件名解析车牌坐标"""
        # CCPD文件名格式示例: 000257620317-90_113-265&349_447&404-447&404_275&414_265&349_438&339-0_0_22_26_27_20_30-124-23.jpg
        try:
            # 提取坐标部分
            parts = img_name.split('-')
            if len(parts) < 3:
                return None
            
            # 第二个部分是车牌坐标: 265&349_447&404
            coord_part = parts[2]
            points = coord_part.split('_')
            
            if len(points) < 2:
                return None
            
            # 获取四个角点坐标
            # 这里简化处理，使用矩形框而不是四边形
            # 提取左上和右下坐标
            x1, y1 = points[0].split('&')
            x2, y2 = points[1].split('&')
            
            return {
                'x1': int(x1),
                'y1': int(y1),
                'x2': int(x2),
                'y2': int(y2)
            }
            
        except Exception as e:
            print(f"解析文件名 {img_name} 时出错: {str(e)}")
            return None
    
    def _convert_to_yolo_format(self, coords, img_width, img_height):
        """将坐标转换为YOLO格式"""
        # YOLO格式: class_id x_center y_center width height
        # 所有值都归一化到0-1之间
        
        x_center = (coords['x1'] + coords['x2']) / 2.0 / img_width
        y_center = (coords['y1'] + coords['y2']) / 2.0 / img_height
        width = (coords['x2'] - coords['x1']) / img_width
        height = (coords['y2'] - coords['y1']) / img_height
        
        # 类别ID为0（车牌）
        return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"


def get_parser():
    parser = argparse.ArgumentParser(description='CCPD数据集处理工具')
    parser.add_argument('--ccpd_root', required=True, help='CCPD数据集根目录')
    parser.add_argument('--output_dir', default='./train', help='输出目录')
    parser.add_argument('--train_ratio', default=0.8, type=float, help='训练集比例')
    
    return parser


def main():
    args = get_parser().parse_args()
    
    processor = CCPDProcessor(args.ccpd_root, args.output_dir)
    processor.process(train_ratio=args.train_ratio)


if __name__ == '__main__':
    main()