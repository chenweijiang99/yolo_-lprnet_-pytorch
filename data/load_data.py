from torch.utils.data import *
from imutils import paths
import numpy as np
import random
import cv2
import os
from PIL import Image as PILImage

# 定义车牌字符集，包括汉字、数字、字母
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

# 创建字符到索引的映射字典，用于标签编码
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

class LPRDataLoader(Dataset):
    """
    车牌识别数据加载器
    继承自PyTorch Dataset类，用于加载和预处理车牌图像数据
    """
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None):
        """
        初始化数据加载器
        
        Args:
            img_dir: 图像目录列表
            imgSize: 图像目标尺寸 (width, height)
            lpr_max_len: 车牌最大长度
            PreprocFun: 图像预处理函数
        """
        self.img_dir = img_dir
        self.img_paths = []
        # 遍历所有目录，收集图像路径
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        random.shuffle(self.img_paths)  # 随机打乱图像路径
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        # 设置预处理函数
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        """返回数据集大小"""
        return len(self.img_paths)

    def __getitem__(self, index):
        """
        获取单个数据项（图像和标签）
        
        Args:
            index: 数据索引
            
        Returns:
            Image: 预处理后的图像
            label: 字符标签列表
            len(label): 标签长度
        """
        filename = self.img_paths[index]
        # 使用PIL库读取图像，更好地支持中文路径
        try:
            # 使用PIL读取图像
            pil_image = PILImage.open(filename)
            # 转换为OpenCV格式（BGR）
            Image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Warning: Cannot read image {filename}. Error: {e}")
            # 返回一张空白图像作为替代
            Image = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
        
        height, width, _ = Image.shape
        # 调整图像尺寸到目标大小
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        Image = self.PreprocFun(Image)

        # 解析文件名获取标签
        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        imgname = imgname.split("-")[0].split("_")[0]
        label = list()
        # 将字符转换为索引
        for c in imgname:
            label.append(CHARS_DICT[c])

        # 检查标签长度是否为8，并验证标签格式
        if len(label) == 8:
            if self.check(label) == False:
                print(imgname)
                assert 0, "Error label ^~^!!!"

        return Image, label, len(label)

    def transform(self, img):
        """
        图像预处理函数
        标准化图像像素值到[-1, 1]范围
        
        Args:
            img: 输入图像
            
        Returns:
            预处理后的图像
        """
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125  # 相当于除以127.5，使范围变为[-1, 1]
        img = np.transpose(img, (2, 0, 1))  # HWC转CHW格式

        return img

    def check(self, label):
        """
        检查车牌标签格式是否正确
        要求第3位和最后1位必须是'D'或'F'
        
        Args:
            label: 标签索引列表
            
        Returns:
            bool: 格式是否正确
        """
        if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
                and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
            print("标签错误，请检查!")
            return False
        else:
            return True
