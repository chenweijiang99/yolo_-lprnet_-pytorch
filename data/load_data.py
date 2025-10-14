from torch.utils.data import *
from imutils import paths
import numpy as np
import random
import cv2
import os

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None):
        print('LPRDataLoader参数：测试图片路径', img_dir, '图片尺寸', imgSize, '车牌最大长度', lpr_max_len)
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
            
        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        # 确保路径格式在Windows上正确
        filename = os.path.normpath(filename)
        # print('~~~~~~~~~', filename)
        
        # 使用cv2.imdecode和numpy.fromfile替代cv2.imread，以支持中文路径
        try:
            # 读取图片数据
            img_data = np.fromfile(filename, dtype=np.uint8)
            # 解码图片
            Image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            
            # 检查是否成功读取图片
            if Image is None:
                print(f"无法读取图片: {filename}")
                # 使用一个默认的黑色图像替代
                Image = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
            else:
                height, width, _ = Image.shape
                if height != self.img_size[1] or width != self.img_size[0]:
                    Image = cv2.resize(Image, self.img_size)
        except Exception as e:
            print(f"读取图片时出错: {filename}, 错误: {e}")
            # 使用一个默认的黑色图像替代
            Image = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
        
        # print('~~~~~~~~~~', Image.shape)
        Image = self.PreprocFun(Image)
        # print('~~~~~~~~~~', Image.shape)

        basename = os.path.basename(filename)   # 返回不带路径的文件名（包含后缀）
        imgname, suffix = os.path.splitext(basename) 
        # print('~~~~~~~~~~imgname, suffix = ', imgname,';;;', suffix )
        imgname = imgname.split("-")[0].split("_")[0]
        # print('~~~~~~~~~', imgname)
        label = list()
        for c in imgname:
            # one_hot_base = np.zeros(len(CHARS))
            # one_hot_base[CHARS_DICT[c]] = 1
            try:
                label.append(CHARS_DICT[c])
            except Exception as e:
                print(imgname, e)
                label.append(CHARS_DICT[c])
        # print('~~~~~~~~~~label: ', label)

        # if len(label) == 8:
        # #     print('len(label) == 8: ', imgname)
        #     if self.check(label) == False:                
        #         assert 0, "Error label ^~^!!!"
        # print('~~~~~~~~~', Image, label, len(label))
        return Image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img

    def check(self, label):
        if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
                and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
            print("车牌第3个字符和最后一个字符不能为D或F")
            return False
        else:
            return True
