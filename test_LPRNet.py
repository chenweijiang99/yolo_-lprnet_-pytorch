
'''
测试LPRNet模型
'''

# 导入必要的库和模块
from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from PIL import Image, ImageDraw, ImageFont
from model.LPRNet import build_lprnet
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import cv2
import os

def get_parser():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='训练网络的参数')
    parser.add_argument('--img_size', default=[94, 24], help='图像尺寸')
    parser.add_argument('--test_img_dirs', default="./data/lprnet/test", help='测试图像路径')
    parser.add_argument('--dropout_rate', default=0, help='dropout率')
    parser.add_argument('--lpr_max_len', default=8, help='车牌号最大长度')
    parser.add_argument('--test_batch_size', default=100, help='测试批次大小')
    parser.add_argument('--phase_train', default=False, type=bool, help='训练或测试阶段标志')
    parser.add_argument('--num_workers', default=8, type=int, help='数据加载使用的工作进程数')
    parser.add_argument('--cuda', default=True, type=bool, help='是否使用cuda训练模型')
    parser.add_argument('--show', default=False, type=bool, help='是否显示测试图像及其预测结果')
    parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='预训练基础模型')

    args = parser.parse_args()
    return args

def collate_fn(batch):
    """
    自定义批处理函数，用于处理不同长度的标签
    
    Args:
        batch: 一批数据样本
        
    Returns:
        tuple: 处理后的图像、标签和长度信息
    """
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def test():
    """
    主测试函数
    """
    args = get_parser()

    # 构建LPRNet模型
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, 
                         class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("成功构建网络!")

    # 加载预训练模型
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model, weights_only=True))
        print("加载预训练模型成功!")
    else:
        print("[错误] 找不到预训练模型，请检查!")
        return False

    # 准备测试数据集
    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)
    try:
        Greedy_Decode_Eval(lprnet, test_dataset, args)
    finally:
        cv2.destroyAllWindows()

def Greedy_Decode_Eval(Net, datasets, args):
    """
    贪心解码评估函数
    
    Args:
        Net: 训练好的LPRNet模型
        datasets: 测试数据集
        args: 命令行参数
    """
    # 计算测试批次数量
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, 
                                   num_workers=args.num_workers, collate_fn=collate_fn))

    # 初始化统计变量
    Tp = 0      # 正确预测数量
    Tn_1 = 0    # 长度不匹配的错误数量
    Tn_2 = 0    # 长度匹配但内容错误的数量
    t1 = time.time()
    
    for i in range(epoch_size):
        # 加载测试数据
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets])
        imgs = images.numpy().copy()

        # 将数据移到GPU（如果可用）
        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # 前向传播
        prebs = Net(images)
        # 贪心解码
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            
            # 去除重复字符和空白字符
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        
        # 计算准确率
        for i, label in enumerate(preb_labels):
            # 显示图像和预测结果（如果启用）
            if args.show:
                show(imgs[i], label, targets[i])
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1
    
    # 输出测试结果
    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print("[信息] 测试准确率: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    t2 = time.time()
    print("[信息] 测试速度: {}秒 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))

def show(img, label, target):
    """
    显示测试图像及其预测结果
    
    Args:
        img: 测试图像
        label: 预测标签
        target: 真实标签
    """
    # 图像反归一化
    img = np.transpose(img, (1, 2, 0))
    img *= 128.
    img += 127.5
    img = img.astype(np.uint8)

    # 将标签转换为字符串
    lb = ""
    for i in label:
        lb += CHARS[i]
    tg = ""
    for j in target.tolist():
        tg += CHARS[int(j)]

    # 判断预测是否正确
    flag = "F"
    if lb == tg:
        flag = "T"
    
    # 在图像上添加文本并显示
    img = cv2ImgAddText(img, lb, (0, 0))
    cv2.imshow("test", img)
    print("目标: ", tg, " ### {} ### ".format(flag), "预测: ", lb)
    cv2.waitKey()
    cv2.destroyAllWindows()

def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    """
    在OpenCV图像上添加中文文本
    
    Args:
        img: OpenCV图像
        text: 要添加的文本
        pos: 文本位置
        textColor: 文本颜色
        textSize: 文本大小
        
    Returns:
        添加文本后的图像
    """
    if (isinstance(img, np.ndarray)):  # 检测是否为OpenCV格式
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def Greedy_Decode(prebs, CHARS):
    """
    贪心解码函数，用于解码单个图像的预测结果
    
    Args:
        prebs: 模型的输出预测结果
        CHARS: 字符集
        
    Returns:
        解码后的车牌字符串
    """
    prebs = prebs.cpu().detach().numpy()
    preb_label = list()
    for j in range(prebs.shape[1]):
        preb_label.append(np.argmax(prebs[:, j], axis=0))
    
    # 去除重复字符和空白字符
    no_repeat_blank_label = list()
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
        plate_str += CHARS[idx]
    
    return plate_str

if __name__ == "__main__":
    test()