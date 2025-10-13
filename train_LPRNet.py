'''
LPRNet车牌识别模型训练
基于Pytorch实现
'''

# 导入必要的库和模块
from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
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
import os

def sparse_tuple_for_ctc(T_length, lengths):
    """
    为CTC损失函数准备输入长度和目标长度元组
    
    Args:
        T_length: 时间步长（序列长度）
        lengths: 目标序列的实际长度列表
        
    Returns:
        tuple: (输入长度元组, 目标长度元组)
    """
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)

def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    """
    根据当前epoch动态调整学习率
    
    Args:
        optimizer: 优化器
        cur_epoch: 当前epoch
        base_lr: 基础学习率
        lr_schedule: 学习率调度计划
        
    Returns:
        float: 调整后的学习率
    """
    lr = 0
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)  # 每个阶段学习率递减10倍
            break
    if lr == 0:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def get_parser():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='LPRNet车牌识别模型训练')
    parser.add_argument('--max_epoch', default=15, type=int, help='训练轮数')
    parser.add_argument('--img_size', default=[94, 24], help='输入图像尺寸')
    parser.add_argument('--train_img_dirs', default="./data/lprnet/train", help='训练图像目录')
    parser.add_argument('--test_img_dirs', default="./data/lprnet/test", help='测试图像目录')
    parser.add_argument('--dropout_rate', default=0.5, help='Dropout率')
    parser.add_argument('--learning_rate', default=0.1, help='学习率')
    parser.add_argument('--lpr_max_len', default=8, help='车牌最大长度')
    parser.add_argument('--train_batch_size', default=128, help='训练批量大小')
    parser.add_argument('--test_batch_size', default=120, help='测试批量大小')
    parser.add_argument('--phase_train', default=True, type=bool, help='训练或测试阶段标志')
    parser.add_argument('--num_workers', default=8, type=int, help='数据加载时使用的工作线程数')
    parser.add_argument('--cuda', default=True, type=bool, help='是否使用CUDA加速训练')
    parser.add_argument('--resume_epoch', default=0, type=int, help='从指定轮数继续训练')
    parser.add_argument('--save_interval', default=2000, type=int, help='保存模型状态字典的间隔')
    parser.add_argument('--test_interval', default=2000, type=int, help='评估模型的间隔')
    parser.add_argument('--momentum', default=0.9, type=float, help='动量因子')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='权重衰减因子')
    parser.add_argument('--lr_schedule', default=[4, 8, 12, 14, 16], help='学习率衰减 schedule')
    parser.add_argument('--save_folder', default='./weights/', help='模型保存目录')
    parser.add_argument('--pretrained_model', default='', help='预训练模型路径')

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
    labels = np.asarray(labels).flatten().astype(np.int_)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def train():
    """
    主训练函数
    """
    args = get_parser()

    T_length = 18 # args.lpr_max_len
    epoch = 0 + args.resume_epoch
    loss_val = 0

    # 创建模型保存目录
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    # 构建LPRNet模型
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, 
                         class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("成功构建网络!")

    # 加载预训练模型或初始化权重
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model, weights_only=True))
        print("成功加载预训练模型!")
    else:
        # 权重初始化函数
        def xavier(param):
            nn.init.xavier_uniform(param)

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = xavier(1)
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0.01

        lprnet.backbone.apply(weights_init)
        lprnet.container.apply(weights_init)
        print("成功初始化网络权重!")

    # 配置优化器（使用RMSprop优化器）
    optimizer = optim.RMSprop(lprnet.parameters(), lr=args.learning_rate, alpha = 0.9, eps=1e-08,
                         momentum=args.momentum, weight_decay=args.weight_decay)
    
    # 准备训练和测试数据集
    train_img_dirs = os.path.expanduser(args.train_img_dirs)
    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    train_dataset = LPRDataLoader(train_img_dirs.split(','), args.img_size, args.lpr_max_len)
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)

    # 计算训练参数
    epoch_size = len(train_dataset) // args.train_batch_size
    max_iter = args.max_epoch * epoch_size

    # 定义CTC损失函数
    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'

    # 设置起始迭代次数
    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    # 开始训练循环
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # 每个epoch开始时创建新的数据迭代器
            batch_iterator = iter(DataLoader(train_dataset, args.train_batch_size, shuffle=True, 
                                           num_workers=args.num_workers, collate_fn=collate_fn))
            loss_val = 0
            epoch += 1

        # 定期保存模型
        if iteration !=0 and iteration % args.save_interval == 0:
            torch.save(lprnet.state_dict(), args.save_folder + 'LPRNet_' + '_iteration_' + repr(iteration) + '.pth')

        # 定期测试模型性能
        if (iteration + 1) % args.test_interval == 0:
            Greedy_Decode_Eval(lprnet, test_dataset, args)
            # lprnet.train() # should be switch to train mode

        start_time = time.time()
        # 加载训练数据
        images, labels, lengths = next(batch_iterator)
        # 为CTC损失函数准备参数
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
        # 更新学习率
        lr = adjust_learning_rate(optimizer, epoch, args.learning_rate, args.lr_schedule)

        # 将数据移到GPU（如果可用）
        if args.cuda:
            images = Variable(images, requires_grad=False).cuda()
            labels = Variable(labels, requires_grad=False).cuda()
        else:
            images = Variable(images, requires_grad=False)
            labels = Variable(labels, requires_grad=False)

        # 前向传播
        logits = lprnet(images)
        log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
        log_probs = log_probs.log_softmax(2).requires_grad_()
        
        # 反向传播
        optimizer.zero_grad()
        loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        if loss.item() == np.inf:
            continue
        loss.backward()
        optimizer.step()
        loss_val += loss.item()
        end_time = time.time()
        
        # 定期打印训练信息
        if iteration % 20 == 0:
            print('轮次:' + repr(epoch) + ' || 轮次迭代: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| 总迭代数 ' + repr(iteration) + ' || 损失: %.4f||' % (loss.item()) +
                  '批次时间: %.4f 秒 ||' % (end_time - start_time) + '学习率: %.8f' % (lr))
    
    # 最终测试
    print("最终测试准确率:")
    Greedy_Decode_Eval(lprnet, test_dataset, args)

    # 保存最终模型参数
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save(lprnet.state_dict(), args.save_folder + f'Final_LPRNet_model_{timestamp}.pth')

def Greedy_Decode_Eval(Net, datasets, args):
    """
    贪心解码评估函数，在测试集上评估模型性能
    
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

if __name__ == "__main__":
    train()