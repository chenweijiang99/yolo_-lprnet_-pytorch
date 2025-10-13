import torch.nn as nn
import torch

class small_basic_block(nn.Module):
    """
    小型基础块，用于构建网络的基本组件
    包含一系列卷积层和激活函数，逐步改变通道数
    """
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),  # 1x1 卷积减少通道数至 ch_out//4
            nn.ReLU(),                                     # ReLU 激活函数
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),  # 垂直方向 3x1 卷积
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),  # 水平方向 1x3 卷积
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),  # 1x1 卷积恢复到原始输出通道数
        )
    
    def forward(self, x):
        return self.block(x)  # 正向传播


class LPRNet(nn.Module):
    """
    车牌识别网络 (LPRNet)
    使用小型基本块构建主干网络，并通过全局上下文增强特征表示
    """
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.phase = phase               # 训练阶段标志 ("train" 或其他)
        self.lpr_max_len = lpr_max_len   # 最大车牌长度
        self.class_num = class_num       # 类别数量（字符种类数）
        
        # 主干网络结构定义
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),  # 输入图像为三通道 RGB 图像
            nn.BatchNorm2d(num_features=64),  # 批归一化
            nn.ReLU(),                        # 激活函数
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),  # 空间池化操作
            
            small_basic_block(ch_in=64, ch_out=128),    # 第一个小型基本块
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),  # 下采样
            
            small_basic_block(ch_in=64, ch_out=256),    # 第二个小型基本块（注意输入通道是64而非128）
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            
            small_basic_block(ch_in=256, ch_out=256),   # 第三个小型基本块
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 进一步下采样
            
            nn.Dropout(dropout_rate),                   # Dropout 层防止过拟合
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 特征图尺寸调整
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),  # 输出类别映射
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),
        )

        # 容器模块，整合不同层级特征并预测最终结果
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448 + self.class_num, out_channels=self.class_num, 
                      kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, x):
        keep_features = list()  # 存储中间特征用于后续融合
        
        # 遍历主干网络每一层，提取特定位置的特征
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:  # 保存关键层输出
                keep_features.append(x)

        global_context = list()
        # 对每个保存下来的特征进行处理以获得全局上下文信息
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)  # 平均池化缩小尺寸
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
                
            f_pow = torch.pow(f, 2)      # 平方计算能量
            f_mean = torch.mean(f_pow)   # 全局平均值
            f = torch.div(f, f_mean)     # 归一化
            
            global_context.append(f)

        # 合并所有全局上下文特征
        x = torch.cat(global_context, 1)
        x = self.container(x)           # 应用容器模块
        logits = torch.mean(x, dim=2)   # 在高度维度上取平均作为最终输出
        
        return logits


def build_lprnet(lpr_max_len=8, phase=False, class_num=66, dropout_rate=0.5):
    """
    构建 LPRNet 实例
    
    参数:
        lpr_max_len: 车牌最大长度，默认为8
        phase: 当前模式 ("train" 表示训练模式)
        class_num: 字符类别总数，默认为66
        dropout_rate: Dropout比率，默认为0.5
    
    返回:
        已设置好运行模式的 LPRNet 实例
    """
    Net = LPRNet(lpr_max_len, phase, class_num, dropout_rate)

    if phase == "train":
        return Net.train()  # 设置为训练模式
    else:
        return Net.eval()   # 设置为评估模式