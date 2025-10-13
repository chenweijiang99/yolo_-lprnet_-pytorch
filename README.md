# LPRNet_Pytorch
# 中文车牌识别系统（YOLO+LPRNet）

一个基于深度学习的中文车牌识别系统，集成了YOLO目标检测和LPRNet字符识别技术，能够从图像中自动检测并识别车牌信息。

完全适用于中国车牌识别（Chinese License Plate Recognition）及国外车牌识别！
目前支持同时识别蓝牌和绿牌即新能源车牌等中国车牌，但可通过扩展训练数据或微调支持其他类型车牌及提高识别准确率！

## 项目结构

```
LPRNet_Pytorch/
├── data/                 # 数据目录
│   ├── CCPD/             # CCPD数据集处理后的YOLO格式数据
│   └── load_data.py      # 数据加载模块
├── model/                # 模型定义目录
│   └── LPRNet.py         # LPRNet模型定义
├── weights/              # 模型权重存储目录
├── train_LPRNet.py       # LPRNet训练脚本
├── test_LPRNet.py        # LPRNet测试脚本
├── train_yolo.py         # YOLO训练脚本
├── prepare_ccpd_data.py  # CCPD数据集处理工具
├── yolo_utils.py         # YOLO工具函数
├── demo_integrated_lpr.py # 集成演示脚本
├── yolo_config.yaml      # YOLO配置文件
├── requirements.txt      # 项目依赖
└── README.md             # 项目说明文档
```

## 依赖安装

使用以下命令安装所有必要的依赖：

```bash
python  install_dependencies.py
```

主要依赖包括：
- torch >= 1.8.0 (PyTorch深度学习框架)
- torchvision >= 0.9.0 (PyTorch视觉库)
- opencv-python >= 4.5.0 (图像处理)
- numpy >= 1.19.0 (数值计算)
- pillow >= 8.0.0 (图像处理，支持中文路径)
- ultralytics >= 8.0.0 (YOLO模型实现)
- matplotlib >= 3.3.0 (可视化)
- pandas >= 1.1.0 (数据处理)
- tqdm >= 4.60.0 (进度条)
- PySide6 >= 6.4.0 (Qt6界面库)
## 数据集准备

本项目使用CCPD（中国城市停车数据集）进行YOLO车牌检测模型的训练。首先需要下载CCPD数据集，然后使用提供的工具脚本将其转换为YOLO训练所需的格式。

### CCPD数据集处理

```bash
python prepare_ccpd_data.py --ccpd_root ./data/CCPD/CCPD2020/ccpd_green --output_dir ./data/yolo --train_ratio 0.8
```

参数说明：
- `--ccpd_root`: CCPD数据集的根目录
- `--output_dir`: 处理后数据的输出目录
- `--train_ratio`: 训练集比例

## 模型训练

### 1. YOLO车牌检测模型训练

```bash
python train_yolo.py --model yolov8n.pt --config ./yolo_config.yaml --epochs 5 --batch_size 16 --img_size 640
```

参数说明：
- `--model`: 预训练模型路径或名称（支持yolov8n、yolov8s、yolov8m等）
- `--config`: YOLO配置文件路径
- `--epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--img_size`: 输入图像大小
- `--lr0`: 初始学习率
- `--device`: 训练设备（留空自动选择GPU或CPU）
- `--name`: 训练结果保存名称
- `--project`: 训练结果保存路径
- `--resume`: 是否从上次训练结果继续

训练完成后，模型权重将保存在`./runs/train/yolo_lpr/weights/`目录下。

### 2. LPRNet字符识别模型训练

LPRNet的训练脚本已经存在于项目中，可以使用以下命令进行训练：

```bash
python train_LPRNet.py --train_img_dirs ./data/train --test_img_dirs ./data/test --pretrained_model ./weights/Final_LPRNet_model.pth
```

## 模型测试与演示

### 1. 集成演示（YOLO检测 + LPRNet识别）

```bash
python demo_integrated_lpr.py --yolo_model ./runs/train/yolo_lpr/weights/best.pt --lpr_model ./weights/Final_LPRNet_model.pth --image ./images/test.jpg --save
```

参数说明：
- `--yolo_model`: YOLO模型权重路径
- `--lpr_model`: LPRNet模型权重路径
- `--image`: 测试图像路径
- `--save`: 是否保存结果图像
- `--conf_threshold`: YOLO检测置信度阈值
- `--iou_threshold`: YOLO检测IoU阈值

### 2. LPRNet单独测试

```bash
python test_LPRNet.py --pretrained_model ./weights/Final_LPRNet_model.pth --test_img_dirs ./data/test
```
## UI界面

项目提供了一个简单的用户界面，用于实时车牌识别。用户可以通过界面上传图像或开启摄像头进行实时检测。

### 运行UI界面

```bash
python main.py
```

## 技术说明

### YOLO车牌检测

- 使用ultralytics库实现的YOLOv8模型进行车牌区域检测
- 支持实时检测和批量处理
- 检测结果包含车牌位置和置信度

### LPRNet字符识别

- 采用端到端的LPRNet模型进行字符识别
- 无需进行复杂的字符分割
- 支持识别中文、字母和数字

### 集成流程

1. 使用YOLO模型检测输入图像中的车牌区域
2. 从图像中裁剪出车牌区域
3. 将裁剪出的车牌图像送入LPRNet进行字符识别
4. 输出车牌位置和识别结果

## 性能

### LPRNet性能

- 测试数据集包含蓝牌和绿牌
- 测试图像数量为27320张
- 模型大小仅为1.7M

|  模型大小 | 测试准确率(%) | GTX 1060推理时间(ms) |
| ------ | --------------------- | ---------------------- |
|  1.7M  |         96.0+         |          0.5-          |

## 注意事项

1. 确保项目依赖已正确安装
2. 训练前需要准备好CCPD数据集
3. 首次训练时可能需要下载YOLO预训练权重
4. 对于中文路径问题，项目已使用PIL库进行支持
5. 模型在不同环境中的表现可能会有所差异，可通过调整参数进行优化

## 常见问题

### 1. 找不到模型文件
确保模型路径正确，训练完成后YOLO模型默认保存在`./runs/train/yolo_lpr/weights/best.pt`

### 2. 检测效果不佳
可尝试调整`--conf_threshold`和`--iou_threshold`参数，或增加训练轮数

### 3. 中文路径问题
项目已使用PIL库替代cv2.imread来解决中文路径问题

### 4. CUDA内存不足
可尝试减小`--batch_size`或`--img_size`参数

## 更新日志

- 2023-XX-XX: 集成YOLOv8车牌检测
- 2023-XX-XX: 添加CCPD数据集处理工具
- 2023-XX-XX: 创建集成演示脚本

## 参考资料

1. [LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/abs/1806.10447v1)
2. [PyTorch中文文档](https://pytorch-cn.readthedocs.io/zh/latest/)
3. [Ultralytics YOLO官方文档](https://docs.ultralytics.com/)

## 说明

如果您觉得这个项目有用，请给我一个star，谢谢！
