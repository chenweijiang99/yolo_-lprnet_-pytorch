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

### 2.YOLO模型测试

```bash
python test_yolo.py --model ./runs/train/yolo_lpr/weights/best.pt --input ./images/test.jpg
```

参数说明：
- `--model`: 训练好的YOLO模型路径
- `--input`: 输入图像或视频路径
- `--conf`: 置信度阈值（默认0.5）
- `--iou`: IoU阈值（默认0.45）
- `--save`: 是否保存结果图像
- `--no-display`: 是否不显示结果图像（默认False）

### 3. LPRNet字符识别模型训练

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