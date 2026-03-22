
# MNIST 手写数字分类器

使用 PyTorch 实现的 MNIST 手写数字分类器

## 项目简介

实现一个简单的 MNIST 手写数字分类器

## 模型架构

使用 3 层全连接神经网络：

```
输入层 (784) 
    ↓
隐藏层 1 (128) + ReLU
    ↓
隐藏层 2 (64) + ReLU
    ↓
输出层 (10)
```

- 输入层：784 个神经元（28×28 像素展平）
- 隐藏层 1：128 个神经元，ReLU 激活
- 隐藏层 2：64 个神经元，ReLU 激活
- 输出层：10 个神经元（数字 0-9）

## 项目成果

- 测试集准确率：**97.5%**
- 目标准确率：≥ 95% ✅

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行项目

```bash
python main.py
```


### 3. 查看结果

- `training_history.png`: 训练过程的损失和准确率曲线
- `sample_images.png`: 样本图像及其预测结果

## 代码结构

```
mnist_classifier/
├── README.md              # 项目说明（本文件）
├── SPECIFICATION.md       # 详细规格说明
├── main.py                # 主程序（包含详细注释）
├── requirements.txt       # 项目依赖
├── training_history.png   # 训练历史图
├── sample_images.png      # 样本预测图
└── data/                  # MNIST 数据集（自动下载）
    └── MNIST/
```

## 技术栈

- Python 3.x
- PyTorch 2.0+
- torchvision
- matplotlib

