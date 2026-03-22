
# MNIST 手写数字分类器

使用 PyTorch 实现的 MNIST 手写数字分类器，专为深度学习零基础开发者设计。

## 项目简介

本项目通过实现一个简单的 MNIST 手写数字分类器，帮助零基础开发者学习深度学习的基本概念和 PyTorch 的使用。代码中包含详细的中文注释，解释每个概念的含义和作用。

## 学习目标

通过本项目，你将学到：
- ✅ PyTorch 核心概念：Tensor、Dataset、DataLoader
- ✅ 如何搭建简单的全连接神经网络
- ✅ 损失函数和优化器的使用
- ✅ 完整的训练循环流程
- ✅ 如何评估模型性能

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

程序会自动：
1. 下载 MNIST 数据集（如果不存在）
2. 训练模型
3. 在测试集上评估
4. 显示训练过程和样本预测

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

## 核心概念详解

代码中包含以下知识点的详细注释：

### 1. Tensor（张量）
PyTorch 的核心数据结构，类似 NumPy 数组，但可以在 GPU 上运行。

### 2. Dataset &amp; DataLoader
- **Dataset**: 表示数据集，负责加载单个样本
- **DataLoader**: 批量加载数据，支持 shuffle 和并行加载

### 3. 模型（Model）
继承自 `nn.Module`，定义网络结构和前向传播。

### 4. 损失函数（Loss Function）
衡量预测与真实值的差距，本项目使用 `CrossEntropyLoss`。

### 5. 优化器（Optimizer）
更新模型参数以最小化损失，本项目使用 `Adam`。

### 6. 训练循环
完整流程：前向传播 → 计算损失 → 反向传播 → 更新参数

## 学习建议

1. **先阅读代码注释**: 每个知识点都有详细的中文解释
2. **运行代码观察**: 实际运行看看每个步骤的输出
3. **尝试修改**: 调整网络结构、学习率等超参数，观察效果
4. **查看可视化**: 理解训练过程和模型预测

## 技术栈

- Python 3.x
- PyTorch 2.0+
- torchvision
- matplotlib

## 下一步

完成本项目后，可以继续学习 `../minGPT/` 项目，深入了解 Transformer 架构和 LLM 原理！
