
# AI Learning Practice

AI 学习实践项目 - 从零基础开始学习深度学习与人工智能

## 项目简介

这是一个深度学习与 AI 零基础学习项目，通过两个完整的实践项目，帮助开发者系统地学习深度学习基础知识、Transformer 架构和 LLM 原理。

## 学习路径

### 第一阶段：MNIST 手写数字分类器
- **项目位置**: `mnist_classifier/`
- **学习目标**: 掌握 PyTorch 基础和全连接神经网络
- **核心知识**:
  - Tensor（张量）
  - Dataset &amp; DataLoader（数据集与数据加载器）
  - 全连接神经网络（FCNN）
  - Loss Function（损失函数）
  - Optimizer（优化器）
  - 训练循环
- **成果**: 3层全连接网络，测试集准确率达 97.5%

### 第二阶段：minGPT - 小型 GPT 实现
- **项目位置**: `minGPT/`
- **学习目标**: 理解 Transformer 架构和 LLM 原理
- **核心知识**:
  - 多头自注意力（Multi-Head Self-Attention）
  - 因果掩码（Causal Mask）
  - 残差连接（Residual Connections）
  - 层归一化（Layer Normalization）
  - 位置编码（Positional Encoding）
  - 字符级语言模型
  - 自回归文本生成
- **成果**: 完整的解码器-only Transformer，可在 Shakespeare 数据集上训练

## 环境要求

- Python 3.11+
- PyTorch 2.0+
- CUDA 11.8+（如需 GPU 加速）

**注意**: 对于 RTX 50 系列显卡，需要使用 PyTorch Nightly 版本以支持 sm_120 计算架构。详见各项目的 GPU 配置文档。

## 项目结构

```
ai-learning-practice/
├── README.md              # 项目总览（本文件）
├── mnist_classifier/      # 第一阶段：MNIST 手写数字分类器
│   ├── README.md
│   ├── main.py
│   ├── requirements.txt
│   ├── data/
│   └── ...
└── minGPT/               # 第二阶段：minGPT 实现
    ├── README.md
    ├── model.py
    ├── dataset.py
    ├── trainer.py
    ├── requirements.txt
    ├── data/
    └── ...
```

## 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/Chao2lyy/ai-learning-practice.git
cd ai-learning-practice
```

### 2. 开始学习
建议按照顺序学习：
1. 先进入 `mnist_classifier/` 学习 PyTorch 基础
2. 再进入 `minGPT/` 学习 Transformer 和 LLM

每个项目都有详细的 README 和代码注释，帮助你理解每一个概念。

## 学习资源

- [PyTorch 官方文档](https://pytorch.org/docs/stable/)
- [Andrej Karpathy 的 minGPT](https://github.com/karpathy/minGPT)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 许可证

MIT License

## 致谢

感谢 Andrej Karpathy 的 minGPT 项目，为本项目提供了重要参考。
