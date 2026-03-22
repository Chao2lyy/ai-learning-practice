
# minGPT - 小型 GPT 实现

参照 Andrej Karpathy 的 minGPT 项目，从零开始实现的简化版 GPT（Generative Pre-trained Transformer），帮助理解 Transformer 架构和 LLM 原理。

## 项目简介

本项目实现了一个完整的解码器-only Transformer 模型，使用 Shakespeare 数据集进行训练，可以生成类似莎士比亚风格的文本。代码中包含详细的中文注释，解释每个 Transformer 组件的作用。

## 学习目标

通过本项目，你将学到：
- ✅ Transformer 解码器架构
- ✅ 多头自注意力机制（Multi-Head Self-Attention）
- ✅ 因果掩码（Causal Mask）
- ✅ 残差连接（Residual Connections）
- ✅ 层归一化（Layer Normalization）
- ✅ 位置编码（Positional Encoding）
- ✅ 字符级语言模型
- ✅ 自回归文本生成

## 项目成果

- 完整的解码器-only Transformer 实现
- 可在 Shakespeare 数据集上训练
- 支持自回归文本生成
- 已训练模型：`shakespeare_model.pth`

## 快速开始

### 1. 创建并激活虚拟环境（推荐）

```bash
# 使用 conda
conda create -n mingpt_env python=3.11
conda activate mingpt_env
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**: 对于 RTX 50 系列显卡，需要使用 PyTorch Nightly 版本。详见 `GPU_SETUP.md`。

### 3. 运行演示

```bash
# 检查 CUDA
python check_cuda.py

# 运行 Shakespeare 训练和生成演示
python shakespeare_demo.py
```

## 项目结构

```
minGPT/
├── README.md              # 项目说明（本文件）
├── SETUP.md               # 环境设置指南
├── GPU_SETUP.md           # GPU 配置说明
├── model.py               # GPT 模型定义（核心代码）
├── dataset.py             # 数据集处理
├── trainer.py             # 训练器
├── shakespeare_demo.py    # Shakespeare 数据集演示
├── check_cuda.py          # CUDA 诊断脚本
├── requirements.txt       # 项目依赖
├── shakespeare_model.pth  # 已训练的模型
├── training_losses.png    # 训练损失曲线
└── data/
    └── shakespeare.txt    # Shakespeare 数据集
```

## 核心组件详解

### model.py - 模型定义

包含完整的 Transformer 解码器实现：

#### 1. MultiHeadAttention（多头自注意力）
- 将输入分成多个头，并行计算注意力
- 每个头学习不同的注意力模式
- 最后拼接所有头的输出

#### 2. FeedForward（前馈网络）
- 两个线性层，中间用 GELU 激活
- 位置-wise 的前馈网络

#### 3. TransformerBlock（Transformer 解码器块）
- 多头自注意力 + 残差连接 + 层归一化
- 前馈网络 + 残差连接 + 层归一化

#### 4. GPT（完整模型）
- 词嵌入层
- 位置编码
- 多层 Transformer 解码器
- 输出层（预测下一个字符）
- `generate()` 方法：自回归文本生成

### dataset.py - 数据集处理

- **CharDataset**: 字符级数据集
  - 将文本转换为字符序列
  - 支持批量采样
- **get_shakespeare_dataset**: 加载 Shakespeare 数据集

### trainer.py - 训练器

- 管理训练循环
- 计算损失
- 更新参数
- 评估验证集
- 保存和加载模型
- 自动设备选择（支持 GPU/CPU）

## Transformer 关键概念

### 自注意力（Self-Attention）
允许模型在生成每个 token 时，关注输入序列中的其他位置。

### 因果掩码（Causal Mask）
防止模型看到未来的信息，确保自回归生成的正确性。

### 残差连接（Residual Connections）
帮助梯度流动，解决深层网络的梯度消失问题。

### 层归一化（Layer Normalization）
稳定训练过程，加速收敛。

### 位置编码（Positional Encoding）
为模型提供位置信息，因为 Transformer 本身没有顺序概念。

## 学习建议

1. **从 mnist_classifier 开始**: 如果你是零基础，建议先学习 `../mnist_classifier/` 项目
2. **阅读代码注释**: 每个组件都有详细的中文解释
3. **运行 shakespeare_demo.py**: 观察训练过程和生成结果
4. **尝试修改**: 调整模型大小、层数、学习率等，观察效果
5. **理解 Attention**: 这是 Transformer 的核心，花时间弄懂它

## GPU 配置

详见 `GPU_SETUP.md`。

**重要提示**:
- RTX 50 系列显卡需要 PyTorch Nightly (cu128) 版本
- 必须坚持使用 GPU 训练，不要使用 CPU

## 技术栈

- Python 3.11+
- PyTorch 2.0+（推荐 Nightly 版本）
- matplotlib
- tqdm

## 学习资源

- [Andrej Karpathy 的 minGPT](https://github.com/karpathy/minGPT)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)（Transformer 原论文）
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/)

## 致谢

感谢 Andrej Karpathy 的 minGPT 项目，为本项目提供了重要参考。

## 下一步

完成本项目后，你已经掌握了 Transformer 和 LLM 的基本原理！可以尝试：
- 训练更大的模型
- 使用其他数据集
- 探索更先进的架构（如 GPT-2、LLaMA 等）
