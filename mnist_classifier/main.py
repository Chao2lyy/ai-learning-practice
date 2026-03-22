import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ============================================================
# 知识点 1：Tensor（张量）
# ------------------------------------------------------------
# Tensor 是 PyTorch 的核心数据结构，类似于 NumPy 数组
# 但 Tensor 可以在 GPU 上运行，加速计算
# ============================================================

# ============================================================
# 知识点 2：数据预处理（Transforms）
# ------------------------------------------------------------
# transforms.Compose 用于组合多个数据变换操作
# - ToTensor(): 将 PIL 图像或 NumPy 数组转换为 Tensor
# - Normalize(): 归一化，这里 mean=0.1307 和 std=0.3081 是 MNIST 数据集的统计值
# ============================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("=" * 60)
print("步骤 1：准备数据集")
print("=" * 60)

# ============================================================
# 知识点 3：Dataset（数据集）
# ------------------------------------------------------------
# Dataset 是 PyTorch 提供的抽象类，用于表示数据集
# torchvision.datasets.MNIST 是 PyTorch 内置的 MNIST 数据集
# 参数说明：
# - root: 数据集存放路径
# - train: True 表示训练集，False 表示测试集
# - download: 如果数据集不存在，是否自动下载
# - transform: 数据预处理操作
# ============================================================
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

print(f"训练集大小: {len(train_dataset)} 张图片")
print(f"测试集大小: {len(test_dataset)} 张图片")

# ============================================================
# 知识点 4：DataLoader（数据加载器）
# ------------------------------------------------------------
# DataLoader 用于批量加载数据，提供以下功能：
# - batch_size: 每批次的样本数量
# - shuffle: 是否打乱数据顺序（训练时通常设为 True）
# - num_workers: 并行加载数据的进程数（Windows 下通常设为 0）
# ============================================================
batch_size = 64
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

print(f"批次大小: {batch_size}")
print(f"训练集批次数: {len(train_loader)}")
print(f"测试集批次数: {len(test_loader)}")

# 可视化几张样本图片
print("\n可视化几张样本图片...")
examples = iter(train_loader)
example_data, example_targets = next(examples)

fig = plt.figure(figsize=(10, 2))
for i in range(6):
    plt.subplot(1, 6, i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title(f"数字: {example_targets[i]}")
    plt.xticks([])
    plt.yticks([])
plt.savefig('sample_images.png')
print("样本图片已保存为 sample_images.png")

print("\n" + "=" * 60)
print("步骤 2：定义神经网络模型")
print("=" * 60)

# ============================================================
# 知识点 5：神经网络模型（nn.Module）
# ------------------------------------------------------------
# 所有神经网络模型都应该继承自 nn.Module
# 需要实现两个方法：
# 1. __init__(): 定义网络层
# 2. forward(): 定义前向传播（数据如何流过网络）
#
# 本项目使用 3 层全连接网络：
# - 输入层：784 个神经元（28x28 像素展平）
# - 隐藏层 1：128 个神经元，ReLU 激活函数
# - 隐藏层 2：64 个神经元，ReLU 激活函数
# - 输出层：10 个神经元（数字 0-9）
# ============================================================
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()
        # nn.Linear 是全连接层（线性层），参数：输入特征数，输出特征数
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        # ReLU 是激活函数，引入非线性，让网络能学习复杂模式
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 前向传播：数据依次通过每一层
        out = self.fc1(x)  # 第一层全连接
        out = self.relu(out)  # ReLU 激活
        out = self.fc2(out)  # 第二层全连接
        out = self.relu(out)  # ReLU 激活
        out = self.fc3(out)  # 第三层全连接（输出层）
        return out

# 超参数设置
input_size = 784  # 28x28
hidden_size1 = 128
hidden_size2 = 64
num_classes = 10
num_epochs = 5  # 训练轮数
learning_rate = 0.001  # 学习率

# 创建模型实例
model = NeuralNet(input_size, hidden_size1, hidden_size2, num_classes)
print(f"模型结构:\n{model}")

# ============================================================
# 知识点 6：损失函数（Loss Function）
# ------------------------------------------------------------
# 损失函数用于衡量模型预测值与真实值之间的差距
# CrossEntropyLoss（交叉熵损失）常用于多分类问题
# 它内部已经包含了 Softmax 激活函数，所以输出层不需要额外加 Softmax
# ============================================================
criterion = nn.CrossEntropyLoss()
print(f"\n损失函数: {criterion}")

# ============================================================
# 知识点 7：优化器（Optimizer）
# ------------------------------------------------------------
# 优化器用于更新模型的参数，以最小化损失函数
# Adam 是一种常用的优化器，自适应学习率，收敛快
# 参数：
# - model.parameters(): 需要优化的模型参数
# - lr: 学习率，控制参数更新的步长
# ============================================================
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(f"优化器: {optimizer}")

print("\n" + "=" * 60)
print("步骤 3：开始训练")
print("=" * 60)

# 记录训练过程
train_losses = []
train_accs = []

# ============================================================
# 知识点 8：训练循环（Training Loop）
# ------------------------------------------------------------
# 完整的训练循环包含以下步骤：
# 1. 前向传播（Forward Pass）：输入数据通过网络得到预测
# 2. 计算损失（Compute Loss）：比较预测与真实值
# 3. 反向传播（Backward Pass）：计算损失对每个参数的梯度
# 4. 更新参数（Update Parameters）：优化器根据梯度更新参数
# ============================================================
total_step = len(train_loader)
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(train_loader):
        # 将图像展平：从 (batch_size, 1, 28, 28) 变为 (batch_size, 784)
        images = images.reshape(-1, input_size)
        
        # --------------------------------------------------------
        # 步骤 1：前向传播
        # --------------------------------------------------------
        outputs = model(images)
        
        # --------------------------------------------------------
        # 步骤 2：计算损失
        # --------------------------------------------------------
        loss = criterion(outputs, labels)
        
        # --------------------------------------------------------
        # 步骤 3：反向传播与参数更新
        # --------------------------------------------------------
        # 清零梯度（避免梯度累积）
        optimizer.zero_grad()
        # 反向传播：计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        
        # 统计训练信息
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 每 100 个批次打印一次
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
    
    # 计算本轮的平均损失和准确率
    epoch_loss = running_loss / total_step
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    
    print(f'Epoch [{epoch+1}/{num_epochs}] 完成! 平均损失: {epoch_loss:.4f}, 训练准确率: {epoch_acc:.2f}%\n')

print("\n" + "=" * 60)
print("步骤 4：在测试集上评估模型")
print("=" * 60)

# ============================================================
# 知识点 9：模型评估
# ------------------------------------------------------------
# 评估时不需要计算梯度，可以节省内存和计算资源
# 使用 torch.no_grad() 上下文管理器
# ============================================================
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    print(f'测试集上的准确率: {test_acc:.2f}%')
    
    # 检查是否达到目标
    if test_acc >= 95:
        print("🎉 恭喜！测试准确率达到 95% 以上！")
    else:
        print("⚠️  测试准确率未达到 95%，可以尝试增加训练轮数或调整超参数")

# 可视化训练过程
print("\n绘制训练过程曲线...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 损失曲线
ax1.plot(train_losses, label='Training Loss', color='blue')
ax1.set_title('训练损失变化')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

# 准确率曲线
ax2.plot(train_accs, label='Training Accuracy', color='red')
ax2.set_title('训练准确率变化')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()

plt.tight_layout()
plt.savefig('training_history.png')
print("训练过程曲线已保存为 training_history.png")

# ============================================================
# 知识点总结
# ============================================================
print("\n" + "=" * 60)
print("🎉 项目完成！核心知识点回顾")
print("=" * 60)
print("1. Tensor: PyTorch 的核心数据结构，可在 GPU 上运行")
print("2. Dataset: 表示数据集，负责加载单个样本")
print("3. DataLoader: 批量加载数据，支持 shuffle 和并行加载")
print("4. nn.Module: 神经网络基类，需实现 __init__ 和 forward")
print("5. Loss Function: 衡量预测与真实值的差距（如 CrossEntropyLoss）")
print("6. Optimizer: 更新模型参数以最小化损失（如 Adam）")
print("7. 训练循环: 前向传播 → 计算损失 → 反向传播 → 更新参数")
print("8. 模型评估: 使用 model.eval() 和 torch.no_grad()")
print("=" * 60)
