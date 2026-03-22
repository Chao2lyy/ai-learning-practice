import torch
import os
import urllib.request
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 知识点：字符级语言模型数据集
# ------------------------------------------------------------
# 我们使用字符级（character-level）的 tokenizer
# 这意味着每个字符都是一个独立的 token
#
# 例如："hello" -> ['h', 'e', 'l', 'l', 'o'] -> [3, 4, 7, 7, 8]
#
# 字符级的优点：
# - 词汇表小（通常只有几十到几百个字符）
# - 不需要复杂的分词算法
# - 可以处理任何字符
#
# 字符级的缺点：
# - 序列更长（一个词可能由多个字符组成）
# - 需要学习更长的依赖关系
# ============================================================


class CharDataset(Dataset):
    def __init__(self, text, block_size):
        """
        Args:
            text: 原始文本字符串
            block_size: 最大序列长度（上下文窗口大小）
        """
        self.block_size = block_size
        
        # =========================================================
        # 步骤 1：构建词汇表
        # ---------------------------------------------------------
        # 找出文本中所有不同的字符
        # =========================================================
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        print(f"词汇表大小: {self.vocab_size}")
        print(f"所有字符: {''.join(chars)}")
        
        # =========================================================
        # 步骤 2：创建字符到整数、整数到字符的映射
        # ---------------------------------------------------------
        # stoi: string to index（字符转整数）
        # itos: index to string（整数转字符）
        # =========================================================
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # =========================================================
        # 步骤 3：将整个文本编码为整数序列
        # =========================================================
        self.data = torch.tensor(self.encode(text), dtype=torch.long)
        print(f"数据长度: {len(self.data)} 个字符")
    
    def encode(self, s):
        """将字符串编码为整数列表"""
        return [self.stoi[c] for c in s]
    
    def decode(self, l):
        """将整数列表解码为字符串"""
        return ''.join([self.itos[i] for i in l])
    
    def __len__(self):
        """数据集的大小（样本数量）"""
        # 我们可以从长序列中滑动窗口生成多个样本
        # 每个样本长度为 block_size + 1（最后一个字符作为 target）
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        """
        获取一个样本
        
        返回:
            x: 输入序列 (block_size,)
            y: 目标序列 (block_size,)
               注意：y 是 x 向右移动一位
               例如：x = [1, 2, 3, 4], y = [2, 3, 4, 5]
               这样模型学习预测下一个字符
        """
        # 取从 idx 开始的 block_size + 1 个字符
        chunk = self.data[idx:idx + self.block_size + 1]
        # 前 block_size 个字符作为输入
        x = chunk[:-1]
        # 后 block_size 个字符作为目标（向右移动一位）
        y = chunk[1:]
        return x, y


# ============================================================
# 辅助函数：创建简单的示例文本
# ------------------------------------------------------------
# 为了演示，我们创建一个简单的重复文本
# 这样小模型也能快速学到东西
# ============================================================
def get_demo_dataset(block_size=32):
    """创建一个演示用的数据集"""
    # 简单的重复文本
    text = """Hello, world! This is a simple demo text for training our minGPT.
We will learn how to generate text character by character.
The quick brown fox jumps over the lazy dog.
1234567890!@#$%^&*()
ABCDEFGHIJKLMNOPQRSTUVWXYZ
abcdefghijklmnopqrstuvwxyz
This is a test. This is only a test.
Repeating patterns help the model learn faster.
Hello again! Welcome to minGPT!
"""
    
    # 重复多次，让数据集更大一些
    text = text * 10
    
    dataset = CharDataset(text, block_size=block_size)
    return dataset


def create_dataloaders(dataset, batch_size=32, val_fraction=0.1):
    """
    创建训练集和验证集的 DataLoader
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        val_fraction: 验证集比例
    """
    # 计算分割点
    n = len(dataset)
    n_val = int(val_fraction * n)
    n_train = n - n_val
    
    # 随机分割（注意：这里简单地用前 n_train 个作为训练集，后 n_val 个作为验证集）
    # 在实际项目中可能需要更高级的分割方式
    from torch.utils.data import random_split
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


# ============================================================
# 莎士比亚数据集
# ------------------------------------------------------------
# 这是 Andrej Karpathy 的 minGPT 项目中使用的经典数据集
# 包含莎士比亚的完整作品
# ============================================================
def get_shakespeare_dataset(block_size=64, data_dir='./data'):
    """
    下载并加载莎士比亚数据集
    
    Args:
        block_size: 最大序列长度
        data_dir: 数据存放目录
    """
    # 确保数据目录存在
    os.makedirs(data_dir, exist_ok=True)
    
    # 数据集 URL（来自 Andrej Karpathy 的 minGPT）
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    file_path = os.path.join(data_dir, 'shakespeare.txt')
    
    # 如果文件不存在，下载它
    if not os.path.exists(file_path):
        print(f"正在下载莎士比亚数据集到 {file_path}...")
        try:
            urllib.request.urlretrieve(url, file_path)
            print("下载完成！")
        except Exception as e:
            print(f"下载失败: {e}")
            print("请手动下载数据集并放置在 data/shakespeare.txt")
            return None
    
    # 读取文本
    print(f"正在读取莎士比亚数据集...")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"数据集大小: {len(text)} 个字符")
    print(f"前 200 个字符预览:\n{repr(text[:200])}\n")
    
    # 创建数据集
    dataset = CharDataset(text, block_size=block_size)
    return dataset

