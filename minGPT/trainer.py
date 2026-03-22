import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# 知识点：训练器
# ------------------------------------------------------------
# 训练器负责：
# 1. 管理训练循环
# 2. 计算损失
# 3. 更新参数
# 4. 评估验证集
# 5. 保存和加载模型
# ============================================================


class Trainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=3e-4, device=None):
        """
        Args:
            model: GPT 模型
            train_loader: 训练集 DataLoader
            val_loader: 验证集 DataLoader
            learning_rate: 学习率
            device: 设备（'cpu' 或 'cuda'）
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        
        # 自动选择设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 如果使用 CPU，给出提示
        if self.device.type == "cpu":
            print("\n" + "="*60)
            print("⚠️  当前使用 CPU 训练")
            print("="*60)
            print("虽然可以正常学习，但 GPU 训练会更快！")
            print("如需使用 GPU，请参考 GPU_SETUP.md 配置环境")
            print("="*60 + "\n")
        
        # 将模型移动到设备
        self.model.to(self.device)
        
        # =========================================================
        # 知识点 1：优化器
        # ---------------------------------------------------------
        # AdamW 是 Adam 的改进版，对权重衰减（Weight Decay）处理更好
        # 是训练 Transformer 的常用选择
        # =========================================================
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
        # 记录训练过程
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        
        # 使用 tqdm 显示进度条
        pbar = tqdm(self.train_loader, desc="训练中")
        
        for x, y in pbar:
            # 将数据移动到设备
            x = x.to(self.device)
            y = y.to(self.device)
            
            # =========================================================
            # 步骤 1：前向传播
            # ---------------------------------------------------------
            # 输入 x，得到 logits 和 loss
            # 当提供 y 时，会自动计算损失
            # =========================================================
            logits, loss = self.model(x, y)
            
            # =========================================================
            # 步骤 2：反向传播
            # ---------------------------------------------------------
            # 清零梯度
            self.optimizer.zero_grad()
            # 反向传播计算梯度
            loss.backward()
            # 更新参数
            self.optimizer.step()
            
            # 统计损失
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """在验证集上评估"""
        self.model.eval()
        total_loss = 0.0
        
        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            
            logits, loss = self.model(x, y)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self, num_epochs):
        """完整训练"""
        print(f"=" * 60)
        print(f"开始训练 {num_epochs} 个 epoch")
        print(f"=" * 60)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        
        print(f"\n训练完成！")
    
    def plot_losses(self, save_path='training_losses.png'):
        """绘制训练和验证损失曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练过程')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        print(f"损失曲线已保存到 {save_path}")
        plt.show()
    
    def save_model(self, path='model.pth'):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
        print(f"模型已保存到 {path}")
    
    def load_model(self, path='model.pth'):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"模型已从 {path} 加载")
