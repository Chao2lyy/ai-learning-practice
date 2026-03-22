import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# ============================================================
# 整体概览：我们要构建什么？
# ------------------------------------------------------------
# 我们要构建一个 GPT（Generative Pre-trained Transformer）模型
# 这是一个"解码器-only"的 Transformer 架构
#
# 主要组件（按执行顺序）：
# 1. 词嵌入（Token Embedding）：把每个词变成向量
# 2. 位置编码（Positional Encoding）：给每个位置添加位置信息
# 3. 多个 Transformer 解码器块堆叠
# 4. 最终的层归一化 + 输出层
#
# 每个 Transformer 解码器块包含：
# - 多头自注意力（Multi-Head Self-Attention）
# - 层归一化（Layer Normalization）
# - 前馈网络（Feed-Forward Network）
# - 残差连接（Residual Connections）
# ============================================================


# ============================================================
# 知识点 1：多头自注意力（Multi-Head Self-Attention）
# ------------------------------------------------------------
# 这是 Transformer 最核心的部分！
#
# 自注意力让模型能够"关注"输入序列中的不同位置
# 多头注意力让模型能够同时关注多个不同的"方面"
#
# 关键概念：
# - Q (Query)：查询 - "我在找什么？"
# - K (Key)：键 - "我这里有什么？"
# - V (Value)：值 - "我的内容是什么？"
#
# 公式：Attention(Q, K, V) = softmax(Q*K^T / sqrt(d_k)) * V
# ============================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size, dropout=0.1):
        super().__init__()
        # d_model: 模型的维度（嵌入向量的大小）
        # num_heads: 注意力头的数量
        # block_size: 最大序列长度
        
        # 确保 d_model 可以被 num_heads 整除
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 线性层：将输入投影到 Q, K, V
        # 这里用一个线性层一次性计算 Q, K, V，效率更高
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        
        # 输出线性层
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Dropout 层，用于正则化
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # =========================================================
        # 因果掩码（Causal Mask）
        # ---------------------------------------------------------
        # 这是解码器的关键！它防止模型看到未来的 token
        # 比如在预测第 5 个词时，只能看到前 4 个词
        #
        # 掩码形状：(1, 1, block_size, block_size)
        # 下三角矩阵，对角线及以下为 True，以上为 False
        # =========================================================
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
                .view(1, 1, block_size, block_size)
        )
    
    def forward(self, x):
        # x 的形状：(batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        
        # =========================================================
        # 步骤 1：计算 Q, K, V
        # ---------------------------------------------------------
        # 将输入通过线性层得到 Q, K, V 的拼接
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3*d_model)
        
        # 将 qkv 拆分成 Q, K, V
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.d_k)
        # 重新排列维度：(3, batch_size, num_heads, seq_len, d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        # 分别取出 Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 每个的形状：(batch_size, num_heads, seq_len, d_k)
        
        # =========================================================
        # 步骤 2：计算注意力分数
        # ---------------------------------------------------------
        # Q * K^T / sqrt(d_k)
        # 为什么要除以 sqrt(d_k)？
        # 为了防止点积结果太大，导致 softmax 进入饱和区（梯度消失）
        # =========================================================
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        # 形状：(batch_size, num_heads, seq_len, seq_len)
        
        # =========================================================
        # 步骤 3：应用因果掩码
        # ---------------------------------------------------------
        # 将未来位置的注意力分数设为 -inf
        # 这样 softmax 后这些位置的权重就是 0
        # =========================================================
        attn_scores = attn_scores.masked_fill(
            self.mask[:, :, :seq_len, :seq_len] == 0,
            float('-inf')
        )
        
        # =========================================================
        # 步骤 4：Softmax 得到注意力权重
        # ---------------------------------------------------------
        # 每一行的和为 1，表示对不同位置的关注程度
        # =========================================================
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 应用 Dropout
        attn_weights = self.attn_dropout(attn_weights)
        
        # =========================================================
        # 步骤 5：用注意力权重对 V 加权求和
        # ---------------------------------------------------------
        # output = attn_weights * V
        # =========================================================
        out = attn_weights @ v
        # 形状：(batch_size, num_heads, seq_len, d_k)
        
        # =========================================================
        # 步骤 6：拼接多头
        # ---------------------------------------------------------
        # 将多个头的结果拼接在一起
        # =========================================================
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # =========================================================
        # 步骤 7：输出线性层 + Dropout
        # =========================================================
        out = self.resid_dropout(self.out_proj(out))
        
        return out


# ============================================================
# 知识点 2：前馈网络（Feed-Forward Network）
# ------------------------------------------------------------
# 这是一个简单的两层全连接网络
# 公式：FFN(x) = max(0, x*W1 + b1) * W2 + b2
#
# 通常中间层的维度是 d_model 的 4 倍
# ============================================================
class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # 第一层：将维度扩大 4 倍
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        # 第二层：将维度变回 d_model
        self.fc2 = nn.Linear(4 * d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 第一层 + ReLU 激活
        x = self.fc1(x)
        x = F.relu(x)
        # 第二层 + Dropout
        x = self.dropout(self.fc2(x))
        return x


# ============================================================
# 知识点 3：Transformer 解码器块（Decoder Block）
# ------------------------------------------------------------
# 这是 Transformer 的基本构建块
# 包含：多头注意力 + 前馈网络，都带有残差连接和层归一化
#
# 我们使用"预归一化"（Pre-LN）结构：
# x = x + Attention(LayerNorm(x))
# x = x + FFN(LayerNorm(x))
#
# 这种结构在实践中训练更稳定
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, block_size, dropout=0.1):
        super().__init__()
        # 层归一化 1（在注意力之前）
        self.ln1 = nn.LayerNorm(d_model)
        # 多头自注意力
        self.attn = MultiHeadAttention(d_model, num_heads, block_size, dropout)
        # 层归一化 2（在前馈网络之前）
        self.ln2 = nn.LayerNorm(d_model)
        # 前馈网络
        self.ffn = FeedForward(d_model, dropout)
    
    def forward(self, x):
        # =========================================================
        # 注意力部分：残差连接
        # ---------------------------------------------------------
        # x = x + Attention(LayerNorm(x))
        # =========================================================
        x = x + self.attn(self.ln1(x))
        
        # =========================================================
        # 前馈网络部分：残差连接
        # ---------------------------------------------------------
        # x = x + FFN(LayerNorm(x))
        # =========================================================
        x = x + self.ffn(self.ln2(x))
        
        return x


# ============================================================
# 知识点 4：层归一化（Layer Normalization）
# ------------------------------------------------------------
# 为什么要用 LayerNorm？
# 1. 加速训练：让每一层的输入分布稳定
# 2. 梯度流动更好：避免梯度消失或爆炸
#
# LayerNorm vs BatchNorm：
# - BatchNorm：在 batch 维度上归一化（适合 CV）
# - LayerNorm：在特征维度上归一化（适合 NLP）
# ============================================================
# （上面的 TransformerBlock 中已经使用了 nn.LayerNorm）


# ============================================================
# 知识点 5：残差连接（Residual Connection）
# ------------------------------------------------------------
# 公式：y = x + F(x)
#
# 为什么重要？
# 1. 解决梯度消失问题：梯度可以直接通过 x 传播
# 2. 允许训练更深的网络
# 3. 让网络更容易学习恒等映射（如果需要的话）
# ============================================================
# （上面的 TransformerBlock 中已经使用了残差连接）


# ============================================================
# 知识点 6：完整的 GPT 模型
# ------------------------------------------------------------
# 把所有组件组合在一起！
# ============================================================
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, block_size, dropout=0.1):
        super().__init__()
        # 保存超参数
        self.block_size = block_size
        
        # =========================================================
        # 1. 词嵌入（Token Embedding）
        # ---------------------------------------------------------
        # 将每个 token（整数）映射为一个 d_model 维的向量
        # 形状：(vocab_size, d_model)
        # =========================================================
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # =========================================================
        # 2. 位置编码（Positional Embedding）
        # ---------------------------------------------------------
        # 为什么需要位置编码？
        # 因为自注意力是"位置无关"的，它不知道词的顺序
        # 位置编码给每个位置添加独特的位置信息
        #
        # 这里我们使用"可学习的位置编码"
        # 形状：(block_size, d_model)
        # =========================================================
        self.position_embedding = nn.Embedding(block_size, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # =========================================================
        # 3. 多个 Transformer 解码器块
        # ---------------------------------------------------------
        # 使用 nn.ModuleList 来堆叠多个层
        # =========================================================
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, block_size, dropout)
            for _ in range(num_layers)
        ])
        
        # =========================================================
        # 4. 最终的层归一化
        # =========================================================
        self.ln_f = nn.LayerNorm(d_model)
        
        # =========================================================
        # 5. 输出线性层（语言建模头）
        # ---------------------------------------------------------
        # 将 d_model 维的向量映射回 vocab_size 维
        # 输出是每个 token 的概率
        # =========================================================
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 打印模型参数数量
        print(f"模型参数总数: {self.get_num_params()/1e6:.2f}M")
    
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_num_params(self):
        """计算模型参数总数（不包括位置编码）"""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def forward(self, idx, targets=None):
        # idx 的形状：(batch_size, seq_len)
        batch_size, seq_len = idx.shape
        
        # 确保序列长度不超过 block_size
        assert seq_len <= self.block_size, f"序列长度 {seq_len} 超过最大长度 {self.block_size}"
        
        # =========================================================
        # 步骤 1：生成位置索引
        # ---------------------------------------------------------
        # 从 0 到 seq_len-1
        # =========================================================
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        
        # =========================================================
        # 步骤 2：词嵌入 + 位置编码
        # ---------------------------------------------------------
        # token_emb: (batch_size, seq_len, d_model)
        # pos_emb: (seq_len, d_model)
        # 广播后相加
        # =========================================================
        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = self.dropout(token_emb + pos_emb)
        
        # =========================================================
        # 步骤 3：通过所有 Transformer 块
        # =========================================================
        for block in self.blocks:
            x = block(x)
        
        # =========================================================
        # 步骤 4：最终的层归一化
        # =========================================================
        x = self.ln_f(x)
        
        # =========================================================
        # 步骤 5：输出层（如果有 targets 的话，计算损失）
        # =========================================================
        if targets is not None:
            # 训练阶段：计算 logits 和损失
            logits = self.lm_head(x)
            # 将 logits 和 targets 展平，计算交叉熵损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            # 推理阶段：只计算最后一个位置的 logits（效率更高）
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归生成文本
        - idx: 初始序列 (batch_size, seq_len)
        - max_new_tokens: 要生成的新 token 数量
        - temperature: 温度参数，控制随机性（>1 更随机，<1 更确定）
        - top_k: 只从 top_k 个概率最高的 token 中采样
        """
        for _ in range(max_new_tokens):
            # 如果序列太长，裁剪到 block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # 前向传播，得到 logits
            logits, _ = self(idx_cond)
            
            # 只取最后一个位置的 logits，并应用温度
            logits = logits[:, -1, :] / temperature
            
            # 如果指定了 top_k，裁剪 logits
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Softmax 得到概率
            probs = F.softmax(logits, dim=-1)
            
            # 从概率分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 将新 token 拼接到序列中
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# ============================================================
# 知识点总结
# ============================================================
print("=" * 60)
print("🎉 minGPT 模型定义完成！核心组件回顾")
print("=" * 60)
print("1. MultiHeadAttention: 多头自注意力，包含因果掩码")
print("2. FeedForward: 两层全连接网络")
print("3. TransformerBlock: 注意力 + 前馈网络，带残差连接和层归一化")
print("4. GPT: 完整模型，包含词嵌入、位置编码、多个 Transformer 块")
print("=" * 60)
