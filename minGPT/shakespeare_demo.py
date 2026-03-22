import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from model import GPT
from dataset import get_shakespeare_dataset, create_dataloaders
from trainer import Trainer

def main():
    print("=" * 60)
    print("minGPT 莎士比亚数据集演示")
    print("参照 Andrej Karpathy 的 minGPT 项目")
    print("=" * 60)
    
    # =========================================================
    # 超参数设置
    # =========================================================
    block_size = 64      # 最大序列长度
    batch_size = 64
    d_model = 128        # 模型维度
    num_heads = 4        # 注意力头数量
    num_layers = 4       # Transformer 块数量
    learning_rate = 3e-4
    num_epochs = 5
    dropout = 0.1
    
    print(f"\n超参数配置:")
    print(f"  block_size: {block_size}")
    print(f"  batch_size: {batch_size}")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_layers: {num_layers}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  num_epochs: {num_epochs}")
    
    # =========================================================
    # 步骤 1：加载莎士比亚数据集
    # =========================================================
    print("\n" + "=" * 60)
    print("步骤 1：加载莎士比亚数据集")
    print("=" * 60)
    dataset = get_shakespeare_dataset(block_size=block_size)
    if dataset is None:
        print("数据集加载失败！")
        return
    
    train_loader, val_loader = create_dataloaders(dataset, batch_size=batch_size)
    
    # =========================================================
    # 步骤 2：创建 GPT 模型
    # =========================================================
    print("\n" + "=" * 60)
    print("步骤 2：创建 GPT 模型")
    print("=" * 60)
    model = GPT(
        vocab_size=dataset.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        block_size=block_size,
        dropout=dropout
    )
    
    # =========================================================
    # 步骤 3：创建训练器并训练
    # =========================================================
    print("\n" + "=" * 60)
    print("步骤 3：训练模型")
    print("=" * 60)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate
    )
    
    trainer.train(num_epochs=num_epochs)
    
    # 绘制损失曲线
    trainer.plot_losses()
    
    # =========================================================
    # 步骤 4：生成莎士比亚风格的文本
    # =========================================================
    print("\n" + "=" * 60)
    print("步骤 4：生成莎士比亚风格的文本")
    print("=" * 60)
    
    # 初始提示 - 来自莎士比亚的经典开头
    prompts = [
        "To be, or not to be",
        "Shall I compare thee to",
        "Now is the winter of",
        "All the world's a stage"
    ]
    
    for prompt in prompts:
        print(f"\n初始提示: '{prompt}'")
        
        # 将提示编码为整数
        context = torch.tensor(dataset.encode(prompt), dtype=torch.long).unsqueeze(0)
        context = context.to(trainer.device)
        
        # 生成文本 - 使用不同的温度参数
        for temp in [0.7, 1.0]:
            generated = model.generate(
                context,
                max_new_tokens=200,
                temperature=temp
            )
            
            generated_text = dataset.decode(generated[0].tolist())
            print(f"\n温度 = {temp}:")
            print("-" * 60)
            print(generated_text)
            print("-" * 60)
    
    # =========================================================
    # 保存模型（可选）
    # =========================================================
    save_choice = input("\n是否保存模型？(y/n): ").strip().lower()
    if save_choice == 'y':
        trainer.save_model('shakespeare_model.pth')
    
    print("\n" + "=" * 60)
    print("🎉 莎士比亚演示完成！")
    print("=" * 60)
    print("\n现在你可以尝试：")
    print("1. 增加 num_epochs 训练更久")
    print("2. 增大 d_model 和 num_layers 让模型更大")
    print("3. 调整 temperature 生成不同风格的文本")
    print("4. 尝试自己的初始提示！")
    print("=" * 60)


if __name__ == "__main__":
    main()
