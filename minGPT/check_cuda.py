import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch

print("=" * 60)
print("PyTorch 和 CUDA 诊断")
print("=" * 60)

print(f"\n1. PyTorch 版本: {torch.__version__}")
print(f"2. CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"3. CUDA 版本: {torch.version.cuda}")
    print(f"4. GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"5. 当前设备: {torch.cuda.current_device()}")
    print(f"   设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    # 测试一下 CUDA 张量
    print("\n6. 测试 CUDA 张量...")
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print(f"   成功创建 CUDA 张量: {x}")
    print(f"   张量设备: {x.device}")
else:
    print("\n⚠️  CUDA 不可用！")
    print("\n可能的原因:")
    print("1. 安装的是 CPU 版本的 PyTorch")
    print("2. NVIDIA 驱动未正确安装")
    print("3. CUDA 工具包未安装或版本不匹配")
    print("\n建议:")
    print("- 访问 https://pytorch.org/get-started/locally/")
    print("- 根据你的 CUDA 版本安装对应的 PyTorch")
    print("- 例如: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

print("\n" + "=" * 60)
