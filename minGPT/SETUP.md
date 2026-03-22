
# minGPT 环境设置指南

## 当前项目结构
```
hwriteNumber/
├── archive_mnist_classifier/  # MNIST 分类器项目归档
├── archive_minGPT/           # minGPT 项目原始文件归档
└── minGPT/                   # 当前使用的 minGPT 项目（推荐）
    ├── model.py              # GPT 模型定义
    ├── dataset.py            # 数据集处理
    ├── trainer.py            # 训练器
    ├── shakespeare_demo.py    # 莎士比亚演示
    ├── check_cuda.py         # CUDA 诊断
    ├── requirements.txt      # 项目依赖
    ├── GPU_SETUP.md         # GPU 配置说明
    ├── README.md             # 项目说明
    ├── SETUP.md              # 本文档
    └── data/                # 数据文件夹
        └── shakespeare.txt  # 莎士比亚数据集
```

## 1. 创建 Python 3.11 虚拟环境（使用 conda）

由于当前环境权限限制，请手动执行以下命令：

### 在 Anaconda Prompt 或 PowerShell 中运行：

```bash
# 创建 Python 3.11 虚拟环境
conda create -n mingpt_env python=3.11

# 激活虚拟环境
conda activate mingpt_env

# 进入项目目录
cd c:\work\demo_work\hwriteNumber\minGPT

# 安装项目依赖
pip install -r requirements.txt
```

## 2. 或者使用 Python 内置 venv（如果不使用 conda）

```bash
# 创建虚拟环境
python -m venv mingpt_env

# 激活虚拟环境（Windows PowerShell）
.\mingpt_env\Scripts\Activate.ps1

# 或激活虚拟环境（Windows CMD）
mingpt_env\Scripts\activate.bat

# 进入项目目录并安装依赖
cd c:\work\demo_work\hwriteNumber\minGPT
pip install -r requirements.txt
```

## 3. 验证安装

```bash
# 检查 Python 版本
python --version  # 应该显示 Python 3.11.x

# 检查 PyTorch 和 CUDA
python check_cuda.py
```

## 4. 运行项目

```bash
# 运行莎士比亚演示
python shakespeare_demo.py
```

## 5. 注意事项

- 请在 `minGPT/` 文件夹下运行所有命令
- 当前环境（TraeAI-4）是 Python 3.13，建议使用 Python 3.11 以获得 CUDA 支持
- 详细的 GPU 配置请参考 `GPU_SETUP.md`
