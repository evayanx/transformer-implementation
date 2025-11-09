<<<<<<< HEAD
# Transformer 文本摘要模型实现（课程作业）

本项目实现了完整的 Transformer 模型，用于 Gigaword 数据集上的轻量级文本摘要任务。支持 Encoder-Decoder 结构、多头注意力、位置编码、残差连接与 LayerNorm，并完成了系统的消融实验。

#项目结构
transformer-implementation/
├── src/ # 源代码
│ ├── model.py # Transformer模型实现
│ ├── train.py # 训练脚本
│ ├── data_loader.py # 数据加载和处理
│ ├── utils.py # 工具函数
│ └── config.py # 配置文件
├── scripts/ # 运行脚本
├── configs/ # 配置文件
├── requirements.txt # 依赖包
└── README.md # 项目说明


#复现实验
python src/train.py \
    --seed 42 \
    --batch_size 32 \
    --num_epochs 50 \
    --learning_rate 0.0001 \
    --d_model 256 \
    --num_heads 8 \
    --num_layers 4 \
    --max_seq_len 128

## 环境要求

### 硬件要求
- **GPU**: NVIDIA GPU 
- **RAM**: ≥ 8GB
- **存储空间**: ≥ 2GB 

### 软件要求
- **操作系统**: Windows 
- **Python**: 3.8 
- **CUDA**: 11.0 

## 安装依赖

```bash
# 创建conda环境
conda create -n transformer python=3.10
conda activate transformer

# 安装PyTorch 
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0

# 安装项目依赖
pip install -r requirements.txt

# 使用配置文件训练
python src/train.py --config configs/base.yaml --seed 42

# 运行消融实验
chmod +x scripts/run_ablation.sh
./scripts/run_ablation.sh
=======
# transformer-implementation
A complete implementation of Transformer model with ablation studies
>>>>>>> 857f0d2d03933838208e72cf43ef38ff5a3f1f20