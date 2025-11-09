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


# 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 运行训练
chmod +x scripts/run.sh
./scripts/run.sh

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
=======
# transformer-implementation
A complete implementation of Transformer model with ablation studies
>>>>>>> 857f0d2d03933838208e72cf43ef38ff5a3f1f20
