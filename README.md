# GC-MS 数据分析深度学习项目

## 项目概述

本项目是一个基于深度学习的气相色谱-质谱（GC-MS）数据分析系统，主要用于对不同类型的化合物进行自动分类识别。该系统采用了基于ResNet的一维卷积神经网络架构，能够有效处理和分析GC-MS产生的光谱数据。

## 技术特性

### 数据处理

- 支持CSV格式的GC-MS数据文件读取
- 实现了Savitzky-Golay滤波进行数据平滑处理
- 使用StandardScaler进行数据标准化
- 支持4种不同药材的部分分类：APCK、FRCK、RACK、RHCK

### 模型架构

- 基于ResNet架构的一维卷积神经网络
- 包含4个残差块层，通道数从32逐步增加到256
- 使用Batch Normalization和Dropout进行正则化
- 采用自适应平均池化进行特征提取

### 训练策略

- 实现了K折交叉验证（默认6折）
- 使用OneCycleLR学习率调度器
- 实现了基于验证损失和F1分数的早停机制
- 支持数据增强：
  - 随机噪声注入
  - 时间序列移位

### 性能监控

- 支持多个评估指标：准确率、精确率、召回率、F1分数
- 自动绘制学习曲线和性能指标变化图
- 保存每个fold的最佳模型和训练结果
- 生成详细的训练报告和模型信息

## 项目结构

```plaintext
.
├── data/               # 数据目录
├── models/            # 模型保存目录
├── AI_GCMS.py        # 主程序文件
└── README.md         # 项目文档
```

## 主要功能模块

### 1. 数据加载与预处理 (load_data)

- 自动识别和加载CSV格式的数据文件
- 进行数据平滑和标准化处理
- 组织数据为适合深度学习的格式

### 2. 数据集管理 (ChromatographyDataset)

- 继承自PyTorch的Dataset类
- 实现数据增强功能
- 支持批量数据加载

### 3. 模型架构 (ResNet1D)

- 实现了一维ResNet网络
- 包含基础残差块和主干网络结构
- 支持可配置的网络深度和宽度

### 4. 训练管理 (train_with_cross_validation)

- 实现完整的训练流程
- 支持交叉验证
- 包含早停机制
- 自动保存最佳模型

### 5. 模型评估与可视化

- 计算多个性能指标
- 生成训练过程的可视化图表
- 保存详细的评估结果

## 使用方法

### 训练模型

```python
from AI_GCMS import train_with_cross_validation

# 使用默认参数进行训练
results = train_with_cross_validation(
    data_dir='data',
    n_folds=6,
    n_epochs=150,
    batch_size=16,
    device='cuda'
)
```

### 加载已训练模型

```python
from AI_GCMS import load_model

model = load_model('models/final_model.pth')
```

## 性能优化

本项目包含多个性能优化策略：

1. 使用OneCycleLR进行学习率调度
2. 实现梯度裁剪防止梯度爆炸
3. 使用Dropout和BatchNorm进行正则化
4. 支持数据增强提高模型鲁棒性
5. 实现早停机制防止过拟合

## 输出文件

项目在训练过程中会生成以下文件：

- `models/best_model_fold{n}.pth`：每个fold的最佳模型
- `models/model_info_fold{n}.txt`：每个fold的详细信息
- `models/final_model.pth`：最终模型
- `models/final_results.txt`：综合训练结果报告
