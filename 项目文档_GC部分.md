# GC-MS 数据分析系统技术文档

## 目录

1. [数据处理模块](#1-数据处理模块)
2. [数据集管理模块](#2-数据集管理模块)
3. [模型架构](#3-模型架构)
4. [训练管理模块](#4-训练管理模块)
5. [评估与可视化模块](#5-评估与可视化模块)
6. [优化建议](#6-优化建议)

## 1. 数据处理模块

### load_data 函数详解

```python
def load_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]
```

#### 功能说明

- 负责读取和预处理GC-MS数据
- 实现数据标准化和平滑处理
- 将原始数据转换为深度学习可用格式

#### 实现细节

1. **类别编码**:
   - 使用字典映射将类别名转换为数值标签
   - 当前支持4个类别：APCK(0), FRCK(1), RACK(2), RHCK(3)
   - 可扩展性：可以通过修改class_to_label字典添加新类别

2. **数据预处理流程**:
   - 第一次遍历：收集所有特征用于标准化
   - 创建StandardScaler进行特征标准化
   - 第二次遍历：处理数据并应用标准化

3. **信号平滑处理**:
   - 使用Savitzky-Golay滤波器
   - 窗口长度：11（可调）
   - 多项式阶数：3（可调）
   - 作用：减少高频噪声，保持峰值特征

#### 优化方向

1. **数据预处理**:
   - 可以添加峰值检测和对齐
   - 实现自适应的平滑参数选择
   - 添加异常值检测和处理

2. **特征工程**:
   - 可以提取峰值特征
   - 添加频域特征
   - 实现自动的特征选择

## 2. 数据集管理模块

### ChromatographyDataset 类详解

```python
class ChromatographyDataset(Dataset)
```

#### 功能说明

- 继承自PyTorch的Dataset类
- 实现数据加载和增强
- 管理训练和验证数据

#### 实现细节

1. **初始化**:

   ```python
   def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False)
   ```

   - X：特征数据，形状为(N, 1, sequence_length)
   - y：标签数据，形状为(N,)
   - augment：是否启用数据增强

2. **数据增强策略**:
   - 随机噪声注入：

     ```python
     noise = torch.randn_like(x) * 0.002
     ```

   - 时间序列移位：

     ```python
     shift = np.random.randint(-3, 4)
     ```

#### 优化方向

1. **数据增强**:
   - 添加更多增强方法：缩放、反转等
   - 实现自适应的噪声强度
   - 添加混合数据增强策略

2. **性能优化**:
   - 实现数据预加载
   - 添加缓存机制
   - 优化内存使用

## 3. 模型架构

### ResNet1D 类详解

```python
class ResNet1D(nn.Module)
```

#### 功能说明

- 实现一维ResNet网络
- 处理时间序列数据
- 提供分类预测

#### 架构详解

1. **初始层**:

   ```python
   self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
   self.bn1 = nn.BatchNorm1d(32)
   self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
   ```

   - 初始卷积：降采样并提取基础特征
   - BatchNorm：加速训练并提供正则化
   - MaxPool：进一步降采样，提取显著特征

2. **残差块结构**:

   ```python
   class BasicBlock1D(nn.Module)
   ```

   - 两个3x1卷积层
   - 批归一化和ReLU激活
   - 跳跃连接
   - Dropout正则化

3. **网络层次**:
   - 4个残差层组
   - 通道数逐层翻倍：32→64→128→256
   - 自适应平均池化
   - 全连接分类层

#### 优化方向

1. **架构优化**:
   - 尝试不同的残差块设计
   - 添加注意力机制
   - 实验不同的池化策略

2. **效率提升**:
   - 模型压缩
   - 知识蒸馏
   - 量化优化

## 4. 训练管理模块

### train_with_cross_validation 函数详解

```python
def train_with_cross_validation(data_dir='data', n_folds=2, n_epochs=150, batch_size=16, device='cuda')
```

#### 功能说明

- 实现交叉验证训练
- 管理模型训练过程
- 保存训练结果和模型

#### 实现细节

1. **优化器配置**:

   ```python
   optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0001)
   ```

   - 使用AdamW优化器
   - 较小的初始学习率
   - 添加权重衰减防止过拟合

2. **学习率调度**:

   ```python
   scheduler = OneCycleLR(
       optimizer,
       max_lr=0.001,
       epochs=n_epochs,
       steps_per_epoch=steps_per_epoch,
       pct_start=0.4,
       anneal_strategy='cos'
   )
   ```

   - 使用OneCycleLR策略
   - 40%时间用于预热
   - 余弦退火降低学习率

3. **早停机制**:

   ```python
   early_stopping = EarlyStopping(patience=15, min_delta=0.0005, min_epochs=10)
   ```

   - 监控验证损失和F1分数
   - 设置最小训练轮数
   - 保存最佳模型

#### 优化方向

1. **训练策略**:
   - 实现学习率自动调整
   - 添加动态批量大小
   - 实现渐进式训练

2. **验证策略**:
   - 添加K折验证的策略选择
   - 实现分层采样
   - 添加交叉验证可视化

## 5. 评估与可视化模块

### 性能评估函数

```python
def calculate_metrics(y_true, y_pred)
```

#### 功能说明

- 计算多个评估指标
- 生成性能报告
- 可视化训练过程

#### 实现细节

1. **评估指标**:
   - 准确率（Accuracy）
   - 精确率（Precision）
   - 召回率（Recall）
   - F1分数

2. **可视化功能**:

   ```python
   def plot_learning_curves(train_losses, val_losses, metrics_history)
   ```

   - 绘制损失曲线
   - 绘制指标变化趋势
   - 支持交互式显示

#### 优化方向

1. **评估指标**:
   - 添加混淆矩阵分析
   - 实现ROC曲线分析
   - 添加统计显著性测试

2. **可视化增强**:
   - 添加特征重要性分析
   - 实现预测结果可视化
   - 添加模型解释性分析

## 6. 优化建议

### 短期优化目标

1. **数据处理优化**:
   - 实现自适应的数据预处理参数
   - 添加更多特征工程方法
   - 优化数据加载效率

2. **模型优化**:
   - 尝试添加注意力机制
   - 实验不同的残差块结构
   - 实现模型压缩

3. **训练优化**:
   - 实现自适应学习率调整
   - 添加更多数据增强方法
   - 优化早停策略

### 长期优化方向

1. **架构升级**:
   - 设计混合模型架构
   - 实现迁移学习功能
   - 添加集成学习支持

2. **功能扩展**:
   - 支持更多数据格式
   - 添加自动化特征选择
   - 实现在线学习能力

3. **工程优化**:
   - 优化内存使用
   - 提高计算效率
   - 添加分布式训练支持
