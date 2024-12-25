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
   - 创建StandardScaler进行特征标准化，确保所有特征在相同的尺度上
   - 第二次遍历：处理数据并应用标准化
   - 返回的X形状为(N, 1, sequence_length)，y形状为(N,)

3. **信号平滑处理**:
   - 使用Savitzky-Golay滤波器
   - 窗口长度：11（可调）
   - 多项式阶数：3（可调）
   - 作用：减少高频噪声，保持峰值特征
   - 在标准化之前应用，以保持信号的原始形态特征

#### 优化方向

1. **数据预处理**:
   - 可以添加峰值检测和对齐
   - 实现自适应的平滑参数选择
   - 添加异常值检测和处理
   - 考虑添加基线校正

2. **特征工程**:
   - 可以提取峰值特征
   - 添加频域特征
   - 实现自动的特征选择
   - 考虑添加峰面积计算

## 2. 数据集管理模块

### ChromatographyDataset 类详解

```python
class ChromatographyDataset(Dataset)
```

#### 功能说明

- 继承自PyTorch的Dataset类
- 实现数据加载和增强
- 管理训练和验证数据
- 支持实时数据增强

#### 实现细节

1. **初始化**:

   ```python
   def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False)
   ```

   - X：特征数据，形状为(N, 1, sequence_length)
   - y：标签数据，形状为(N,)
   - augment：是否启用数据增强，默认为False

2. **数据增强策略**:
   - 随机噪声注入：
     ```python
     noise = torch.randn_like(x) * 0.002  # 噪声强度为0.002
     ```
   - 时间序列移位：
     ```python
     shift = np.random.randint(-3, 4)  # 随机左右移动-3到3个点
     ```
   - 增强概率：50%（每个样本有0.5的概率被增强）

3. **数据获取**:
   ```python
   def __getitem__(self, idx):
   ```
   - 返回(特征, 标签)对
   - 特征为torch.FloatTensor类型
   - 标签为torch.LongTensor类型
   - 如果启用增强，每次获取时随机应用增强

#### 优化方向

1. **数据增强**:
   - 添加更多增强方法：缩放、反转等
   - 实现自适应的噪声强度
   - 添加混合数据增强策略
   - 考虑添加基于物理特性的增强方法

2. **性能优化**:
   - 实现数据预加载
   - 添加缓存机制
   - 优化内存使用
   - 实现多进程数据加载

## 3. 模型架构

### ResNet1D 类详解

```python
class ResNet1D(nn.Module)
```

#### 功能说明

- 实现一维ResNet网络
- 处理时间序列数据
- 提供分类预测
- 支持dropout正则化

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
   - Dropout：防止过拟合（dropout_rate可配置）

2. **残差块结构**:

   ```python
   class BasicBlock1D(nn.Module)
   ```

   - 两个3x1卷积层
   - 批归一化和ReLU激活
   - 跳跃连接
   - Dropout正则化（每个残差块都有独立的dropout）
   - 支持下采样操作

3. **网络层次**:
   - 4个残差层组
   - 通道数逐层翻倍：32→64→128→256
   - 自适应平均池化
   - 全连接分类层
   - 权重初始化使用kaiming_normal_

#### 优化方向

1. **架构优化**:
   - 尝试不同的残差块设计
   - 添加注意力机制
   - 实验不同的池化策略
   - 考虑添加SE（Squeeze-and-Excitation）模块

2. **效率提升**:
   - 模型压缩
   - 知识蒸馏
   - 量化优化
   - 考虑使用MobileNet风格的轻量级设计

## 4. 训练管理模块

### train_with_cross_validation 函数详解

```python
def train_with_cross_validation(data_dir='data', n_folds=2, n_epochs=150, batch_size=16, device='cuda')
```

#### 功能说明

- 实现交叉验证训练
- 管理模型训练过程
- 保存训练结果和模型
- 支持早停机制

#### 实现细节

1. **优化器配置**:

   ```python
   optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0001)
   ```

   - 使用AdamW优化器
   - 初始学习率：0.0005
   - 权重衰减：0.0001
   - 梯度裁剪：max_norm=0.5

2. **学习率调度**:

   ```python
   scheduler = OneCycleLR(
       optimizer,
       max_lr=0.001,
       epochs=n_epochs,
       steps_per_epoch=steps_per_epoch,
       pct_start=0.4,
       anneal_strategy='cos',
       div_factor=25.0,
       final_div_factor=1000.0
   )
   ```

   - 使用OneCycleLR策略
   - 40%时间用于预热
   - 余弦退火降低学习率
   - 初始学习率为max_lr/25.0
   - 最终学习率为初始学习率/1000.0

3. **早停机制**:

   ```python
   early_stopping = EarlyStopping(patience=15, min_delta=0.0005, min_epochs=10)
   ```

   - 监控验证损失和F1分数
   - 最小训练轮数：10轮
   - 耐心值：15轮
   - 最小改善阈值：0.0005
   - 保存最佳模型状态

4. **模型保存**:
   - 每个fold保存最佳模型
   - 保存训练信息到文本文件
   - 记录每个fold的最佳指标
   - 保存最终的综合结果

#### 优化方向

1. **训练策略**:
   - 实现学习率自动调整
   - 添加动态批量大小
   - 实现渐进式训练
   - 添加混合精度训练支持

2. **验证策略**:
   - 添加K折验证的策略选择
   - 实现分层采样
   - 添加交叉验证可视化
   - 支持自定义验证指标

## 5. 评估与可视化模块

### 性能评估函数

```python
def calculate_metrics(y_true, y_pred)
```

#### 功能说明

- 计算多个评估指标
- 生成性能报告
- 可视化训练过程
- 支持宏平均和微平均计算

#### 实现细节

1. **评估指标**:
   - 准确率（Accuracy）
   - 精确率（Precision）：宏平均
   - 召回率（Recall）：宏平均
   - F1分数：宏平均
   - 所有指标都处理了零除情况

2. **可视化功能**:

   ```python
   def plot_learning_curves(train_losses, val_losses, metrics_history)
   ```

   - 绘制损失曲线
   - 绘制指标变化趋势
   - 支持交互式显示
   - 双子图布局展示多个指标

3. **结果保存**:
   ```python
   def save_final_model(model, fold_results, save_dir='models')
   ```
   - 保存模型状态字典
   - 记录交叉验证结果
   - 保存综合评估指标
   - 生成详细的结果报告

#### 优化方向

1. **评估指标**:
   - 添加混淆矩阵分析
   - 实现ROC曲线分析
   - 添加统计显著性测试
   - 支持自定义评估指标

2. **可视化增强**:
   - 添加特征重要性分析
   - 实现预测结果可视化
   - 添加模型解释性分析
   - 支持交互式结果探索

## 6. 优化建议

### 短期优化目标

1. **数据处理优化**:
   - 实现自适应的数据预处理参数
   - 添加更多特征工程方法
   - 优化数据加载效率
   - 添加数据质量检查机制

2. **模型优化**:
   - 尝试添加注意力机制
   - 实验不同的残差块结构
   - 实现模型压缩
   - 添加模型集成支持

3. **训练优化**:
   - 实现自适应学习率调整
   - 添加更多数据增强方法
   - 优化早停策略
   - 实现分布式训练支持

### 长期优化方向

1. **架构升级**:
   - 设计混合模型架构
   - 实现迁移学习功能
   - 添加集成学习支持
   - 探索新型网络结构

2. **功能扩展**:
   - 支持更多数据格式
   - 添加自动化特征选择
   - 实现在线学习能力
   - 添加模型解释工具

3. **工程优化**:
   - 优化内存使用
   - 提高计算效率
   - 添加分布式训练支持
   - 实现自动化部署流程
