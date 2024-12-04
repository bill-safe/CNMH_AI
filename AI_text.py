# 导入必要的库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器
from torch.utils.data import Dataset, DataLoader  # 数据加载工具
import numpy as np  # 数值计算库
from sklearn.model_selection import KFold  # K折交叉验证
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # 评估指标
import matplotlib.pyplot as plt  # 绘图库
from typing import List, Tuple  # 类型注解
import copy  # 深拷贝
import pandas as pd  # 数据处理库
import os  # 操作系统接口
from torch.optim.lr_scheduler import OneCycleLR  # 学习率调度器
from scipy.signal import savgol_filter  # 信号平滑滤波
from sklearn.preprocessing import StandardScaler  # 数据标准化

def load_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载所有数据文件并组织数据
    
    Args:
        data_dir: 数据文件目录
        
    Returns:
        X: 形状为(N, 1, sequence_length)的特征数据
        y: 形状为(N,)的标签数据
    """
    # 定义类别到标签的映射字典
    class_to_label = {
        'APCK': 0,
        'FRCK': 1,
        'RACK': 2,
        'RHCK': 3
    }
    
    all_features = []  # 存储所有特征
    all_labels = []    # 存储所有标签
    all_raw_features = []  # 存储原始特征用于标准化
    
    # 第一次遍历：收集所有特征用于标准化
    for filename in os.listdir(data_dir):
        if not filename.endswith('.CSV'):
            continue
            
        class_name = filename[:4]
        if class_name not in class_to_label:
            continue
            
        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath, skiprows=1)  # 跳过第一行
        features = df['Y(Counts)'].values  # 提取Y轴计数值
        all_raw_features.append(features)
    
    # 创建并拟合标准化器
    scaler = StandardScaler()
    scaler.fit(np.concatenate(all_raw_features).reshape(-1, 1))
    
    # 第二次遍历：处理数据
    for filename in os.listdir(data_dir):
        if not filename.endswith('.CSV'):
            continue
            
        class_name = filename[:4]
        if class_name not in class_to_label:
            continue
            
        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath, skiprows=1)
        
        # 使用Savitzky-Golay滤波进行平滑处理，减少噪声
        features = savgol_filter(df['Y(Counts)'].values, window_length=11, polyorder=3)
        
        # 标准化特征
        features = scaler.transform(features.reshape(-1, 1)).ravel()
        
        all_features.append(features)
        all_labels.append(class_to_label[class_name])
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    # 添加通道维度，转换为(N, 1, sequence_length)的形状
    X = X[:, np.newaxis, :]
    
    return X, y

class ChromatographyDataset(Dataset):
    """
    色谱图数据集类，用于加载和预处理数据
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        """
        初始化数据集
        
        Args:
            X: 特征数据
            y: 标签数据
            augment: 是否进行数据增强
        """
        self.X = torch.FloatTensor(X)  # 转换为PyTorch张量
        self.y = torch.LongTensor(y)
        self.augment = augment
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.y)
    
    def __getitem__(self, idx):
        """
        获取单个数据样本
        
        Args:
            idx: 索引
            
        Returns:
            (x, y): 特征和标签对
        """
        x = self.X[idx].clone()
        y = self.y[idx]
        
        # 数据增强：添加随机噪声和随机移位
        if self.augment and np.random.random() > 0.5:
            # 添加高斯噪声
            noise = torch.randn_like(x) * 0.002
            x = x + noise
            
            # 随机移位
            shift = np.random.randint(-3, 4)
            if shift > 0:
                x = torch.cat([x[:, shift:], x[:, :shift]], dim=1)
            elif shift < 0:
                x = torch.cat([x[:, shift:], x[:, :shift]], dim=1)
        
        return x, y

class BasicBlock1D(nn.Module):
    """
    一维ResNet的基本块
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0.1):
        """
        初始化基本块
        
        Args:
            inplanes: 输入通道数
            planes: 输出通道数
            stride: 卷积步长
            downsample: 下采样层
            dropout_rate: dropout比率
        """
        super(BasicBlock1D, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)  # 批归一化
        self.relu = nn.ReLU(inplace=True)  # ReLU激活
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout正则化
        # 第二个卷积层
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """前向传播"""
        identity = x  # 保存输入用于残差连接

        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 添加残差连接
        out = self.relu(out)

        return out

class ResNet1D(nn.Module):
    """
    一维ResNet模型
    """
    def __init__(self, block, layers, num_classes=4, dropout_rate=0.1):
        """
        初始化ResNet模型
        
        Args:
            block: 基本块类型
            layers: 每层的基本块数量
            num_classes: 分类数量
            dropout_rate: dropout比率
        """
        super(ResNet1D, self).__init__()
        self.inplanes = 32
        self.dropout_rate = dropout_rate
        
        # 初始卷积层
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # 残差层
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        
        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        创建残差层
        
        Args:
            block: 基本块类型
            planes: 输出通道数
            blocks: 基本块数量
            stride: 步长
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.dropout_rate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout_rate=self.dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播"""
        # 初始层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 分类层
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

class EarlyStopping:
    """
    早停类，用于防止过拟合
    """
    def __init__(self, patience=10, min_delta=0.001, save_dir='models'):
        """
        初始化早停器
        
        Args:
            patience: 容忍的轮数
            min_delta: 最小改善阈值
            save_dir: 模型保存目录
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_f1 = None
        self.early_stop = False
        self.best_model = None
        self.best_epoch = 0
        self.best_metrics = None
        self.no_improvement_epochs = 0
        self.save_dir = save_dir
        
        # 创建保存模型的目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def __call__(self, val_loss, val_metrics, model, epoch, fold=None):
        """
        检查是否应该早停
        
        Args:
            val_loss: 验证损失
            val_metrics: 验证指标
            model: 模型
            epoch: 当前轮数
            fold: 当前折数
        """
        current_f1 = val_metrics['f1']
        
        if self.best_loss is None:  # 首次调用
            self.best_loss = val_loss
            self.best_f1 = current_f1
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.best_metrics = val_metrics
            self._save_model(model, epoch, val_metrics, fold)
            return
        
        # 同时考虑损失和F1分数的改善
        loss_improved = val_loss < (self.best_loss - self.min_delta)
        f1_improved = current_f1 > (self.best_f1 + self.min_delta)
        
        if loss_improved or f1_improved:
            # 如果任一指标有显著改善
            if loss_improved:
                self.best_loss = val_loss
            if f1_improved:
                self.best_f1 = current_f1
            
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.best_metrics = val_metrics
            self.no_improvement_epochs = 0
            
            # 保存最佳模型
            self._save_model(model, epoch, val_metrics, fold)
        else:
            self.no_improvement_epochs += 1
            
        # 检查是否应该早停
        if self.no_improvement_epochs >= self.patience:
            self.early_stop = True
            
    def _save_model(self, model, epoch, metrics, fold=None):
        """
        保存模型和相关信息
        
        Args:
            model: 模型
            epoch: 当前轮数
            metrics: 评估指标
            fold: 当前折数
        """
        fold_str = f'_fold{fold}' if fold is not None else ''
        model_path = os.path.join(self.save_dir, f'best_model{fold_str}.pth')
        
        # 保存模型状态和相关信息
        save_dict = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'fold': fold
        }
        torch.save(save_dict, model_path)
        
        # 保存模型信息到文本文件
        info_path = os.path.join(self.save_dir, f'model_info{fold_str}.txt')
        with open(info_path, 'w') as f:
            f.write(f'Best Epoch: {epoch}\n')
            f.write('Metrics:\n')
            for metric_name, metric_value in metrics.items():
                f.write(f'{metric_name}: {metric_value:.4f}\n')

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备(CPU/GPU)
        
    Returns:
        epoch_loss: 平均训练损失
        metrics: 训练指标
    """
    model.train()
    running_loss = 0.0
    predictions = []
    true_labels = []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader.dataset)
    metrics = calculate_metrics(true_labels, predictions)
    return epoch_loss, metrics

def validate(model, val_loader, criterion, device):
    """
    验证模型
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备(CPU/GPU)
        
    Returns:
        epoch_loss: 平均验证损失
        metrics: 验证指标
    """
    model.eval()
    running_loss = 0.0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader.dataset)
    metrics = calculate_metrics(true_labels, predictions)
    return epoch_loss, metrics

def calculate_metrics(y_true, y_pred):
    """
    计算评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        dict: 包含各项评估指标的字典
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

def plot_learning_curves(train_losses, val_losses, metrics_history):
    """
    绘制学习曲线
    
    Args:
        train_losses: 训练损失历史
        val_losses: 验证损失历史
        metrics_history: 评估指标历史
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Learning Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # 绘制评估指标曲线
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        values = [metrics[metric] for metrics in metrics_history]
        ax2.plot(values, label=metric)
    ax2.set_title('Performance Metrics')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def save_final_model(model, fold_results, save_dir='models'):
    """
    保存最终模型和综合结果
    
    Args:
        model: 模型
        fold_results: 各折的结果
        save_dir: 保存目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'fold_results': fold_results
    }, final_model_path)
    
    # 保存综合结果到文本文件
    results_path = os.path.join(save_dir, 'final_results.txt')
    with open(results_path, 'w') as f:
        f.write('Cross-Validation Results:\n\n')
        for i, fold in enumerate(fold_results):
            f.write(f'Fold {i+1}:\n')
            f.write(f'Best Validation Loss: {fold["best_val_loss"]:.4f}\n')
            f.write(f'Best Epoch: {fold["best_epoch"]}\n')
            f.write('Metrics:\n')
            for metric_name, metric_value in fold['final_val_metrics'].items():
                f.write(f'{metric_name}: {metric_value:.4f}\n')
            f.write('\n')

def load_model(model_path, model=None):
    """
    加载保存的模型
    
    Args:
        model_path: 模型文件路径
        model: 模型实例（可选）
        
    Returns:
        model: 加载的模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
        
    checkpoint = torch.load(model_path)
    
    if model is None:
        model = ResNet1D(BasicBlock1D, [1, 1, 1, 1])
        
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    return model

def train_with_cross_validation(data_dir='data', n_folds=6, n_epochs=150, batch_size=16, device='cuda'):
    """
    使用交叉验证训练模型
    
    Args:
        data_dir: 数据目录
        n_folds: 交叉验证折数
        n_epochs: 训练轮数
        batch_size: 批次大小
        device: 设备(CPU/GPU)
        
    Returns:
        fold_results: 各折的结果
        avg_metrics: 平均评估指标
        std_metrics: 评估指标标准差
    """
    # 加载数据
    X, y = load_data(data_dir)
    
    # 创建K折交叉验证
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    # 创建最终模型（用于保存最后一个fold的状态）
    final_model = ResNet1D(BasicBlock1D, [1, 1, 1, 1]).to(device)
    
    # 开始K折交叉验证
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\nFold {fold+1}/{n_folds}")
        
        # 准备数据
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 创建数据集和数据加载器
        train_dataset = ChromatographyDataset(X_train, y_train, augment=True)
        val_dataset = ChromatographyDataset(X_val, y_val, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 初始化模型、损失函数和优化器
        model = ResNet1D(BasicBlock1D, [1, 1, 1, 1], dropout_rate=0.1).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        
        # 使用OneCycleLR调度器
        steps_per_epoch = len(train_loader)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.001,
            epochs=n_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # 初始化早停器
        early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        
        # 记录训练过程
        train_losses = []
        val_losses = []
        metrics_history = []
        
        # 开始训练
        for epoch in range(n_epochs):
            train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_metrics = validate(model, val_loader, criterion, device)
            
            scheduler.step()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            metrics_history.append(val_metrics)
            
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Validation Metrics: {val_metrics}")
            
            # 检查早停
            early_stopping(val_loss, val_metrics, model, epoch, fold)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                print(f"Best epoch was {early_stopping.best_epoch+1} with metrics: {early_stopping.best_metrics}")
                model.load_state_dict(early_stopping.best_model)
                break
        
        # 记录本折结果
        fold_results.append({
            'final_val_metrics': early_stopping.best_metrics,
            'best_val_loss': early_stopping.best_loss,
            'best_epoch': early_stopping.best_epoch
        })
        
        # 保存最后一个fold的模型状态到final_model
        if fold == n_folds - 1:
            final_model.load_state_dict(model.state_dict())
        
        # 绘制学习曲线
        plot_learning_curves(train_losses, val_losses, metrics_history)
    
    # 保存最终模型和结果
    save_final_model(final_model, fold_results)
    
    # 计算平均指标和标准差
    avg_metrics = {
        metric: np.mean([fold['final_val_metrics'][metric] for fold in fold_results])
        for metric in ['accuracy', 'precision', 'recall', 'f1']
    }
    std_metrics = {
        metric: np.std([fold['final_val_metrics'][metric] for fold in fold_results])
        for metric in ['accuracy', 'precision', 'recall', 'f1']
    }
    
    # 打印最终结果
    print("\nCross-Validation Results:")
    for metric in avg_metrics:
        print(f"{metric}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")
    
    return fold_results, avg_metrics, std_metrics

if __name__ == "__main__":
    # 设置设备（GPU/CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 开始训练
    results = train_with_cross_validation(device=device)
