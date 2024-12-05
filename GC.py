import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from typing import List, Tuple
import copy
import pandas as pd
import os
from torch.optim.lr_scheduler import OneCycleLR
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

def load_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载所有数据文件并组织数据
    
    Args:
        data_dir: 数据文件目录
        
    Returns:
        X: 形状为(N, 1, sequence_length)的特征数据
        y: 形状为(N,)的标签数据
    """
    class_to_label = {
        'APCK': 0,
        'FRCK': 1,
        'RACK': 2,
        'RHCK': 3
    }
    
    all_features = []
    all_labels = []
    
    # 收集所有特征用于标准化
    all_raw_features = []
    
    # 第一次遍历：收集所有特征
    for filename in os.listdir(data_dir):
        if not filename.endswith('.CSV'):
            continue
            
        class_name = filename[:4]
        if class_name not in class_to_label:
            continue
            
        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath, skiprows=1)
        features = df['Y(Counts)'].values
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
        
        # 使用Savitzky-Golay滤波进行平滑处理
        features = savgol_filter(df['Y(Counts)'].values, window_length=11, polyorder=3)
        
        # 标准化
        features = scaler.transform(features.reshape(-1, 1)).ravel()
        
        all_features.append(features)
        all_labels.append(class_to_label[class_name])
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    X = X[:, np.newaxis, :]
    
    return X, y

class ChromatographyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        
        if self.augment and np.random.random() > 0.5:
            noise = torch.randn_like(x) * 0.002
            x = x + noise
            
            shift = np.random.randint(-3, 4)
            if shift > 0:
                x = torch.cat([x[:, shift:], x[:, :shift]], dim=1)
            elif shift < 0:
                x = torch.cat([x[:, shift:], x[:, :shift]], dim=1)
        
        return x, y

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0.1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=4, dropout_rate=0.1):
        super(ResNet1D, self).__init__()
        self.inplanes = 32  
        self.dropout_rate = dropout_rate
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0005, min_epochs=15, save_dir='models'):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs  # 最小训练轮数
        self.counter = 0
        self.best_loss = float('inf')  # 初始化为无穷大
        self.best_f1 = 0  # 初始化为0
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
        current_f1 = val_metrics['f1']
        
        # 如果还没达到最小训练轮数，直接更新最佳值
        if epoch < self.min_epochs:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_f1 = current_f1
                self.best_model = copy.deepcopy(model.state_dict())
                self.best_epoch = epoch
                self.best_metrics = val_metrics
                self._save_model(model, epoch, val_metrics, fold)
            return False
        
        # 检查是否有显著改善
        loss_improved = val_loss < (self.best_loss - self.min_delta)
        f1_improved = current_f1 > (self.best_f1 + self.min_delta)
        
        if loss_improved or f1_improved:
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
            
        return self.early_stop
            
    def _save_model(self, model, epoch, metrics, fold=None):
        """保存模型和相关信息"""
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
        
        # 梯度裁剪
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
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

def plot_learning_curves(train_losses, val_losses, metrics_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Learning Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
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
    """保存最终的模型和综合结果"""
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
    """加载保存的模型"""
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

def train_with_cross_validation(data_dir='data', n_folds=2, n_epochs=150, batch_size=16, device='cuda'):
    X, y = load_data(data_dir)
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    # 创建最终模型（用于保存最后一个fold的状态）
    final_model = ResNet1D(BasicBlock1D, [1, 1, 1, 1]).to(device)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\nFold {fold+1}/{n_folds}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_dataset = ChromatographyDataset(X_train, y_train, augment=True)
        val_dataset = ChromatographyDataset(X_val, y_val, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        model = ResNet1D(BasicBlock1D, [1, 1, 1, 1], dropout_rate=0.1).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0001)  # 降低初始学习率
        
        # 使用OneCycleLR调度器，调整参数
        steps_per_epoch = len(train_loader)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.001,
            epochs=n_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.4,  # 增加预热阶段
            anneal_strategy='cos',
            div_factor=25.0,  # 初始学习率将是max_lr/25.0
            final_div_factor=1000.0  # 最终学习率将是初始学习率/1000.0
        )
        
        early_stopping = EarlyStopping(patience=15, min_delta=0.0005, min_epochs=10)
        
        train_losses = []
        val_losses = []
        metrics_history = []
        
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
            
            if early_stopping(val_loss, val_metrics, model, epoch, fold):
                print(f"Early stopping triggered at epoch {epoch+1}")
                print(f"Best epoch was {early_stopping.best_epoch+1} with metrics: {early_stopping.best_metrics}")
                model.load_state_dict(early_stopping.best_model)
                break
        
        fold_results.append({
            'final_val_metrics': early_stopping.best_metrics,
            'best_val_loss': early_stopping.best_loss,
            'best_epoch': early_stopping.best_epoch
        })
        
        # 保存最后一个fold的模型状态到final_model
        if fold == n_folds - 1:
            final_model.load_state_dict(model.state_dict())
        
        plot_learning_curves(train_losses, val_losses, metrics_history)
    
    # 保存最终模型和结果
    save_final_model(final_model, fold_results)
    
    avg_metrics = {
        metric: np.mean([fold['final_val_metrics'][metric] for fold in fold_results])
        for metric in ['accuracy', 'precision', 'recall', 'f1']
    }
    std_metrics = {
        metric: np.std([fold['final_val_metrics'][metric] for fold in fold_results])
        for metric in ['accuracy', 'precision', 'recall', 'f1']
    }
    
    print("\nCross-Validation Results:")
    for metric in avg_metrics:
        print(f"{metric}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")
    
    return fold_results, avg_metrics, std_metrics

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = train_with_cross_validation(device=device)
