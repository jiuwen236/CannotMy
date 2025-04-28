import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt


class ArknightsDataset(Dataset):
    def __init__(self, csv_file, normalize=True):
        data = pd.read_csv(csv_file, header=None)
        features = data.iloc[:, :-1].values.astype(np.float32)
        labels = data.iloc[:, -1].map({'L': 0, 'R': 1}).values
        
        # 处理可能的无效标签（如第一行的'69'）
        labels = np.where((labels != 0) & (labels != 1), 0, labels).astype(np.float32)

        # 分离左右双方并保留符号信息
        feature_count = features.shape[1]
        midpoint = feature_count // 2  # 应该是34
        
        # 符号信息：1表示己方，-1表示敌方
        self.left_signs = np.sign(features[:, :midpoint])
        self.right_signs = np.sign(features[:, midpoint:])
        
        # 数量信息
        self.left_counts = np.abs(features[:, :midpoint])
        self.right_counts = np.abs(features[:, midpoint:])
        
        
        self.labels = labels
        print(f"数据加载完成! 特征维度: {feature_count}, 样本数量: {len(labels)}")
        print(f"左侧特征平均非零数量: {np.count_nonzero(self.left_counts > 0)/len(self.left_counts):.2f}")
        print(f"右侧特征平均非零数量: {np.count_nonzero(self.right_counts > 0)/len(self.right_counts):.2f}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.left_signs[idx]),
            torch.tensor(self.left_counts[idx]),
            torch.tensor(self.right_signs[idx]),
            torch.tensor(self.right_counts[idx]),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )


class UnitAwareTransformer(nn.Module):
    def __init__(self, feature_dim=34, embed_dim=64, num_heads=4, num_layers=2, dropout_rate=0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # 特征嵌入层
        self.feature_embed = nn.Linear(feature_dim, embed_dim)
        nn.init.xavier_uniform_(self.feature_embed.weight, gain=0.1)
        nn.init.constant_(self.feature_embed.bias, 0.0)
        
        # 位置编码 - 用于区分不同位置的特征
        self.position_embed = nn.Parameter(torch.randn(1, feature_dim, embed_dim) * 0.01)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # 全局平均池化之后的全连接层
        self.fc1 = nn.Linear(embed_dim * 2, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(embed_dim, 1)
        
        # 初始化全连接层
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.1)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.constant_(self.fc2.bias, 0.0)

    def encode_side(self, sign, count):
        # 将符号和数量信息结合
        batch_size = sign.size(0)
        
        # 给数量加上微小值避免全零向量
        count = count + 1e-6
        
        # 结合符号和数量信息 [batch_size, feature_dim]
        features = sign * count
        
        # 使用特征嵌入层进行编码 [batch_size, feature_dim, embed_dim]
        features = self.feature_embed(features).unsqueeze(1)
        features = features + self.position_embed[:, :features.size(1), :]
        
        # 使用Transformer编码 [batch_size, feature_dim, embed_dim]
        features = self.layer_norm1(features)
        features = self.transformer(features)
        features = self.layer_norm2(features)
        
        # 全局平均池化 [batch_size, embed_dim]
        features = torch.mean(features, dim=1)
        
        return features

    def forward(self, left_sign, left_count, right_sign, right_count):
        # 编码左右两侧
        left_features = self.encode_side(left_sign, left_count)
        right_features = self.encode_side(right_sign, right_count)
        
        # 拼接左右两侧特征
        combined = torch.cat([left_features, right_features], dim=1)
        
        # 全连接层处理
        x = self.fc1(combined)
        x = torch.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x).squeeze(-1)
        
        return logits


def train_one_epoch(model, train_loader, criterion, optimizer, device, clip_grad_norm=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    valid_batches = 0

    for ls, lc, rs, rc, labels in train_loader:
        ls, lc, rs, rc, labels = [x.to(device) for x in (ls, lc, rs, rc, labels)]

        optimizer.zero_grad()
        
        # 前向传播
        try:
            logits = model(ls, lc, rs, rc)
            
            # 检查并处理logits中的NaN或Inf
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 损失计算
            loss = criterion(logits, labels)
            
            # 检查损失是否有效
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                
                # 梯度裁剪避免梯度爆炸
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                
                optimizer.step()
                
                # 累加损失
                total_loss += loss.item()
                valid_batches += 1
                
                # 计算准确率
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            else:
                print(f"警告: 跳过无效损失 {loss.item()}")
        
        except RuntimeError as e:
            print(f"警告: 处理批次时出错 - {str(e)}")
            continue

    avg_loss = total_loss / max(1, valid_batches)
    accuracy = 100 * correct / max(1, total)
    
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    valid_batches = 0

    with torch.no_grad():
        for ls, lc, rs, rc, labels in data_loader:
            ls, lc, rs, rc, labels = [x.to(device) for x in (ls, lc, rs, rc, labels)]
            
            try:
                logits = model(ls, lc, rs, rc)
                
                # 确保logits不包含NaN或Inf
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 损失计算
                loss = criterion(logits, labels)
                
                # 累加有效损失
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    valid_batches += 1
                
                # 计算准确率
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            except RuntimeError as e:
                print(f"警告: 验证时处理批次出错 - {str(e)}")
                continue

    avg_loss = total_loss / max(1, valid_batches)
    accuracy = 100 * correct / max(1, total)
    
    return avg_loss, accuracy


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='training_history.png'):
    """绘制训练历史并保存图表"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('损失曲线')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('准确率曲线')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"训练历史已保存到 {save_path}")


def main():
    # 配置参数
    config = {
        'data_file': 'arknights.csv',
        'batch_size': 32,
        'feature_dim': 34,
        'embed_dim': 64,
        'num_heads': 4,
        'num_layers': 2,
        'dropout_rate': 0.2,
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'epochs': 100,
        'patience': 80,
        'seed': 42,
        'save_dir': 'models'
    }

    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
        
        # 设置确定性计算以增加稳定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("警告: 未检测到GPU，将在CPU上运行训练，这可能会很慢!")

    # 加载数据集
    dataset = ArknightsDataset(config['data_file'], normalize=False)
    
    # 数据集分割
    train_indices, val_indices = train_test_split(
        range(len(dataset)), 
        test_size=0.2,  # 提高验证集比例以更好评估模型
        random_state=config['seed'],
        stratify=dataset.labels  # 保证训练集和验证集标签分布一致
    )
    
    print(f"训练集大小: {len(train_indices)}, 验证集大小: {len(val_indices)}")
    print(f"训练集标签分布: {np.bincount(dataset.labels[train_indices].astype(int))}")
    print(f"验证集标签分布: {np.bincount(dataset.labels[val_indices].astype(int))}")

    # 数据加载器
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=config['batch_size'],
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 初始化模型
    model = UnitAwareTransformer(
        feature_dim=config['feature_dim'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.99),
        eps=1e-8
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )

    # 训练历史记录
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # 训练设置
    best_acc = 0
    best_loss = float('inf')
    patience_counter = 0
    
    # 训练循环
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, clip_grad_norm=1.0)
        
        # 验证
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 保存最佳模型（基于准确率）
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best_model_acc.pth'))
            torch.save(model, os.path.join(config['save_dir'], 'best_model_full.pth'))
            patience_counter = 0
            print("保存了新的最佳准确率模型!")
        else:
            patience_counter += 1
            
        # 保存最佳模型（基于损失）
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best_model_loss.pth'))
            print("保存了新的最佳损失模型!")
            
        # 保存最新模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'config': config
        }, os.path.join(config['save_dir'], 'latest_checkpoint.pth'))

        # 打印训练信息
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"早停计数器: {patience_counter}/{config['patience']}")
        print("-" * 40)
        
        # 绘制并保存训练历史
        # if (epoch + 1) % 5 == 0 or epoch == config['epochs'] - 1:
        #     plot_training_history(
        #         train_losses, val_losses, train_accs, val_accs,
        #         save_path=os.path.join(config['save_dir'], 'training_history.png')
        #     )
        
        # 早停
        if patience_counter >= config['patience']:
            print(f"验证准确率 {config['patience']} 个 epoch 未改善，提前停止训练")
            break

    print(f"训练完成! 最佳验证准确率: {best_acc:.2f}%, 最佳验证损失: {best_loss:.4f}")
    
    # 保存最终训练历史
    # plot_training_history(
    #     train_losses, val_losses, train_accs, val_accs,
    #     save_path=os.path.join(config['save_dir'], 'final_training_history.png')
    # )


if __name__ == "__main__":
    main() 