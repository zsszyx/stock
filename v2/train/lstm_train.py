import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets.dataset import MultiTableTorchIterableDataset
from model.lstm import ImprovedStockLSTMClassifier, StockPatternLoss
from bao_data import DB_PATH
from bao_data.prepare import get_table_names_with_connection
from datasets.preprocess import feature_fields

def print_gpu_memory_info(device, stage=""):
    """打印GPU显存使用情况"""
    if torch.cuda.is_available() and device.type == 'cuda':
        print(f"\n{'='*20} GPU显存信息 {stage} {'='*20}")
        
        # 获取当前GPU
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        # 显存信息（以MB为单位）
        allocated = torch.cuda.memory_allocated(current_device) / 1024**2
        cached = torch.cuda.memory_reserved(current_device) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(current_device) / 1024**2
        total_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**2
        
        print(f"GPU设备: {device_name}")
        print(f"总显存: {total_memory:.1f} MB")
        print(f"已分配: {allocated:.1f} MB ({allocated/total_memory*100:.1f}%)")
        print(f"已缓存: {cached:.1f} MB ({cached/total_memory*100:.1f}%)")
        print(f"峰值分配: {max_allocated:.1f} MB ({max_allocated/total_memory*100:.1f}%)")
        print(f"可用显存: {total_memory-cached:.1f} MB")
        print("="*60)
    else:
        print(f"\n{'='*20} 使用CPU训练 {stage} {'='*20}")

class StockTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 device, save_dir='./checkpoints', log_dir='./logs', positive_class=1):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.positive_class = positive_class  # 正样本类别（通常是1，表示上涨）
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # 训练历史记录 - 重点关注正样本指标
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            # 正样本专门指标
            'positive_precision': [],    # 正样本精确率
            'positive_recall': [],       # 正样本召回率 
            'positive_f1': [],           # 正样本F1
            'positive_specificity': [],  # 特异性（负样本正确率）
            'positive_support': [],      # 正样本数量
            'train_positive_acc': [],    # 训练时正样本准确率
            'val_positive_acc': []       # 验证时正样本准确率
        }
        
        self.best_val_acc = 0.0
        self.best_positive_f1 = 0.0  # 最佳正样本F1
        self.best_model_state = None
        
    def calculate_positive_metrics(self, targets, predictions):
        """计算正样本相关指标"""
        # 转换为numpy数组
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
            
        # 正样本mask
        positive_mask = targets == self.positive_class
        negative_mask = targets != self.positive_class
        
        # 基础统计
        tp = np.sum((predictions == self.positive_class) & (targets == self.positive_class))  # 真正例
        fp = np.sum((predictions == self.positive_class) & (targets != self.positive_class))  # 假正例
        tn = np.sum((predictions != self.positive_class) & (targets != self.positive_class))  # 真负例
        fn = np.sum((predictions != self.positive_class) & (targets == self.positive_class))  # 假负例
        
        # 正样本指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # 精确率
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0      # 召回率（敏感性）
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # 特异性
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 正样本准确率
        positive_acc = np.sum(predictions[positive_mask] == targets[positive_mask]) / np.sum(positive_mask) if np.sum(positive_mask) > 0 else 0.0
        
        # 正样本支持度
        support = np.sum(positive_mask)
        
        return {
            'precision': precision * 100,
            'recall': recall * 100,
            'specificity': specificity * 100,
            'f1': f1 * 100,
            'accuracy': positive_acc * 100,
            'support': support,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        }
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # 使用tqdm显示进度条
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, (data, targets) in enumerate(pbar):
            # 移动数据到设备
            data = data.to(self.device)
            targets = targets.to(self.device).long()
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs, attention_weights = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # 更新进度条 - 显示正样本指标
            if batch_idx % 20 == 0:
                current_targets = np.array(all_targets)
                current_preds = np.array(all_predictions)
                pos_metrics = self.calculate_positive_metrics(current_targets, current_preds)
                
                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Pos_Rec': f'{pos_metrics["recall"]:.1f}',
                    'Pos_Prec': f'{pos_metrics["precision"]:.1f}'
                })
        
        # 计算epoch指标
        num_batches = batch_idx + 1
        epoch_loss = running_loss / num_batches
        
        # 整体准确率
        accuracy = accuracy_score(all_targets, all_predictions)
        
        # 正样本指标
        pos_metrics = self.calculate_positive_metrics(all_targets, all_predictions)
        
        return epoch_loss, accuracy * 100, pos_metrics
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        all_attention_weights = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
            
            for batch_idx, (data, targets) in enumerate(pbar):
                data = data.to(self.device)
                targets = targets.to(self.device).long()
                
                outputs, attention_weights = self.model(data)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                if attention_weights is not None:
                    all_attention_weights.append(attention_weights.cpu().numpy())
                
                # 更新进度条
                if batch_idx % 20 == 0:
                    pbar.set_postfix({'Loss': f'{running_loss/(batch_idx+1):.4f}'})
        
        # 计算epoch指标
        num_batches = batch_idx + 1
        epoch_loss = running_loss / num_batches
        
        # 整体指标
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted', zero_division=0
        )
        
        # 正样本指标
        pos_metrics = self.calculate_positive_metrics(all_targets, all_predictions)
        
        # 打印分类报告
        if epoch % 5 == 0:  # 每5个epoch打印一次详细报告
            print(f"\n=== Epoch {epoch+1} 分类报告 ===")
            print(classification_report(all_targets, all_predictions, 
                                      target_names=[f'Class_{i}' for i in range(max(all_targets)+1)]))
            print(f"正样本详细指标:")
            print(f"  TP: {pos_metrics['tp']}, FP: {pos_metrics['fp']}")
            print(f"  TN: {pos_metrics['tn']}, FN: {pos_metrics['fn']}")
            print(f"  正样本支持度: {pos_metrics['support']}")
        
        return (epoch_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100, 
                pos_metrics, all_attention_weights, all_targets, all_predictions)
    
    def plot_positive_metrics(self):
        """绘制正样本指标趋势图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 正样本精确率和召回率
        axes[0, 0].plot(self.history['positive_precision'], label='Precision', color='green', linewidth=2)
        axes[0, 0].plot(self.history['positive_recall'], label='Recall', color='red', linewidth=2)
        axes[0, 0].plot(self.history['positive_f1'], label='F1-Score', color='blue', linewidth=2)
        axes[0, 0].set_title('正样本核心指标趋势', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Score (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 100])
        
        # 正样本准确率对比
        axes[0, 1].plot(self.history['train_positive_acc'], label='Train Pos Acc', color='blue', linewidth=2)
        axes[0, 1].plot(self.history['val_positive_acc'], label='Val Pos Acc', color='red', linewidth=2)
        axes[0, 1].set_title('正样本准确率对比', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 100])
        
        # 敏感性和特异性
        axes[1, 0].plot(self.history['positive_recall'], label='敏感性(Sensitivity)', color='orange', linewidth=2)
        axes[1, 0].plot(self.history['positive_specificity'], label='特异性(Specificity)', color='purple', linewidth=2)
        axes[1, 0].set_title('敏感性 vs 特异性', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 100])
        
        # 正样本支持度
        axes[1, 1].plot(self.history['positive_support'], label='正样本数量', color='brown', linewidth=2)
        axes[1, 1].set_title('正样本支持度', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Sample Count')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'positive_metrics_trends.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_checkpoint(self, epoch, is_best=False, is_best_positive=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'best_positive_f1': self.best_positive_f1
        }
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(self.save_dir, 'latest_checkpoint.pth'))
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
            print(f"✓ Best overall model saved with validation accuracy: {self.best_val_acc:.2f}%")
            
        # 保存最佳正样本模型
        if is_best_positive:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_positive_model.pth'))
            print(f"✓ Best positive model saved with positive F1: {self.best_positive_f1:.2f}%")
    
    def train(self, num_epochs, save_every=5, scheduler=None):
        """完整训练流程"""
        print(f"开始训练模型，共 {num_epochs} 个epoch")
        print(f"设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"正样本类别: {self.positive_class}")
        print("-" * 80)

        # 显示训练开始前的显存情况
        print_gpu_memory_info(self.device, "训练开始前")
        
        # 尝试运行一个小批次来估算显存使用
        try:
            print("正在估算显存需求...")
            self.model.train()
            
            # 获取一个批次的数据
            sample_batch = next(iter(self.train_loader))
            data, targets = sample_batch
            data = data.to(self.device)
            targets = targets.to(self.device).long()
            
            # 前向传播
            with torch.no_grad():
                outputs, attention_weights = self.model(data)
            
            print_gpu_memory_info(self.device, "加载一个batch后")
            
            # 模拟一次完整的前向+反向传播
            self.optimizer.zero_grad()
            outputs, attention_weights = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            print_gpu_memory_info(self.device, "完成一次前向+反向传播后")
            
            # 清理
            del data, targets, outputs, loss
            if attention_weights is not None:
                del attention_weights
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"显存估算过程中出现警告: {e}")
            print("继续训练...")
        
        print("-" * 80)
        
        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc, train_pos_metrics = self.train_epoch(epoch)
            
            # 验证
            (val_loss, val_acc, val_precision, val_recall, val_f1, 
             val_pos_metrics, attention_weights, val_targets, val_predictions) = self.validate_epoch(epoch)
            
            # 记录历史
            self.history['train_loss'].append(float(train_loss))
            self.history['train_acc'].append(float(train_acc))
            self.history['val_loss'].append(float(val_loss))
            self.history['val_acc'].append(float(val_acc))
            self.history['val_precision'].append(float(val_precision))
            self.history['val_recall'].append(float(val_recall))
            self.history['val_f1'].append(float(val_f1))
            
            # 正样本指标记录
            self.history['positive_precision'].append(float(val_pos_metrics['precision']))
            self.history['positive_recall'].append(float(val_pos_metrics['recall']))
            self.history['positive_f1'].append(float(val_pos_metrics['f1']))
            self.history['positive_specificity'].append(float(val_pos_metrics['specificity']))
            self.history['positive_support'].append(float(val_pos_metrics['support']))
            self.history['train_positive_acc'].append(float(train_pos_metrics['accuracy']))
            self.history['val_positive_acc'].append(float(val_pos_metrics['accuracy']))
            
            # 保存注意力权重样本
            if attention_weights and len(attention_weights) > 0:
                self.sample_attention = np.concatenate(attention_weights, axis=0)[:20]
            
            # 打印结果 - 重点显示正样本指标
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Pos_Acc: {train_pos_metrics['accuracy']:.2f}%")
            print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Pos_Acc: {val_pos_metrics['accuracy']:.2f}%")
            print(f"  正样本 - Prec: {val_pos_metrics['precision']:.2f}%, Rec: {val_pos_metrics['recall']:.2f}%, F1: {val_pos_metrics['f1']:.2f}%")
            print(f"  正样本 - Spec: {val_pos_metrics['specificity']:.2f}%, Support: {val_pos_metrics['support']}")
            
            # 更新学习率
            if scheduler:
                scheduler.step(val_pos_metrics['f1'])  # 基于正样本F1调整学习率
            
            # 检查是否是最佳模型
            is_best = val_acc > self.best_val_acc
            is_best_positive = val_pos_metrics['f1'] > self.best_positive_f1
            
            if is_best:
                self.best_val_acc = val_acc
                
            if is_best_positive:
                self.best_positive_f1 = val_pos_metrics['f1']
                self.best_model_state = self.model.state_dict().copy()
            
            # 保存检查点
            if (epoch + 1) % save_every == 0 or is_best or is_best_positive:
                self.save_checkpoint(epoch, is_best, is_best_positive)
            
            print("-" * 80)
        
        # 保存训练历史图
        self.plot_training_history()
        self.plot_positive_metrics()  # 新增正样本指标图
        
        # 保存训练历史到JSON
        history_path = os.path.join(self.log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"训练完成！")
        print(f"最佳验证准确率: {self.best_val_acc:.2f}%")
        print(f"最佳正样本F1: {self.best_positive_f1:.2f}%")
        print(f"模型和日志保存在: {self.save_dir}, {self.log_dir}")

    def plot_training_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc', color='blue')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 精确率、召回率、F1
        axes[1, 0].plot(self.history['val_precision'], label='Precision', color='green')
        axes[1, 0].plot(self.history['val_recall'], label='Recall', color='orange')
        axes[1, 0].plot(self.history['val_f1'], label='F1', color='purple')
        axes[1, 0].set_title('Validation Metrics')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 注意力权重可视化（如果有的话）
        if hasattr(self, 'sample_attention'):
            axes[1, 1].imshow(self.sample_attention[:5], cmap='Blues', aspect='auto')
            axes[1, 1].set_title('Sample Attention Weights')
            axes[1, 1].set_xlabel('Time Steps')
            axes[1, 1].set_ylabel('Samples')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Attention Data', 
                           horizontalalignment='center', verticalalignment='center')
            axes[1, 1].set_title('Attention Weights')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_history.png'))
        plt.close()

def create_data_loaders(batch_size=32, window_size=30, train_ratio=0.8):
    """创建数据加载器"""
    print("正在创建数据加载器...")
    
    # 获取所有表名
    table_names = get_table_names_with_connection()
    print(f"找到 {len(table_names)} 个数据表")
    
    # 分割训练和验证集
    split_idx = int(len(table_names) * train_ratio)
    train_tables = table_names[:split_idx]
    val_tables = table_names[split_idx:]
    
    print(f"训练表数量: {len(train_tables)}, 验证表数量: {len(val_tables)}")
    
    # 创建数据集
    train_dataset = MultiTableTorchIterableDataset(
        DB_PATH, train_tables, window_size=window_size
    )
    val_dataset = MultiTableTorchIterableDataset(
        DB_PATH, val_tables, window_size=window_size
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    
    return train_loader, val_loader

def main():
    """主训练函数"""
    # 训练参数
    BATCH_SIZE = 2048*2*2
    WINDOW_SIZE = 30
    FEATURE_DIM = len(feature_fields) - 1  # 减去label字段
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    NUM_CLASSES = 2  # 根据你的数据调整
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    POSITIVE_CLASS = 1  # 正样本类别（上涨）
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        batch_size=BATCH_SIZE, 
        window_size=WINDOW_SIZE
    )
    
    # 创建模型
    model = ImprovedStockLSTMClassifier(
        feature_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES,
        use_attention=True,
        use_residual=True
    ).to(device)
    
    # 创建损失函数和优化器
    criterion = StockPatternLoss(alpha=0.5, gamma=1.5, class_weights=torch.tensor([0.5, 1]).to(device))

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    # 学习率调度器 - 基于正样本F1调整
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # 创建训练器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'./checkpoints/lstm_{timestamp}'
    log_dir = f'./logs/lstm_{timestamp}'
    
    trainer = StockTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=save_dir,
        log_dir=log_dir,
        positive_class=POSITIVE_CLASS
    )
    
    # 开始训练
    try:
        trainer.train(num_epochs=NUM_EPOCHS, save_every=5, scheduler=scheduler)
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        trainer.save_checkpoint(len(trainer.history['train_loss']) - 1)
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("训练结束")

if __name__ == "__main__":
    main()