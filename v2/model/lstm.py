import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ImprovedStockLSTMClassifier(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers=2, dropout=0.3, 
                 num_classes=3, use_attention=True, use_residual=True):
        """
        改进的股票LSTM分类器
        
        Args:
            feature_dim: 输入特征维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout: dropout概率
            num_classes: 分类数量 (3: 上涨/下跌/持平)
            use_attention: 是否使用注意力机制
            use_residual: 是否使用残差连接
        """
        super(ImprovedStockLSTMClassifier, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # 特征预处理层
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 注意力机制
        if self.use_attention:
            self.attention = AttentionModule(hidden_dim * 2)
        
        # 残差连接的全连接层
        fc_input_dim = hidden_dim * 2
        
        self.fc_layers = nn.ModuleList([
            nn.Linear(fc_input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, num_classes)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """修复的权重初始化"""
        for name, param in self.named_parameters():
            if param.dim() >= 2:  # 只对二维及以上的参数进行Xavier/Kaiming初始化
                if 'lstm' in name and 'weight' in name:
                    # LSTM权重使用Xavier初始化
                    nn.init.xavier_uniform_(param)
                elif 'weight' in name:
                    # 其他权重使用He初始化
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif param.dim() == 1:  # 一维参数（bias等）
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                else:
                    # 其他一维参数使用正态分布初始化
                    nn.init.normal_(param, 0, 0.01)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch_size, window_size, feature_dim]
        
        Returns:
            output: [batch_size, num_classes]
            attention_weights: [batch_size, window_size] (if use_attention)
        """
        batch_size, window_size, _ = x.shape
        
        # 特征预处理
        # x = self.feature_norm(x)
        x = self.feature_projection(x)
        x = F.relu(x)
        
        # LSTM处理
        lstm_out, (hidden, cell) = self.lstm(x)  # [batch_size, window_size, hidden_dim*2]
        
        # 注意力机制或简单池化
        if self.use_attention:
            attended_out, attention_weights = self.attention(lstm_out)
            pooled_out = attended_out
        else:
            # 使用最后一个时间步和最大池化的组合
            last_out = lstm_out[:, -1, :]  # [batch_size, hidden_dim*2]
            max_out = torch.max(lstm_out, dim=1)[0]  # [batch_size, hidden_dim*2]
            pooled_out = (last_out + max_out) / 2
            attention_weights = None
        
        # 全连接层处理
        out = pooled_out
        for i, fc in enumerate(self.fc_layers[:-1]):
            # 修复残差连接的维度检查
            if self.use_residual and i > 0 and out.shape[-1] == fc.out_features:
                residual = out
            else:
                residual = None
                
            out = fc(out)
            
            if i == 0:
                out = self.layer_norm(out)
            
            out = F.relu(out)
            out = self.dropout(out)
            
            if residual is not None:
                out = out + residual
        
        # 最后的分类层
        out = self.fc_layers[-1](out)
        
        return out, attention_weights

class AttentionModule(nn.Module):
    """注意力机制模块"""
    def __init__(self, hidden_dim):
        super(AttentionModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_weights = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_out):
        """
        Args:
            lstm_out: [batch_size, window_size, hidden_dim]
        
        Returns:
            attended_out: [batch_size, hidden_dim]
            attention_weights: [batch_size, window_size]
        """
        # 计算注意力分数
        attention_scores = self.attention_weights(lstm_out)  # [batch_size, window_size, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, window_size]
        
        # 应用softmax
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, window_size]
        
        # 加权求和
        attended_out = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)  # [batch_size, hidden_dim]
        
        return attended_out, attention_weights

class StockPatternLoss(nn.Module):
    """
    Computes the focal loss with optional class weights for stock pattern recognition.
    Args:
        predictions (torch.Tensor): Predicted logits of shape (batch_size, num_classes).
        targets (torch.Tensor): Ground truth labels of shape (batch_size,).
        alpha (float): Focal loss alpha parameter, default is 0.25.
        gamma (float): Focal loss gamma parameter, default is 2.0.
        class_weights (torch.Tensor, optional): Class weights for the loss function.
    Returns:
        torch.Tensor: Scalar tensor representing the mean focal loss over the batch.
    """
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super(StockPatternLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        
    def forward(self, predictions, targets):
        """
        Focal Loss + 类别权重
        """
        ce_loss = F.cross_entropy(predictions, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return ce_loss.mean()

# 示例用法和测试
if __name__ == "__main__":
    # 模型参数
    batch_size = 32
    window_size = 30
    feature_dim = 5  # 开盘价、收盘价、最高价、最低价、成交量、技术指标等
    hidden_dim = 64
    num_classes = 2  # 上涨、下跌、持平
    
    # 创建改进的模型
    model = ImprovedStockLSTMClassifier(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.3,
        num_classes=num_classes,
        use_attention=True,
        use_residual=True
    )
    
    # 测试输入
    sample_input = torch.randn(batch_size, window_size, feature_dim)
    sample_targets = torch.randint(0, num_classes, (batch_size,))
    
    # 前向传播
    try:
        with torch.no_grad():
            output, attention_weights = model(sample_input)
            print(f"Output shape: {output.shape}")  # [batch_size, num_classes]
            if attention_weights is not None:
                print(f"Attention weights shape: {attention_weights.shape}")  # [batch_size, window_size]
        
        # 测试损失函数
        criterion = StockPatternLoss(alpha=0.25, gamma=2.0)
        loss = criterion(output, sample_targets)
        print(f"Loss: {loss.item()}")
        
        # 打印模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        print("模型测试成功!")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()