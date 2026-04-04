#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生存分析模型定义
包含三种模型架构：
- v0: MLPVanilla (标准MLP)
- v1: ResNetStyle (带残差连接的ResNet架构)
- v2: AttentionNetwork (带自注意力机制的Transformer风格架构)
"""

import math
import torch
import torch.nn as nn
import torchtuples as tt
from pycox.models import CoxPH
from pycox.models.loss import BCESurvLoss


# ========== v0: MLPVanilla (标准MLP) ==========
def create_mlp_model(in_features, num_nodes, out_features=1, 
                     batch_norm=True, dropout=0.3, output_bias=False):
    """
    创建标准MLP模型
    
    Args:
        in_features: 输入特征数
        num_nodes: 隐藏层节点数列表，如 [32, 64, 32]
        out_features: 输出特征数，默认1
        batch_norm: 是否使用批归一化
        dropout: Dropout率
        output_bias: 输出层是否使用偏置
    
    Returns:
        net: 神经网络模型
    """
    net = tt.practical.MLPVanilla(
        in_features=in_features,
        num_nodes=num_nodes,
        out_features=out_features,
        batch_norm=batch_norm,
        dropout=dropout,
        output_bias=output_bias
    )
    return net


# ========== v1: ResNetStyle (带残差连接) ==========
class ResidualBlock(nn.Module):
    """残差块 - ResNet风格"""
    def __init__(self, in_features, out_features, dropout=0.3, batch_norm=False):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features) if batch_norm else None
        self.bn2 = nn.BatchNorm1d(out_features) if batch_norm else None
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # 如果输入输出维度不同，需要投影层
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else None
    
    def forward(self, x):
        identity = x
        
        # 第一层
        out = self.fc1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # 第二层
        out = self.fc2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        
        # 残差连接
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        
        out = out + identity  # 残差连接
        out = self.activation(out)
        out = self.dropout(out)
        out = torch.clamp(out, -20, 20)
        
        return out


class ResNetStyle(nn.Module):
    """ResNet风格的网络 - 带残差连接"""
    def __init__(self, in_features, num_blocks=3, hidden_dim=64, out_features=1, 
                 dropout=0.3, batch_norm=False, output_bias=False):
        super().__init__()
        
        # 输入投影层
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim) if batch_norm else None
        self.activation = nn.GELU()
        self.dropout_input = nn.Dropout(dropout)
        
        # 多个残差块
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, dropout, batch_norm)
            for _ in range(num_blocks)
        ])
        
        # 输出层
        self.output = nn.Linear(hidden_dim, out_features, bias=output_bias)
    
    def forward(self, x):
        # 输入投影
        x = self.input_proj(x)
        if self.bn_input is not None:
            x = self.bn_input(x)
        x = self.activation(x)
        x = self.dropout_input(x)
        
        # 通过残差块
        for block in self.blocks:
            x = block(x)
        
        # 输出
        x = self.output(x)
        # x = torch.clamp(x, -20, 20)
        return x


def create_resnet_model(in_features, num_blocks=3, hidden_dim=64, out_features=1,
                        dropout=0.3, batch_norm=True, output_bias=False):
    """
    创建ResNet风格模型
    
    Args:
        in_features: 输入特征数
        num_blocks: 残差块数量，默认3
        hidden_dim: 隐藏层维度，默认64
        out_features: 输出特征数，默认1
        dropout: Dropout率
        batch_norm: 是否使用批归一化
        output_bias: 输出层是否使用偏置
    
    Returns:
        net: 神经网络模型
    """
    net = ResNetStyle(
        in_features=in_features,
        num_blocks=num_blocks,
        hidden_dim=hidden_dim,
        out_features=out_features,
        dropout=dropout,
        batch_norm=batch_norm,
        output_bias=output_bias
    )
    return net


# ========== v2: AttentionNetwork (带自注意力机制) ==========
class SelfAttention(nn.Module):
    """自注意力层 - 学习特征重要性"""
    def __init__(self, input_dim, attention_dim=32):
        super().__init__()
        self.attention_dim = attention_dim
        
        # 查询、键、值的投影
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        
        # 输出投影
        self.output_proj = nn.Linear(attention_dim, input_dim)
        
        self.scale = attention_dim ** 0.5
        
        # 使用 Xavier 初始化，防止梯度爆炸
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，使用 Xavier uniform 初始化"""
        for module in [self.query, self.key, self.value, self.output_proj]:
            nn.init.xavier_uniform_(module.weight, gain=0.5)  # 使用较小的gain
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # x: (batch_size, features)
        batch_size = x.size(0)
        
        # 计算 Q, K, V
        Q = self.query(x)  # (batch, attention_dim)
        K = self.key(x)    # (batch, attention_dim)
        V = self.value(x)  # (batch, attention_dim)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q.unsqueeze(1), K.unsqueeze(2)) / self.scale
        attention_weights = torch.softmax(attention_scores.squeeze(), dim=-1)
        
        # 应用注意力
        attended = attention_weights.unsqueeze(1) * V
        
        # 输出投影
        output = self.output_proj(attended)
        
        return output


class AttentionBlock(nn.Module):
    """带注意力机制的块"""
    def __init__(self, hidden_dim, dropout=0.3, batch_norm=True):
        super().__init__()
        
        # 自注意力
        self.attention = SelfAttention(hidden_dim, attention_dim=hidden_dim // 2)
        
        # 前馈网络
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 归一化和正则化
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2) if batch_norm else None
        self.bn2 = nn.BatchNorm1d(hidden_dim) if batch_norm else None
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 使用GELU激活函数
        
        # Layer Normalization (比BatchNorm更适合注意力机制)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化前馈网络权重"""
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0)
    
    def forward(self, x):
        # 注意力子层 + 残差连接
        attended = self.attention(x)
        x = self.ln1(x + self.dropout(attended))
        
        # 前馈子层 + 残差连接
        ffn = self.fc1(x)
        if self.bn1 is not None:
            ffn = self.bn1(ffn)
        ffn = self.activation(ffn)
        ffn = self.dropout(ffn)
        
        ffn = self.fc2(ffn)
        if self.bn2 is not None:
            ffn = self.bn2(ffn)
        
        x = self.ln2(x + self.dropout(ffn))
        
        return x


class AttentionNetwork(nn.Module):
    """带自注意力机制的深度网络"""
    def __init__(self, in_features, num_blocks=3, hidden_dim=64, out_features=1, 
                 dropout=0.3, batch_norm=True, output_bias=True):
        super().__init__()
        
        # 输入嵌入层
        self.input_embed = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 多个注意力块
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(hidden_dim, dropout, batch_norm)
            for _ in range(num_blocks)
        ])
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_features, bias=output_bias)
        )
    
    def forward(self, x):
        # 输入嵌入
        x = self.input_embed(x)
        
        # 通过注意力块
        for block in self.attention_blocks:
            x = block(x)
        
        # 输出
        x = self.output(x)
        # 限幅以避免 Cox 部分似然中的 exp 溢出
        x = torch.clamp(x, -20.0, 20.0)
        return x


def create_attention_model(in_features, num_blocks=2, hidden_dim=16, out_features=1,
                           dropout=0.3, batch_norm=False, output_bias=False):
    """
    创建注意力机制模型
    
    Args:
        in_features: 输入特征数
        num_blocks: 注意力块数量，默认2
        hidden_dim: 隐藏层维度，默认16
        out_features: 输出特征数，默认1
        dropout: Dropout率
        batch_norm: 是否使用批归一化
        output_bias: 输出层是否使用偏置
    
    Returns:
        net: 神经网络模型
    """
    net = AttentionNetwork(
        in_features=in_features,
        num_blocks=num_blocks,
        hidden_dim=hidden_dim,
        out_features=out_features,
        dropout=dropout,
        batch_norm=batch_norm,
        output_bias=output_bias
    )
    return net


# ========== v3: CrossAttentionCox (与 PUMC_11_17/train/models/model.py 对齐，用于加载 v3 权重) ==========
class FeatureTokenizer(nn.Module):
    """将 (batch, num_features) 转为 (batch, num_features, embed_dim)，供 v3 Transformer 分支使用。"""

    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.weights = nn.Parameter(torch.randn(1, num_features, embed_dim))
        self.biases = nn.Parameter(torch.zeros(1, num_features, embed_dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x):
        x_expanded = x.unsqueeze(-1)
        return x_expanded * self.weights + self.biases


class CrossNet(nn.Module):
    """Deep Cross Network：x_{l+1} = x_0 * (W_l^T x_l + b_l) + x_l"""

    def __init__(self, in_features, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.kernels = nn.ParameterList([
            nn.Parameter(torch.randn(in_features, 1)) for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(in_features, 1)) for _ in range(num_layers)
        ])
        for p in self.kernels:
            nn.init.xavier_normal_(p)

    def forward(self, x):
        x_0 = x.unsqueeze(2)
        x_l = x_0
        for i in range(self.num_layers):
            xl_w = torch.matmul(x_l.transpose(1, 2), self.kernels[i])
            dot_ = torch.matmul(x_0, xl_w)
            x_l = dot_ + self.biases[i] + x_l
        return x_l.squeeze(2)


class GatingMechanism(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.gate = nn.Linear(in_features, in_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.gate(x))


class CrossAttentionCox(nn.Module):
    """Transformer 分支 + CrossNet 分支 + 融合 MLP（与训练脚本 v3 一致）。"""

    def __init__(self, in_features, num_blocks=2, hidden_dim=32, out_features=1,
                 dropout=0.1, batch_norm=False, output_bias=True, num_heads=4,
                 num_cross_layers=2):
        super().__init__()
        self.gating = GatingMechanism(in_features)
        self.tokenizer = FeatureTokenizer(in_features, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.trans_flatten_dim = in_features * hidden_dim
        self.trans_norm = nn.LayerNorm(self.trans_flatten_dim)
        self.cross_net = CrossNet(in_features, num_layers=num_cross_layers)
        fusion_dim = self.trans_flatten_dim + in_features
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features, bias=output_bias)
        )

    def forward(self, x):
        x_g = self.gating(x)
        x_emb = self.tokenizer(x_g)
        x_trans = self.transformer(x_emb)
        x_trans_flat = x_trans.reshape(x_trans.size(0), -1)
        x_trans_out = self.trans_norm(x_trans_flat)
        x_cross_out = self.cross_net(x_g)
        x_combined = torch.cat([x_trans_out, x_cross_out], dim=1)
        return self.head(x_combined)


def create_cross_attention_model(in_features, num_blocks=2, hidden_dim=32, out_features=1,
                                 dropout=0.1, batch_norm=False, output_bias=False, num_cross_layers=2):
    num_heads = 4
    if hidden_dim % num_heads != 0:
        hidden_dim = (hidden_dim // num_heads + 1) * num_heads
    return CrossAttentionCox(
        in_features=in_features,
        num_blocks=num_blocks,
        hidden_dim=hidden_dim,
        out_features=out_features,
        dropout=dropout,
        batch_norm=batch_norm,
        output_bias=output_bias,
        num_heads=num_heads,
        num_cross_layers=num_cross_layers
    )


# ========== 模型配置类 ==========
class ModelConfig:
    """
    模型配置类，用于控制选择哪种模型架构
    
    支持的模型类型:
    - 'v0' 或 'mlp': MLPVanilla (标准MLP)
    - 'v1' 或 'resnet': ResNetStyle (带残差连接)
    - 'v2' 或 'attention': AttentionNetwork (带自注意力机制)
    - 'v3' 或 'cross_attention': CrossAttentionCox (Transformer + DCN + Gating)
    """
    
    def __init__(self, model_type='v0', **kwargs):
        """
        初始化模型配置
        
        Args:
            model_type: 模型类型，可选 'v0', 'v1', 'v2', 'mlp', 'resnet', 'attention'
            **kwargs: 模型特定参数
                - 对于 v0/mlp:
                    num_nodes: 隐藏层节点数列表，默认 [32, 64, 32]
                    batch_norm: 是否使用批归一化，默认 True
                    dropout: Dropout率，默认 0.3
                    output_bias: 输出层是否使用偏置，默认 False
                
                - 对于 v1/resnet:
                    num_blocks: 残差块数量，默认 3
                    hidden_dim: 隐藏层维度，默认 64
                    batch_norm: 是否使用批归一化，默认 True
                    dropout: Dropout率，默认 0.3
                    output_bias: 输出层是否使用偏置，默认 False
                
                - 对于 v2/attention:
                    num_blocks: 注意力块数量，默认 2
                    hidden_dim: 隐藏层维度，默认 16
                    batch_norm: 是否使用批归一化，默认 False
                    dropout: Dropout率，默认 0.3
                    output_bias: 输出层是否使用偏置，默认 False
        """
        # 标准化模型类型名称
        model_type = model_type.lower()
        if model_type in ['v0', 'mlp']:
            self.model_type = 'v0'
        elif model_type in ['v1', 'resnet']:
            self.model_type = 'v1'
        elif model_type in ['v2', 'attention']:
            self.model_type = 'v2'
        elif model_type in ['v3', 'cross_attention']:
            self.model_type = 'v3'
        else:
            raise ValueError(
                f"不支持的模型类型: {model_type}。支持的类型: "
                f"'v0', 'v1', 'v2', 'v3', 'mlp', 'resnet', 'attention', 'cross_attention'"
            )
        
        # 根据模型类型设置默认参数
        if self.model_type == 'v0':
            self.num_nodes = kwargs.get('num_nodes', [32, 64, 32])
            self.batch_norm = kwargs.get('batch_norm', True)
            self.dropout = kwargs.get('dropout', 0.3)
            self.output_bias = kwargs.get('output_bias', False)
        elif self.model_type == 'v1':
            self.num_blocks = kwargs.get('num_blocks', 3)
            self.hidden_dim = kwargs.get('hidden_dim', 64)
            self.batch_norm = kwargs.get('batch_norm', True)
            self.dropout = kwargs.get('dropout', 0.3)
            self.output_bias = kwargs.get('output_bias', False)
        elif self.model_type == 'v2':
            self.num_blocks = kwargs.get('num_blocks', 2)
            self.hidden_dim = kwargs.get('hidden_dim', 16)
            self.batch_norm = kwargs.get('batch_norm', False)
            self.dropout = kwargs.get('dropout', 0.3)
            self.output_bias = kwargs.get('output_bias', False)
        elif self.model_type == 'v3':
            self.num_blocks = kwargs.get('num_blocks', 2)
            self.hidden_dim = kwargs.get('hidden_dim', 32)
            self.batch_norm = kwargs.get('batch_norm', False)
            self.dropout = kwargs.get('dropout', 0.1)
            self.output_bias = kwargs.get('output_bias', False)
            self.num_cross_layers = kwargs.get('num_cross_layers', 2)

        self.weight_init = kwargs.get('weight_init', 'default')

    def _apply_weight_initialization(self, net):
        """
        根据 weight_init 策略重新初始化网络权重
        """
        strategy = (self.weight_init or 'default').lower()
        if strategy in ('default', 'none'):
            return net

        for module in net.modules():
            if isinstance(module, nn.Linear):
                if strategy == 'zero':
                    nn.init.constant_(module.weight, 0.0)
                elif strategy == 'random':
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                elif strategy == 'kaiming':
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                else:
                    raise ValueError(f"未知的初始化策略: {self.weight_init}")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        return net
    
    def create_network(self, in_features, out_features=1):
        """
        根据配置创建网络
        
        Args:
            in_features: 输入特征数
            out_features: 输出特征数，默认1
        
        Returns:
            net: 神经网络模型
        """
        if self.model_type == 'v0':
            net = create_mlp_model(
                in_features=in_features,
                num_nodes=self.num_nodes,
                out_features=out_features,
                batch_norm=self.batch_norm,
                dropout=self.dropout,
                output_bias=self.output_bias
            )
        elif self.model_type == 'v1':
            net = create_resnet_model(
                in_features=in_features,
                num_blocks=self.num_blocks,
                hidden_dim=self.hidden_dim,
                out_features=out_features,
                dropout=self.dropout,
                batch_norm=self.batch_norm,
                output_bias=self.output_bias
            )
        elif self.model_type == 'v2':
            net = create_attention_model(
                in_features=in_features,
                num_blocks=self.num_blocks,
                hidden_dim=self.hidden_dim,
                out_features=out_features,
                dropout=self.dropout,
                batch_norm=self.batch_norm,
                output_bias=self.output_bias
            )
        elif self.model_type == 'v3':
            net = create_cross_attention_model(
                in_features=in_features,
                num_blocks=self.num_blocks,
                hidden_dim=self.hidden_dim,
                out_features=out_features,
                dropout=self.dropout,
                batch_norm=self.batch_norm,
                output_bias=self.output_bias,
                num_cross_layers=self.num_cross_layers
            )
        else:
            raise ValueError(f"未知的模型类型: {self.model_type}")

        return self._apply_weight_initialization(net)
    
    def create_model(self, in_features, prediction_target='survival', optimizer=None, device=None):
        """
        创建完整的模型（包括网络和优化器）
        
        Args:
            in_features: 输入特征数
            prediction_target: 预测目标，可选 'survival', 'event_only', 'time_only'
            optimizer: 优化器，如果为None则使用默认的Adam
            device: 计算设备（torch.device 或字符串），默认根据环境自动选择
        
        Returns:
            model: 完整的模型对象
            net: 神经网络对象
            model_type: 模型类型字符串
        """
        import torch.nn as nn
        
        # 创建网络
        if prediction_target == 'survival':
            out_features = 1
            output_bias = self.output_bias
        elif prediction_target == 'event_only':
            out_features = 1
            output_bias = True  # 分类任务需要偏置
        elif prediction_target == 'time_only':
            out_features = 1
            output_bias = True  # 回归任务需要偏置
        else:
            raise ValueError(f"未知的预测目标: {prediction_target}")
        
        net = self.create_network(in_features, out_features)
        
        # 创建模型
        if prediction_target == 'survival':
            if optimizer is None:
                model = CoxPH(net, tt.optim.Adam, device=device)
            else:
                model = CoxPH(net, optimizer, device=device)
            model_type = 'survival'
        elif prediction_target == 'event_only':
            if optimizer is None:
                model = tt.Model(net, nn.BCEWithLogitsLoss(), tt.optim.Adam, device=device)
            else:
                model = tt.Model(net, nn.BCEWithLogitsLoss(), optimizer, device=device)
            model_type = 'classification'
        elif prediction_target == 'time_only':
            if optimizer is None:
                model = tt.Model(net, nn.MSELoss(), tt.optim.Adam, device=device)
            else:
                model = tt.Model(net, nn.MSELoss(), optimizer, device=device)
            model_type = 'regression'
        
        return model, net, model_type
    
    def __repr__(self):
        """返回配置的字符串表示"""
        if self.model_type == 'v0':
            return (f"ModelConfig(model_type='{self.model_type}', "
                    f"num_nodes={self.num_nodes}, batch_norm={self.batch_norm}, "
                    f"dropout={self.dropout}, output_bias={self.output_bias}, "
                    f"weight_init='{self.weight_init}')")
        elif self.model_type == 'v1':
            return (f"ModelConfig(model_type='{self.model_type}', "
                    f"num_blocks={self.num_blocks}, hidden_dim={self.hidden_dim}, "
                    f"batch_norm={self.batch_norm}, dropout={self.dropout}, "
                    f"output_bias={self.output_bias}, weight_init='{self.weight_init}')")
        elif self.model_type == 'v2':
            return (f"ModelConfig(model_type='{self.model_type}', "
                    f"num_blocks={self.num_blocks}, hidden_dim={self.hidden_dim}, "
                    f"batch_norm={self.batch_norm}, dropout={self.dropout}, "
                    f"output_bias={self.output_bias}, weight_init='{self.weight_init}')")
        elif self.model_type == 'v3':
            return (f"ModelConfig(model_type='{self.model_type}', "
                    f"num_blocks={self.num_blocks}, hidden_dim={self.hidden_dim}, "
                    f"num_cross_layers={self.num_cross_layers}, "
                    f"batch_norm={self.batch_norm}, dropout={self.dropout}, "
                    f"output_bias={self.output_bias}, weight_init='{self.weight_init}')")


# ========== 便捷函数 ==========
def create_model_from_config(in_features, model_type='v0', prediction_target='survival', device=None, **kwargs):
    """
    便捷函数：根据配置创建模型
    
    Args:
        in_features: 输入特征数
        model_type: 模型类型，可选 'v0', 'v1', 'v2', 'mlp', 'resnet', 'attention'
        prediction_target: 预测目标，可选 'survival', 'event_only', 'time_only'
        device: 计算设备
        **kwargs: 模型特定参数
    
    Returns:
        model: 完整的模型对象
        net: 神经网络对象
        model_type: 模型类型字符串
        config: ModelConfig对象
    """
    config = ModelConfig(model_type=model_type, **kwargs)
    model, net, model_type_str = config.create_model(in_features, prediction_target, device=device)
    return model, net, model_type_str, config


# ========== 示例用法 ==========
if __name__ == '__main__':
    # 示例1: 使用v0模型
    print("=" * 60)
    print("示例1: 使用v0模型 (MLPVanilla)")
    print("=" * 60)
    config_v0 = ModelConfig(model_type='v0', num_nodes=[32, 64, 32], dropout=0.3)
    net_v0 = config_v0.create_network(in_features=10)
    print(f"v0模型配置: {config_v0}")
    print(f"v0网络: {net_v0}")
    print()
    
    # 示例2: 使用v1模型
    print("=" * 60)
    print("示例2: 使用v1模型 (ResNet)")
    print("=" * 60)
    config_v1 = ModelConfig(model_type='v1', num_blocks=3, hidden_dim=64, dropout=0.3)
    net_v1 = config_v1.create_network(in_features=10)
    print(f"v1模型配置: {config_v1}")
    print(f"v1网络: {net_v1}")
    print()
    
    # 示例3: 使用v2模型
    print("=" * 60)
    print("示例3: 使用v2模型 (Attention)")
    print("=" * 60)
    config_v2 = ModelConfig(model_type='v2', num_blocks=2, hidden_dim=16, dropout=0.3)
    net_v2 = config_v2.create_network(in_features=10)
    print(f"v2模型配置: {config_v2}")
    print(f"v2网络: {net_v2}")
    print()
    
    # 示例4: 创建完整模型（包括优化器）
    print("=" * 60)
    print("示例4: 创建完整模型（生存分析）")
    print("=" * 60)
    model, net, model_type = config_v0.create_model(in_features=10, prediction_target='survival')
    print(f"模型类型: {model_type}")
    print(f"模型: {model}")
    print()

