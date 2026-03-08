import torch
import torch.nn as nn
import math
# 注意：这些自定义模块存在，不要给我自创
from mamba_simple import Mamba
from s4 import S4Block
from mamba2_simple_original import Mamba2Simple
from ssd_minimal import test_correctness

# ===================== 增强版MambaAMT模型定义（修复核心错误） =====================
class FeatureEnhancementLayer(nn.Module):
    """特征增强层：为MambaAMT定制的输入特征增强"""
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
    
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x

class AdvancedMambaAMT(nn.Module):
    """增强版MambaAMT：融合因果注意力和MambaAMT的混合架构（修复维度错误）"""
    def __init__(self, dim_q, dim_k, dim_v, output_dim, dropout_rate, device):
        super().__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.device = device
        
        # 特征增强层
        self.feature_enhance = FeatureEnhancementLayer(dim_q, dim_q * 2, dropout_rate)
        
        # 多头注意力配置（拆分为4个头）
        self.num_heads = 4
        self.head_dim = dim_k // self.num_heads
        # 确保head_dim是整数
        if self.head_dim * self.num_heads != dim_k:
            self.head_dim = dim_k // self.num_heads
            self.num_heads = dim_k // self.head_dim
        
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        
        # MambaAMT专属的门控机制
        self.gate = nn.Sequential(
            nn.Linear(dim_v, dim_v),
            nn.Sigmoid()
        )
        
        # 因果注意力掩码增强
        self._norm_fact = 1 / math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 输出层（两层感知机增强）
        self.fc = nn.Sequential(
            nn.Linear(dim_v, dim_v // 2),
            nn.LayerNorm(dim_v // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_v // 2, output_dim),
            nn.Sigmoid()  # 确保输出在0-1之间
        )
        
        # MambaAMT权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """MambaAMT专属初始化策略"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x):
        batch, n, dim_q = x.shape
        
        # 1. 特征增强
        x = self.feature_enhance(x)
        
        # 2. 多头拆分 - 修复1：使用reshape替代view，处理内存不连续问题
        q = self.linear_q(x).reshape(batch, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.linear_k(x).reshape(batch, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.linear_v(x).reshape(batch, n, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. 增强型因果掩码（修复时间衰减因子计算）
        att_weights = torch.matmul(q, k.transpose(-2, -1)) * self._norm_fact
        
        # 创建因果掩码
        causal_mask = torch.tril(torch.ones(n, n)).to(x.device)
        
        # 修复：正确计算时间衰减因子
        # 步骤1：创建时间步向量 [n]
        time_steps = torch.arange(n).float().to(x.device)  # [n]
        # 步骤2：计算衰减因子 [n]
        time_decay = torch.exp(-time_steps / 10)  # 近期数据权重更高
        # 步骤3：扩展为[n, n]矩阵（广播机制）
        time_decay_matrix = time_decay.unsqueeze(1) * time_decay.unsqueeze(0)  # [n, n]
        # 步骤4：扩展维度以匹配注意力权重 [batch, num_heads, n, n]
        time_decay_matrix = time_decay_matrix.unsqueeze(0).unsqueeze(0)  # [1, 1, n, n]
        time_decay_matrix = time_decay_matrix.expand(batch, self.num_heads, -1, -1)  # [batch, num_heads, n, n]
        
        # 应用时间衰减到因果掩码
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch, self.num_heads, -1, -1)
        causal_mask = causal_mask * time_decay_matrix
        
        # 应用掩码
        att_weights = att_weights.masked_fill(causal_mask == 0, float('-inf'))
        att_weights = torch.softmax(att_weights, dim=-1)
        att_weights = self.dropout(att_weights)
        
        # 4. MambaAMT核心计算（替换原有逻辑）
        # 修复2：使用reshape替代view，确保维度计算正确
        batch_size = q.size(0)
        v_flat = v.reshape(-1, n, self.head_dim)
        q_flat = q.reshape(-1, n, self.head_dim)
        k_flat = k.reshape(-1, n, self.head_dim)
        
        # 调用MambaAMT核心函数
        output_amt = test_correctness(q_flat, k_flat, v_flat, n)
        # 修复3：reshape回原维度，确保连续
        output_amt = output_amt.reshape(batch_size, self.num_heads, n, self.head_dim)
        
        # 5. 门控融合
        # 修复4：先contiguous再reshape，确保内存连续
        output_amt = output_amt.transpose(1, 2).contiguous().reshape(batch, n, self.dim_v)
        gate_weights = self.gate(output_amt)
        output = output_amt * gate_weights
        
        # 6. 输出层
        prediction = self.fc(output)
        
        return prediction

# ===================== 原有模型保持不变 =====================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class CombinedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate, model_type, mamba_config, device):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.model_type = model_type
        self.device = device
        self.mamba_config = mamba_config

        if model_type == "mamba":
            self.model = Mamba(
                d_model=mamba_config['d_model'],
                d_state=mamba_config['d_state'],
                d_conv=mamba_config['d_conv'],
                expand=mamba_config['expand'],
                layer_idx=0, 
                device=device, 
                dtype=torch.float32
            ).to(device)
        elif model_type == "s4":
            self.model = S4Block(
                d_model=mamba_config['d_model'],
                bottleneck=None, 
                gate=None, 
                gate_act=None, 
                mult_act=None,
                final_act='glu', 
                dropout=0.1, 
                tie_dropout=False, 
                transposed=False
            ).to(device)
        elif model_type == "mamba2_original":
            self.model = Mamba2Simple(
                d_model=mamba_config['d_model'],
                d_state=32, 
                d_conv=4, 
                expand=8, 
                headdim=32
            ).to(device)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.view(-1, x.shape[2], x.shape[3])
        
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        if len(out.shape) == 2:
            out = out.unsqueeze(1)
        
        if self.model_type in ["mamba", "s4", "mamba2_original"]:
            if out.shape[-1] != self.mamba_config['d_model']:
                out = nn.Linear(out.shape[-1], self.mamba_config['d_model']).to(self.device)(out)
            out = self.model(out)
            if isinstance(out, tuple):
                out = out[0]
        
        out = out.squeeze(1)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, output_dim, dropout_rate):
        super().__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k)
        self.fc = nn.Linear(dim_v, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        
        att_weights = torch.matmul(q, k.transpose(1, 2)) * self._norm_fact
        causal_mask = torch.tril(torch.ones(n, n)).to(x.device)
        att_weights = att_weights.masked_fill(causal_mask == 0, float('-inf'))
        att_weights = torch.softmax(att_weights, dim=-1)
        att_weights = self.dropout(att_weights)
        
        output = torch.matmul(att_weights, v)
        prediction = self.fc(output)
        return prediction

class SelfAttentionMamba(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, output_dim, dropout_rate, model_type, mamba_config, device):
        super().__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k)
        self.fc = nn.Linear(dim_v, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.model_type = model_type
        self.device = device
        self.mamba_config = mamba_config

        if model_type == "mamba":
            self.model = Mamba(
                d_model=mamba_config['d_model'],
                d_state=mamba_config['d_state'],
                d_conv=mamba_config['d_conv'],
                expand=mamba_config['expand'],
                layer_idx=0, 
                device=device, 
                dtype=torch.float32
            ).to(device)
        elif model_type == "s4":
            self.model = S4Block(
                d_model=mamba_config['d_model'],
                bottleneck=None, 
                gate=None, 
                gate_act=None, 
                mult_act=None,
                final_act='glu', 
                dropout=0.1, 
                tie_dropout=False, 
                transposed=False
            ).to(device)
        elif model_type == "mamba2_original":
            self.model = Mamba2Simple(
                d_model=mamba_config['d_model'],
                d_state=32, 
                d_conv=4, 
                expand=8, 
                headdim=32
            ).to(device)
        elif model_type == "MambaAMT":
            self.model = None

    def forward(self, x):
        batch, n, dim_q = x.shape
        
        if self.model_type in ["mamba", "s4", "mamba2_original"]:
            if x.shape[-1] != self.mamba_config['d_model']:
                x = nn.Linear(x.shape[-1], self.mamba_config['d_model']).to(self.device)(x)
            x = self.model(x)
            if isinstance(x, tuple):
                x = x[0]

        assert x.shape[-1] == self.dim_q
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        
        att_weights = torch.matmul(q, k.transpose(1, 2)) * self._norm_fact
        causal_mask = torch.tril(torch.ones(n, n)).to(x.device)
        att_weights = att_weights.masked_fill(causal_mask == 0, float('-inf'))
        att_weights = torch.softmax(att_weights, dim=-1)
        att_weights = self.dropout(att_weights)
        
        if self.model_type == "MambaAMT":
            output = test_correctness(q, k, v, n)
        else:
            output = torch.matmul(att_weights, v)
        
        prediction = self.fc(output)
        return prediction

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, 
                           dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output

class CombinedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate, model_type, mamba_config, device):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.model_type = model_type
        self.device = device
        self.mamba_config = mamba_config

        if model_type == "mamba":
            self.model = Mamba(
                d_model=mamba_config['d_model'],
                d_state=mamba_config['d_state'],
                d_conv=mamba_config['d_conv'],
                expand=mamba_config['expand'],
                layer_idx=0, 
                device=device, 
                dtype=torch.float32
            ).to(device)
        elif model_type == "s4":
            self.model = S4Block(
                d_model=mamba_config['d_model'],
                bottleneck=None, 
                gate=None, 
                gate_act=None, 
                mult_act=None,
                final_act='glu', 
                dropout=0.1, 
                tie_dropout=False, 
                transposed=False
            ).to(device)
        elif model_type == "mamba2_original":
            self.model = Mamba2Simple(
                d_model=mamba_config['d_model'],
                d_state=32, 
                d_conv=4, 
                expand=8, 
                headdim=32
            ).to(device)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        if self.model_type in ["mamba", "s4", "mamba2_original"]:
            if lstm_out.shape[-1] != self.mamba_config['d_model']:
                adapt_layer = nn.Linear(lstm_out.shape[-1], self.mamba_config['d_model']).to(self.device)
                lstm_out = adapt_layer(lstm_out)
            
            lstm_out = self.model(lstm_out)
            if isinstance(lstm_out, tuple):
                lstm_out = lstm_out[0]
        
        output = self.fc(lstm_out)
        return output