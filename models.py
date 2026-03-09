import torch
import torch.nn as nn
import math
# 注意：这些自定义模块存在，不要给我自创
from mamba_simple import Mamba
from s4 import S4Block
from mamba2_simple_original import Mamba2Simple
from ssd_minimal import test_correctness

# ===================== 增强版MambaAMT模型定义（深度优化版） =====================
class FeatureEnhancementLayer(nn.Module):
    """特征增强层：为MambaAMT定制的输入特征增强（优化版）"""
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()
        # 双通道特征增强
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.fc_shortcut = nn.Linear(input_dim, input_dim)  # 短路连接
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
        # 特征校准系数
        self.gamma = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        residual = x
        shortcut = self.fc_shortcut(residual)
        
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.layer_norm2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        # 融合短路连接和残差
        x = self.gamma * x + residual + 0.1 * shortcut
        x = self.layer_norm1(x)
        return x


class AdvancedMambaAMT(nn.Module):
    """
    超增强版 MambaAMT
    Attention + MambaAMT + TemporalConv + RelativeTimeBias
    """

    def __init__(self, dim_q, dim_k, dim_v, output_dim, dropout_rate, device):
        super().__init__()

        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.device = device

        # ===== Feature Enhance =====
        self.feature_enhance = FeatureEnhancementLayer(dim_q, dim_q * 2, dropout_rate)

        # ===== Multi-head =====
        self.num_heads = 4
        self.head_dim = dim_k // self.num_heads

        # ===== QKV =====
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)

        # ===== Temporal Conv (local trend) =====
        self.temporal_conv = nn.Conv1d(
            dim_v,
            dim_v,
            kernel_size=3,
            padding=1,
            groups=dim_v
        )

        # ===== Relative Time Bias =====
        self.rel_bias = nn.Parameter(torch.zeros(self.num_heads, 512, 512))

        # ===== Attention norm =====
        self._norm_fact = 1 / math.sqrt(self.head_dim)

        self.dropout = nn.Dropout(dropout_rate)

        # ===== Gate Fusion =====
        self.gate = nn.Sequential(
            nn.Linear(dim_v * 2, dim_v),
            nn.Sigmoid()
        )

        # ===== Residual Gate =====
        self.residual_gate = nn.Sequential(
            nn.Linear(dim_v, dim_v),
            nn.Sigmoid()
        )

        # ===== LayerNorm =====
        self.norm = nn.LayerNorm(dim_v)

        # ===== Output =====
        self.fc = nn.Sequential(
            nn.Linear(dim_v, dim_v // 2),
            nn.LayerNorm(dim_v // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_v // 2, output_dim),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        batch, n, _ = x.shape

        # ===== Feature Enhance =====
        x = self.feature_enhance(x)

        # ===== QKV =====
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        # ===== Multi-head reshape =====
        q = q.reshape(batch, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, n, self.num_heads, self.head_dim).transpose(1, 2)
        v_head = v.reshape(batch, n, self.num_heads, self.head_dim).transpose(1, 2)

        # ===== Causal Mask =====
        causal_mask = torch.tril(torch.ones(n, n, device=x.device))

        # ===== Attention =====
        att = torch.matmul(q, k.transpose(-2, -1)) * self._norm_fact

        rel_bias = self.rel_bias[:, :n, :n]
        att = att + rel_bias.unsqueeze(0)

        att = att.masked_fill(causal_mask == 0, float('-inf'))

        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        att_output = torch.matmul(att, v_head)

        att_output = att_output.transpose(1, 2).contiguous().reshape(batch, n, self.dim_v)

        # ===== Temporal Conv =====
        conv_input = v.transpose(1, 2)
        conv_output = self.temporal_conv(conv_input)
        conv_output = conv_output.transpose(1, 2)

        att_output = att_output + conv_output

        # ===== AMT Path =====
        q_flat = q.reshape(-1, n, self.head_dim)
        k_flat = k.reshape(-1, n, self.head_dim)
        v_flat = v_head.reshape(-1, n, self.head_dim)

        amt_output = test_correctness(q_flat, k_flat, v_flat, n)

        amt_output = amt_output.reshape(batch, self.num_heads, n, self.head_dim)
        amt_output = amt_output.transpose(1, 2).contiguous().reshape(batch, n, self.dim_v)

        # ===== Fusion =====
        fusion = torch.cat([att_output, amt_output], dim=-1)

        gate = self.gate(fusion)

        output = gate * amt_output + (1 - gate) * att_output

        # ===== Gated Residual =====
        res_gate = self.residual_gate(output)

        output = x + res_gate * output

        output = self.norm(output)

        # ===== Prediction =====
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
        self.device = device
        self.model_type = model_type
        self.mamba_config = mamba_config

        # ===== QKV =====
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)

        self._norm_fact = 1 / math.sqrt(dim_k)

        self.dropout = nn.Dropout(dropout_rate)

        # ===== Temporal Conv (local trend extractor) =====
        self.temporal_conv = nn.Conv1d(
            dim_v,
            dim_v,
            kernel_size=3,
            padding=1,
            groups=dim_v
        )

        # ===== Relative Time Bias =====
        self.rel_bias = nn.Parameter(torch.zeros(1, 512, 512))

        # ===== RWKV Gate =====
        self.rwkv_gate = nn.Sequential(
            nn.Linear(dim_v * 2, dim_v),
            nn.Sigmoid()
        )

        # ===== Residual Gate =====
        self.res_gate = nn.Sequential(
            nn.Linear(dim_v, dim_v),
            nn.Sigmoid()
        )

        # ===== LayerNorm =====
        self.norm = nn.LayerNorm(dim_v)

        # ===== Output =====
        self.fc = nn.Sequential(
            nn.Linear(dim_v, dim_v // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_v // 2, output_dim)
        )

        # ===== Sequence models =====
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

        else:
            self.model = None

    def forward(self, x):

        batch, n, dim_q = x.shape

        # ===== Sequence model =====
        if self.model_type in ["mamba", "s4", "mamba2_original"]:

            if x.shape[-1] != self.mamba_config['d_model']:
                x = nn.Linear(x.shape[-1], self.mamba_config['d_model']).to(self.device)(x)

            x = self.model(x)

            if isinstance(x, tuple):
                x = x[0]

        # ===== QKV =====
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        # ===== Attention =====
        att = torch.matmul(q, k.transpose(1, 2)) * self._norm_fact

        # relative time bias
        bias = self.rel_bias[:, :n, :n]

        att = att + bias

        # causal mask
        causal_mask = torch.tril(torch.ones(n, n, device=x.device))

        att = att.masked_fill(causal_mask == 0, float('-inf'))

        att = torch.softmax(att, dim=-1)

        att = self.dropout(att)

        att_output = torch.matmul(att, v)

        # ===== Temporal Conv =====
        conv_input = v.transpose(1, 2)

        conv_output = self.temporal_conv(conv_input)

        conv_output = conv_output.transpose(1, 2)

        att_output = att_output + conv_output

        # ===== MambaAMT =====
        if self.model_type in ["MambaAMT", "MambaAMT-Enhanced"]:

            amt_output = test_correctness(q, k, v, n)

            fusion = torch.cat([att_output, amt_output], dim=-1)

            gate = self.rwkv_gate(fusion)

            output = gate * amt_output + (1 - gate) * att_output

        else:

            output = att_output

        # ===== Gated Residual =====
        res = self.res_gate(output)

        output = x + res * output

        output = self.norm(output)

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