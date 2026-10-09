"""Model zoo for the rework: clean definitions, not the tangle in the submitted models.py.

Every model takes windows (N, T, F) and returns a single logit per window. The proposed model is
a gated fusion of a causal attention branch and a selective state-space branch, which is what the
code did once the second branch was made real. The branch itself is the one verified by the
causality regression test in ssm_branch.py.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssm_branch import SelectiveSSMBranch, causal_depthwise_conv


class MLPClassifier(nn.Module):
    def __init__(self, n_feat, seq_len, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_feat * seq_len, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LSTMClassifier(nn.Module):
    def __init__(self, n_feat, seq_len, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(n_feat, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class AttentionClassifier(nn.Module):
    """Causal single-head attention over the window, with a left-padded depthwise convolution."""

    def __init__(self, n_feat, seq_len, d_model=64, dropout=0.1, conv=True):
        super().__init__()
        self.in_proj = nn.Linear(n_feat, d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.conv = (nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, padding=0)
                     if conv else None)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.in_proj(x)
        n = h.shape[1]
        att = (self.q(h) @ self.k(h).transpose(1, 2)) / math.sqrt(h.shape[-1])
        mask = torch.tril(torch.ones(n, n, device=h.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        out = self.drop(torch.softmax(att, dim=-1)) @ self.v(h)
        if self.conv is not None:
            out = out + causal_depthwise_conv(h, self.conv, 3)
        return self.head(self.norm(out[:, -1])).squeeze(-1)


class DLinear(nn.Module):
    """Zeng et al., Are Transformers Effective for Time Series Forecasting? (2023).

    Trend and remainder are separated by a moving average and each is mapped to the horizon by a
    single linear layer; no attention and no recurrence.
    """

    def __init__(self, n_feat, seq_len, kernel=25):
        super().__init__()
        self.kernel = kernel
        self.season = nn.Linear(seq_len, 1)
        self.trend = nn.Linear(seq_len, 1)
        self.scale = nn.Parameter(torch.ones(n_feat))
        self.bias = nn.Parameter(torch.zeros(n_feat))
        self.head = nn.Linear(n_feat, 1)

    def forward(self, x):
        x = x * self.scale + self.bias
        pad = (self.kernel - 1) // 2
        xt = x.transpose(1, 2)
        trend = F.avg_pool1d(F.pad(xt, (pad, pad), mode="replicate"), self.kernel, stride=1)
        trend = trend.transpose(1, 2)
        season = x - trend
        s = self.season(season.transpose(1, 2)).squeeze(-1)
        t = self.trend(trend.transpose(1, 2)).squeeze(-1)
        return self.head(s + t).squeeze(-1)


class SSMOnly(nn.Module):
    """The selective state-space branch alone, with a linear head.

    Ablation: it isolates how much of the fused model comes from the state-space branch rather
    than from the attention branch it is fused with.
    """

    def __init__(self, n_feat, seq_len, d_state=16, dropout=0.1, **kw):
        super().__init__()
        self.ssm = SelectiveSSMBranch(d_model=n_feat, d_state=d_state, d_conv=4, expand=2,
                                      headdim=n_feat, block_len=60, guidance=False)
        self.norm = nn.LayerNorm(n_feat)
        self.head = nn.Linear(n_feat, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.ssm(x)
        return self.head(self.drop(self.norm(out[:, -1]))).squeeze(-1)


class PatchTST(nn.Module):
    """Nie, Nguyen, Sinthong and Kalagnanam (ICLR 2023), "A Time Series is Worth 64 Words".

    The series is split into sub-series patches, each patch is embedded by a linear layer, a
    Transformer encoder runs over the patch sequence, and the flattened patch representation is
    mapped to the output. Channels are processed independently, as in the published model; the
    per-channel representations are averaged before the head because the task here is a single
    binary output per window rather than a multi-variate forecast.
    """

    def __init__(self, n_feat, seq_len, patch=16, stride=8, d_model=64, heads=4, layers=2,
                 dropout=0.1, **kw):
        super().__init__()
        patch = min(patch, seq_len)
        stride = max(1, min(stride, patch))
        self.patch, self.stride = patch, stride
        self.n_patch = max(1, (seq_len - patch) // stride + 1)
        self.embed = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patch, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model * self.n_patch, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        n, t, f = x.shape
        z = x.transpose(1, 2).reshape(n * f, t)
        z = z.unfold(1, self.patch, self.stride)                 # (N*F, P, patch)
        z = self.embed(z) + self.pos
        z = self.norm(self.enc(z))
        z = z.reshape(n, f, -1).mean(dim=1)                      # pool over channels
        return self.head(self.drop(z)).squeeze(-1)


class ITransformer(nn.Module):
    """Liu et al. (ICLR 2024), "iTransformer: Inverted Transformers Are Effective for Time
    Series Forecasting".

    The axes are inverted: each variate's whole look-back series is embedded into one token and
    attention runs across variates, which is the opposite of the usual time-token arrangement.
    """

    def __init__(self, n_feat, seq_len, d_model=64, heads=4, layers=2, dropout=0.1, **kw):
        super().__init__()
        self.embed = nn.Linear(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 1)                        # per-variate prediction
        self.head = nn.Linear(n_feat, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        z = self.embed(x.transpose(1, 2))                        # (N, F, d_model)
        z = self.norm(self.enc(z))                               # attention across variates
        z = self.proj(z).squeeze(-1)                             # (N, F)
        return self.head(self.drop(z)).squeeze(-1)


class _InceptionBlock2D(nn.Module):
    """The inception block TimesNet uses inside its 2-D variation block."""

    def __init__(self, in_ch, out_ch, num_kernels=6):
        super().__init__()
        self.branches = nn.ModuleList()
        for i in range(num_kernels):
            k = 2 * i + 1
            self.branches.append(nn.Conv2d(in_ch, out_ch, kernel_size=(k, k),
                                           padding=(k // 2, k // 2)))
        self.project = nn.Conv2d(out_ch * num_kernels, out_ch, kernel_size=1)

    def forward(self, x):
        return self.project(torch.cat([b(x) for b in self.branches], dim=1))


class TimesNet(nn.Module):
    """Wu et al. (ICLR 2023), "TimesNet: Temporal 2D-Variation Modeling for General Time Series
    Analysis".

    Periods are selected from the amplitude spectrum of the look-back window, the 1-D series is
    folded into a 2-D tensor of shape (period, period-length), inception blocks are applied, and
    the results are aggregated with weights taken from the same amplitudes.

    Deviation from the published implementation, stated so the comparison is not overstated: the
    published model selects candidate periods separately for every sample. Here one period set is
    used for the whole batch, taken as the median of the per-sample selections, so that the 2-D
    reshaping stays rectangular and the batch can be processed in one pass.
    """

    def __init__(self, n_feat, seq_len, d_model=32, k_periods=3, num_kernels=6, layers=2,
                 dropout=0.1, **kw):
        super().__init__()
        self.k = k_periods
        self.seq_len = seq_len
        self.blocks = nn.ModuleList([
            nn.Sequential(_InceptionBlock2D(n_feat, n_feat, num_kernels), nn.GELU())
            for _ in range(layers)])
        self.head = nn.Linear(n_feat, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        n, t, f = x.shape
        xt = x.transpose(1, 2)                                   # (N, F, T)
        spec = torch.fft.rfft(xt, dim=-1)
        amp = spec.abs().mean(dim=1)                             # (N, T//2+1)
        amp[:, 0] = 0.0
        kk = min(self.k, amp.shape[1] - 1)
        if kk < 1:
            pooled = xt.mean(dim=-1)
            return self.head(self.drop(pooled)).squeeze(-1)
        top = torch.topk(amp, kk, dim=-1)                        # amplitudes and positions
        freqs = top.indices.float()
        periods = torch.clamp((t / freqs.clamp(min=1.0)).round().long(), 2, max(2, t // 2))
        weights = torch.softmax(top.values, dim=-1)

        outs = []
        for j in range(kk):
            p = int(periods[:, j].float().median().item())
            length = max(1, t // p)
            used = p * length
            fold = xt[..., :used].reshape(n, f, p, length)
            h = fold
            for blk in self.blocks:
                h = blk(h) + h              # the block preserves the (p, length) shape
            h = h.reshape(n, f, used)
            if used < t:
                h = F.pad(h, (0, t - used))
            outs.append(h * weights[:, j].reshape(n, 1, 1))
        agg = torch.stack(outs, dim=0).sum(dim=0)                # (N, F, T)
        feat = agg.transpose(1, 2)[:, -1]                        # last day, all variates
        return self.head(self.drop(feat)).squeeze(-1)


class GatedFusionSSM(nn.Module):
    """The proposed model, as the code actually implements it.

    A causal attention branch with a left-padded depthwise convolution, a selective state-space
    branch driven by the same window, and a sigmoid gate that mixes the two branch outputs. The
    gate decides per channel how much of each branch to keep; it does not modulate the state
    update of the state-space branch.

    `guidance=True` is the variant that does modulate it: the attention representation at each
    position is fed to the state-space branch, where it scales the discretisation step and shifts
    the input and output matrices, both through zero-initialised projections. That is the mechanism
    the submitted manuscript claimed for the proposed model; here it is a separate, measurable
    variant rather than a description of the plain gated model.
    """

    def __init__(self, n_feat, seq_len, d_model=64, d_state=16, dropout=0.1, guidance=False):
        super().__init__()
        self.att = AttentionClassifier(n_feat, seq_len, d_model=d_model, dropout=dropout, conv=True)
        self.ssm = SelectiveSSMBranch(d_model=n_feat, d_state=d_state, d_conv=4, expand=2,
                                      headdim=n_feat, block_len=60, guidance=guidance,
                                      guide_dim=d_model)
        self.guidance = guidance
        self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        self.ssm_proj = nn.Linear(n_feat, d_model)
        # score head fed by the fused representation
        self.fuse_head = nn.Linear(d_model, 1)

    def forward(self, x, return_parts=False):
        # attention branch representation
        h = self.att.in_proj(x)
        n = h.shape[1]
        att = (self.att.q(h) @ self.att.k(h).transpose(1, 2)) / math.sqrt(h.shape[-1])
        mask = torch.tril(torch.ones(n, n, device=h.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        att_out = self.att.drop(torch.softmax(att, dim=-1)) @ self.att.v(h)
        att_out = att_out + causal_depthwise_conv(h, self.att.conv, 3)
        att_seq = self.att.norm(att_out)
        att_last = att_seq[:, -1]

        guide = att_seq if self.guidance else None
        ssm_out = self.ssm(x, guidance=guide)
        ssm_last = self.ssm_proj(ssm_out[:, -1])

        gate = self.gate(torch.cat([att_last, ssm_last], dim=-1))
        fused = gate * ssm_last + (1 - gate) * att_last
        logit = self.fuse_head(fused).squeeze(-1)
        if return_parts:
            return logit, att_last, ssm_last, gate
        return logit


def build(name, n_feat, seq_len, **kw):
    name = name.lower()
    if name == "mlp":
        return MLPClassifier(n_feat, seq_len, **kw)
    if name == "lstm":
        return LSTMClassifier(n_feat, seq_len, **kw)
    if name == "attention":
        return AttentionClassifier(n_feat, seq_len, conv=False, **kw)
    if name == "attention+conv":
        return AttentionClassifier(n_feat, seq_len, conv=True, **kw)
    if name == "dlinear":
        return DLinear(n_feat, seq_len, **kw)
    if name == "gatedfusion":
        return GatedFusionSSM(n_feat, seq_len, **kw)
    if name == "gatedfusion-guided":
        return GatedFusionSSM(n_feat, seq_len, guidance=True, **kw)
    if name == "ssm-only":
        return SSMOnly(n_feat, seq_len, **kw)
    if name == "patchtst":
        return PatchTST(n_feat, seq_len, **kw)
    if name == "itransformer":
        return ITransformer(n_feat, seq_len, **kw)
    if name == "timesnet":
        return TimesNet(n_feat, seq_len, **kw)
    raise ValueError("unknown model: %s" % name)
