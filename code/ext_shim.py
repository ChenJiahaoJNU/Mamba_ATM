"""Pure-PyTorch stand-ins for the mamba-ssm CUDA/Triton entry points.

The published mamba-ssm package needs its Triton kernels, which are not available on this
Windows machine (nvcc 11.8 is installed but Triton is not). These replacements keep the same
call signatures and compute the same quantities with the reference chunked scan, so the baseline
models can run. They are slower than the fused kernels but numerically equivalent up to
float32 rounding.

The scan itself is the reference minimal SSD implementation by Albert Gu and Tri Dao
(ssd_minimal.py, Copyright (c) 2024, Apache-2.0).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def segsum(x):
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    return x_segsum.masked_fill(~mask, -torch.inf)


def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]
    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state


def _pad_along_time(x, pad):
    """Pad dim 1 (time) by `pad` on the right; F.pad runs from the last dim to the first."""
    if not pad:
        return x
    shape = [0] * (2 * x.dim())
    shape[2 * (x.dim() - 2) + 1] = pad
    return F.pad(x, tuple(shape))


def _pick_block(l, chunk_size):
    """Largest divisor of l not exceeding chunk_size, so no padding is needed."""
    limit = int(chunk_size)
    for c in range(min(limit, l), 0, -1):
        if l % c == 0:
            return c
    return l


def mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size=None, D=None, z=None,
                              seq_idx=None, initial_states=None, dt_limit=(0.0, float("inf")),
                              return_final_states=False, **kwargs):
    """x (b,l,h,p), dt (b,l,h), A (h,) or (b,l,h), B/C (b,l,g,n)."""
    block = _pick_block(x.shape[1], chunk_size or x.shape[1])
    X = x.float()
    dtf = dt.float()
    Af = A.float()
    Bf = B.float()
    Cf = C.float()
    l = X.shape[1]
    pad = (-l) % block
    if pad:
        X = _pad_along_time(X, pad)
        dtf = _pad_along_time(dtf, pad)
        Bf = _pad_along_time(Bf, pad)
        Cf = _pad_along_time(Cf, pad)
    if Af.dim() == 1:
        Af = Af[None, None, :].expand(X.shape[0], X.shape[1], -1)
    else:
        Af = _pad_along_time(Af, pad)

    Y, final = ssd_minimal_discrete(X * dtf.unsqueeze(-1), Af * dtf, Bf, Cf, block,
                                    initial_states)
    Y = Y[:, :l]
    x = x.float()
    if D is not None:
        Y = Y + x * D.float()[None, None, :, None]
    if z is not None:
        Y = Y * F.silu(z.float())
    return (Y, final) if return_final_states else Y


def mamba_split_conv1d_scan_combined(*args, **kwargs):
    raise NotImplementedError(
        "the fused split-path kernel is unavailable; construct the module with "
        "use_mem_eff_path=False so the slow path, which this shim supports, is taken")


def causal_conv1d_fn(x, weight, bias=None, activation=None, **kwargs):
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=weight.shape[-1] - 1)
    out = out[..., :x.shape[-1]]
    if activation in ("silu", "swish"):
        out = F.silu(out)
    return out


def causal_conv1d_update(*args, **kwargs):
    raise NotImplementedError("incremental decoding is not used by these experiments")


def selective_scan_fn(*args, **kwargs):
    raise NotImplementedError("use SelectiveSSMBranch, which implements the scan in PyTorch")


def mamba_inner_fn(*args, **kwargs):
    raise NotImplementedError("use SelectiveSSMBranch, which implements the inner path in PyTorch")


def selective_state_update(*args, **kwargs):
    raise NotImplementedError("incremental decoding is not used by these experiments")


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x, gate=None, **kwargs):
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        if gate is not None:
            out = out * F.silu(gate)
        return out


LayerNorm = nn.LayerNorm


def layer_norm_fn(x, weight, bias, residual=None, prenorm=False, residual_in_fp32=False,
                  eps=1e-5, **kwargs):
    if residual is not None:
        x = x + residual
    out = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    return (out, x) if prenorm else out


def rms_norm_fn(x, weight, bias=None, residual=None, prenorm=False, eps=1e-5, **kwargs):
    if residual is not None:
        x = x + residual
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
    return (out, x) if prenorm else out
