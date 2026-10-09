"""A selective state-space branch with learned, input-dependent parameters.

The chunked scan below is the reference minimal SSD implementation by Albert Gu and Tri Dao
(ssd_minimal.py, Copyright (c) 2024, Apache-2.0, "this is the same as Listing 1 from the paper").
It is copied here unchanged except for the module-level import of the fused Triton kernel, which
the forward path never used, so that this branch runs without the mamba-ssm extension.

The difference from the version of the branch in the original submission: every parameter below
is learned or derived from the input. In the original the discretisation step and the state
transition were re-drawn from a fixed pseudo-random seed on every forward pass and the state
inputs were the attention block's own Q, K and V, which left the branch with no parameters of
its own.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# --------------------------------------------------------------------------- reference scan
def segsum(x):
    """Stable segment sum over the last axis."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    return x_segsum.masked_fill(~mask, -torch.inf)


def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    """X (b, l, h, p), A (b, l, h), B (b, l, g, n), C (b, l, g, n) -> Y (b, l, h, p)."""
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0, "sequence length must be a multiple of block_len"
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


# --------------------------------------------------------------------------- the branch
def causal_depthwise_conv(x, conv, kernel_size):
    """Left-padded depthwise convolution, so position t never reads beyond t."""
    x = x.transpose(1, 2)
    x = F.pad(x, (kernel_size - 1, 0))
    return conv(x).transpose(1, 2)


class SelectiveSSMBranch(nn.Module):
    """Mamba2-style branch: dt, B and C come from the input, A and D are learned.

    `guidance` is optional. When it is given, the attention representation modulates the
    discretisation step and the state matrices before the scan, which is the mechanism the
    original manuscript claimed. Passing None gives the plain branch.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, headdim=64,
                 block_len=60, guidance=False, guide_dim=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.block_len = block_len
        self.guidance = guidance
        self.guide_dim = guide_dim if guide_dim is not None else d_model

        self.d_inner = int(expand * d_model)
        assert self.d_inner % headdim == 0, "d_inner must be a multiple of headdim"
        self.nheads = self.d_inner // headdim
        self.dt_rank = max(1, d_model // 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                              groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.nheads, bias=True)

        # one learnable log-decay per head, matching the reference scan's A (b, l, h) contract
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.nheads + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(self.d_inner)

        if guidance:
            # both projections start at zero, so a guided run begins as the plain branch and the
            # modulation has to be learned; that is what makes the guided/unguided pair a
            # controlled comparison rather than two different initialisations
            self.guide_dt = nn.Linear(self.guide_dim, self.nheads)
            self.guide_bc = nn.Linear(self.guide_dim, 2 * d_state)
            for m in (self.guide_dt, self.guide_bc):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        # dt bias initialised so that softplus(bias) is small, as in the reference implementation
        dt = torch.exp(torch.rand(self.nheads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x, guidance=None):
        b, l, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        x_conv = causal_depthwise_conv(x_in, self.conv, self.d_conv)
        x_act = F.silu(x_conv)

        dt_bcn = self.x_proj(x_act)
        dt, B_, C_ = torch.split(dt_bcn, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        if guidance is not None and self.guidance:
            # per-step guidance: the attention representation at position t modulates the
            # discretisation step and the state matrices used at position t
            dt = dt * (1.0 + torch.tanh(self.guide_dt(guidance)))
            gbc = self.guide_bc(guidance)
            gb, gc = gbc.chunk(2, dim=-1)
            B_ = B_ + gb
            C_ = C_ + gc

        A = -torch.exp(self.A_log)

        # pick a block that divides the sequence length so no padding is needed
        block = self.block_len
        while l % block != 0 and block > 1:
            block -= 1
        pad = (-l) % block
        if pad:
            x_s = F.pad(x_act, (0, 0, 0, pad))
            dt_s = F.pad(dt, (0, 0, 0, pad))
            B_s = F.pad(B_, (0, 0, 0, pad))
            C_s = F.pad(C_, (0, 0, 0, pad))
        else:
            x_s, dt_s, B_s, C_s = x_act, dt, B_, C_

        X = rearrange(x_s, "b l (h p) -> b l h p", h=self.nheads)
        Bg = rearrange(B_s, "b l n -> b l 1 n")
        Cg = rearrange(C_s, "b l n -> b l 1 n")
        dt_h = repeat(dt_s, "b l h -> b l h")

        Y, _ = ssd_minimal_discrete(X * dt_h.unsqueeze(-1), A.unsqueeze(0) * dt_h,
                                    Bg, Cg, block)
        Y = Y[:, :l]
        Y = Y + X[:, :l] * self.D[None, :, None]
        Y = rearrange(Y, "b l h p -> b l (h p)")
        Y = Y * F.silu(z)
        return self.out_proj(self.norm(Y))
