"""Pure-PyTorch selective_scan_fn (Mamba-1 formulation).

Appended to the shim so mamba_simple.py can run without the CUDA kernel. The recurrence is
    h_t = exp(dt_t * A) * h_{t-1} + dt_t * (B_t^T u_t)
    y_t = C_t . h_t + D * u_t
with A of shape (d_inner, d_state) and dt broadcast over d_inner. The loop runs over the
sequence, which is only 120 steps here.
"""

import torch
import torch.nn.functional as F


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                       return_last_state=False, **kwargs):
    """u (b, d_inner, l), delta (b, d_inner, l), A (d_inner, d_state),
    B/C (b, d_state, l), D (d_inner,)."""
    b, d_inner, l = u.shape
    d_state = A.shape[1]

    dt = delta
    if delta_bias is not None:
        dt = dt + delta_bias.view(1, -1, 1)
    if delta_softplus:
        dt = F.softplus(dt)
    dt = dt.float()
    A = A.float()
    B = B.float()
    C = C.float()
    u = u.float()

    # (b, d_inner, d_state, l)
    dA = torch.exp(dt.unsqueeze(2) * A.unsqueeze(0).unsqueeze(-1))
    dB_u = (dt.unsqueeze(2) * B.unsqueeze(1)) * u.unsqueeze(2)      # (b, d_inner, d_state, l)
    h = torch.zeros(b, d_inner, d_state, device=u.device, dtype=torch.float32)
    ys = []
    for t in range(l):
        h = dA[..., t] * h + dB_u[..., t]
        ys.append(torch.einsum("bdn,bn->bd", h, C[..., t]))
    y = torch.stack(ys, dim=-1)                                     # (b, d_inner, l)
    if D is not None:
        y = y + u * D.float().view(1, -1, 1)
    if z is not None:
        y = y * F.silu(z.float())
    if return_last_state:
        return y, h
    return y
