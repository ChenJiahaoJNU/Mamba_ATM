"""The function the submitted model called to produce its second branch, kept verbatim.

Copied from the repository's ssd_minimal.py (Copyright (c) 2024, Albert Gu and Tri Dao) so that
the submitted class can be exercised for comparison. Nothing here is learned and nothing depends
on the input beyond the reuse of q, k and v:

    torch.manual_seed(42)                       # re-seeds the global RNG on every forward pass
    dt = softplus(randn(...) - 4)               # random, not a parameter
    A  = -exp(rand(...))                        # random, not a parameter
    x, B, C = V, K, Q                           # the attention block's own projections

Only the call into the chunked scan differs: the comment shows the call into the fused Triton
kernel, and the line below it uses the pure-PyTorch reference scan with the same arguments.
"""
import torch
import torch.nn.functional as F

from ext_shim import ssd_minimal_discrete


def test_correctness(Q, K, V, chunk_size):
    torch.manual_seed(42)
    batch, seqlen, dim = Q.shape
    headdim = chunk_size
    nheads = 1
    ngroups = dim // headdim
    dstate = headdim
    device = Q.device

    x = V.split(nheads, dim=2)
    x = torch.stack(x, dim=2)
    B = K.split(nheads, dim=2)
    B = torch.stack(B, dim=2)
    C = Q.split(nheads, dim=2)
    C = torch.stack(C, dim=2)

    dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4)
    A = -torch.exp(torch.rand(nheads, dtype=torch.float32, device=device))

    # y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None)
    y_min, _ = ssd_minimal_discrete(x * dt.unsqueeze(-1), A * dt, B, C, chunk_size)
    return y_min.view(y_min.size(0), y_min.size(1), -1)
