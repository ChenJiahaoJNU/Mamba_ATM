"""Causality regression test.

With all parameters randomised, inputs at positions >= c are perturbed and the outputs at
positions < c must be identical. This is the test the manuscript claims every model passes.
"""
import sys

import torch

sys.path.insert(0, r"H:\..\libs")


def causality_test(model, seq_len=120, features=16, cutoff=60, scale=10.0, device="cpu"):
    model = model.to(device).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, 0.0, 0.1)
    x = torch.randn(2, seq_len, features, device=device)
    with torch.no_grad():
        y_ref = model(x)
        x_pert = x.clone()
        x_pert[:, cutoff:, :] += scale * torch.randn_like(x_pert[:, cutoff:, :])
        y_pert = model(x_pert)
        delta_past = (y_ref[:, :cutoff] - y_pert[:, :cutoff]).abs().max().item()
        delta_future = (y_ref[:, cutoff:] - y_pert[:, cutoff:]).abs().max().item()
    return delta_past, delta_future


if __name__ == "__main__":
    sys.path.insert(0, r".\libs")
    sys.path.insert(0, r".\code")
    from ssm_branch import SelectiveSSMBranch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("torch %s, device %s" % (torch.__version__, dev))

    m = SelectiveSSMBranch(d_model=16, d_state=16, headdim=16, block_len=40)
    n_param = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print("SelectiveSSMBranch: %d learnable parameters" % n_param)

    past, future = causality_test(m, features=16, device=dev)
    print("causality: max change before the cutoff = %.6f" % past)
    print("causality: max change after  the cutoff = %.6f" % future)
    print("VERDICT:", "PASS (past outputs unchanged)" if past == 0.0 else "FAIL")

    x = torch.randn(4, 120, 16, device=dev)
    y = m(x)
    print("forward shape: %s -> %s" % (tuple(x.shape), tuple(y.shape)))
    loss = y.pow(2).mean()
    loss.backward()
    grads = {n: (p.grad.abs().mean().item() if p.grad is not None else None)
             for n, p in m.named_parameters()}
    dead = [n for n, g in grads.items() if g is None or g == 0.0]
    print("parameters receiving no gradient: %s" % (dead if dead else "none"))

    g = SelectiveSSMBranch(d_model=16, d_state=16, headdim=16, block_len=40, guidance=True)
    gp, gf = causality_test(g, features=16, device=dev)
    print("with guidance: past change %.6f, VERDICT %s"
          % (gp, "PASS" if gp == 0.0 else "FAIL"))
