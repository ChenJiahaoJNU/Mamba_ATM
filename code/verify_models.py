"""Run the causality regression test on the submitted model and on the corrected one.

The test is the one the manuscript claims every model passes: with all parameters randomised,
inputs at positions >= cutoff are perturbed and the outputs at positions < cutoff must be
unchanged.
"""
import sys

import torch

sys.path.insert(0, r".\libs")
sys.path.insert(0, r".\code")

from models import AdvancedMambaAMT, SelfAttentionMamba            # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFG = {"d_model": 64, "d_state": 16, "d_conv": 4, "expand": 2}
SEQ, FEAT, CUT = 120, 64, 60


def test(model, feat=FEAT, label=""):
    model = model.to(DEV).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, 0.0, 0.1)
    x = torch.randn(2, SEQ, feat, device=DEV)
    with torch.no_grad():
        a = model(x)
        xp = x.clone()
        xp[:, CUT:, :] += 10.0 * torch.randn_like(xp[:, CUT:, :])
        b = model(xp)
        past = (a[:, :CUT] - b[:, :CUT]).abs().max().item()
        fut = (a[:, CUT:] - b[:, CUT:]).abs().max().item()
    print("   %-46s past %.6f  future %.6f  %s"
          % (label, past, fut, "PASS" if past == 0.0 else "FAILS (future information reaches the past)"))
    return past


print("device %s, torch %s" % (DEV, torch.__version__))
print()

print("A. the model that produced the reported MambaATM numbers")
print("   SelfAttentionMamba(model_type='MambaAMT'), equal q/v width so no re-randomised layer")
test(SelfAttentionMamba(64, 64, 64, 1, 0.1, "MambaAMT", CFG, DEV),
     label="submitted, dim_q == dim_v")
print("   same class, the configuration main.py actually uses (dim_q=INPUT_DIM, dim_v=64)")
test(SelfAttentionMamba(16, 64, 64, 1, 0.1, "MambaAMT", CFG, DEV), feat=16,
     label="submitted, dim_q != dim_v")
print()

print("B. the corrected model")
test(AdvancedMambaAMT(16, 64, 64, 1, 0.1, DEV), feat=16,
     label="AdvancedMambaAMT with the real SSM branch")
print()

print("C. how many learnable parameters does the sequence-model slot hold?")
for name, m in [("submitted SelfAttentionMamba(MambaAMT)",
                 SelfAttentionMamba(64, 64, 64, 1, 0.1, "MambaAMT", CFG, DEV)),
                ("corrected AdvancedMambaAMT", AdvancedMambaAMT(64, 64, 64, 1, 0.1, DEV))]:
    total = sum(p.numel() for p in m.parameters())
    seq = getattr(m, "model", None)
    seq_p = sum(p.numel() for p in seq.parameters()) if seq is not None else 0
    ssm_p = sum(p.numel() for p in m.ssm.parameters()) if hasattr(m, "ssm") else 0
    print("   %-40s total %7d   sequence-model slot %6d   ssm branch %6d"
          % (name, total, seq_p, ssm_p))
