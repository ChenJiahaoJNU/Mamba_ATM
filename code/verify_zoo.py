"""Checks the model zoo has to pass before any number it produces is used.

Four checks, each one answering a defect that was measured in the submitted code:

1. every model builds for every sequence length used in the paper, returns one logit per window,
   and holds a non-zero number of trainable parameters (the submitted model held zero);
2. no module is constructed inside `forward`, which is how the submitted model ended up with
   untrained, re-randomised weights;
3. within-window causality: for the two models that carry a convolutional or recurrent path, the
   per-position outputs before a cutoff must not move when inputs at or after the cutoff are
   perturbed;
4. the standardiser is fitted on the training block only, so no test-period statistic reaches the
   training inputs.

Usage: python verify_zoo.py
"""
import json
import os
import sys

import numpy as np
import torch
sys.path.insert(0, r"D:\研究生学习资料\暨南大学-博士学习资料\论文\重投\MambaATM\_rework\libs")
sys.path.insert(0, r"D:\研究生学习资料\暨南大学-博士学习资料\论文\重投\MambaATM\_rework\code")

import model_zoo  # noqa: E402
import pipeline as P  # noqa: E402
from ssm_branch import causal_depthwise_conv  # noqa: E402

MODELS = ["mlp", "lstm", "attention", "attention+conv", "dlinear", "patchtst",
          "itransformer", "timesnet", "ssm-only", "gatedfusion", "gatedfusion-guided"]
SEQ_LENS = [20, 60, 120]
N_FEAT = 8


def param_report():
    print("model                 params  trainable  forward-out-shape")
    rows = []
    for name in MODELS:
        for T in SEQ_LENS:
            torch.manual_seed(0)
            model = model_zoo.build(name, N_FEAT, T)
            n_all = sum(p.numel() for p in model.parameters())
            n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            out = model(torch.randn(4, T, N_FEAT))
            assert out.shape == (4,), f"{name} T={T}: shape {tuple(out.shape)}"
            assert torch.isfinite(out).all(), f"{name} T={T}: non-finite output"
            if T == 60:
                print(f"  {name:<20}{n_all:>8,}{n_tr:>11,}  {tuple(out.shape)}")
                rows.append((name, n_all))
        assert n_tr > 0, f"{name}: zero trainable parameters"
    return rows


def no_modules_built_in_forward():
    """A module constructed inside forward is not in the optimizer's parameter list."""
    bad = []
    for name in MODELS:
        model = model_zoo.build(name, N_FEAT, 60)
        before = {id(p) for p in model.parameters()}
        model(torch.randn(2, 60, N_FEAT))
        after = {id(p) for p in model.parameters()}
        if after != before:
            bad.append(name)
    print("modules created during forward:", bad if bad else "none")
    return bad


def window_causality(name, T=60, cutoff=40, reps=3, tol=0.0):
    """Per-position outputs before `cutoff` must not move when inputs from `cutoff` change."""
    worst = 0.0
    for r in range(reps):
        torch.manual_seed(r)
        model = model_zoo.build(name, N_FEAT, T).double().eval()
        x = torch.randn(2, T, N_FEAT, dtype=torch.float64)

        def per_position(inp):
            if name.startswith("gatedfusion") or name == "ssm-only":
                return model.ssm(inp)
            if name in ("attention", "attention+conv"):
                h = model.in_proj(inp)
                out = h
                if model.conv is not None:
                    out = out + causal_depthwise_conv(h, model.conv, 3)
                return out
            return None

        ref = per_position(x)
        if ref is None:
            return None
        x2 = x.clone()
        x2[:, cutoff:, :] += 5.0 * torch.randn_like(x2[:, cutoff:, :])
        with torch.no_grad():
            moved = (per_position(x2)[:, :cutoff] - ref[:, :cutoff]).abs().max().item()
        worst = max(worst, moved)
    return worst


def standardiser_leak_test(T=20, n=200):
    rng = np.random.RandomState(0)
    X = rng.randn(n, T, N_FEAT)
    tr, te = np.arange(0, 150), np.arange(150, n)
    a, b = P.standardise(X[tr], X[te])
    X2 = X.copy()
    X2[150:] += 100.0                      # change only the test block
    a2, b2 = P.standardise(X2[tr], X2[te])
    same_train = np.allclose(a, a2)
    print(f"train inputs unchanged when test block is rewritten: {same_train}")
    X3 = X.copy()
    X3[:150] += 100.0                      # change only the train block
    a3, b3 = P.standardise(X3[tr], X3[te])
    print(f"test inputs change when train block is rewritten:      "
          f"{not np.allclose(b, b3)}")
    return same_train and not np.allclose(b, b3)


def main():
    summary = {}
    print("== 1/2. build, shape, parameter count ==")
    rows = param_report()
    summary["parameters"] = dict(rows)
    bad = no_modules_built_in_forward()
    summary["models_creating_modules_in_forward"] = bad
    summary["pass_no_forward_modules"] = not bad

    print("\n== 3. within-window causality (max change before the cutoff) ==")
    verdicts = {}
    for name in ["attention", "attention+conv", "ssm-only", "gatedfusion", "gatedfusion-guided"]:
        v = window_causality(name)
        verdicts[name] = v
        print(f"  {name:<20} {v:.6f}  {'PASS' if v == 0.0 else 'FAIL'}")
    summary["causality_max_change"] = verdicts
    summary["pass_causality"] = all(v == 0.0 for v in verdicts.values())

    print("\n== 4. standardiser ==")
    ok = standardiser_leak_test()
    summary["pass_standardiser"] = bool(ok)

    summary["all_pass"] = bool(not bad and summary["pass_causality"] and ok)
    print("\nsummary:", "PASS" if summary["all_pass"] else "FAIL")

    out = os.environ.get("VERIFY_OUT")
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("written", out)


if __name__ == "__main__":
    main()
