"""Accuracy by calendar year of the test blocks, for the proposed model and the references.

A model that tracks the majority-class rate of the year is tracking the drift of the sample rather
than a stable signal. This script measures that co-movement from the saved predictions, so the claim
in the sensitivity section is a measurement and not an impression.

Usage: python by_year.py --preds out/main/preds.npz --out out/main/by_year.json [--model gatedfusion]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODELS = ["mlp", "lstm", "attention+conv", "dlinear", "patchtst", "itransformer", "timesnet",
          "ssm-only", "gatedfusion", "gatedfusion-guided"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.preds)
    models = [m for m in MODELS if any(k.startswith(m + "|") for k in z.files)]
    out = {}
    for m in models:
        codes = sorted({k.split("|")[1] for k in z.files if k.startswith(m + "|")})
        # dates are stored as days since the epoch, so they have to be reinterpreted as such
        dates = np.concatenate([z[f"{m}|{c}|dates"] for c in codes]).astype("datetime64[D]")
        pos = np.concatenate([z[f"{m}|{c}|pos"] for c in codes]).astype(int)
        ret = np.concatenate([z[f"{m}|{c}|ret"] for c in codes]).astype(float)
        years = dates.astype("datetime64[Y]").astype(int) + 1970
        truth = (ret > 0).astype(int)
        for y in np.unique(years):
            sel = years == y
            rec = out.setdefault(int(y), {"models": {}, "n": 0, "majority": 0.0})
            rec["models"][m] = float((pos[sel] == truth[sel]).mean())
            rec["n"] = int(sel.sum())
            rec["majority"] = float(max(truth[sel].mean(), 1.0 - truth[sel].mean()))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"{'year':>6}{'n':>8}{'maj':>8}{'MambaATM':>10}{'MLP':>8}{'PatchTST':>10}{'DLinear':>9}")
    for y in sorted(out):
        r = out[y]
        g = lambda k: r["models"].get(k, float("nan"))
        print(f"{y:>6}{r['n']:>8}{r['majority']:>8.3f}{g('gatedfusion'):>10.3f}"
              f"{g('mlp'):>8.3f}{g('patchtst'):>10.3f}{g('dlinear'):>9.3f}")
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
