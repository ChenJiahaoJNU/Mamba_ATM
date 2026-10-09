"""Append the characteristic factors to a saved score file, so the references can be evaluated at
the same holding periods and cost levels as the learned models without retraining anything.

Usage: python xsec_add_chars.py --scores out/xsec_main/xsec_scores.npz --panel <panel> --seq 60 \
    --stocks 232 --out out/xsec_main/xsec_scores_all.npz
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "libs"))
sys.path.insert(0, HERE)

import xsection as X  # noqa: E402

DEFAULT_PANEL = (r"."
                 r"\data\adj")
CHARACTERISTICS = ["short_reversal", "reversal_5", "reversal_20", "momentum", "low_volatility",
                   "volatility_20", "low_turnover", "small_amount", "low_price"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--seq", type=int, default=60)
    ap.add_argument("--stocks", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.scores, allow_pickle=False)
    models = sorted({k.split("|")[0] for k in z.files if "|" in k})
    te_rows = z["te_rows"]
    ref = models[0]

    data_all = X.load_panel(args.panel)
    codes = sorted(data_all)[:args.stocks] if args.stocks else sorted(data_all)
    data = {c: data_all[c] for c in codes}
    ds = X.build_dataset(data, args.seq)
    if len(ds.y) != len(X.tradable_mask(data, ds)):
        raise SystemExit("panel rebuild does not match the saved run")

    out = {}
    for k in z.files:
        out[k] = z[k]
    for ch in CHARACTERISTICS:
        out[f"{ch}|score"] = X.characteristic_vector(data, ds, ch)[te_rows]
    np.savez_compressed(args.out, **out)
    print(f"added {len(CHARACTERISTICS)} characteristics to {args.out}")
    print("models in file:", models)


if __name__ == "__main__":
    main()
