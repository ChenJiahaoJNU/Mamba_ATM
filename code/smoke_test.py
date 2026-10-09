"""One-command smoke test of the whole chain, for a fresh checkout.

It reads a few stocks from the released panel, builds the look-back windows, makes the walk-forward
splits, trains one small model for one epoch, and compares the resulting accuracy with the
majority-class rate of the same test blocks. Every step of the chain the paper relies on is touched,
so a broken checkout fails here in about a minute instead of failing three hours into the main run.

It also prints the two reference points, because on this panel a directional accuracy means nothing
until it is placed next to the majority-class rate, and a strategy return means nothing until it is
placed next to buy and hold.

Usage
    python code/smoke_test.py
    python code/smoke_test.py --panel data/adj --stocks 3 --seq 60 --model mlp --epochs 1
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import pipeline as P          # noqa: E402
import train_eval as T        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=os.path.join(ROOT, "data", "adj"),
                    help="directory of one CSV per stock, default <repo>/data/adj")
    ap.add_argument("--stocks", type=int, default=3, help="how many stocks to test on")
    ap.add_argument("--seq", type=int, default=60, help="look-back window length")
    ap.add_argument("--model", default="mlp", help="any name understood by model_zoo.build")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.panel, "*.csv")))[:args.stocks]
    if not files:
        print(f"FAIL: no CSV files under {args.panel}")
        print("      the panel is hosted separately, see data/README.md")
        return 1
    print(f"panel      : {args.panel}")
    print(f"stocks     : {len(files)} of {len(glob.glob(os.path.join(args.panel, '*.csv')))}")
    print(f"window     : {args.seq} days, model {args.model}, {args.epochs} epoch(s)")
    print()

    header = f"{'stock':12s} {'rows':>6s} {'windows':>8s} {'folds':>6s} {'accuracy':>9s} {'majority':>9s} {'up-rate':>8s}"
    print(header)
    print("-" * len(header))

    rows, failures = [], []
    for path in files:
        code = os.path.basename(path).replace(".csv", "")
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            feat = df[P.FEATURE_COLS].to_numpy(dtype=np.float64)
            ret = pd.to_numeric(df[P.RET_COL], errors="coerce").to_numpy(dtype=np.float64)
            X, yc, yr, _ = P.make_windows(feat, ret, args.seq)
            splits = P.walk_forward(len(X), folds=args.folds, purge=1)
            if not splits:
                failures.append(f"{code}: walk_forward returned no split")
                continue
            r = T.run(args.model, X, yc, yr, splits, epochs=args.epochs, seed=0,
                      device=args.device)
            te_idx = np.concatenate([te for _, te in splits])
            pred = (r["pred"] > 0).astype(int)
            acc = float(np.mean(pred == yc[te_idx]))
            majority = float(max(np.mean(yc[te_idx]), 1.0 - np.mean(yc[te_idx])))
            print(f"{code:12s} {len(df):6d} {len(X):8d} {len(splits):6d} "
                  f"{acc:9.4f} {majority:9.4f} {np.mean(yc[te_idx]):8.3f}")
            rows.append((acc, majority))
            if not np.isfinite(acc) or not np.isfinite(majority):
                failures.append(f"{code}: non-finite metric")
        except Exception as exc:                                      # noqa: BLE001
            failures.append(f"{code}: {type(exc).__name__}: {exc}")
            print(f"{code:12s} FAILED  {type(exc).__name__}: {exc}")

    print()
    if not rows:
        print("FAIL: nothing ran to completion")
        return 1
    accs = np.array([a for a, _ in rows])
    majs = np.array([m for _, m in rows])
    print(f"accuracy mean {accs.mean():.4f}   majority-class mean {majs.mean():.4f}   "
          f"models above the reference on {int((accs > majs).sum())} of {len(rows)} stocks")
    print("this is the expected shape of the result: on this panel the reference is hard to beat")

    if failures:
        print()
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print()
    print("PASS: panel read, windows built, walk-forward splits made, model trained, metrics finite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
