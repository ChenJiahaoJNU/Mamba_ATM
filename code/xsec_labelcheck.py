"""Diagnostic: which label is predictable, and by what?

Before spending hours retraining ten architectures on a new target, this asks the cheap question:
across the extended factor library, how much cross-sectional information is there in the
close-to-close label, in the overnight leg, and in the intraday leg?

The models cannot be evaluated on the new labels yet, because they were trained on the close-to-close
label; the factor library can, because a factor needs no training. If the overnight label turns out to
be markedly more predictable than the label the models were trained on, retraining is worth the
compute. If it is not, the money is better spent elsewhere.

Usage: python xsec_labelcheck.py --panel <panel> --seq 60 [--stocks 232] [--out DIR]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "libs"))
sys.path.insert(0, HERE)

import factors as FA  # noqa: E402
import xsection as X  # noqa: E402

DEFAULT_PANEL = (r"."
                 r"\data\adj")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--seq", type=int, default=60)
    ap.add_argument("--stocks", type=int, default=232)
    ap.add_argument("--min-names", type=int, default=30)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    data_all = X.load_panel(args.panel)
    codes = sorted(data_all)[:args.stocks]
    data = {c: data_all[c] for c in codes}
    ds = X.build_dataset(data, args.seq)

    # equal-weight panel return per calendar date, needed for the market-exposure factor
    per_date = {}
    for code, (dates, feat, ret) in data.items():
        for d, r in zip(dates, ret):
            per_date.setdefault(d, []).append(r)
    mean_by_date = {d: float(np.mean(v)) for d, v in per_date.items()}

    def market_ret_by_date(dates):
        return np.array([mean_by_date.get(d, 0.0) for d in dates])

    F, names = FA.factor_arrays(data, ds, market_ret_by_date)
    tg = FA.targets(data, ds)
    print(f"{len(codes)} stocks, {len(ds.X):,} windows, {len(np.unique(ds.date_idx)):,} dates, "
          f"{len(names)} factors")
    for key in ("cc", "on", "id"):
        v = tg[key]
        print(f"  label {key}: mean {np.nanmean(v) * 100:+.4f}%/day, "
              f"sd {np.nanstd(v) * 100:.3f}%, non-null {np.isfinite(v).mean() * 100:.1f}%")

    dates = np.unique(ds.date_idx)
    order = np.argsort(ds.date_idx, kind="stable")
    starts = np.searchsorted(ds.date_idx[order], dates)
    ends = np.append(starts[1:], len(order))

    def rank(v):
        return stats.rankdata(v).astype(np.float64)

    rows = []
    for key in ("cc", "on", "id"):
        y = tg[key]
        for n in names:
            ics = []
            for s, e in zip(starts, ends):
                idx = order[s:e]
                if len(idx) < args.min_names:
                    continue
                yy = y[idx]
                xx = F[n][idx]
                ok = np.isfinite(yy) & np.isfinite(xx)
                if ok.sum() < args.min_names or np.std(xx[ok]) < 1e-12:
                    continue
                ics.append(np.corrcoef(rank(xx[ok]), rank(yy[ok]))[0, 1])
            v = np.asarray(ics)
            if len(v) < 50:
                continue
            rows.append({"factor": n, "label": key, "ic": float(v.mean()),
                         "t": float(v.mean() / v.std(ddof=1) * np.sqrt(len(v))),
                         "positive_share": float((v > 0).mean()), "days": len(v)})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    piv_ic = df.pivot(index="factor", columns="label", values="ic")[["cc", "on", "id"]]
    piv_t = df.pivot(index="factor", columns="label", values="t")[["cc", "on", "id"]]
    joined = piv_ic.round(4).join(piv_t.round(1), rsuffix="_t")
    joined.columns = ["ic_cc", "ic_on", "ic_id", "t_cc", "t_on", "t_id"]
    print("\n=== cross-sectional rank IC of every factor, by label ===")
    print(joined.sort_values("ic_on", ascending=False).to_string())

    print("\n=== best factor per label ===")
    for key in ("cc", "on", "id"):
        sub = df[df.label == key].sort_values("t", ascending=False).head(5)
        print(f"  {key}: " + ", ".join(f"{r.factor} ({r.ic:+.4f}, t={r.t:+.1f})"
                                       for r in sub.itertuples()))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        df.to_csv(os.path.join(args.out, "labelcheck.csv"), index=False)
        with open(os.path.join(args.out, "labelcheck.json"), "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "rows": rows}, f, indent=2)
        print(f"\nwritten {os.path.join(args.out, 'labelcheck.csv')}")


if __name__ == "__main__":
    main()
