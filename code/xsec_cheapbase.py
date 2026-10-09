"""Does a sequence model still add anything once the comparison is restricted to the cheap factors?

The question this answers, in the form a referee would ask it:

    at weekly rebalancing, on a liquid universe, net of A-share fees, stamp duty, square-root impact
    and borrow, does any learned model beat a two-line factor built from turnover and turnover value,
    and does it add information that those two factors do not already contain?

Three things are computed for every score:

1. rank IC on the weekly signal, and the share of it that is borrowed from the two cheap factors
   (residual after a daily cross-sectional regression on low_turnover and small_amount),
2. the long-short book at weekly rebalancing with the realistic cost model, at two liquidity
   universes (all eligible names, and the most liquid half), reported as net return and the book size
   at which impact eats the edge,
3. the long-only top-quantile excess over the equal-weight universe, which is the implementable
   version of the same signal.

Usage: python xsec_cheapbase.py --scores out/xsec_main/xsec_scores_all.npz --panel <panel> \
    --seq 60 --quantile 0.1 --rebalance 5 --out out/xsec_cheapbase
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

import costs as C  # noqa: E402
import xsection as X  # noqa: E402

DEFAULT_PANEL = (r"."
                 r"\data\adj")
CHEAP = ["low_turnover", "small_amount"]
MODELS = ["dlinear", "mlp", "lstm", "attention+conv", "patchtst", "itransformer", "timesnet",
          "ssm-only", "gatedfusion", "gatedfusion-guided"]


def rank_within(v):
    return stats.rankdata(v).astype(np.float64)


def daily_ic(score, y, di, min_names=30):
    ics = []
    for d in np.unique(di):
        m = di == d
        if m.sum() < min_names or np.std(score[m]) < 1e-12:
            continue
        ics.append(np.corrcoef(rank_within(score[m]), rank_within(y[m]))[0, 1])
    v = np.asarray(ics)
    return float(v.mean()), float(v.mean() / v.std(ddof=1) * np.sqrt(len(v))), len(v)


def residualise(targets, base, y, di, min_names=30):
    """Cross-sectional residual of every target on the base factor set, computed per day."""
    out = np.full_like(targets, np.nan, dtype=np.float64)
    for d in np.unique(di):
        m = di == d
        if m.sum() < min_names:
            continue
        B = base[m]
        sdB = B.std(axis=0)
        keep = sdB > 1e-12
        if not keep.any():
            continue
        Bz = (B[:, keep] - B[:, keep].mean(axis=0)) / sdB[keep]
        T = targets[m]
        sdT = T.std(axis=0)
        good = sdT > 1e-12
        Tz = np.zeros_like(T)
        Tz[:, good] = (T[:, good] - T[:, good].mean(axis=0)) / sdT[good]
        Xm = np.column_stack([np.ones(m.sum()), Bz])
        beta, *_ = np.linalg.lstsq(Xm, Tz, rcond=None)
        out[m] = Tz - Xm @ beta
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--seq", type=int, default=60)
    ap.add_argument("--quantile", type=float, default=0.1)
    ap.add_argument("--rebalance", type=int, default=5)
    ap.add_argument("--liquid", default="1.0,0.5")
    ap.add_argument("--aum", type=float, default=1e8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    z = np.load(args.scores, allow_pickle=False)
    ref = next(k.split("|")[0] for k in z.files if k.endswith("|y"))
    y, di = z[f"{ref}|y"], z[f"{ref}|date"]
    te_rows = z["te_rows"]

    data_all = X.load_panel(args.panel)
    ds = X.build_dataset(data_all, args.seq)
    tradable = X.tradable_mask(data_all, ds)[te_rows]
    stock_ids = z["te_stocks"] if "te_stocks" in z.files else ds.stock_idx[te_rows]
    adv, sigma = C.liquidity_arrays(data_all, ds)
    adv, sigma = adv[te_rows], sigma[te_rows]

    have = [m for m in MODELS if f"{m}|score" in z.files]
    scores_raw = {m: z[f"{m}|score"] for m in have}
    base = np.column_stack([z[f"{c}|score"] for c in CHEAP])
    scores_raw["cheap_factor_combo"] = base.mean(axis=1)      # two-line practitioner baseline

    # residuals of every score on the two cheap factors, and of each cheap factor on the other
    names = list(scores_raw)
    T = np.column_stack([scores_raw[n] for n in names])
    resid = residualise(T, base, y, di)
    scores_res = {n: resid[:, j] for j, n in enumerate(names)}
    # the cheap factors themselves, each on the other one
    for j, c in enumerate(CHEAP):
        other = base[:, [1 - j]]
        scores_res[c] = residualise(base[:, [j]], other, y, di, min_names=30)[:, 0]

    rows = []
    for tag, table in [("raw", scores_raw), ("residual_on_cheap", scores_res)]:
        for n, s in table.items():
            if not np.isfinite(s).all():
                pass
            ic, t, nd = daily_ic(s, y, di)
            for liq in [float(x) for x in args.liquid.split(",")]:
                base_kw = dict(quantile=args.quantile, rebalance=args.rebalance, tradable=tradable,
                               stock_ids=stock_ids, top_liquidity=liq)
                zero = C.CostModel(commission=0, transfer=0, slippage=0, stamp_before=0,
                                   stamp_after=0, borrow_annual=0, impact_coef=0, aum=args.aum)
                real = C.CostModel(aum=args.aum)
                r0 = C.replay(s, y, di, ds.grid, adv=adv, sigma=sigma, cm=zero, **base_kw)
                r = C.replay(s, y, di, ds.grid, adv=adv, sigma=sigma, cm=real, **base_kw)
                lo = C.replay(s, y, di, ds.grid, adv=adv, sigma=sigma, cm=real, long_only=True,
                              **base_kw)
                lo0 = C.replay(s, y, di, ds.grid, adv=adv, sigma=sigma, cm=zero, long_only=True,
                               **base_kw)
                cap = C.breakeven_aum(r0["gross_ann"], r["fee_ann"], r["borrow_ann"],
                                      r["impact_ann"], args.aum)
                rows.append({
                    "set": tag, "score": n, "ic": ic, "ic_t": t, "liquid_frac": liq,
                    "gross_ls": r0["gross_ann"], "fee": r["fee_ann"],
                    "impact": r["impact_ann"], "borrow": r["borrow_ann"],
                    "net_ls": r["net_ann"], "breakeven_aum_M": (cap / 1e6) if cap else 0.0,
                    "long_only_excess_gross": lo0["gross_ann"],
                    "long_only_excess_net": lo["net_ann"],
                    "part_p95_pct": r["participation_p95"] * 100,
                })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 260)
    for liq in [float(x) for x in args.liquid.split(",")]:
        sub = df[df.liquid_frac == liq].sort_values(["set", "gross_ls"], ascending=[True, False])
        print(f"\n===== weekly rebalancing, liquidity universe {liq:.0%}, "
              f"AUM {args.aum / 1e6:.0f}M per leg, figures in per cent per year =====")
        print(sub[["set", "score", "ic", "ic_t", "gross_ls", "fee", "impact", "borrow", "net_ls",
                   "breakeven_aum_M", "long_only_excess_net"]].round(2).to_string(index=False))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        df.to_csv(os.path.join(args.out, "cheapbase.csv"), index=False)
        with open(os.path.join(args.out, "cheapbase.json"), "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "rows": rows}, f, indent=2)
        print(f"\nwritten {os.path.join(args.out, 'cheapbase.csv')}")


if __name__ == "__main__":
    main()
