"""The break-even table with a cost model that reflects how A-share trades are actually charged.

For every score (models, characteristics, and the factor-neutralised versions of both) and for daily
and weekly rebalancing, the script reports the decomposition of the cost into fees, stamp duty,
square-root impact and borrow, the net return at three book sizes, and the book size at which impact
eats the remaining edge.

Usage: python xsec_realcost.py --scores out/xsec_main/xsec_scores_all.npz --panel <panel> --seq 60 \
    --rebalances 1,5 [--neutral-scores out/xsec_main/xsec_neutral_scores.npz] --out out/xsec_main
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "libs"))
sys.path.insert(0, HERE)

import costs as C  # noqa: E402
import xsection as X  # noqa: E402

DEFAULT_PANEL = (r"D:\研究生学习资料\暨南大学-博士学习资料\论文\重投\MambaATM"
                 r"\代码\data_panel_adj")
AUM_SIZES = [1e7, 1e8, 5e8, 1e9]


def load_scores(path):
    z = np.load(path, allow_pickle=False)
    names = sorted({k.split("|")[0] for k in z.files
                    if "|" in k and not k.endswith(("|y", "|date", "|raw"))})
    ref = next(k.split("|")[0] for k in z.files if k.endswith("|y"))
    return z, names, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--neutral-scores", default="")
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--seq", type=int, default=60)
    ap.add_argument("--quantile", type=float, default=0.1)
    ap.add_argument("--rebalances", default="1,5")
    ap.add_argument("--impact-coef", type=float, default=0.5)
    ap.add_argument("--borrow", type=float, default=0.08)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    z, names, ref = load_scores(args.scores)
    y, di = z[f"{ref}|y"], z[f"{ref}|date"]
    te_rows = z["te_rows"] if "te_rows" in z.files else np.arange(len(y))

    data_all = X.load_panel(args.panel)
    data = {c: data_all[c] for c in sorted(data_all)}
    ds = X.build_dataset(data, args.seq)
    tradable = X.tradable_mask(data, ds)[te_rows]
    stock_ids = z["te_stocks"] if "te_stocks" in z.files else ds.stock_idx[te_rows]
    adv, sigma = C.liquidity_arrays(data, ds)
    adv, sigma = adv[te_rows], sigma[te_rows]
    print(f"{len(names)} scores, {len(y):,} rows, {len(np.unique(di)):,} dates, "
          f"median daily turnover value {np.median(adv) / 1e6:.0f} M CNY")

    sets = [("raw", args.scores)]
    if args.neutral_scores and os.path.exists(args.neutral_scores):
        sets.append(("neutralised", args.neutral_scores))

    out = []
    for tag, path in sets:
        zz, nn, rr = load_scores(path)
        yy = zz[f"{rr}|y"]
        dd = zz[f"{rr}|date"]
        for score_name in nn:
            scores = zz[f"{score_name}|score"]
            for h in [int(x) for x in args.rebalances.split(",")]:
                base = dict(quantile=args.quantile, rebalance=h, tradable=tradable,
                            stock_ids=stock_ids)
                cm = C.CostModel(impact_coef=args.impact_coef, aum=AUM_SIZES[1],
                                 borrow_annual=args.borrow)
                r = C.replay(scores, yy, dd, ds.grid, adv=adv, sigma=sigma, cm=cm, **base)
                zero = C.CostModel(commission=0.0, transfer=0.0, slippage=0.0, stamp_before=0.0,
                                   stamp_after=0.0, borrow_annual=0.0, impact_coef=0.0,
                                   aum=AUM_SIZES[1])
                r0 = C.replay(scores, yy, dd, ds.grid, adv=adv, sigma=sigma, cm=zero, **base)
                # impact grows as sqrt(size), so one evaluation at the reference size gives the
                # whole capacity curve without replaying the book at every size
                nets = {aum: r["gross_ann"] - r["fee_ann"] - r["borrow_ann"]
                              - r["impact_ann"] * np.sqrt(aum / AUM_SIZES[1])
                        for aum in AUM_SIZES}
                cap = C.breakeven_aum(r0["gross_ann"], r["fee_ann"], r["borrow_ann"],
                                      r["impact_ann"], AUM_SIZES[1])
                if tag == "neutralised":
                    lon = C.replay(scores, yy, dd, ds.grid, adv=adv, sigma=sigma, cm=cm,
                                   long_only=True, **base)
                    lon0 = C.replay(scores, yy, dd, ds.grid, adv=adv, sigma=sigma, cm=zero,
                                    long_only=True, **base)
                else:
                    lon = lon0 = None
                out.append({
                    "set": tag, "score": score_name, "hold_days": h,
                    "gross_ann": r0["gross_ann"],
                    "fee_ann": r["fee_ann"], "impact_ann_at_100M": r["impact_ann"],
                    "borrow_ann": r["borrow_ann"],
                    "net_10M": nets[AUM_SIZES[0]], "net_100M": nets[AUM_SIZES[1]],
                    "net_500M": nets[AUM_SIZES[2]], "net_1B": nets[AUM_SIZES[3]],
                    "breakeven_aum_M": (cap / 1e6) if cap else 0.0,
                    "part_mean_pct": r["participation_mean"] * 100,
                    "part_p95_pct": r["participation_p95"] * 100,
                    "traded_above_10pct_share": r["traded_above_threshold_share"] * 100,
                    "long_only_excess_gross": (lon0["gross_ann"] if lon0 else np.nan),
                    "long_only_excess_net": (lon["net_ann"] if lon else np.nan),
                })
    df = pd.DataFrame(out)
    pd.set_option("display.width", 250)
    show = df.copy()
    for c in ["gross_ann", "fee_ann", "impact_ann_at_100M", "borrow_ann", "net_10M", "net_100M",
              "net_500M", "net_1B", "long_only_excess_gross", "long_only_excess_net"]:
        show[c] = show[c].round(1)
    for c in ["breakeven_aum_M", "part_mean_pct", "part_p95_pct"]:
        show[c] = show[c].round(2)
    cols = ["set", "score", "hold_days", "gross_ann", "fee_ann", "impact_ann_at_100M", "borrow_ann",
            "net_10M", "net_100M", "net_500M", "net_1B", "breakeven_aum_M", "part_p95_pct", "traded_above_10pct_share",
            "long_only_excess_net"]
    print("\nall figures are annualised per cent, except breakeven_aum_M and participation")
    print(show[cols].to_string(index=False))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        df.to_csv(os.path.join(args.out, "xsec_realcost.csv"), index=False)
        with open(os.path.join(args.out, "xsec_realcost.json"), "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "rows": out}, f, indent=2)
        print(f"\nwritten {os.path.join(args.out, 'xsec_realcost.csv')}")


if __name__ == "__main__":
    main()
