"""What the commissions have to be for each strategy to break even.

Every return reported so far is net of a constant one-way cost per position change. That cost is an
assumption, so the useful quantity is not the net return at one assumed rate but the rate at which
the net return reaches zero. Because the cost is linear in the rate, the break-even rate can be read
off exactly from two evaluations: gross return, and the cost charged per year at a reference rate.

Realistic A-share rates, for comparison, in basis points one way:

    commission (retail, negotiated)      2.5 - 3.0
    stamp duty, sell side only           5.0 since August 2023, 10.0 before
    transfer fee                         0.1
    slippage, liquid names               several, and the largest unknown of the four

so a round trip is roughly 8-11 bp for the sell-heavy side and 3-5 bp for the buy side, and a
symmetric charge of 6 bp one way is close to the average of the two.

Usage: python xsec_breakeven.py --scores out/xsec_pilot/xsec_scores.npz --panel <panel> \
    --stocks 40 --seq 60 --quantile 0.2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "libs"))
sys.path.insert(0, HERE)

import xsection as X  # noqa: E402

DEFAULT_PANEL = (r"D:\研究生学习资料\暨南大学-博士学习资料\论文\重投\MambaATM"
                 r"\代码\data_panel_adj")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--stocks", type=int, default=0)
    ap.add_argument("--seq", type=int, default=60)
    ap.add_argument("--quantile", type=float, default=0.1)
    ap.add_argument("--reference-bp", type=float, default=10.0)
    ap.add_argument("--rebalances", default="1,5,20")
    args = ap.parse_args()

    z = np.load(args.scores, allow_pickle=False)
    models = sorted({k.split("|")[0] for k in z.files if "|" in k})
    ref = next(k.split("|")[0] for k in z.files if k.endswith("|y"))
    data_all = X.load_panel(args.panel)
    codes = sorted(data_all)[:args.stocks] if args.stocks else sorted(data_all)
    data = {c: data_all[c] for c in codes}
    ds = X.build_dataset(data, args.seq)
    if "te_rows" in z.files:
        te_rows = z["te_rows"]
    else:
        first_test = ds.grid[int(len(ds.grid) * 0.4)]
        splits = X.date_splits(ds.grid, first_test, folds=5, purge_days=1)
        te_rows = np.concatenate([np.where(np.isin(ds.grid[ds.date_idx], list(te)))[0]
                                  for _, te in splits])
    tradable = X.tradable_mask(data, ds)[te_rows]
    stock_ids = z["te_stocks"] if "te_stocks" in z.files else ds.stock_idx[te_rows]

    ref_cost = args.reference_bp / 1e4
    rows = []
    for m in models:
        score = z[f"{m}|score"]
        y = z[f"{m}|y"] if f"{m}|y" in z.files else z[f"{ref}|y"]
        di = z[f"{m}|date"] if f"{m}|date" in z.files else z[f"{ref}|date"]
        for rb in [int(x) for x in args.rebalances.split(",")]:
            gross = X.decile_portfolio(score, y, di, cost=0.0, quantile=args.quantile,
                                       tradable=tradable, rebalance=rb, stock_ids=stock_ids)
            net = X.decile_portfolio(score, y, di, cost=ref_cost, quantile=args.quantile,
                                     tradable=tradable, rebalance=rb, stock_ids=stock_ids)
            drag = gross["ls_ann"] - net["ls_ann"]            # cost paid per year at the reference
            breakeven_bp = (gross["ls_ann"] / drag * args.reference_bp) if drag > 0 else np.nan
            rows.append({
                "model": m, "hold_days": rb,
                "gross_ann_pct": gross["ls_ann"] * 100,
                "net_ann_pct_at_ref": net["ls_ann"] * 100,
                "cost_pct_year_at_ref": drag * 100,
                "breakeven_bp_one_way": breakeven_bp,
                "long_only_excess_gross_pct": gross["top_minus_market_ann"] * 100,
                "long_only_excess_net_pct": net["top_minus_market_ann"] * 100,
            })
    df = pd.DataFrame(rows)
    print(f"reference cost {args.reference_bp:.0f} bp one way\n")
    print(df.round(2).to_string(index=False))
    print("\nbreakeven_bp_one_way is the symmetric one-way commission at which the long-short book "
          "nets zero;\nthe long-only columns compare the top quantile with the equal-weight universe")


if __name__ == "__main__":
    main()
