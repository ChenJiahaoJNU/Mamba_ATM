"""How often the portfolios actually trade, measured from the holdings rather than asserted.

For each holding period the script reports, per model:

* `names_per_leg`   how many stocks a leg holds (the quantile size),
* `replaced_pct`    the share of a leg replaced on an average day,
* `trades_per_day`  names bought plus names sold on an average day, both legs together,
* `cost_bp_day`     the cost charged that day, in basis points of one unit of book,
* `cost_pct_year`   the same annualised,
* `days_traded`     the share of days on which the book changed at all.

A leg that is fully replaced costs 2.0 in sum|dw| (one to sell the old names, one to buy the new
ones), so the share of a leg replaced is sum|dw| / 2.

Usage: python xsec_frequency.py --scores out/xsec_pilot/xsec_scores.npz --panel <panel> \
    --stocks 40 --seq 60 --quantile 0.2 --cost 0.0010
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

DEFAULT_PANEL = (r"."
                 r"\data\adj")


def daily_book(scores, y, date_idx, stock_ids, tradable, quantile, rebalance, min_names=10):
    """Replay the book day by day and return the per-day trading statistics."""
    dates = np.unique(date_idx)
    rows = []
    prev_t, prev_b, held_t, held_b = {}, {}, {}, {}
    for step, d in enumerate(dates):
        m = date_idx == d
        if tradable is not None:
            m = m & tradable
        idxs = np.where(m)[0]
        if not idxs.size:
            rows.append((d, 0, 0.0, 0.0))
            continue
        need = max(min_names, int(np.ceil(1.0 / quantile)) * 2)
        if (step % max(1, rebalance) == 0 or not held_t) and len(idxs) >= need:
            k = max(1, int(round(quantile * len(idxs))))
            order = np.argsort(-scores[idxs])
            held_t = {int(stock_ids[idxs[o]]): 1.0 / k for o in order[:k]}
            held_b = {int(stock_ids[idxs[o]]): 1.0 / k for o in order[-k:]}
            t_cost = sum(abs(held_t.get(i, 0.0) - prev_t.get(i, 0.0))
                         for i in set(held_t) | set(prev_t))
            b_cost = sum(abs(held_b.get(i, 0.0) - prev_b.get(i, 0.0))
                         for i in set(held_b) | set(prev_b))
            prev_t, prev_b = held_t, held_b
            k_leg = len(held_t)
        else:
            t_cost = b_cost = 0.0
            k_leg = len(held_t) or 0
        rows.append((d, k_leg, t_cost, b_cost))
    return pd.DataFrame(rows, columns=["date", "names_per_leg", "top_cost", "bot_cost"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--stocks", type=int, default=0)
    ap.add_argument("--seq", type=int, default=60)
    ap.add_argument("--quantile", type=float, default=0.1)
    ap.add_argument("--cost", type=float, default=0.0010)
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
        # files written before the test rows were saved: rebuild the same folds, which are
        # deterministic given the date grid and the fold rule
        first_test = ds.grid[int(len(ds.grid) * 0.4)]
        splits = X.date_splits(ds.grid, first_test, folds=5, purge_days=1)
        te_rows = np.concatenate([np.where(np.isin(ds.grid[ds.date_idx], list(te)))[0]
                                  for _, te in splits])
    tradable = X.tradable_mask(data, ds)[te_rows]
    stock_ids = (z["te_stocks"] if "te_stocks" in z.files else ds.stock_idx[te_rows])

    out = []
    for m in models:
        score = z[f"{m}|score"]
        y = z[f"{m}|y"] if f"{m}|y" in z.files else z[f"{ref}|y"]
        di = z[f"{m}|date"] if f"{m}|date" in z.files else z[f"{ref}|date"]
        for rb in [int(x) for x in args.rebalances.split(",")]:
            b = daily_book(score, y, di, stock_ids, tradable, args.quantile, rb)
            t_cost, b_cost = b.top_cost.to_numpy(), b.bot_cost.to_numpy()
            turn = 0.5 * (t_cost + b_cost)                 # average share of a leg replaced
            trades = (t_cost + b_cost) / 2 * b.names_per_leg.to_numpy()
            cost_day = (t_cost + b_cost) * args.cost
            out.append({
                "model": m, "hold_days": rb,
                "names_per_leg": float(b.names_per_leg.replace(0, np.nan).mean()),
                "replaced_pct": float(turn.mean() * 100),
                "trades_per_day": float(np.nanmean(trades)),
                "trades_per_year": float(np.nanmean(trades) * 252),
                "cost_bp_day": float(cost_day.mean() * 1e4),
                "cost_pct_year": float(cost_day.mean() * 252 * 100),
                "days_traded_pct": float((cost_day > 0).mean() * 100),
            })
    df = pd.DataFrame(out)
    print(df.round(2).to_string(index=False))
    print("\nleg replaced = sum|dw| / 2, so 100 per cent means the leg is fully turned over")
    print("trades per day counts a buy and a sell of the same name as two")


if __name__ == "__main__":
    main()
