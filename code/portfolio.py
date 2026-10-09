"""Portfolio construction and cost sensitivity from the saved predictions.

Two levels are reported because they answer different questions:

* per stock: the positions of one model on one stock, so the dispersion across the panel is visible;
* equal weight portfolio: every date, the position is the share of stocks that are long that day and
  the portfolio return is the mean return of those stocks. This is the level at which a claim about
  risk-adjusted return has to be made, because the per-stock Sharpe of a daily long/flat rule is
  dominated by the idiosyncratic volatility of that single name.

Costs are charged whenever the position changes, at a one-way rate that covers commission, stamp
duty and slippage. Several levels are reported, because the sign of a net return on a daily strategy
is sensitive to it, which is exactly what a referee is entitled to check.

Usage: python portfolio.py --preds out/main/preds.npz --per-stock out/main/per_stock.csv --out out/main
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P  # noqa: E402

COST_LEVELS = {"zero": 0.0, "low_6bp": 0.0006, "base_10bp": 0.0010, "high_15bp": 0.0015}


def load_preds(path):
    z = np.load(path)
    models = sorted({k.split("|")[0] for k in z.files})
    out = {}
    for m in models:
        keys = sorted({k.split("|")[1] for k in z.files if k.startswith(m + "|")})
        out[m] = [{"code": c,
                   "dates": z[f"{m}|{c}|dates"],
                   "pos": z[f"{m}|{c}|pos"],
                   "ret": z[f"{m}|{c}|ret"].astype(np.float64)}
                  for c in keys]
    return out


def portfolio_series(recs):
    """Equal-weight daily portfolio: mean position and mean return across the stocks present."""
    dates = np.concatenate([r["dates"] for r in recs])
    pos = np.concatenate([r["pos"].astype(np.float64) for r in recs])
    ret = np.concatenate([r["ret"] for r in recs])
    order = np.argsort(dates, kind="stable")
    dates, pos, ret = dates[order], pos[order], ret[order]
    uniq, start = np.unique(dates, return_index=True)
    ends = np.append(start[1:], len(dates))
    daily_pos = np.array([pos[s:e].mean() for s, e in zip(start, ends)])
    daily_ret = np.array([ret[s:e].mean() for s, e in zip(start, ends)])
    counts = np.array([e - s for s, e in zip(start, ends)])
    return uniq, daily_pos, daily_ret, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figs", default="")
    args = ap.parse_args()

    preds = load_preds(args.preds)
    report = {"cost_levels": COST_LEVELS, "portfolio": {}, "per_stock_mean": {}}
    curves = {}

    for model, recs in preds.items():
        uniq, dpos, dret, counts = portfolio_series(recs)
        report["portfolio"][model] = {}
        for label, cost in COST_LEVELS.items():
            report["portfolio"][model][label] = P.trading_metrics(dpos, dret, cost=cost)
        report["portfolio"][model]["days"] = int(len(uniq))
        report["portfolio"][model]["avg_breadth"] = float(counts.mean())
        report["per_stock_mean"][model] = {}
        for label, cost in COST_LEVELS.items():
            vals = [P.trading_metrics(r["pos"].astype(float), r["ret"], cost=cost)
                    for r in recs]
            report["per_stock_mean"][model][label] = {
                k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        # equity curve at the base cost, for the figure
        cost = COST_LEVELS["base_10bp"]
        net = dpos * dret - np.abs(np.diff(np.concatenate([[0.0], dpos]))) * cost
        curves[model] = (1.0 + net).cumprod()

    # buy and hold, equal weight, same dates and same breadth as the models
    ref = next(iter(preds.values()))
    uniq, dpos, dret, counts = portfolio_series(ref)
    report["buy_hold_equal_weight"] = P.trading_metrics(np.ones_like(dpos), dret, cost=0.0)
    curves["buy_hold"] = (1.0 + dret).cumprod()
    report["days"] = int(len(uniq))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "portfolio.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"portfolio level, {report['days']} dates, average breadth "
          f"{report['portfolio'][next(iter(preds))]['avg_breadth']:.1f} stocks per date\n")
    print(f"{'model':<22}{'cost':>10}{'total%':>9}{'ann%':>8}{'sharpe':>8}{'maxDD%':>8}{'turn':>7}")
    for model, rec in report["portfolio"].items():
        for label in COST_LEVELS:
            m = rec[label]
            print(f"{model:<22}{label:>10}{m['total_return']*100:>9.1f}{m['annual_return']*100:>8.1f}"
                  f"{m['sharpe']:>8.2f}{m['max_drawdown']*100:>8.1f}{m['turnover']:>7.3f}")
    bh = report["buy_hold_equal_weight"]
    print(f"{'buy_hold':<22}{'-':>10}{bh['total_return']*100:>9.1f}{bh['annual_return']*100:>8.1f}"
          f"{bh['sharpe']:>8.2f}{bh['max_drawdown']*100:>8.1f}{0.0:>7.3f}")

    if args.figs:
        os.makedirs(args.figs, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7.0, 4.0))
        for name, curve in sorted(curves.items()):
            style = dict(lw=2.2, color="black", ls="--") if name == "buy_hold" else dict(lw=1.4)
            plt.plot(curve, label=name, **style)
        plt.yscale("log")
        plt.xlabel("trading day")
        plt.ylabel("equity (log scale)")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        out = os.path.join(args.figs, "portfolio_curves.png")
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"\nwritten {out}")

    print(f"written {os.path.join(args.out, 'portfolio.json')}")


if __name__ == "__main__":
    main()
