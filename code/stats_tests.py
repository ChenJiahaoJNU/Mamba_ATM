"""Statistical comparison of the models across stocks.

The unit of observation is one stock, so the tests are the ones for a complete block design with
many treatments and many blocks:

* Friedman test on the per-stock metric, to ask whether the models differ at all;
* mean ranks, to order them without assuming normality;
* Wilcoxon signed-rank of the proposed model against each other model, paired by stock, with a
  Holm correction over the family of comparisons;
* the same tests on the trading metrics, so a claim about profitability is tested on the same
  footing as a claim about accuracy.

Usage: python stats_tests.py <per_stock.csv> [--metric accuracy] [--proposed gatedfusion] [--out DIR]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r".\libs")
from scipy import stats  # noqa: E402


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, idx in enumerate(order):
        val = (len(p) - i) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("per_stock")
    ap.add_argument("--metrics", default="accuracy,f1,total_return,sharpe,max_drawdown")
    ap.add_argument("--proposed", default="gatedfusion")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.per_stock)
    models = sorted(df.model.unique())
    report = {}

    for metric in args.metrics.split(","):
        pivot = df.pivot_table(index="stock", columns="model", values=metric).dropna()
        if pivot.shape[1] < 2 or pivot.shape[0] < 3:
            continue
        cols = [c for c in models if c in pivot.columns]
        pivot = pivot[cols]
        fr = stats.friedmanchisquare(*[pivot[c].to_numpy() for c in cols])
        # mean rank, rank 1 = best (highest for returns/accuracy, lowest for drawdown).
        # The unit of observation is one stock, so the ranking runs across models within each stock
        # and is then averaged over stocks; axis=1, not axis=0.
        arr = pivot.to_numpy(dtype=float)
        higher_better = metric not in ("max_drawdown",)
        ranks = np.apply_along_axis(
            lambda r: pd.Series(r).rank(ascending=not higher_better).to_numpy(), 1, arr)
        mean_rank = dict(zip(cols, ranks.mean(axis=0).round(3)))

        rec = {"friedman_stat": float(fr.statistic), "friedman_p": float(fr.pvalue),
               "n_stocks": int(pivot.shape[0]), "mean_rank": mean_rank, "paired": {}}
        base = pivot[args.proposed].to_numpy(dtype=float) if args.proposed in cols else None
        if base is not None:
            pairs, ps = [], []
            for c in cols:
                if c == args.proposed:
                    continue
                a, b = base, pivot[c].to_numpy(dtype=float)
                if np.allclose(a, b):
                    p = 1.0
                else:
                    p = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
                pairs.append((c, float(np.mean(a - b)), p))
                ps.append(p)
            adj = holm(ps)
            for (c, diff, p), pa in zip(pairs, adj):
                rec["paired"][c] = {"mean_diff": diff, "p_raw": p, "p_holm": float(pa),
                                    "significant_5pct": bool(pa < 0.05)}
        report[metric] = rec

        print(f"\n=== {metric} ===")
        print(f"  Friedman chi2 {fr.statistic:.1f}, p {fr.pvalue:.3g}, "
              f"{pivot.shape[0]} stocks, {len(cols)} models")
        ranked = sorted(mean_rank.items(), key=lambda kv: kv[1])
        for name, rk in ranked:
            print(f"    rank {rk:5.2f}  {name}")
        if base is not None:
            print(f"  paired vs {args.proposed} (Holm-corrected):")
            for c, d in sorted(rec["paired"].items(), key=lambda kv: kv[1]["p_holm"]):
                star = "*" if d["significant_5pct"] else " "
                print(f"    {star} {c:<20} mean diff {d['mean_diff']:+.4f}  "
                      f"p_holm {d['p_holm']:.3g}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "stats_tests.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nwritten {os.path.join(args.out, 'stats_tests.json')}")


if __name__ == "__main__":
    main()
