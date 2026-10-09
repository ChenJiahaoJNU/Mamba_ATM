"""Re-evaluate saved cross-sectional scores under several holding periods and cost levels.

The scores are stored by `run_xsec.py`; this script does not retrain anything. It exists because the
economic reading of a signal depends on how often the book is turned over, and that is a reporting
choice rather than a property of the model.

Usage: python xsec_report.py --scores out/xsec_pilot/xsec_scores.npz --panel <panel> --seq 60 \
    --quantile 0.2 --out out/xsec_pilot
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

import xsection as X  # noqa: E402

DEFAULT_PANEL = (r"D:\研究生学习资料\暨南大学-博士学习资料\论文\重投\MambaATM"
                 r"\代码\data_panel_adj")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--stocks", type=int, default=0, help="how many stocks were used, 0 = all in file")
    ap.add_argument("--seq", type=int, default=60)
    ap.add_argument("--quantile", type=float, default=0.1)
    ap.add_argument("--costs", default="0.0006,0.0010,0.0015")
    ap.add_argument("--rebalances", default="1,5,20")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    z = np.load(args.scores, allow_pickle=False)
    models = sorted({k.split("|")[0] for k in z.files if "|" in k})
    ref = next(k.split("|")[0] for k in z.files if k.endswith("|y"))
    grid = z["grid"]

    data_all = X.load_panel(args.panel)
    codes = sorted(data_all)[:args.stocks] if args.stocks else sorted(data_all)
    data = {c: data_all[c] for c in codes}
    ds = X.build_dataset(data, args.seq)
    all_tradable = X.tradable_mask(data, ds)
    if "te_rows" in z.files:
        te_rows = z["te_rows"]
    else:
        # reconstruct the test rows of the run: same fold rule, so the selection is identical
        first_test = ds.grid[int(len(ds.grid) * 0.4)]
        splits = X.date_splits(ds.grid, first_test, folds=5, purge_days=1)
        te_rows = np.concatenate([np.where(np.isin(ds.grid[ds.date_idx], list(te)))[0]
                                  for _, te in splits])
    tradable = all_tradable[te_rows]
    stock_ids = z["te_stocks"] if "te_stocks" in z.files else ds.stock_idx[te_rows]
    print(f"{len(te_rows):,} pooled test rows, {tradable.mean() * 100:.1f}% tradable")

    rows = []
    for m in models:
        score = z[f"{m}|score"]
        y = z[f"{m}|y"] if f"{m}|y" in z.files else z[f"{ref}|y"]
        di = z[f"{m}|date"] if f"{m}|date" in z.files else z[f"{ref}|date"]
        ic = X.daily_ic(score, y, di)
        rec = {"model": m, **X.ic_summary(ic)}
        for rb in [int(x) for x in args.rebalances.split(",")]:
            for c in [float(x) for x in args.costs.split(",")]:
                p = X.decile_portfolio(score, y, di, cost=c, quantile=args.quantile,
                                       tradable=tradable, rebalance=rb, stock_ids=stock_ids)
                for k, v in p.items():
                    rec[f"h{rb}_c{int(c * 10000)}bp_{k}"] = v
        rows.append(rec)

    df = pd.DataFrame(rows).set_index("model")
    show = [c for c in df.columns if c.startswith(("ic_", "h1_", "h5_", "h20_"))
            and c.endswith(("ls_ann", "ls_sharpe", "ls_turnover", "top_minus_market_ann"))]
    show = ["ic_mean", "ic_t"] + [c for c in show if "c10bp" in c]
    print(df[show].round(4).to_string())

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        df.to_csv(os.path.join(args.out, "xsec_report.csv"))
        with open(os.path.join(args.out, "xsec_report.json"), "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "metrics": df.to_dict(orient="index")}, f,
                      indent=2, ensure_ascii=False)
        print(f"\nwritten {os.path.join(args.out, 'xsec_report.csv')}")


if __name__ == "__main__":
    main()
