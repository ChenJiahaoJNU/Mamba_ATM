"""Panel statistics quoted in the paper: sizes, suspensions, feature-return association, autocorrelation.

Writes `panel_stats.json` next to the requested output so that the table and the prose read the same
numbers instead of recomputing them differently.

Usage: python panel_stats.py --panel <dir> --out out/main/panel_stats.json
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats as sps

FEATURES = ["当日开盘", "收盘价", "最高价", "最低价", "成交量", "成交额", "换手率", "昨收"]
RET = "涨跌幅"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.panel, "*.csv")))
    per_stock, rank_corrs = [], []
    autocorr = []
    all_dates = set()
    for path in files:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        all_dates.update(df["日期"].astype(str).tolist())
        ret = pd.to_numeric(df[RET], errors="coerce")
        vol = pd.to_numeric(df["成交量"], errors="coerce")
        keep = ret.notna() & vol.notna() & (vol > 0)
        nxt = ret.shift(-1)
        use = keep & nxt.notna()
        f = df[FEATURES].apply(pd.to_numeric, errors="coerce")[use]
        y = nxt[use].to_numpy()
        per_stock.append({
            "code": os.path.basename(path).replace(".csv", ""),
            "rows": int(len(df)),
            "usable": int(use.sum()),
            "susp": int((vol == 0).sum()),
            # the three categories of removed row, counted separately, together with the one
            # overlap that exists, so that the removal lines of the table sum to the usable total
            "no_prev_close": int(ret.isna().sum()),
            "no_label": int(nxt.isna().sum()),
            "no_prev_and_susp": int((ret.isna() & (vol == 0)).sum()),
            "first": str(df["日期"].iloc[0]),
            "last": str(df["日期"].iloc[-1]),
            "frac_up": float((y > 0).mean()) if len(y) else np.nan,
        })
        if len(y) > 30:
            for c in FEATURES:
                x = f[c].to_numpy()
                if np.std(x) > 0:
                    rank_corrs.append(abs(sps.spearmanr(x, y).statistic))
            r = ret[use].to_numpy()
            if len(r) > 30:
                autocorr.append(float(pd.Series(r).autocorr(lag=1)))

    ds = pd.DataFrame(per_stock)
    out = {
        "n_stocks": int(len(ds)),
        "n_dates": int(len(all_dates)),
        "rows_total": int(ds.rows.sum()),
        "usable_total": int(ds.usable.sum()),
        "susp_total": int(ds.susp.sum()),
        "susp_pct": float(100.0 * ds.susp.sum() / ds.rows.sum()),
        "no_prev_total": int(ds.no_prev_close.sum()),
        "no_label_total": int(ds.no_label.sum()),
        "overlap_total": int(ds.no_prev_and_susp.sum()),
        "removed_total": int(ds.rows.sum() - ds.usable.sum()),
        "usable_min": int(ds.usable.min()),
        "usable_median": int(ds.usable.median()),
        "usable_max": int(ds.usable.max()),
        "first_date": str(ds["first"].mode()[0]),
        "last_date": str(ds["last"].mode()[0]),
        "frac_up_mean": float(ds.frac_up.mean()),
        "frac_up_min": float(ds.frac_up.min()),
        "frac_up_max": float(ds.frac_up.max()),
        "max_rank_corr": float(np.max(rank_corrs)) if rank_corrs else float("nan"),
        "median_rank_corr": float(np.median(rank_corrs)) if rank_corrs else float("nan"),
        "autocorr_lag1_mean": float(np.mean(autocorr)) if autocorr else float("nan"),
        "autocorr_lag1_min": float(np.min(autocorr)) if autocorr else float("nan"),
        "autocorr_lag1_max": float(np.max(autocorr)) if autocorr else float("nan"),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
