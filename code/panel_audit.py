"""Audit the rebuilt A-share panel before any model touches it.

The previous shipped dataset failed four checks in a row (label column inconsistent with the price
columns, 248 constant columns, a 500-row window described as 2,435, and unnamed assets). This script
applies the same checks to the panel the rework will actually use, so that the fix is measured
rather than assumed.

Checks
------
1. schema: columns present, one file per stock, rows per file;
2. missing: NaN and empty cells per column, and where they sit;
3. label consistency: corr(涨跌幅, close/prev_close - 1) and the mean absolute difference;
4. arithmetic: fraction of rows where 昨收 equals the previous row's 收盘价;
5. constant columns: how many features carry no information, after removing the rows with no label;
6. screen: largest absolute daily move implied by the adjusted prices;
7. degenerate series: stocks whose return series is (almost) always the same sign.

Usage: python panel_audit.py [panel_dir]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

DEFAULT_PANEL = (r"D:\研究生学习资料\暨南大学-博士学习资料\论文\重投\MambaATM"
                 r"\代码\data_panel_adj")
FEATURE_COLS = ["当日开盘", "收盘价", "最高价", "最低价", "成交量", "成交额", "换手率", "昨收"]
RET_COL = "涨跌幅"


def audit_one(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    code = os.path.basename(path).replace(".csv", "")
    rec = {"code": code, "rows_raw": len(df), "cols": len(df.columns)}
    missing_cols = [c for c in ["日期"] + FEATURE_COLS + [RET_COL] if c not in df.columns]
    rec["missing_cols"] = missing_cols
    if missing_cols:
        return rec

    close = pd.to_numeric(df["收盘价"], errors="coerce")
    ret = pd.to_numeric(df[RET_COL], errors="coerce")
    prev = pd.to_numeric(df["昨收"], errors="coerce")
    rec["nan_dates"] = int(pd.to_datetime(df["日期"], errors="coerce").isna().sum())
    rec["nan_ret"] = int(ret.isna().sum())
    rec["nan_prev"] = int(prev.isna().sum())
    rec["nan_feat"] = int(df[FEATURE_COLS].isna().sum().sum())
    rec["zero_vol_rows"] = int((pd.to_numeric(df["成交量"], errors="coerce") == 0).sum())

    # 3. label vs prices
    implied = close / close.shift(1) - 1.0
    unit = 100.0 if ret.abs().max() > 0.5 else 1.0        # percent or fraction
    m = np.isfinite(implied.to_numpy()) & np.isfinite(ret.to_numpy())
    if m.sum() > 10:
        rec["corr_label_price"] = float(np.corrcoef(implied.to_numpy()[m],
                                                    (ret / unit).to_numpy()[m])[0, 1])
        rec["mad_label_price"] = float(np.mean(np.abs(implied.to_numpy()[m]
                                                      - (ret / unit).to_numpy()[m])))
    else:
        rec["corr_label_price"] = np.nan
        rec["mad_label_price"] = np.nan

    # 4. 昨收 continuity
    m2 = np.isfinite(prev.to_numpy()) & np.isfinite(close.shift(1).to_numpy())
    rec["frac_prev_equals_prevclose"] = float(np.mean(np.abs(
        prev.to_numpy()[m2] - close.shift(1).to_numpy()[m2]) < 1e-6)) if m2.sum() else np.nan

    # 5-7 on the rows that will actually be used
    feat = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    usable = (~ret.isna()) & feat.notna().all(axis=1)
    rec["rows_usable"] = int(usable.sum())
    f = feat[usable]
    rec["constant_cols"] = [c for c in FEATURE_COLS if f[c].std() < 1e-8]
    r = ret[usable]
    rec["frac_up"] = float((r > 0).mean()) if len(r) else np.nan
    adj_ret = implied[usable]
    rec["max_abs_move"] = float(np.nanmax(np.abs(adj_ret.to_numpy()))) if len(adj_ret) else np.nan
    rec["first_date"] = str(df["日期"].iloc[0])
    rec["last_date"] = str(df["日期"].iloc[-1])
    return rec


def main():
    panel = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PANEL
    files = sorted(glob.glob(os.path.join(panel, "*.csv")))
    print(f"panel: {panel}\nfiles: {len(files)}\n")
    rows = [audit_one(p) for p in files]
    df = pd.DataFrame(rows)

    bad_schema = df[df["missing_cols"].apply(len) > 0]
    print(f"[1] schema: {len(df) - len(bad_schema)} ok, {len(bad_schema)} with missing columns")
    for _, r in bad_schema.iterrows():
        print("    ", r["code"], r["missing_cols"])
    print(f"    rows (raw): min {df.rows_raw.min()} median {int(df.rows_raw.median())} max {df.rows_raw.max()}")

    print(f"\n[2] missing: 涨跌幅 NaN total {int(df.nan_ret.sum())} "
          f"(stocks with any: {int((df.nan_ret > 0).sum())}), 昨收 NaN total {int(df.nan_prev.sum())}, "
          f"feature NaN total {int(df.nan_feat.sum())}")
    print(f"    zero-volume rows total: {int(df.zero_vol_rows.sum())} "
          f"(stocks with any: {int((df.zero_vol_rows > 0).sum())})")

    print(f"\n[3] label consistency: corr(涨跌幅, close/prev-1) "
          f"min {df.corr_label_price.min():.4f} median {df.corr_label_price.median():.4f} "
          f"max {df.corr_label_price.max():.4f}")
    print(f"    mean |difference|: median {df.mad_label_price.median():.3e} "
          f"max {df.mad_label_price.max():.3e}")
    worst = df.nsmallest(3, "corr_label_price")[["code", "corr_label_price", "mad_label_price"]]
    print("    worst three:\n", worst.to_string(index=False))

    print(f"\n[4] 昨收_t == 收盘价_(t-1): median {df.frac_prev_equals_prevclose.median():.4f} "
          f"min {df.frac_prev_equals_prevclose.min():.4f}")

    const = df[df["constant_cols"].apply(len) > 0]
    print(f"\n[5] constant feature columns: {len(const)} stocks affected")
    if len(const):
        for _, r in const.head(10).iterrows():
            print("    ", r["code"], r["constant_cols"])

    print(f"\n[6] screen: max |daily move| median {df.max_abs_move.median():.4f} "
          f"max {df.max_abs_move.max():.4f}; stocks above 0.25: "
          f"{int((df.max_abs_move > 0.25).sum())}")

    print(f"\n[7] usable rows: min {df.rows_usable.min()} median {int(df.rows_usable.median())} "
          f"max {df.rows_usable.max()}")
    print(f"    fraction of up days: min {df.frac_up.min():.4f} median {df.frac_up.median():.4f} "
          f"max {df.frac_up.max():.4f}  (a value far from 0.5 means a near-constant label)")
    print(f"    date range: {df.first_date.mode()[0]} .. {df.last_date.mode()[0]}")

    total_days = int(df.rows_usable.sum())
    print(f"\nstock-days usable in total: {total_days:,}")


if __name__ == "__main__":
    main()
