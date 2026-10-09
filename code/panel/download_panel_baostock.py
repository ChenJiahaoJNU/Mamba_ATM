"""Download A-share daily bars from baostock into per-stock CSVs for the leak-free benchmark.

Why this replaces the Yahoo version.  The first panel was pulled from Yahoo's chart API, which
returns a raw `close` alongside an adjusted one, and the raw close was written into 收盘价.  Ex-dates
therefore looked like large negative returns, and the only defence was to drop rows moving more than
20%, which misses the common cases in this market, since a large stock dividend moves the raw price
about 9% and a cash dividend about 2%.  The 换手率 column was also written as a constant zero because
Yahoo does not serve it.  baostock serves 前复权 OHLC, the exchange turnover rate and the official
percentage change for A-shares, so all three problems disappear at the source.

Adjustment.  adjustflag '2' is 前复权.  It rescales the whole history when a new distribution occurs,
but the scaling factor is constant across the series at download time, so returns, and therefore every
feature and label in the benchmark, are identical to the 后复权 series up to a constant scale.  Daily
returns are the quantity the model sees.

Universe.  Codes are taken by stepping through the Shenzhen main board, the SME board and the Shanghai
main board at a fixed stride and keeping whichever return enough history.  This is a convenience
sample rather than an index membership list, so stocks that delisted during the sample, that listed
late, or that the source serves incompletely are absent.  That is survivorship bias, and the paper
states it as a limitation rather than describing the panel as a representative cross-section.
"""
import argparse
import os
import time

import pandas as pd

FIELDS = 'date,open,high,low,close,volume,amount,turn,pctChg'


def build_universe(step=3):
    codes = []
    for i in range(1, 1000, step):
        codes.append(f'sz.{i:06d}')
    for i in range(1, 1000, step):
        codes.append(f'sz.002{i:03d}')
    for i in range(0, 1000, step):
        codes.append(f'sh.600{i:03d}')
    for i in range(0, 1000, step):
        codes.append(f'sh.601{i:03d}')
    for i in range(0, 1000, step):
        codes.append(f'sh.603{i:03d}')
    return codes


def fetch(bs, code, start, end):
    rs = bs.query_history_k_data_plus(code, FIELDS, start_date=start, end_date=end,
                                      frequency='d', adjustflag='2')
    if rs.error_code != '0':
        raise RuntimeError(f'{rs.error_code} {rs.error_msg}')
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def to_panel_csv(df, ticker):
    df = df[df['close'] != ''].copy()
    out = pd.DataFrame({
        '日期': df['date'],
        '当日开盘': pd.to_numeric(df['open'], errors='coerce'),
        '收盘价': pd.to_numeric(df['close'], errors='coerce'),
        '最高价': pd.to_numeric(df['high'], errors='coerce'),
        '最低价': pd.to_numeric(df['low'], errors='coerce'),
        '成交量': pd.to_numeric(df['volume'], errors='coerce'),
        '成交额': pd.to_numeric(df['amount'], errors='coerce'),
        '涨跌幅': pd.to_numeric(df['pctChg'], errors='coerce') / 100.0,
        '换手率': pd.to_numeric(df['turn'], errors='coerce'),
    })
    out['昨收'] = out['收盘价'].shift(1)
    out = out.dropna(subset=['收盘价', '当日开盘', '最高价', '最低价', '成交量'])
    # no row is dropped for moving too much: with adjusted prices a large move is a real move
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--want', type=int, default=240)
    ap.add_argument('--step', type=int, default=3)
    ap.add_argument('--min-bars', type=int, default=1200)
    ap.add_argument('--start', default='2015-01-01')
    ap.add_argument('--end', default='2025-12-31')
    args = ap.parse_args()

    import baostock as bs

    os.makedirs(args.out, exist_ok=True)
    lg = bs.login()
    if lg.error_code != '0':
        raise SystemExit(f'baostock login failed: {lg.error_msg}')

    codes = build_universe(args.step)
    got = 0
    t0 = time.time()
    try:
        for t in codes:
            if got >= args.want:
                break
            short = t.replace('sz.', '').replace('sh.', '')
            suffix = '.SZ' if t.startswith('sz.') else '.SS'
            dest = os.path.join(args.out, f'{short}{suffix}.csv')
            if os.path.exists(dest):
                got += 1
                continue
            try:
                df = fetch(bs, t, args.start, args.end)
            except Exception:
                continue
            if len(df) < args.min_bars:
                continue
            panel = to_panel_csv(df, t)
            if len(panel) < args.min_bars:
                continue
            panel.to_csv(dest, index=False)
            got += 1
            if got % 20 == 0:
                el = time.time() - t0
                print(f'  {got}/{args.want} stocks   ({el:.0f}s)', flush=True)
    finally:
        bs.logout()
    print(f'DONE: {got} stocks -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
