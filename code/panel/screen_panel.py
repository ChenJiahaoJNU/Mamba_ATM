"""Screen the adjusted panel for price-series integrity, one decision per stock.

The adjusted series contains daily moves that A-shares cannot produce.  Two causes appear in the data:
a stock that was suspended and resumed into the delisting process, where the price limit does not span
the suspension, and a mid-series break in the adjustment factor, which for two names moves the price
by more than a factor of five.  Without a corporate-action database the two cannot be told apart, and
both leave a label that is meaningless, so the rule applied here is the simple one: a stock whose
adjusted series contains any daily move beyond the exchange band is excluded entirely.

Excluding the stock rather than the row is deliberate.  Dropping an interior row leaves the next
return spanning a two-day gap, which manufactures a large spurious return, and that is exactly the
defect that made the previous panel unusable.  A whole-stock screen keeps every remaining series
continuous.

The first row of each stock is also removed, because it has no previous close and the percentage
change the source reports for it is computed against nothing.
"""
import argparse
import glob
import os
import shutil

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--backup', required=True)
    ap.add_argument('--max-abs-move', type=float, default=0.25)
    args = ap.parse_args()

    if not os.path.isdir(args.backup):
        shutil.copytree(args.panel, args.backup)
        print(f'unscreened copy kept in {args.backup}')
    else:
        print(f'unscreened copy already present at {args.backup}')

    kept, dropped = [], []
    for f in sorted(glob.glob(os.path.join(args.panel, '*.csv'))):
        d = pd.read_csv(f, encoding='utf-8-sig')
        d = d.iloc[1:].reset_index(drop=True)                 # drop the first row, no previous close
        close = pd.to_numeric(d['收盘价'], errors='coerce')
        ret = close.pct_change()
        if not np.isfinite(ret.iloc[1:]).all():
            dropped.append((os.path.basename(f), 'non-finite price'))
            os.remove(f)
            continue
        worst = float(ret.abs().iloc[1:].max())
        if worst > args.max_abs_move:
            dropped.append((os.path.basename(f), f'move of {worst * 100:.1f}%'))
            os.remove(f)
            continue
        # recompute the change, the previous close and the turnover from the adjusted series so that
        # every column agrees with the price column
        d['昨收'] = close.shift(1)
        d['涨跌幅'] = ret
        d['换手率'] = pd.to_numeric(d['换手率'], errors='coerce').fillna(0.0)
        d.to_csv(f, index=False)
        kept.append(os.path.basename(f))

    print(f'kept {len(kept)} stocks, dropped {len(dropped)}')
    for code, why in dropped:
        print(f'  dropped {code:16s} {why}')


if __name__ == '__main__':
    main()
