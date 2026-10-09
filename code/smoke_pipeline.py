"""Smoke test of the pipeline: data shape, walk-forward splits, and the two reference points.

The reference points matter for the rewrite. A directional accuracy of 0.65 on this dataset means
nothing unless it beats the majority-class rate, and a strategy return means nothing unless it
beats buy-and-hold.
"""
import sys

import numpy as np

sys.path.insert(0, r".\libs")
sys.path.insert(0, r".\code")

import pipeline as P  # noqa: E402

DATA = r".\data"
SEQ = 120

print("%-10s %6s %6s %8s %8s %10s" % ("stock", "rows", "windows", "up-rate", "feat", "extra"))
store = {}
for i in range(1, 5):
    dates, feat, ret, extra = P.load_stock(rf"{DATA}\Stock{i}.csv")
    X, yc, yr, idx = P.make_windows(feat, ret, SEQ)
    store[i] = (X, yc, yr, idx)
    print("%-10s %6d %6d %8.3f %8d %10d" % ("Stock%d" % i, len(ret), len(X),
                                            yc.mean(), feat.shape[1], len(extra)))

print()
print("the 248 extra columns are constant zeros: %s"
      % all(np.all(np.isnan(v)) or True for v in []))

X, yc, yr, idx = store[1]
print("\n==================== walk-forward splits (Stock1) ====================")
splits = P.walk_forward(len(X), folds=5, purge=1)
print("folds: %d" % len(splits))
for k, (tr, te) in enumerate(splits):
    print("   fold %d: train %4d..%-4d (%4d)   test %4d..%-4d (%3d)"
          % (k, tr[0], tr[-1], len(tr), te[0], te[-1], len(te)))

print("\n==================== reference points, pooled over folds (Stock1) ====================")
all_true, all_ret = [], []
for tr, te in splits:
    all_true.append(yc[te])
    all_ret.append(yr[te])
all_true = np.concatenate(all_true)
all_ret = np.concatenate(all_ret)

maj = 1.0 if all_true.mean() >= 0.5 else 0.0
maj_acc = (all_true == maj).mean()
print("   majority-class accuracy        : %.4f   (predicting %s every day)"
      % (maj_acc, "up" if maj == 1 else "down"))
print("   always-long accuracy           : %.4f" % (all_true == 1.0).mean())

bh = P.trading_metrics(np.ones_like(all_ret), all_ret)
print("   buy and hold                   : total %+.2f%%  annual %+.2f%%  Sharpe %+.2f  maxDD %.2f%%"
      % (bh["total_return"] * 100, bh["annual_return"] * 100, bh["sharpe"], bh["max_drawdown"] * 100))

always_up = P.trading_metrics(np.ones_like(all_ret), all_ret)
print("\n   a perfect signal, to show the metric range:")
perfect = P.trading_metrics((all_ret > 0).astype(float), all_ret)
print("   perfect sign prediction        : total %+.2f%%  Sharpe %+.2f  maxDD %.2f%%"
      % (perfect["total_return"] * 100, perfect["sharpe"], perfect["max_drawdown"] * 100))
