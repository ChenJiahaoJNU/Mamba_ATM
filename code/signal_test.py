"""Is there any directional signal in this data at all?

Two questions, both answerable without training anything:
  1. do the shipped feature columns carry information about the next day's return;
  2. does the return series itself show autocorrelation, i.e. is a momentum or reversal rule
     any better than a coin.
If the answers are no, then no architecture can honestly produce the accuracies the submitted
manuscript reports on this dataset.
"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
DATA = r".\data"
FEATS = ["当日开盘", "收盘价", "最高价", "最低价", "成交量", "成交额", "换手率", "昨收"]


def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


print("==================== 1. do the shipped features predict the next-day return? ====================")
print("%-8s %-12s %10s" % ("stock", "feature", "corr(with next-day return)"))
for i in range(1, 5):
    df = pd.read_csv(rf"{DATA}\Stock{i}.csv", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    r_next = df["涨跌幅"].shift(-1).to_numpy()[:-1]
    line = []
    for c in FEATS:
        v = df[c].to_numpy()[:-1]
        line.append((c, spearman(v, r_next)))
    best = max(line, key=lambda t: abs(t[1]))
    print("%-8s %-12s %10.4f   (largest magnitude of the eight)" % ("Stock%d" % i, best[0], best[1]))

print("\n==================== 2. autocorrelation of the daily return ====================")
print("%-8s %8s %8s %8s %8s %8s" % ("stock", "lag1", "lag2", "lag3", "lag5", "lag10"))
for i in range(1, 5):
    df = pd.read_csv(rf"{DATA}\Stock{i}.csv", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    r = df["涨跌幅"].to_numpy()
    r = r - r.mean()
    ac = [float(np.corrcoef(r[:-k], r[k:])[0, 1]) for k in (1, 2, 3, 5, 10)]
    print("%-8s %+8.4f %+8.4f %+8.4f %+8.4f %+8.4f" % (("Stock%d" % i, *ac)))

print("\n==================== 3. how often does the sign of the return repeat? ====================")
print("%-8s %12s %12s" % ("stock", "same as prev", "expect 0.5"))
for i in range(1, 5):
    df = pd.read_csv(rf"{DATA}\Stock{i}.csv", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    s = (df["涨跌幅"].to_numpy() > 0).astype(int)
    print("%-8s %12.4f" % ("Stock%d" % i, float((s[1:] == s[:-1]).mean())))

print("\n==================== 4. scale check: is the z-scoring informative? ====================")
df = pd.read_csv(rf"{DATA}\Stock1.csv", encoding="utf-8-sig")
df.columns = [c.strip() for c in df.columns]
print("收盘价 and 昨收 are the same series shifted by one day, per stock:")
print("   corr(close_t, prevclose_{t+1}) = %.6f" % float(np.corrcoef(
    df["收盘价"].to_numpy()[:-1], df["昨收"].to_numpy()[1:])[0, 1]))
print("   but each column was z-scored with its own mean and sd:")
print("      close   mean %.4f sd %.4f" % (df["收盘价"].mean(), df["收盘价"].std()))
print("      prevclose mean %.4f sd %.4f" % (df["昨收"].mean(), df["昨收"].std()))
