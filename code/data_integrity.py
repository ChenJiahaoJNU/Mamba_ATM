"""Data integrity: is the return column consistent with the price columns it ships with?

If 昨收 is the previous close, then 涨跌幅 should equal close/prevclose - 1. If it does not, the
file mixes two price sources and the feature matrix cannot be interpreted as a price history.
The Stock3 autocorrelation of 0.25 is also checked, since that is implausibly high for a daily
equity return and usually indicates repeated or interpolated values.
"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
DATA = r".\data"

print("==================== is 涨跌幅 = close/prevclose - 1? ====================")
print("%-8s %14s %14s %14s" % ("stock", "corr(implied,r)", "mean|diff|", "max|diff|"))
for i in range(1, 5):
    df = pd.read_csv(rf"{DATA}\Stock{i}.csv", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    r = df["涨跌幅"].to_numpy()
    c = df["收盘价"].to_numpy()
    pc = df["昨收"].to_numpy()
    implied = np.diff(c) / c[:-1]          # close-to-close return implied by the price columns
    d = np.abs(implied - r[1:])
    print("%-8s %14.4f %14.4f %14.4f" % ("Stock%d" % i, np.corrcoef(implied, r[1:])[0, 1],
                                         d.mean(), d.max()))

print("\n==================== is 涨跌幅 = close/prevclose - 1 within the same row? ====================")
for i in range(1, 5):
    df = pd.read_csv(rf"{DATA}\Stock{i}.csv", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    r = df["涨跌幅"].to_numpy()
    c = df["收盘价"].to_numpy()
    pc = df["昨收"].to_numpy()
    implied = c / pc - 1.0
    print("   Stock%d: corr %.4f   mean|diff| %.4f" % (i, np.corrcoef(implied, r)[0, 1],
                                                       np.abs(implied - r).mean()))

print("\n==================== staleness: how often is the price unchanged? ====================")
for i in range(1, 5):
    df = pd.read_csv(rf"{DATA}\Stock{i}.csv", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    c = df["收盘价"].to_numpy()
    r = df["涨跌幅"].to_numpy()
    print("   Stock%d: identical consecutive closes %.4f   identical consecutive returns %.4f"
          % (i, float((np.diff(c) == 0).mean()), float((np.diff(r) == 0).mean())))

print("\n==================== Stock3's strong lag-1 autocorrelation, in context ====================")
df = pd.read_csv(rf"{DATA}\Stock3.csv", encoding="utf-8-sig")
df.columns = [c.strip() for c in df.columns]
r = df["涨跌幅"].to_numpy()
print("   first 12 returns:", np.round(r[:12], 5))
print("   last 6 returns  :", np.round(r[-6:], 5))
print("   |r| mean %.5f   r sd %.5f   share of |r|>0.05: %.4f"
      % (np.abs(r).mean(), r.std(), float((np.abs(r) > 0.05).mean())))
for i in range(1, 5):
    df = pd.read_csv(rf"{DATA}\Stock{i}.csv", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    rr = df["涨跌幅"].to_numpy()
    print("   Stock%d: r sd %.5f, |r| mean %.5f, max |r| %.4f, share |r| in (0.099,0.11] %.4f"
          % (i, rr.std(), np.abs(rr).mean(), np.abs(rr).max(),
             float(((np.abs(rr) > 0.099) & (np.abs(rr) <= 0.11)).mean())))
