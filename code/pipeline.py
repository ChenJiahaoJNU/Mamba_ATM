"""Data pipeline and evaluation protocol for the MambaATM rework.

Dataset, as it actually is:
  * four A-share stocks, 500 trading days each, 2023-01-10 to 2025-02-07;
  * eight informative exchange fields, already z-scored across the sample per stock
    (open, close, high, low, volume, turnover value, turnover rate, previous close);
  * one return column (the daily close-to-close return, in natural units);
  * 248 further columns named extra_feature_* that are identically zero in every row and are
    therefore dropped here rather than fed to the models as constant inputs.

Protocol: walk-forward with an expanding training window and a purge of `purge` days between
train and test, which is the evaluation the reviewer asked for in place of a single chronological
split. Everything is reported per fold as well as pooled.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

DATE_COL = "日期"
RET_COL = "涨跌幅"
# the eight informative fields, in file order
FEATURE_COLS = ["当日开盘", "收盘价", "最高价", "最低价", "成交量", "成交额", "换手率", "昨收"]


def load_stock(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    dates = pd.to_datetime(df[DATE_COL]).to_numpy()
    feat = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    ret = df[RET_COL].to_numpy(dtype=np.float64)
    extra = [c for c in df.columns if c.startswith("extra_feature_")]
    return dates, feat, ret, extra


def make_windows(feat, ret, seq_len, label_mode="next_day"):
    """X (N, T, F) of trailing windows.

    label_mode "next_day"  : label is the sign of the return AFTER the window, which is the
                             leak-free choice; the trading return is the same day's return.
    label_mode "same_day"  : label is the sign of the return ON the last day of the window,
                             which is the definition in the submitted data loader. That return
                             is a function of the close and previous close carried in the last
                             row of the window, so the label is partly present in the input.
    """
    n = len(ret)
    xs, yc, yr, idx = [], [], [], []
    for t in range(seq_len - 1, n - 1):
        xs.append(feat[t - seq_len + 1:t + 1])
        if label_mode == "same_day":
            yc.append(1.0 if ret[t] > 0 else 0.0)
        else:
            yc.append(1.0 if ret[t + 1] > 0 else 0.0)
        yr.append(ret[t + 1])
        idx.append(t + 1)
    return (np.asarray(xs), np.asarray(yc), np.asarray(yr), np.asarray(idx))


def walk_forward(n, folds=5, purge=1, min_train_frac=0.4):
    """Expanding-window walk-forward splits over the window index, with a purge gap."""
    first_test = int(n * min_train_frac)
    usable = n - first_test
    block = max(1, usable // folds)
    splits = []
    for k in range(folds):
        te0 = first_test + k * block
        te1 = n if k == folds - 1 else te0 + block
        tr1 = te0 - purge
        if tr1 <= 10 or te1 <= te0:
            continue
        splits.append((np.arange(0, tr1), np.arange(te0, te1)))
    return splits


def standardise(train_x, test_x):
    mu = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    sd = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (train_x - mu) / sd, (test_x - mu) / sd


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    n: int


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return Metrics(acc, prec, rec, f1, len(y_true))


def trading_metrics(signals, returns, cost=0.0003 * 2, periods_per_year=252):
    """Long when the signal is +1, flat when it is 0 or -1 (long-only, no shorting).

    Reports annualised return, annualised Sharpe and maximum drawdown, all net of `cost` charged
    whenever the position changes.
    """
    signals = np.asarray(signals).astype(float)
    returns = np.asarray(returns).astype(float)
    pos = (signals > 0).astype(float)
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    gross = pos * returns
    net = gross - turnover * cost
    equity = np.cumprod(1.0 + net)
    total = equity[-1] - 1.0
    years = len(net) / periods_per_year
    ann = (equity[-1]) ** (1.0 / max(years, 1e-9)) - 1.0
    sd = net.std(ddof=1) if len(net) > 1 else 0.0
    sharpe = (net.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    mdd = float((1.0 - equity / peak).max()) if len(equity) else 0.0
    return {"total_return": float(total), "annual_return": float(ann),
            "sharpe": float(sharpe), "max_drawdown": mdd, "turnover": float(turnover.mean())}


def to_tensor(x, device):
    return torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
