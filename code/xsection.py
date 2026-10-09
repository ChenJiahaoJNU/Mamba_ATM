"""Cross-sectional evaluation: predict relative, not absolute, next-day performance.

Why this exists
---------------
The time-series protocol in `pipeline.py` asks each model to predict the sign of one stock's
next-day return. On the rebuilt panel that target is dominated by the drift of the sample: the
majority-class rate of the test blocks is 0.528, every model lands between 0.507 and 0.521, and in
eight of the nine test years the majority-class rate is above every model's accuracy. The reason is
that an absolute-direction target keeps the market factor in the label, and the label of a rising
period is easy while the label of a falling period is not.

The cross-sectional target removes the market factor instead of predicting it: within one trading
day, a model is asked to rank the stocks against each other. The label is the next-day return
demeaned and scaled across the stocks of that day, the metric is the rank information coefficient
(Spearman between score and realised return, computed per day), and the economic evaluation is a
decile long-short portfolio with turnover charged at the same cost rates as before.

Two consequences worth stating before the numbers are seen:

* Because evaluation is rank-based, the choice of cross-sectional transform of the label affects the
  training loss only, never the metric. A model cannot gain by the transform.
* Nothing about the day is known at prediction time except the model's features: the transform uses
  the realised cross-section of the day being predicted, which is the definition of a relative
  target, not an input.

Reused from the time-series work: the same panel, the same eight features, the same window
construction, the same expanding-window folds with a purge, the same train-only standardisation, and
the same model zoo.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

import pipeline as P

DATE_COL = P.DATE_COL
RET_COL = P.RET_COL
FEATURE_COLS = P.FEATURE_COLS
MASKS = ["date", "sample"]


# --------------------------------------------------------------------------- loading
def load_panel(panel_dir, codes=None, min_rows=1200):
    """Per stock: dates, features, returns, with the unlabelled first row dropped."""
    import glob
    import os
    files = sorted(glob.glob(os.path.join(panel_dir, "*.csv")))
    if codes:
        want = {c.replace(".csv", "") for c in codes}
        files = [f for f in files if os.path.basename(f).replace(".csv", "") in want]
    out = {}
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        ret = pd.to_numeric(df[RET_COL], errors="coerce")
        vol = pd.to_numeric(df["成交量"], errors="coerce")
        keep = ret.notna() & vol.notna() & (vol > 0)
        df = df[keep].reset_index(drop=True)
        if len(df) < min_rows:
            continue
        code = os.path.basename(f).replace(".csv", "")
        dates = pd.to_datetime(df[DATE_COL]).to_numpy().astype("datetime64[D]")
        feat = df[FEATURE_COLS].to_numpy(dtype=np.float64)
        r = pd.to_numeric(df[RET_COL], errors="coerce").to_numpy(dtype=np.float64)
        out[code] = (dates, feat, r)
    return out


# --------------------------------------------------------------------------- windows
@dataclass
class Dataset:
    X: np.ndarray          # (N, T, F)
    y: np.ndarray          # (N,)  next-day return, in natural units
    date_idx: np.ndarray   # (N,)  index of the target date in `grid`
    stock_idx: np.ndarray  # (N,)  index into `codes`
    end_idx: np.ndarray    # (N,)  index of the last day of the window inside its own stock series
    grid: np.ndarray       # sorted unique target dates
    codes: list


def build_dataset(data, seq, min_date=None, max_date=None):
    """Every stock's trailing windows that have a next-day return, aligned on the target date."""
    all_dates = np.unique(np.concatenate([d for d, _, _ in data.values()]))
    grid = all_dates
    pos = {d: i for i, d in enumerate(grid)}
    X, y, di, si, ei, codes = [], [], [], [], [], []
    for k, (code, (dates, feat, ret)) in enumerate(sorted(data.items())):
        codes.append(code)
        for t in range(seq - 1, len(ret) - 1):
            td = dates[t + 1]
            j = pos.get(td)
            if j is None:
                continue
            if min_date is not None and td < min_date:
                continue
            if max_date is not None and td > max_date:
                continue
            X.append(feat[t - seq + 1:t + 1])
            y.append(ret[t + 1])
            di.append(j)
            si.append(k)
            ei.append(t)
    return Dataset(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float64),
                   np.asarray(di, dtype=np.int64), np.asarray(si, dtype=np.int64),
                   np.asarray(ei, dtype=np.int64), grid, codes)


def xsec_target(y, date_idx, winsor=3.0):
    """Demean and scale the realised return within each date, then clip.

    Evaluation is rank based, so this transform changes the training loss and not the metric.
    """
    z = np.zeros_like(y)
    for d in np.unique(date_idx):
        m = date_idx == d
        v = y[m]
        if len(v) < 3:
            continue
        sd = v.std()
        z[m] = (v - v.mean()) / (sd if sd > 1e-12 else 1.0)
    return np.clip(z, -winsor, winsor)


def date_splits(grid, first_test_date, folds=5, purge_days=1):
    """Expanding folds over the date grid; a purge of `purge_days` dates precedes each test block."""
    idx = np.searchsorted(grid, np.datetime64(first_test_date))
    usable = len(grid) - idx
    block = max(1, usable // folds)
    splits = []
    for k in range(folds):
        te0 = idx + k * block
        te1 = len(grid) if k == folds - 1 else te0 + block
        tr_end = te0 - purge_days
        if tr_end <= 1 or te1 <= te0:
            continue
        splits.append((grid[:tr_end], set(grid[te0:te1])))
    return splits


# --------------------------------------------------------------------------- evaluation
def daily_ic(scores, y, date_idx, min_names=5):
    """Spearman information coefficient per date, plus the number of names in that date."""
    from scipy import stats
    rows = []
    for d in np.unique(date_idx):
        m = date_idx == d
        if m.sum() < min_names:
            continue
        s, r = scores[m], y[m]
        if np.std(s) < 1e-12 or np.std(r) < 1e-12:
            rows.append((d, 0.0, int(m.sum())))
            continue
        rows.append((d, float(stats.spearmanr(s, r).statistic), int(m.sum())))
    return pd.DataFrame(rows, columns=["date", "ic", "n"])


def ic_summary(ic):
    if ic.empty:
        return {}
    v = ic.ic.to_numpy()
    n = len(v)
    mean, sd = float(v.mean()), float(v.std(ddof=1)) if n > 1 else 0.0
    ir = mean / sd if sd > 0 else 0.0
    return {"ic_mean": mean, "ic_sd": sd, "ic_ir": ir, "ic_t": ir * np.sqrt(n),
            "ic_positive_share": float((v > 0).mean()), "days": n,
            "names_per_day": float(ic.n.mean())}


def tradable_mask(data, ds, limit=0.098):
    """Rows whose entry would not have been possible, using only information available at the time.

    A stock that closed at the price limit on the day the signal is formed cannot reliably be bought
    at the next open, and in China a limit-up name frequently opens limit-up again. The rule here
    uses the last day of the window, which is known when the signal is produced, and never the day
    being predicted.
    """
    out = np.ones(len(ds.y), dtype=bool)
    for k, code in enumerate(ds.codes):
        _, _, ret = data[code]
        sel = ds.stock_idx == k
        out[sel] = np.abs(ret[ds.end_idx[sel]]) < limit
    return out


def decile_portfolio(scores, y, date_idx, cost=0.0010, quantile=0.1, min_names=10, tradable=None,
                     rebalance=1, stock_ids=None):
    """Equal-weight long-short portfolio on the extreme quantiles, charged per position change.

    Long the top quantile and short the bottom quantile. `rebalance` is the holding period in trading
    days: the legs are chosen from the signal on the rebalance date and held unchanged in between, so
    the cost is charged only when the book changes.

    `stock_ids` must identify the asset of each row. Holdings have to be tracked by asset: a row index
    identifies an observation, not a name, so a position carried from one day to the next changes its
    row index and would otherwise be counted as a full replacement. Getting this wrong makes turnover
    and therefore every net return meaningless, in either direction.
    """
    dates = np.unique(date_idx)
    top_ret, bot_ret, mkt_ret, t_cost_l, b_cost_l = [], [], [], [], []
    prev_t, prev_b = {}, {}
    held_t, held_b = {}, {}
    flat_days = 0
    eligible_min = None
    for step, d in enumerate(dates):
        m = date_idx == d
        if tradable is not None:
            m = m & tradable
        idxs = np.where(m)[0]
        if not idxs.size:
            # no tradable name at all on this date: the book is flat and nothing is charged
            top_ret.append(0.0)
            bot_ret.append(0.0)
            mkt_ret.append(np.nan)
            t_cost_l.append(0.0)
            b_cost_l.append(0.0)
            flat_days += 1
            continue
        eligible_min = len(idxs) if eligible_min is None else min(eligible_min, len(idxs))
        need = max(min_names, int(np.ceil(1.0 / quantile)) * 2)
        if (step % max(1, rebalance) == 0 or not held_t) and len(idxs) >= need:
            s = scores[idxs]
            k = max(1, int(round(quantile * len(s))))
            order = np.argsort(-s)
            key = (lambda i: int(stock_ids[i])) if stock_ids is not None else (lambda i: int(i))
            held_t = {key(i): 1.0 / k for i in idxs[order[:k]]}
            held_b = {key(i): 1.0 / k for i in idxs[order[-k:]]}
            t_cost = sum(abs(held_t.get(i, 0.0) - prev_t.get(i, 0.0))
                         for i in set(held_t) | set(prev_t))
            b_cost = sum(abs(held_b.get(i, 0.0) - prev_b.get(i, 0.0))
                         for i in set(held_b) | set(prev_b))
            prev_t, prev_b = held_t, held_b
        elif not held_t:
            continue
        else:
            t_cost = b_cost = 0.0
        today = {}
        if stock_ids is not None:
            today = {int(stock_ids[i]): y[i] for i in idxs}
        else:
            today = {int(i): y[i] for i in idxs}
        # a name that is not tradable today cannot be held today; when none of the book is tradable
        # the position is flat for the day, which is a real outcome on market-wide limit days
        avail_t = [today[i] for i in held_t if i in today]
        avail_b = [today[i] for i in held_b if i in today]
        if not avail_t or not avail_b:
            flat_days += 1
        top_ret.append(float(np.mean(avail_t)) if avail_t else 0.0)
        bot_ret.append(float(np.mean(avail_b)) if avail_b else 0.0)
        mkt_ret.append(y[idxs].mean())
        t_cost_l.append(t_cost)
        b_cost_l.append(b_cost)
    if not top_ret:
        return {"ls_ann": float("nan"), "ls_sharpe": float("nan"), "ls_maxdd": float("nan"),
                "ls_turnover": float("nan"), "top_ann": float("nan"),
                "market_ann": float("nan"), "market_sharpe": float("nan"),
                "top_minus_market_ann": float("nan"), "days": 0}
    after = np.argsort(dates)
    dates = dates[after]
    top_ret = np.asarray(top_ret)[after]
    bot_ret = np.asarray(bot_ret)[after]
    mkt_ret = np.asarray(mkt_ret)[after]
    t_cost = np.asarray(t_cost_l)[after]
    b_cost = np.asarray(b_cost_l)[after]
    ls = top_ret - bot_ret - (t_cost + b_cost) * cost
    turn = 0.5 * (t_cost + b_cost)
    out = {"ls_ann": float(np.nanmean(ls) * 252), "ls_sharpe": _sharpe(ls),
           "ls_maxdd": _maxdd(ls), "ls_turnover": float(turn.mean()),
           "top_ann": float((top_ret - t_cost * cost).mean() * 252),
           "market_ann": float(np.nanmean(mkt_ret) * 252), "market_sharpe": _sharpe(mkt_ret),
           "flat_days": int(flat_days), "min_eligible": int(eligible_min or 0),
           "days": int(len(ls))}
    out["top_minus_market_ann"] = out["top_ann"] - out["market_ann"]
    return out


def _sharpe(x):
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return float(x.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0


def _maxdd(x):
    eq = np.cumprod(1.0 + x)
    peak = np.maximum.accumulate(eq)
    return float((1.0 - eq / peak).max()) if len(eq) else 0.0


# --------------------------------------------------------------------------- characteristic factors
def characteristic_vector(data, ds, name):
    """Score every row of `ds` from a published characteristic, so the references sit on the same
    windows as the learned models and can be compared row by row.

    short_reversal : negative of the return on the last day of the window
    momentum       : cumulative return over the window
    low_volatility : negative of the return standard deviation over the window
    low_turnover   : negative of the turnover rate on the last day of the window
    """
    out = np.zeros(len(ds.y), dtype=np.float64)
    turn_col = FEATURE_COLS.index("换手率")
    amount_col = FEATURE_COLS.index("成交额")
    close_col = FEATURE_COLS.index("收盘价")
    L = ds.X.shape[1]
    for k, code in enumerate(ds.codes):
        dates, feat, ret = data[code]
        sel = ds.stock_idx == k
        idx = ds.end_idx[sel]
        if name == "short_reversal":                      # 1-day reversal
            v = -ret[idx]
        elif name == "reversal_5":                        # 5-day reversal
            v = np.array([-float(np.prod(1.0 + ret[t - 4:t + 1]) - 1.0) for t in idx])
        elif name == "reversal_20":                       # 20-day reversal
            v = np.array([-float(np.prod(1.0 + ret[t - 19:t + 1]) - 1.0) for t in idx])
        elif name == "momentum":                          # cumulative return over the window
            v = np.array([float(np.prod(1.0 + ret[t - (L - 1):t + 1]) - 1.0) for t in idx])
        elif name == "low_volatility":                    # 60-day return standard deviation, negated
            v = np.array([-float(np.std(ret[t - (L - 1):t + 1])) for t in idx])
        elif name == "volatility_20":
            v = np.array([-float(np.std(ret[t - 19:t + 1])) for t in idx])
        elif name == "low_turnover":                      # turnover rate on the last day, negated
            v = -feat[idx, turn_col]
        elif name == "small_amount":                      # log turnover value, negated (size proxy)
            v = -np.log(np.maximum(feat[idx, amount_col], 1.0))
        elif name == "low_price":                         # log close, negated
            v = -np.log(np.maximum(feat[idx, close_col], 1e-6))
        else:
            raise ValueError(f"unknown characteristic: {name}")
        out[sel] = v
    return out
