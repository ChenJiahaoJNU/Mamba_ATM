"""An extended characteristic library, and the overnight/intraday decomposition of the label.

Two things are built here.

**More characteristics.** The nine factors used so far are not a factor library, they are a sample.
Everything below is computed from the eight columns the panel already carries, using only days up to
and including the last day of the window, so nothing looks ahead:

    reversals        1, 5, 20 days, sign-flipped so that a higher score means a higher expected return
    momentum         cumulative return over the 60-day window
    volatility      standard deviation over 20 and 60 days, downside-only deviation, daily range
    shape           skewness, kurtosis, close-location value, upper and lower shadows
    liquidity       turnover level and variability, Amihud illiquidity, log turnover value,
                    volume spike against its own 20-day mean
    intraday/overnight split of the stock's own returns, and their volatilities
    market          correlation of the stock's return with the equal-weight panel return

**Two more labels.** Within a day, the return from close to close decomposes into an overnight leg and
an intraday leg:

    overnight_{t+1} = open_{t+1} / close_t  - 1      ordered at the close of t, filled in the auction
    intraday_{t+1}  = close_{t+1} / open_{t+1} - 1   bought at the open, sold at the close
    close-to-close  = the label used everywhere else

The overnight leg is tradable under T+1 (you buy at the next open and can sell the day after). The
intraday leg is not, for a long-only account: shares bought at the open of t+1 cannot be sold at the
close of t+1. It is reported because the literature finds the two legs carry different information,
not because it can be harvested as stated.
"""
import numpy as np
import pandas as pd

from xsection import FEATURE_COLS

IDX = {c: i for i, c in enumerate(FEATURE_COLS)}


def _stock_series(feat, ret):
    close = feat[:, IDX["收盘价"]]
    open_ = feat[:, IDX["当日开盘"]]
    high = feat[:, IDX["最高价"]]
    low = feat[:, IDX["最低价"]]
    vol = feat[:, IDX["成交量"]]
    amount = np.maximum(feat[:, IDX["成交额"]], 1.0)
    turn = feat[:, IDX["换手率"]]
    prev = feat[:, IDX["昨收"]]
    rng = np.maximum(high - low, 1e-12)
    return {
        "r": ret,
        "close": close, "open": open_, "high": high, "low": low,
        "vol": vol, "amount": amount, "turn": turn, "prev": prev, "range_": rng,
        "gap": open_ / np.maximum(prev, 1e-12) - 1.0,
        "idr": close / np.maximum(open_, 1e-12) - 1.0,
        "clv": (close - low) / rng,
        "upper": (high - np.maximum(open_, close)) / rng,
        "lower": (np.minimum(open_, close) - low) / rng,
    }


def _roll(x, w, fn):
    return pd.Series(x).rolling(w, min_periods=max(3, w // 2)).apply(fn, raw=True).to_numpy()


def factor_arrays(data, ds, market_ret_by_date):
    """Every extended factor, as an array aligned with the rows of `ds`."""
    L = ds.X.shape[1]
    names = ["rev1", "rev5", "rev20", "mom60", "vol20", "vol60", "downvol60", "maxret20",
             "skew60", "kurt60", "turn1", "turn20", "turnstd20", "amihud20", "logamt1", "logamt20",
             "volspike", "clv1", "clv5", "uppershadow5", "lowershadow5", "gap1", "gap5", "idr1",
             "idr5", "idrvol60", "onvol60", "range20", "mktcorr60"]
    out = {n: np.zeros(len(ds.y), dtype=np.float64) for n in names}
    for k, code in enumerate(ds.codes):
        dates, feat, ret = data[code]
        s = _stock_series(feat, ret)
        r = s["r"]
        close = s["close"]
        cum = lambda w: pd.Series(1.0 + r).rolling(w, min_periods=max(3, w // 2)).apply(
            np.prod, raw=True).to_numpy() - 1.0
        vals = {
            "rev1": -r,
            "rev5": -cum(5),
            "rev20": -cum(20),
            "mom60": cum(L),
            "vol20": -_roll(r, 20, np.std),
            "vol60": -_roll(r, L, np.std),
            "downvol60": -_roll(r, L, lambda v: np.std(v[v < 0]) if (v < 0).sum() > 2 else 0.0),
            "maxret20": -_roll(r, 20, np.max),
            "skew60": -_roll(r, L, lambda v: pd.Series(v).skew() if np.std(v) > 0 else 0.0),
            "kurt60": -_roll(r, L, lambda v: pd.Series(v).kurt() if np.std(v) > 0 else 0.0),
            "turn1": -s["turn"],
            "turn20": -_roll(s["turn"], 20, np.mean),
            "turnstd20": -_roll(s["turn"], 20, np.std),
            "amihud20": _roll(np.abs(r) / s["amount"] * 1e8, 20, np.mean),
            "logamt1": -np.log(s["amount"]),
            "logamt20": -np.log(_roll(s["amount"], 20, np.mean)),
            "volspike": -s["vol"] / np.maximum(_roll(s["vol"], 20, np.mean), 1.0),
            "clv1": s["clv"],
            "clv5": _roll(s["clv"], 5, np.mean),
            "uppershadow5": -_roll(s["upper"], 5, np.mean),
            "lowershadow5": _roll(s["lower"], 5, np.mean),
            "gap1": -s["gap"],
            "gap5": -_roll(s["gap"], 5, np.mean),
            "idr1": -s["idr"],
            "idr5": -_roll(s["idr"], 5, np.mean),
            "idrvol60": -_roll(s["idr"], L, np.std),
            "onvol60": -_roll(s["gap"], L, np.std),
            "range20": -_roll(s["range_"] / np.maximum(close, 1e-12), 20, np.mean),
        }
        # market exposure: rolling beta of the stock on the equal-weight panel return, negated so
        # that a lower beta scores higher, which is the direction of the low-beta anomaly
        mr = np.asarray(market_ret_by_date(dates), dtype=np.float64)
        cov = pd.Series(r).rolling(L, min_periods=max(3, L // 2)).cov(pd.Series(mr)).to_numpy()
        var = pd.Series(mr).rolling(L, min_periods=max(3, L // 2)).var().to_numpy()
        vals["mktcorr60"] = -np.divide(cov, var, out=np.zeros_like(cov), where=var > 1e-12)
        sel = ds.stock_idx == k
        idx = ds.end_idx[sel]
        for n in names:
            out[n][sel] = vals[n][idx]
    return out, names


def targets(data, ds):
    """The three labels for the row set of `ds`: close-to-close, overnight, intraday."""
    n = len(ds.y)
    cc = np.full(n, np.nan)
    on = np.full(n, np.nan)
    idr = np.full(n, np.nan)
    for k, code in enumerate(ds.codes):
        dates, feat, ret = data[code]
        s = _stock_series(feat, ret)
        sel = ds.stock_idx == k
        t = ds.end_idx[sel]
        nxt = t + 1
        ok = nxt < len(ret)
        cc[sel] = np.where(ok, ret[np.minimum(nxt, len(ret) - 1)], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            on_v = s["open"][np.minimum(nxt, len(ret) - 1)] / np.maximum(s["close"][t], 1e-12) - 1.0
            id_v = s["close"][np.minimum(nxt, len(ret) - 1)] / np.maximum(
                s["open"][np.minimum(nxt, len(ret) - 1)], 1e-12) - 1.0
        on[sel] = np.where(ok, on_v, np.nan)
        idr[sel] = np.where(ok, id_v, np.nan)
    return {"cc": cc, "on": on, "id": idr}
