"""Two attacks on the constraint that actually binds: turnover.

The signal is real but the book is expensive, and the earlier attribution showed why: the return is
earned in the overnight jump, while a daily rotation pays a full round trip every day to chase it.
Both experiments below try to keep the overnight edge while paying for fewer trades.

**Smoothing.** A daily score is noisy; an exponential moving average over consecutive days keeps the
persistent part and suppresses the part that only changes the ranking from one day to the next. It is
the cheapest possible turnover penalty, because it needs no retraining, and it is measured at
halflives of one, five and twenty days. If smoothing at a five-day halflife keeps the information and
halves the turnover, a trained turnover penalty has something to aim at.

**Combination.** The model score and the cheap factor scores are standardised within the day and
averaged with a weight that is scanned from zero (factors only) to one (model only). The earlier
results suggest the two carry different information: the model is mostly a linear combination of the
factor library, but the library's best members have information the model does not. The scan shows
whether a combination dominates every component.

Everything is evaluated with the realistic cost model at weekly and daily rebalancing, on the full
eligible universe and on its most liquid half.

Usage: python xsec_combine.py --panel <panel> --scores out/xsec_main/xsec_scores_all.npz --seq 60 \
    [--out DIR]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "libs"))
sys.path.insert(0, HERE)

import costs as C  # noqa: E402
import factors as FA  # noqa: E402
import xsection as X  # noqa: E402

DEFAULT_PANEL = (r"."
                 r"\data\adj")
MODELS = ["gatedfusion-guided", "gatedfusion", "attention+conv", "lstm", "ssm-only", "timesnet",
          "dlinear", "mlp"]
CHEAP = ["turn1", "logamt1", "range20", "mom60"]


def zscore_by_date(x, di):
    out = np.zeros_like(x, dtype=np.float64)
    for d in np.unique(di):
        m = di == d
        v = x[m]
        sd = v.std()
        out[m] = (v - v.mean()) / (sd if sd > 1e-12 else 1.0)
    return out


def ema_by_stock(score, stock_ids, dates, halflife):
    """Exponential moving average along each stock's own time line, reset across fold gaps."""
    out = np.empty_like(score, dtype=np.float64)
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    gap = np.timedelta64(5, "D")
    for k in np.unique(stock_ids):
        rows = np.where(stock_ids == k)[0]
        rows = rows[np.argsort(dates[rows])]
        ema, prev = 0.0, None
        for i in rows:
            if prev is None or (dates[i] - prev) > gap:
                ema = score[i]
            else:
                ema = alpha * score[i] + (1.0 - alpha) * ema
            out[i] = ema
            prev = dates[i]
    return out


def ic_of(score, y, di, min_names=30):
    vals = []
    for d in np.unique(di):
        m = di == d
        if m.sum() < min_names:
            continue
        ok = np.isfinite(score[m]) & np.isfinite(y[m])
        if ok.sum() < min_names:
            continue
        a, b = score[m][ok], y[m][ok]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        vals.append(np.corrcoef(stats.rankdata(a), stats.rankdata(b))[0, 1])
    v = np.asarray(vals)
    return (float(v.mean()), float(v.mean() / v.std(ddof=1) * np.sqrt(len(v)))) if len(v) > 30 \
        else (float("nan"), float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--seq", type=int, default=60)
    ap.add_argument("--aum", type=float, default=1e8)
    ap.add_argument("--quantile", type=float, default=0.1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    z = np.load(args.scores, allow_pickle=False)
    ref = next(k.split("|")[0] for k in z.files if k.endswith("|y"))
    te_rows = z["te_rows"]
    di = z[f"{ref}|date"]
    stock_ids = z["te_stocks"]

    data_all = X.load_panel(args.panel)
    data = {c: data_all[c] for c in sorted(data_all)}
    ds = X.build_dataset(data, args.seq)
    tg = FA.targets(data, ds)
    per_date = {}
    for code, (dates, feat, ret) in data.items():
        for d, r in zip(dates, ret):
            per_date.setdefault(d, []).append(r)
    mean_by_date = {d: float(np.mean(v)) for d, v in per_date.items()}
    F, fnames = FA.factor_arrays(data, ds, lambda dates: np.array([mean_by_date.get(d, 0.0)
                                                                   for d in dates]))
    adv, sigma = C.liquidity_arrays(data, ds)
    adv, sigma = adv[te_rows], sigma[te_rows]
    y_cc, y_on = tg["cc"][te_rows], tg["on"][te_rows]
    dates = ds.grid[np.asarray(di)]

    def zf(name):
        return zscore_by_date(F[name][te_rows], di)

    cheap = np.mean([zf(c) for c in CHEAP], axis=0)

    variants = {}
    for m in MODELS:
        if f"{m}|score" not in z.files:
            continue
        s = z[f"{m}|score"]
        zs = zscore_by_date(s, di)
        variants[m] = zs
        for hl in (1, 5, 20):
            variants[f"{m} + EMA{hl}"] = zscore_by_date(
                ema_by_stock(zs, stock_ids, dates, hl), di)
    variants["cheap factors only"] = cheap
    for w in (0.25, 0.5, 0.75):
        for m in ("gatedfusion-guided", "gatedfusion"):
            if m in variants:
                variants[f"{w:.2f} {m} + {1 - w:.2f} cheap"] = (
                    w * variants[m] + (1 - w) * cheap)

    rows = []
    for name, s in variants.items():
        ic_cc = ic_of(s, y_cc, di)
        ic_on = ic_of(s, y_on, di)
        for h in (1, 5):
            for liq in (1.0, 0.5):
                kw = dict(quantile=args.quantile, rebalance=h, stock_ids=stock_ids,
                          top_liquidity=liq)
                cm = C.CostModel(aum=args.aum)
                zero = C.CostModel(commission=0, transfer=0, slippage=0, stamp_before=0,
                                   stamp_after=0, borrow_annual=0, impact_coef=0, aum=args.aum)
                r = C.replay(s, y_cc, di, ds.grid, adv=adv, sigma=sigma, cm=cm, **kw)
                r0 = C.replay(s, y_cc, di, ds.grid, adv=adv, sigma=sigma, cm=zero, **kw)
                cap = C.breakeven_aum(r0["gross_ann"], r["fee_ann"], r["borrow_ann"],
                                      r["impact_ann"], args.aum)
                rows.append({"variant": name, "hold_days": h, "liquid": liq,
                             "ic_cc": ic_cc[0], "t_cc": ic_cc[1],
                             "ic_on": ic_on[0], "t_on": ic_on[1],
                             "turnover": r0.get("traded_above_threshold_share", np.nan),
                             "gross": r0["gross_ann"], "fee": r["fee_ann"],
                             "impact": r["impact_ann"], "net": r["net_ann"],
                             "breakeven_aum_M": (cap / 1e6) if cap else 0.0,
                             "p95_participation": r["participation_p95"] * 100})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    for h, liq in [(5, 0.5), (5, 1.0), (1, 0.5)]:
        sub = df[(df.hold_days == h) & (df.liquid == liq)].sort_values("net", ascending=False)
        print(f"\n===== hold {h} day(s), liquidity universe {liq:.0%}, "
              f"AUM {args.aum / 1e6:.0f}M per leg, per cent per year =====")
        print(sub[["variant", "ic_cc", "ic_on", "t_on", "gross", "fee", "impact", "net",
                   "breakeven_aum_M", "p95_participation"]].round(2).to_string(index=False))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        df.to_csv(os.path.join(args.out, "combine.csv"), index=False)
        with open(os.path.join(args.out, "combine.json"), "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "rows": rows}, f, indent=2)
        print(f"\nwritten {os.path.join(args.out, 'combine.csv')}")


if __name__ == "__main__":
    main()
