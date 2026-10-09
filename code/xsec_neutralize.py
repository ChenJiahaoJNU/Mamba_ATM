"""Is a model's cross-sectional information anything more than the nine characteristics?

For every test day the script regresses the model's score on the nine characteristic factors, both
standardised within the day, and keeps the residual. The residual score is then correlated with the
realised return in the same way as before, so the comparison is:

    raw IC          what the model does on its own
    neutralised IC  what it does after its linear exposure to the nine factors is removed
    factor IC       what the nine factors alone explain, that is the IC of the fitted part
    R^2             the share of the model's cross-sectional variance that is linear in the factors

If the neutralised IC falls to zero, the model is re-deriving characteristics that are already
published, and the honest claim is a replication rather than a contribution. If it survives, the
model carries information that the factor set does not, and that is the number worth reporting.

Each characteristic is neutralised against the other eight as well, so the factors are held to the
same standard as the models.

Usage: python xsec_neutralize.py --scores out/xsec_main/xsec_scores_all.npz --panel <panel> \
    --seq 60 [--out out/xsec_main]
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

import xsection as X  # noqa: E402

DEFAULT_PANEL = (r"D:\研究生学习资料\暨南大学-博士学习资料\论文\重投\MambaATM"
                 r"\代码\data_panel_adj")
FACTORS = ["short_reversal", "reversal_5", "reversal_20", "momentum", "low_volatility",
           "volatility_20", "low_turnover", "small_amount", "low_price"]


def rank_within(values):
    """Ranks within a 1-D array, ties averaged, returned as float."""
    return stats.rankdata(values).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--seq", type=int, default=60)
    ap.add_argument("--stocks", type=int, default=0)
    ap.add_argument("--min-names", type=int, default=30)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    z = np.load(args.scores, allow_pickle=False)
    models = sorted({k.split("|")[0] for k in z.files if "|" in k and not k.endswith(("|y", "|date"))})
    ref = next(k.split("|")[0] for k in z.files if k.endswith("|y"))
    y = z[f"{ref}|y"]
    di = z[f"{ref}|date"]

    missing = [f for f in FACTORS if f"{f}|score" not in z.files]
    if missing:
        raise SystemExit(f"score file lacks these factors: {missing}")

    names = models + [f for f in FACTORS if f not in models]
    S = np.column_stack([z[f"{n}|score"] for n in names])
    F = np.column_stack([z[f"{f}|score"] for f in FACTORS])
    print(f"{len(names)} scores (models + factors), {len(y):,} rows, {len(np.unique(di)):,} dates")

    dates = np.unique(di)
    order = np.argsort(di, kind="stable")
    starts = np.searchsorted(di[order], dates)
    ends = np.append(starts[1:], len(order))

    n_score = len(names)
    ic_raw = np.full((len(dates), n_score), np.nan)
    ic_neu = np.full((len(dates), n_score), np.nan)
    ic_fit = np.full((len(dates), n_score), np.nan)
    r2 = np.full((len(dates), n_score), np.nan)
    load = np.zeros((n_score, len(FACTORS)))
    neu = [np.full((len(y), n_score), np.nan)]
    used = 0

    for k, (s, e) in enumerate(zip(starts, ends)):
        rows = order[s:e]
        if len(rows) < args.min_names:
            continue
        yd = y[rows]
        if np.std(yd) < 1e-12:
            continue
        Yd = S[rows]
        Fd = F[rows]
        sdF = Fd.std(axis=0)
        keep = sdF > 1e-12
        keep_cols = [c for c in range(len(keep)) if keep[c]]
        Fz = (Fd[:, keep] - Fd[:, keep].mean(axis=0)) / sdF[keep]
        sdY = Yd.std(axis=0)
        good = sdY > 1e-12
        Yz = np.zeros_like(Yd)
        Yz[:, good] = (Yd[:, good] - Yd[:, good].mean(axis=0)) / sdY[good]
        Xd = np.column_stack([np.ones(len(rows)), Fz])
        beta, *_ = np.linalg.lstsq(Xd, Yz, rcond=None)
        fit_all = Xd @ beta
        beta_loo = np.zeros_like(beta)
        for jj in range(n_score):
            if names[jj] not in FACTORS:
                beta_loo[:, jj] = beta[:, jj]
                continue
            # leave the factor out of its own regression, otherwise its residual is zero by
            # construction and the row says nothing about whether the factor is redundant
            drop = FACTORS.index(names[jj])
            cols = [c for c in range(len(keep_cols)) if keep_cols[c] != drop]
            X2 = np.column_stack([np.ones(len(rows)), Fz[:, cols]])
            b2, *_ = np.linalg.lstsq(X2, Yz[:, jj:jj + 1], rcond=None)
            beta_loo[0, jj] = b2[0, 0]
            for c, col in enumerate(cols):
                beta_loo[1 + col, jj] = b2[1 + c, 0]
        fit = np.column_stack([
            (Xd @ beta_loo[:, jj]) if names[jj] in FACTORS else fit_all[:, jj]
            for jj in range(n_score)])
        res = Yz - fit
        neu[0][rows, :] = res
        load[:, keep_cols] += beta[1:].T
        used += 1

        ry = rank_within(yd)
        for j in range(n_score):
            if not good[j]:
                continue
            a = rank_within(Yd[:, j])
            b = rank_within(res[:, j])
            c = rank_within(fit[:, j])
            ic_raw[k, j] = np.corrcoef(a, ry)[0, 1]
            ic_neu[k, j] = np.corrcoef(b, ry)[0, 1]
            ic_fit[k, j] = np.corrcoef(c, ry)[0, 1]
            ss_tot = np.sum((Yz[:, j] - 0) ** 2)
            r2[k, j] = 1.0 - np.sum(res[:, j] ** 2) / ss_tot if ss_tot > 0 else np.nan
    load /= max(1, used)

    def summ(series):
        v = series[np.isfinite(series)]
        if len(v) < 10:
            return np.nan, np.nan, np.nan
        m, sd = v.mean(), v.std(ddof=1)
        return m, (m / sd * np.sqrt(len(v)) if sd > 0 else np.nan), len(v)

    rows = []
    for j, n in enumerate(names):
        raw_m, raw_t, nd = summ(ic_raw[:, j])
        neu_m, neu_t, _ = summ(ic_neu[:, j])
        fit_m, fit_t, _ = summ(ic_fit[:, j])
        diff = ic_raw[:, j] - ic_neu[:, j]
        d = diff[np.isfinite(diff)]
        t_diff = d.mean() / d.std(ddof=1) * np.sqrt(len(d)) if len(d) > 10 and d.std(ddof=1) > 0 else np.nan
        rows.append({"score": n, "raw_ic": raw_m, "raw_t": raw_t,
                     "neutralised_ic": neu_m, "neutralised_t": neu_t,
                     "factor_ic": fit_m, "factor_ic_t": fit_t,
                     "raw_minus_neutralised": d.mean(), "t_of_difference": t_diff,
                     "mean_r2_on_factors": np.nanmean(r2[:, j]), "days": nd})
    df = pd.DataFrame(rows).set_index("score")
    pd.set_option("display.width", 200)
    print("\n=== raw against neutralised information ===")
    print(df.round(4).to_string())

    print("\n=== strongest loadings of each model on the factors (standardised beta) ===")
    for j, n in enumerate(models):
        idx = np.argsort(-np.abs(load[j]))[:3]
        print(f"  {n:<20} " + ", ".join(f"{FACTORS[i]} {load[j, i]:+.2f}" for i in idx))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        np.savez_compressed(os.path.join(args.out, "xsec_neutral_scores.npz"),
                            **{f"{n}|score": neu[0][:, j] for j, n in enumerate(names)},
                            **{f"{n}|raw": S[:, j] for j, n in enumerate(names)},
                            **{f"{ref}|y": y}, **{f"{ref}|date": di},
                            te_rows=z["te_rows"] if "te_rows" in z.files else np.arange(len(y)),
                            te_stocks=z["te_stocks"] if "te_stocks" in z.files else np.zeros(len(y)))
        df.to_csv(os.path.join(args.out, "xsec_neutralised.csv"))
        with open(os.path.join(args.out, "xsec_neutralised.json"), "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "metrics": df.to_dict(orient="index"),
                       "loadings": {n: dict(zip(FACTORS, load[j].tolist()))
                                    for j, n in enumerate(names)}}, f, indent=2)
        print(f"\nwritten {os.path.join(args.out, 'xsec_neutralised.csv')}")


if __name__ == "__main__":
    main()
