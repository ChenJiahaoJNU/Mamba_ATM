"""Two follow-ups on the overnight finding.

**A. Does the model's incremental information also live overnight?**

The factor-neutralised residual of a model score had a small but significant close-to-close IC. Here
the same residualisation is repeated against the full twenty-nine-factor library and measured on the
overnight label. If the model has anything of its own, it should show up there, because that is where
the predictability is. The same test is run for the factors themselves with leave-one-out, so the
comparison is fair in both directions.

**B. Where does the money come from, and what would trading at the open cost?**

For the weekly-rebalanced book the gross return is split into the overnight leg and the intraday leg,
with the fee, impact and borrow charges kept separate, and the whole thing is repeated for a book that
enters at the open: a name added at a rebalance then earns only its intraday return on the first day,
because it was bought after the overnight jump it was selected for.

Usage: python xsec_attrib.py --panel <panel> --scores out/xsec_main/xsec_scores_all.npz --seq 60 \
    [--rebalances 1,5] [--out DIR]
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

DEFAULT_PANEL = (r"D:\研究生学习资料\暨南大学-博士学习资料\论文\重投\MambaATM"
                 r"\代码\data_panel_adj")
MODELS = ["dlinear", "mlp", "lstm", "attention+conv", "patchtst", "itransformer", "timesnet",
          "ssm-only", "gatedfusion", "gatedfusion-guided"]


def ranks(v):
    return stats.rankdata(v).astype(np.float64)


def ic_of(score, y, di, min_names=30, mask=None):
    vals = []
    for d in np.unique(di):
        m = di == d
        if mask is not None:
            m = m & mask
        if m.sum() < min_names:
            continue
        a, b = score[m], y[m]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < min_names or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
            continue
        vals.append(np.corrcoef(ranks(a[ok]), ranks(b[ok]))[0, 1])
    v = np.asarray(vals)
    if len(v) < 30:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.mean() / v.std(ddof=1) * np.sqrt(len(v)))


def residualise(target, base, di, min_names=30, exclude_self=None):
    """Cross-sectional residual of `target` on `base`, day by day, optional leave-one-out column."""
    out = np.full_like(target, np.nan, dtype=np.float64)
    for d in np.unique(di):
        m = di == d
        if m.sum() < min_names:
            continue
        B = base[m]
        if exclude_self is not None:
            B = np.delete(B, exclude_self, axis=1)
        sdB = B.std(axis=0)
        keep = sdB > 1e-12
        Bz = (B[:, keep] - B[:, keep].mean(axis=0)) / sdB[keep]
        t = target[m]
        if np.std(t) < 1e-12:
            continue
        tz = (t - t.mean()) / t.std()
        X = np.column_stack([np.ones(m.sum()), Bz])
        beta, *_ = np.linalg.lstsq(X, tz, rcond=None)
        out[m] = tz - X @ beta
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--seq", type=int, default=60)
    ap.add_argument("--rebalances", default="1,5")
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

    y_cc, y_on, y_id = tg["cc"][te_rows], tg["on"][te_rows], tg["id"][te_rows]
    Fte = np.column_stack([F[n][te_rows] for n in fnames])
    scores = {m: z[f"{m}|score"] for m in MODELS if f"{m}|score" in z.files}
    scores["cheap_combo"] = (F["turn1"][te_rows] + F["logamt1"][te_rows]) / 2
    print(f"{len(scores)} scores, {len(fnames)} factors, {len(di):,} test rows\n")

    # ---------------------------------------------------------------- A. neutralised, by label
    rows = []
    for n, s in scores.items():
        raw_on = ic_of(s, y_on, di)
        raw_cc = ic_of(s, y_cc, di)
        res_on = residualise(s, Fte, di)
        neu_on = ic_of(res_on, y_on, di)
        neu_cc = ic_of(res_on, y_cc, di)
        # share of the score that is linear in the factor library
        m = np.isfinite(res_on) & np.isfinite(s)
        r2 = 1.0 - np.nanvar(res_on[m]) / np.nanvar((s[m] - s[m].mean()) / s[m].std())
        rows.append({"score": n, "ic_cc": raw_cc[0], "t_cc": raw_cc[1],
                     "ic_on": raw_on[0], "t_on": raw_on[1],
                     "resid_ic_cc": neu_cc[0], "t_resid_cc": neu_cc[1],
                     "resid_ic_on": neu_on[0], "t_resid_on": neu_on[1],
                     "factor_r2": r2})
    for j, fn in enumerate(fnames):
        s = Fte[:, j]
        rr = residualise(s, Fte, di, exclude_self=j)
        rows.append({"score": f"[factor] {fn}", "ic_cc": ic_of(s, y_cc, di)[0],
                     "t_cc": ic_of(s, y_cc, di)[1], "ic_on": ic_of(s, y_on, di)[0],
                     "t_on": ic_of(s, y_on, di)[1],
                     "resid_ic_cc": ic_of(rr, y_cc, di)[0], "t_resid_cc": ic_of(rr, y_cc, di)[1],
                     "resid_ic_on": ic_of(rr, y_on, di)[0], "t_resid_on": ic_of(rr, y_on, di)[1],
                     "factor_r2": np.nan})
    A = pd.DataFrame(rows).set_index("score")
    pd.set_option("display.width", 220)
    print("=== A. raw and factor-neutralised IC, on the close-to-close label and on the overnight "
          "label ===")
    print(A.sort_values("resid_ic_on", ascending=False).round(4).to_string())

    # ---------------------------------------------------------------- B. attribution
    rows = []
    for n in ["gatedfusion-guided", "gatedfusion", "cheap_combo"]:
        s = scores[n]
        for h in [int(x) for x in args.rebalances.split(",")]:
            for liq in (1.0, 0.5):
                kw = dict(quantile=args.quantile, rebalance=h, tradable=None,
                          stock_ids=stock_ids, top_liquidity=liq, pnl_parts=(y_on, y_id),
                          open_entry=True)
                cm = C.CostModel()
                r = C.replay(s, y_cc, di, ds.grid, adv=adv, sigma=sigma, cm=cm, **kw)
                rows.append({"score": n, "hold_days": h, "liquid": liq,
                             "gross_total": r["gross_ann"], "gross_overnight": r["gross_on_ann"],
                             "gross_intraday": r["gross_id_ann"],
                             "gross_if_entered_at_open": r["gross_open_entry_ann"],
                             "fee": r["fee_ann"], "impact": r["impact_ann"], "borrow": r["borrow_ann"],
                             "net": r["net_ann"]})
    B = pd.DataFrame(rows)
    print("\n=== B. attribution of the long-short book, per cent per year ===")
    print(B.round(2).to_string(index=False))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        A.to_csv(os.path.join(args.out, "attrib_neutral.csv"))
        B.to_csv(os.path.join(args.out, "attrib_pnl.csv"), index=False)
        with open(os.path.join(args.out, "attrib.json"), "w", encoding="utf-8") as f:
            json.dump({"neutral": A.to_dict(orient="index"), "pnl": rows}, f, indent=2)
        print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
