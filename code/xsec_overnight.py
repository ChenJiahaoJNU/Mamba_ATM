"""Where does the cross-sectional predictability of A-shares actually live?

The label decomposition in `factors.py` split the next-day return into an overnight leg
(previous close to open) and an intraday leg (open to close). On this panel the overnight leg is two
to three times more predictable than the close-to-close return the models were trained on, which is
either the most interesting result of the project or an artefact of illiquid and stale prices. This
script tries to kill it, in the five ways a referee would:

1. **The dilution identity.** If only the overnight leg is predictable and the intraday leg is noise,
   then ``IC(close-to-close) ~ IC(overnight) * sd(overnight) / sd(close-to-close)``. If the identity
   holds for every score, the decomposition explains the modest close-to-close IC rather than
   contradicting it.
2. **Extreme moves.** Winsorise the overnight label and drop the tail, in case a handful of
   limit-up opens drive the result.
3. **Stale prices.** Recompute on the liquid half only; a stale close manufactures overnight moves
   for illiquid names.
4. **Static cross-sectional differences.** Demean every score and the label within each stock over
   the test period, so a permanent difference between low-turnover and high-turnover names cannot
   earn anything, and see what is left.
5. **Stability.** Report the IC year by year.

Usage: python xsec_overnight.py --panel <panel> --scores out/xsec_main/xsec_scores_all.npz \
    --seq 60 [--out DIR]
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
MODELS = ["dlinear", "mlp", "lstm", "attention+conv", "patchtst", "itransformer", "timesnet",
          "ssm-only", "gatedfusion", "gatedfusion-guided"]
EXTRA_FACTORS = ["turn1", "logamt1", "vol20", "range20", "rev5", "gap1", "clv5"]


def ranks(v):
    return stats.rankdata(v).astype(np.float64)


def ic_series(score, y, di, min_names=30, mask=None):
    out = []
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
        out.append((d, np.corrcoef(ranks(a[ok]), ranks(b[ok]))[0, 1]))
    return pd.DataFrame(out, columns=["date", "ic"])


def summarise(ic):
    v = ic.ic.to_numpy()
    if len(v) < 30:
        return float("nan"), float("nan"), len(v)
    return float(v.mean()), float(v.mean() / v.std(ddof=1) * np.sqrt(len(v))), len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--seq", type=int, default=60)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    z = np.load(args.scores, allow_pickle=False)
    ref = next(k.split("|")[0] for k in z.files if k.endswith("|y"))
    te_rows = z["te_rows"]
    di = z[f"{ref}|date"]
    stock_ids = z["te_stocks"] if "te_stocks" in z.files else None

    data_all = X.load_panel(args.panel)
    data = {c: data_all[c] for c in sorted(data_all)}
    ds = X.build_dataset(data, args.seq)
    tg = FA.targets(data, ds)
    per_date = {}
    for code, (dates, feat, ret) in data.items():
        for d, r in zip(dates, ret):
            per_date.setdefault(d, []).append(r)
    mean_by_date = {d: float(np.mean(v)) for d, v in per_date.items()}
    F, _ = FA.factor_arrays(data, ds, lambda dates: np.array([mean_by_date.get(d, 0.0)
                                                              for d in dates]))
    adv, sigma = C.liquidity_arrays(data, ds)

    scores = {m: z[f"{m}|score"] for m in MODELS if f"{m}|score" in z.files}
    for f in EXTRA_FACTORS:
        scores[f] = F[f][te_rows]

    y_cc = tg["cc"][te_rows]
    y_on = tg["on"][te_rows]
    y_id = tg["id"][te_rows]
    adv = adv[te_rows]
    dates_all = np.unique(di)
    # di holds positions in the date grid rather than dates, so map back before taking the year
    years = ds.grid[np.asarray(di)].astype("datetime64[Y]").astype(int) + 1970
    print(f"{len(scores)} scores, {len(di):,} test rows")
    print(f"label means per day: cc {np.nanmean(y_cc) * 100:+.4f}%  on {np.nanmean(y_on) * 100:+.4f}%"
          f"  id {np.nanmean(y_id) * 100:+.4f}%")
    print(f"label sd:            cc {np.nanstd(y_cc) * 100:.3f}%  on {np.nanstd(y_on) * 100:.3f}%"
          f"  id {np.nanstd(y_id) * 100:.3f}%\n")

    sd_cc, sd_on = np.nanstd(y_cc), np.nanstd(y_on)
    rows = []
    for n, s in scores.items():
        ic_cc = summarise(ic_series(s, y_cc, di))
        ic_on = summarise(ic_series(s, y_on, di))
        ic_id = summarise(ic_series(s, y_id, di))
        # 1. dilution identity
        predicted_cc = ic_on[0] * sd_on / sd_cc
        # 2. winsorised overnight label
        w = np.clip(y_on, -0.05, 0.05)
        ic_on_w = summarise(ic_series(s, w, di))
        # 3. liquid half only
        liq_hi = np.zeros(len(adv), dtype=bool)
        thr = np.nanmedian(adv)
        liq_hi = adv >= thr
        ic_on_liq = summarise(ic_series(s, y_on, di, mask=liq_hi))
        # 4. within-stock demeaning
        if stock_ids is not None:
            s_dm = s.copy()
            y_dm = y_on.copy()
            for k in np.unique(stock_ids):
                m = stock_ids == k
                s_dm[m] = s[m] - np.nanmean(s[m])
                y_dm[m] = y_on[m] - np.nanmean(y_on[m])
            ic_on_dm = summarise(ic_series(s_dm, y_dm, di))
        else:
            ic_on_dm = (np.nan, np.nan, 0)
        rows.append({"score": n, "ic_cc": ic_cc[0], "t_cc": ic_cc[1],
                     "ic_on": ic_on[0], "t_on": ic_on[1],
                     "ic_id": ic_id[0], "t_id": ic_id[1],
                     "identity_pred_cc": predicted_cc,
                     "ic_on_winsor": ic_on_w[0], "ic_on_liquidhalf": ic_on_liq[0],
                     "ic_on_within_stock": ic_on_dm[0], "t_within": ic_on_dm[1]})
    df = pd.DataFrame(rows).set_index("score")
    pd.set_option("display.width", 220)
    print("=== close-to-close against the two legs, and the checks ===")
    print(df.round(4).to_string())

    print("\n=== overnight IC by year, best three scores ===")
    for n in df.sort_values("ic_on", ascending=False).head(3).index:
        line = []
        for yr in sorted(set(years)):
            m = years == yr
            ic = summarise(ic_series(scores[n], y_on, di, mask=m))
            line.append(f"{yr}:{ic[0]:+.3f}")
        print(f"  {n:<20} " + "  ".join(line))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        df.to_csv(os.path.join(args.out, "overnight.csv"))
        with open(os.path.join(args.out, "overnight.json"), "w", encoding="utf-8") as f:
            json.dump({"rows": df.to_dict(orient="index"),
                       "label_means": {"cc": float(np.nanmean(y_cc)),
                                       "on": float(np.nanmean(y_on)),
                                       "id": float(np.nanmean(y_id))},
                       "label_sds": {"cc": float(sd_cc), "on": float(sd_on),
                                     "id": float(np.nanstd(y_id))}}, f, indent=2)
        print(f"\nwritten {os.path.join(args.out, 'overnight.csv')}")


if __name__ == "__main__":
    main()
