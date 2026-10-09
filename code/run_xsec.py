"""Run the cross-sectional experiment: rank stocks within a day, evaluate with IC and deciles.

Usage
-----
python run_xsec.py --panel <dir> --stocks 40 --models dlinear,mlp,gatedfusion --seq 60 \
    --folds 5 --epochs 30 --out out/xsec_pilot

The protocol, in order:

1. every stock contributes trailing windows; a window is labelled by the date it predicts, so windows
   from different stocks whose target date coincides form one cross-section;
2. the training label is the next-day return of that date, demeaned and scaled across the stocks of
   the same date, clipped at three standard deviations;
3. expanding folds cut the date grid, not the window list, with a purge of one date before each test
   block; the feature standardiser is fitted on the training windows of the fold only;
4. evaluation is the per-date rank information coefficient between score and realised return, the
   information ratio of that series, and an equal-weight decile long-short portfolio charged the same
   cost per position change as the time-series experiments;
5. four published characteristics are scored on exactly the same windows as the learned models, so the
   economic references are not a different sample.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "libs"))
sys.path.insert(0, HERE)

import model_zoo  # noqa: E402
import xsection as X  # noqa: E402

DEFAULT_PANEL = (r"."
                 r"\data\adj")
CHARACTERISTICS = ["short_reversal", "reversal_5", "reversal_20", "momentum",
                   "low_volatility", "volatility_20", "low_turnover", "small_amount",
                   "low_price"]


def standardise(train_x, test_x):
    mu = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    sd = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (train_x - mu) / sd, (test_x - mu) / sd


def train_one(name, ds, z, tr_mask, te_mask, seq, epochs, lr, batch, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_feat = ds.X.shape[2]
    Xtr, Xte = standardise(ds.X[tr_mask], ds.X[te_mask])
    ztr, zte = z[tr_mask], z[te_mask]
    order = np.argsort(ds.date_idx[tr_mask], kind="stable")
    Xtr, ztr = Xtr[order], ztr[order]
    n_val = max(16, int(0.1 * len(Xtr)))
    Xtr_in, ztr_in = Xtr[:-n_val], ztr[:-n_val]
    Xva, zva = Xtr[-n_val:], ztr[-n_val:]

    model = model_zoo.build(name, n_feat, seq).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    # tensors stay on the host and are moved per batch: at 232 stocks the whole training set does not
    # fit beside the model on a 12 GB card, and a batch of 256 windows is a few megabytes
    Xt = torch.tensor(Xtr_in, dtype=torch.float32)
    zt = torch.tensor(ztr_in, dtype=torch.float32)
    Xv = torch.tensor(Xva, dtype=torch.float32)
    zv = torch.tensor(zva, dtype=torch.float32)

    def val_loss():
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(Xv), 4096):
                xb = Xv[i:i + 4096].to(device)
                zb = zv[i:i + 4096].to(device)
                tot += lossf(model(xb), zb).item() * len(xb)
                n += len(xb)
        return tot / max(1, n)

    best, best_state, patience = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(Xt[idx].to(device)), zt[idx].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vl = val_loss()
        if vl < best - 1e-5:
            best, patience, best_state = vl, 0, {k: v.detach().clone()
                                                 for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 5:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(Xte_t), 4096):
            out.append(model(Xte_t[i:i + 4096].to(device)).cpu().numpy())
    return (np.concatenate(out) if out else np.zeros(0)).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--stocks", type=int, default=40, help="first N stocks of the sorted panel")
    ap.add_argument("--models", default="dlinear,mlp,lstm,gatedfusion")
    ap.add_argument("--seq", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--first-test-frac", type=float, default=0.4)
    ap.add_argument("--cost", type=float, default=0.0010)
    ap.add_argument("--limit", type=float, default=0.098,
                    help="a stock that closed at or beyond this move is excluded from the\n                          next day's entry, because a limit-up name cannot reliably be bought")
    ap.add_argument("--quantile", type=float, default=0.1,
                    help="fraction per leg of the long-short portfolio; 0.2 for a small cross-section")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    models = [m for m in args.models.split(",") if m]
    t0 = time.time()

    data_all = X.load_panel(args.panel)
    codes = sorted(data_all)[:args.stocks]
    data = {c: data_all[c] for c in codes}
    ds = X.build_dataset(data, args.seq)
    z = X.xsec_target(ds.y, ds.date_idx)
    print(f"panel: {len(codes)} stocks, {len(ds.X):,} windows, {len(ds.grid):,} dates "
          f"({ds.grid[0]} to {ds.grid[-1]})")
    tradable = X.tradable_mask(data, ds, limit=args.limit)
    print(f'tradable rows: {tradable.mean() * 100:.1f}% of {len(tradable):,}')
    names_per_date = pd.Series(ds.date_idx).value_counts()
    print(f"cross-section: median {int(names_per_date.median())} stocks per date, "
          f"min {int(names_per_date.min())}")

    first_test = ds.grid[int(len(ds.grid) * args.first_test_frac)]
    splits = X.date_splits(ds.grid, first_test, folds=args.folds, purge_days=1)
    print(f"folds: {len(splits)}, first test block starts {first_test}")

    # ------------------------------------------------------------------ learned models
    results, pooled = {}, {}
    te_rows_all = None
    for name in models:
        te_dates, te_y, te_di, te_score, te_rows = [], [], [], [], []
        for k, (tr_dates, te_set) in enumerate(splits):
            tr_mask = np.isin(ds.grid[ds.date_idx], tr_dates)
            te_mask = np.isin(ds.grid[ds.date_idx], list(te_set))
            te_rows.append(np.where(te_mask)[0])
            s = train_one(name, ds, z, tr_mask, te_mask, args.seq, args.epochs, args.lr,
                          args.batch, args.seed, args.device)
            te_score.append(s)
            te_di.append(ds.date_idx[te_mask])
            te_y.append(ds.y[te_mask])
            ic = X.daily_ic(s, ds.y[te_mask], ds.date_idx[te_mask])
            sm = X.ic_summary(ic)
            print(f"  [{name}] fold {k + 1}/{len(splits)}: IC {sm.get('ic_mean', 0):+.4f} "
                  f"(t {sm.get('ic_t', 0):+.2f}, {sm.get('days', 0)} days)")
        score = np.concatenate(te_score)
        di = np.concatenate(te_di)
        y = np.concatenate(te_y)
        te_rows_all = np.concatenate(te_rows)
        pooled[name] = (score, y, di)
        ic = X.daily_ic(score, y, di)
        rec = {"model": name, **X.ic_summary(ic),
               **X.decile_portfolio(score, y, di, cost=args.cost, quantile=args.quantile,
                                            tradable=tradable[te_rows_all],
                                            stock_ids=ds.stock_idx[te_rows_all])}
        results[name] = rec
        print(f"  {name:<20} IC {rec['ic_mean']:+.4f}  IR {rec['ic_ir']:+.2f}  "
              f"t {rec['ic_t']:+.1f}  L/S {rec['ls_ann'] * 100:+.1f}%/yr  "
              f"Sharpe {rec['ls_sharpe']:+.2f}  turnover {rec['ls_turnover']:.2f}")

    # ------------------------------------------------- economic references on the same windows
    # Same rows, not merely the same dates: the characteristic is scored on every window the learned
    # models were scored on, so the comparison is row for row.
    ref_score, ref_y, ref_di = next(iter(pooled.values()))
    for ch in CHARACTERISTICS:
        sc = X.characteristic_vector(data, ds, ch)[te_rows_all]
        ic = X.daily_ic(sc, ref_y, ref_di)
        rec = {"model": ch, **X.ic_summary(ic),
               **X.decile_portfolio(sc, ref_y, ref_di, cost=args.cost, quantile=args.quantile,
                                            tradable=tradable[te_rows_all],
                                            stock_ids=ds.stock_idx[te_rows_all])}
        results[ch] = rec
        print(f"  {ch:<20} IC {rec['ic_mean']:+.4f}  IR {rec['ic_ir']:+.2f}  "
              f"t {rec['ic_t']:+.1f}  L/S {rec['ls_ann'] * 100:+.1f}%/yr  "
              f"Sharpe {rec['ls_sharpe']:+.2f}  turnover {rec['ls_turnover']:.2f}")

    # ------------------------------------------------------------------------- output
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(args.out, "xsec_metrics.csv"))
    np.savez_compressed(
        os.path.join(args.out, "xsec_scores.npz"),
        **{f"{k}|score": v[0] for k, v in pooled.items()},
        **{f"{k}|y": v[1] for k, v in pooled.items()},
        **{f"{k}|date": v[2] for k, v in pooled.items()},
        te_rows=te_rows_all,
        te_stocks=ds.stock_idx[te_rows_all],
        grid=ds.grid)
    with open(os.path.join(args.out, "xsec_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "metrics": results,
                   "minutes": (time.time() - t0) / 60}, f, indent=2, ensure_ascii=False)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
