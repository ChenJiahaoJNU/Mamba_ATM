"""Walk-forward evaluation of the whole model zoo across the A-share panel.

Protocol, identical for every model:

* one CSV per stock from the panel directory, forward-adjusted prices, real turnover rate;
* rows with no label (the first row) and rows with zero volume (suspension days, which cannot be
  traded and whose return is an artefact of the suspension) are dropped;
* features are the eight exchange fields of the day; the label is the sign of the return of the
  next trading day, and the return used for the trading metrics is that same next-day return;
* five expanding-window folds per stock, the first starting after 40 percent of the windows, with a
  one-day purge between the training block and the test block;
* the standardiser is fitted on the training block of the fold and applied unchanged to the test
  block, so no test-period statistic reaches a training input;
* the last tenth of each training block is held out, in time order, for early stopping.

Outputs, written to --out:

* `per_stock.csv`  one row per (stock, model): classification and trading metrics;
* `per_fold.csv`   the same metrics per fold, for the walk-forward table;
* `preds.npz`      dates, positions and returns per stock and model, so the equal-weight portfolio
  backtest can be rebuilt without retraining;
* `summary.json`   cross-stock mean, standard deviation and median, pooled metrics, and the
  reference points (majority class, always long, buy and hold).

Usage:
  python run_panel.py --panel <dir> --models mlp,dlinear --seq 60 --limit 3 --out out/smoke
"""
import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "libs"))
sys.path.insert(0, HERE)

import pipeline as P  # noqa: E402
import train_eval as T  # noqa: E402

DEFAULT_PANEL = (r"D:\研究生学习资料\暨南大学-博士学习资料\论文\重投\MambaATM"
                 r"\代码\data_panel_adj")
DEFAULT_MODELS = ["mlp", "lstm", "attention+conv", "dlinear", "patchtst", "itransformer",
                  "timesnet", "ssm-only", "gatedfusion", "gatedfusion-guided"]


def load_clean(path):
    """Read one panel file and drop the rows that cannot carry a tradable label."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    ret = pd.to_numeric(df[P.RET_COL], errors="coerce")
    vol = pd.to_numeric(df["成交量"], errors="coerce")
    keep = ret.notna() & vol.notna() & (vol > 0)
    df = df[keep].reset_index(drop=True)
    dates = pd.to_datetime(df[P.DATE_COL]).to_numpy()
    feat = df[P.FEATURE_COLS].to_numpy(dtype=np.float64)
    ret = df[P.RET_COL].to_numpy(dtype=np.float64)
    return dates, feat, ret


def evaluate_stock(args):
    path, models, seq, folds, epochs, seed, device = args
    code = os.path.basename(path).replace(".csv", "")
    out_rows, fold_rows, preds = [], [], {}
    t0 = time.time()
    try:
        dates, feat, ret = load_clean(path)
        X, yc, yr, idx = P.make_windows(feat, ret, seq)
        splits = P.walk_forward(len(X), folds=folds, purge=1)
        if not splits or len(X) < 60:
            return {"code": code, "skipped": f"too few windows ({len(X)})"}

        te_idx = np.concatenate([te for _, te in splits])
        ref = T.reference_points(yc[te_idx], yr[te_idx])
        preds["_ref"] = {
            "dates": dates[idx[te_idx]].astype("datetime64[D]").astype(np.int64),
            "ret": yr[te_idx],
            "majority": float(np.mean(yc[te_idx]) >= 0.5),
            "y": yc[te_idx],
        }

        for name in models:
            r = T.run(name, X, yc, yr, splits, epochs=epochs, seed=seed, device=device)
            out_rows.append({
                "stock": code, "model": name, "n": r["n"], "n_windows": len(X), "seq": seq,
                "accuracy": r["accuracy"], "precision": r["precision"], "recall": r["recall"],
                "f1": r["f1"], "total_return": r["total_return"],
                "annual_return": r["annual_return"], "sharpe": r["sharpe"],
                "max_drawdown": r["max_drawdown"], "turnover": r["turnover"],
                "seconds": r["seconds"],
                "majority_accuracy": ref["majority_accuracy"],
                "always_long_accuracy": ref["always_long_accuracy"],
                "buy_hold_return": ref["buy_hold"]["total_return"],
                "buy_hold_sharpe": ref["buy_hold"]["sharpe"],
                "buy_hold_mdd": ref["buy_hold"]["max_drawdown"],
                "ic": float(np.corrcoef(r["pred"], r["ret"])[0, 1]) if r["n"] > 2 else np.nan,
            })
            # per-fold metrics
            pos = 0
            for k, (_, te) in enumerate(splits):
                p = r["pred"][pos:pos + len(te)]
                pos += len(te)
                m = P.classification_metrics(yc[te], (p > 0).astype(int))
                tm = P.trading_metrics((p > 0).astype(float), yr[te])
                fold_rows.append({
                    "stock": code, "model": name, "fold": k, "n": int(len(te)),
                    "accuracy": m.accuracy, "f1": m.f1, "total_return": tm["total_return"],
                    "sharpe": tm["sharpe"], "max_drawdown": tm["max_drawdown"],
                    "buy_hold_return": P.trading_metrics(np.ones_like(yr[te]), yr[te])["total_return"],
                })
            preds[name] = {
                "code": code,
                "dates": dates[idx[te_idx]].astype("datetime64[D]").astype(np.int64),
                "pos": (r["pred"] > 0).astype(np.int8),
                "ret": r["ret"],
                "logit": r["pred"].astype(np.float32),
            }
    except Exception:
        return {"code": code, "error": traceback.format_exc(limit=3)}
    return {"code": code, "rows": out_rows, "folds": fold_rows, "preds": preds,
            "seconds": time.time() - t0}


def portfolio_metrics(preds_by_stock, cost=0.0003 * 2):
    """Equal-weight long-only portfolio built from the per-stock positions of one model.

    Every date, the position is the fraction of stocks that are long that day, and the portfolio
    return is the mean return of those stocks; a change in the aggregate position is charged the
    same round-trip cost as a single-stock trade.
    """
    dates, pos, ret = [], [], []
    for rec in preds_by_stock:
        dates.append(rec["dates"])
        pos.append(rec["pos"].astype(np.float64))
        ret.append(rec["ret"].astype(np.float64))
    dates = np.concatenate(dates)
    pos = np.concatenate(pos)
    ret = np.concatenate(ret)
    order = np.argsort(dates, kind="stable")
    dates, pos, ret = dates[order], pos[order], ret[order]
    uniq, start = np.unique(dates, return_index=True)
    ends = np.append(start[1:], len(dates))
    daily_pos = np.array([pos[s:e].mean() for s, e in zip(start, ends)])
    daily_ret = np.array([ret[s:e].mean() for s, e in zip(start, ends)])
    return P.trading_metrics(daily_pos, daily_ret, cost=cost)


def summarise(per_stock, preds_all, models):
    out = {"models": {}, "references": {}}
    df = pd.DataFrame(per_stock)
    for name in models:
        sub = df[df.model == name]
        if sub.empty:
            continue
        rec = {}
        for col in ["accuracy", "f1", "precision", "recall", "sharpe", "max_drawdown",
                    "total_return", "annual_return", "turnover", "ic"]:
            v = sub[col].astype(float)
            rec[col + "_mean"] = float(v.mean())
            rec[col + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            rec[col + "_median"] = float(v.median())
        rec["stocks"] = int(len(sub))
        rec["stocks_beating_buy_hold"] = int((sub.total_return > sub.buy_hold_return).sum())
        rec["stocks_beating_majority"] = int((sub.accuracy > sub.majority_accuracy).sum())
        rec["portfolio"] = portfolio_metrics(preds_all[name])
        out["models"][name] = rec

    for col in ["majority_accuracy", "always_long_accuracy", "buy_hold_return", "buy_hold_sharpe",
                "buy_hold_mdd"]:
        v = df[col].astype(float).drop_duplicates()
        out["references"][col + "_mean"] = float(df[col].astype(float).mean())
        out["references"][col + "_median"] = float(df[col].astype(float).median())
        del v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--seq", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N stocks")
    ap.add_argument("--stocks", default="", help="explicit comma-separated codes")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m]
    if args.stocks:
        files = [os.path.join(args.panel, c if c.endswith(".csv") else c + ".csv")
                 for c in args.stocks.split(",")]
    else:
        files = sorted(f for f in os.listdir(args.panel) if f.endswith(".csv"))
        files = [os.path.join(args.panel, f) for f in files]
    if args.limit:
        files = files[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    print(f"panel {args.panel}\nstocks {len(files)}  models {len(models)}  seq {args.seq}  "
          f"folds {args.folds}  epochs {args.epochs}  workers {args.workers}  device {args.device}")

    jobs = [(f, models, args.seq, args.folds, args.epochs, args.seed, args.device) for f in files]
    per_stock, per_fold, skipped, errors = [], [], [], []
    preds_by_stock_model = {m: [] for m in models}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(evaluate_stock, j): j[0] for j in jobs}
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            if "error" in r:
                errors.append({"code": r["code"], "error": r["error"]})
                print(f"  [{done}/{len(jobs)}] {r['code']} ERROR")
            elif "skipped" in r:
                skipped.append(r)
                print(f"  [{done}/{len(jobs)}] {r['code']} skipped: {r['skipped']}")
            else:
                per_stock.extend(r["rows"])
                per_fold.extend(r["folds"])
                for m in models:
                    if m in r["preds"]:
                        preds_by_stock_model[m].append(r["preds"][m])
                el = time.time() - t0
                print(f"  [{done}/{len(jobs)}] {r['code']} {len(r['rows'])} models "
                      f"in {r['seconds']:.0f}s   elapsed {el/60:.1f} min")

    pd.DataFrame(per_stock).to_csv(os.path.join(args.out, "per_stock.csv"), index=False)
    pd.DataFrame(per_fold).to_csv(os.path.join(args.out, "per_fold.csv"), index=False)
    np.savez_compressed(
        os.path.join(args.out, "preds.npz"),
        **{f"{m}|{rec['code']}|dates": rec["dates"] for m, lst in preds_by_stock_model.items()
           for rec in lst},
        **{f"{m}|{rec['code']}|pos": rec["pos"] for m, lst in preds_by_stock_model.items()
           for rec in lst},
        **{f"{m}|{rec['code']}|ret": rec["ret"].astype(np.float32)
           for m, lst in preds_by_stock_model.items() for rec in lst})

    summary = summarise(per_stock, preds_by_stock_model, models) if per_stock else {}
    summary["config"] = vars(args)
    summary["skipped"] = skipped
    summary["errors"] = errors
    summary["minutes"] = (time.time() - t0) / 60
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\ndone in {summary.get('minutes', 0):.1f} min -> {args.out}")
    for name, rec in summary.get("models", {}).items():
        print(f"  {name:<20} acc {rec['accuracy_mean']:.4f}+-{rec['accuracy_std']:.4f}  "
              f"sharpe {rec['sharpe_mean']:+.2f}  portSharpe {rec['portfolio']['sharpe']:+.2f}  "
              f"portRet {rec['portfolio']['total_return']*100:+.1f}%")


if __name__ == "__main__":
    main()
