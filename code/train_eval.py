"""Train and evaluate one model under the walk-forward protocol, and report honest references.

Per fold: standardise on the training block, hold out its last 10% for early stopping, train on
the rest, predict the test block. Predictions are pooled across folds in time order, and the
trading metrics are computed on that pooled series with a long-only, cost-charged rule.
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, r".\libs")
sys.path.insert(0, r".\code")

import model_zoo  # noqa: E402
import pipeline as P  # noqa: E402


def run(name, X, yc, yr, splits, epochs=30, lr=1e-3, batch=64, seed=0, device="cuda",
        verbose=False, **kw):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_feat, seq_len = X.shape[2], X.shape[1]
    pooled_pred, pooled_true, pooled_ret = [], [], []
    t0 = time.time()
    for k, (tr, te) in enumerate(splits):
        xtr, xte = P.standardise(X[tr], X[te])
        ytr = yc[tr]
        n_val = max(8, int(0.1 * len(tr)))
        xtr_in, ytr_in = xtr[:-n_val], ytr[:-n_val]
        xva, yva = xtr[-n_val:], ytr[-n_val:]

        model = model_zoo.build(name, n_feat, seq_len, **kw).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        lossf = nn.BCEWithLogitsLoss()
        Xt = P.to_tensor(xtr_in, device)
        yt = P.to_tensor(ytr_in, device)
        Xv = P.to_tensor(xva, device)
        yv = P.to_tensor(yva, device)

        best, best_state, patience = 1e9, None, 0
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(Xt), device=device)
            for i in range(0, len(Xt), batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                loss = lossf(model(Xt[idx]), yt[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = lossf(model(Xv), yv).item()
            if vl < best - 1e-5:
                best, patience = vl, 0
                best_state = {k2: v.detach().clone() for k2, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 5:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            logit = model(P.to_tensor(xte, device)).cpu().numpy()
        pooled_pred.append(logit)
        pooled_true.append(yc[te])
        pooled_ret.append(yr[te])
        if verbose:
            acc = P.classification_metrics(yc[te], (logit > 0).astype(int)).accuracy
            print("      fold %d: n=%3d  acc=%.4f  (%.1fs)"
                  % (k, len(te), acc, time.time() - t0))

    pred = np.concatenate(pooled_pred)
    true = np.concatenate(pooled_true)
    ret = np.concatenate(pooled_ret)
    m = P.classification_metrics(true, (pred > 0).astype(int))
    tm = P.trading_metrics((pred > 0).astype(float), ret)
    return {"model": name, "accuracy": m.accuracy, "precision": m.precision,
            "recall": m.recall, "f1": m.f1, "n": m.n, **tm,
            "seconds": time.time() - t0, "pred": pred, "true": true, "ret": ret}


def reference_points(yc, yr):
    maj = 1.0 if np.mean(yc) >= 0.5 else 0.0
    bh = P.trading_metrics(np.ones_like(yr), yr)
    return {"majority_accuracy": float((yc == maj).mean()),
            "always_long_accuracy": float((yc == 1.0).mean()),
            "buy_hold": bh}


def main():
    DATA = r".\data"
    SEQ, FOLDS, EPOCHS = 120, 5, 30
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["mlp", "dlinear", "gatedfusion"]
    stocks = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["1"])]

    for sidx in stocks:
        dates, feat, ret, extra = P.load_stock(rf"{DATA}\Stock{sidx}.csv")
        X, yc, yr, idx = P.make_windows(feat, ret, SEQ)
        splits = P.walk_forward(len(X), folds=FOLDS, purge=1)
        ref = reference_points(yc[np.concatenate([s[1] for s in splits])],
                               yr[np.concatenate([s[1] for s in splits])])
        print("\n================= Stock%d =================" % sidx)
        print("   references: majority %.4f   always-long %.4f   buy&hold total %+.2f%%  Sharpe %+.2f  maxDD %.2f%%"
              % (ref["majority_accuracy"], ref["always_long_accuracy"],
                 ref["buy_hold"]["total_return"] * 100, ref["buy_hold"]["sharpe"],
                 ref["buy_hold"]["max_drawdown"] * 100))
        print("   %-20s %8s %8s %7s %7s %9s %8s %8s %7s" %
              ("model", "acc", "prec", "recall", "f1", "total%", "sharpe", "maxDD%", "sec"))
        for nm in names:
            r = run(nm, X, yc, yr, splits, epochs=EPOCHS, verbose=False)
            print("   %-20s %8.4f %8.4f %7.4f %7.4f %+9.2f %+8.2f %8.2f %7.1f"
                  % (nm, r["accuracy"], r["precision"], r["recall"], r["f1"],
                     r["total_return"] * 100, r["sharpe"], r["max_drawdown"] * 100, r["seconds"]))


if __name__ == "__main__":
    main()
