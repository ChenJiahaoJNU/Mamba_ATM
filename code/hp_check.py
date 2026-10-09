"""Hyper-parameter sanity check: can the MLP at least reach the majority-class rate?

A model that scores clearly below the majority rate on a near-random task usually means the
configured capacity or budget is wrong rather than that the task is informative. This checks a
small grid on one stock before the full run, so the reported grid is not an artefact of one bad
configuration.
"""
import sys

import numpy as np

sys.path.insert(0, r".\libs")
sys.path.insert(0, r".\code")

import pipeline as P      # noqa: E402
import train_eval as TE   # noqa: E402

DATA = r".\data"
SEQ = 120

dates, feat, ret, extra = P.load_stock(rf"{DATA}\Stock1.csv")
X, yc, yr, idx = P.make_windows(feat, ret, SEQ)
splits = P.walk_forward(len(X), folds=5, purge=1)
ref = TE.reference_points(yc[np.concatenate([s[1] for s in splits])],
                          yr[np.concatenate([s[1] for s in splits])])
print("majority-class accuracy on the pooled test blocks: %.4f" % ref["majority_accuracy"])
print()
print("%-12s %-28s %8s %8s" % ("model", "config", "accuracy", "f1"))
for hidden in (32, 64, 128):
    for ep, lr in ((30, 1e-3), (80, 1e-3), (80, 3e-4)):
        r = TE.run("mlp", X, yc, yr, splits, epochs=ep, lr=lr, hidden=hidden)
        print("%-12s %-28s %8.4f %8.4f" % ("mlp", "hidden=%d ep=%d lr=%.0e" % (hidden, ep, lr),
                                           r["accuracy"], r["f1"]))
print()
for ep in (30, 80):
    r = TE.run("dlinear", X, yc, yr, splits, epochs=ep)
    print("%-12s %-28s %8.4f %8.4f" % ("dlinear", "ep=%d" % ep, r["accuracy"], r["f1"]))
for ep in (30, 80):
    r = TE.run("lstm", X, yc, yr, splits, epochs=ep)
    print("%-12s %-28s %8.4f %8.4f" % ("lstm", "ep=%d" % ep, r["accuracy"], r["f1"]))
for ep in (30, 80):
    r = TE.run("attention", X, yc, yr, splits, epochs=ep)
    print("%-12s %-28s %8.4f %8.4f" % ("attention", "ep=%d" % ep, r["accuracy"], r["f1"]))
