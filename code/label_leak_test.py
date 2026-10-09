"""Does the label definition in the submitted data loader inflate the reported accuracy?

The loader labels each window with the sign of the return on the LAST day of that window, while
the window itself carries that day's open, high, low, close and previous close. The return is a
function of two of those columns, so part of the label is inside the input. This script runs the
same models under both label definitions and compares.
"""
import sys

import numpy as np

sys.path.insert(0, r".\libs")
sys.path.insert(0, r".\code")

import pipeline as P            # noqa: E402
import train_eval as TE         # noqa: E402

DATA = r".\data"
SEQ, FOLDS, EPOCHS = 120, 5, 30

print("%-8s %-12s %8s %8s %8s %8s   %s" %
      ("stock", "label", "majority", "accuracy", "f1", "sharpe", "model"))
for sidx in [1, 2]:
    dates, feat, ret, extra = P.load_stock(rf"{DATA}\Stock{sidx}.csv")
    for mode in ["same_day", "next_day"]:
        X, yc, yr, idx = P.make_windows(feat, ret, SEQ, label_mode=mode)
        splits = P.walk_forward(len(X), folds=FOLDS, purge=1)
        ref = TE.reference_points(yc[np.concatenate([s[1] for s in splits])],
                                  yr[np.concatenate([s[1] for s in splits])])
        for nm in ["mlp", "dlinear"]:
            r = TE.run(nm, X, yc, yr, splits, epochs=EPOCHS)
            print("%-8s %-12s %8.4f %8.4f %8.4f %+8.2f   %s"
                  % ("Stock%d" % sidx, mode, ref["majority_accuracy"], r["accuracy"],
                     r["f1"], r["sharpe"], nm))
    print()
