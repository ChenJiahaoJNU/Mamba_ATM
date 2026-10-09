"""Result figures for the reworked manuscript, all drawn from the result files.

Figures written to <figs>:
  accuracy_dist.png  per-stock accuracy of every model, against the majority-class rate
  cost_curve.png     annualised portfolio return against the cost level, with buy and hold
  by_year.png        accuracy by test year against the majority-class rate of that year
  ablation.png       branches and fused variants with cross-stock standard deviations
  sharpe_scatter.png per-stock Sharpe against per-stock maximum drawdown

Usage: python make_figs.py --dir out/main --figs <rework dir>/figs
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DISPLAY = {
    "mlp": "MLP", "lstm": "LSTM", "attention": "Attn", "attention+conv": "Attn+conv",
    "dlinear": "DLinear", "patchtst": "PatchTST", "itransformer": "iTransf",
    "timesnet": "TimesNet", "ssm-only": "SSM only", "gatedfusion": "MambaATM",
    "gatedfusion-guided": "MambaATM-g",
}
ORDER = ["dlinear", "mlp", "lstm", "attention+conv", "timesnet", "patchtst", "itransformer",
         "ssm-only", "gatedfusion-guided", "gatedfusion"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--figs", required=True)
    args = ap.parse_args()
    os.makedirs(args.figs, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ps = pd.read_csv(os.path.join(args.dir, "per_stock.csv"))
    with open(os.path.join(args.dir, "summary.json"), encoding="utf-8") as f:
        S = json.load(f)
    models = [m for m in ORDER if m in set(ps.model)]

    # ---------------------------------------------------------------- accuracy distribution
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    data = [ps[ps.model == m].accuracy.to_numpy() for m in models]
    bp = ax.boxplot(data, tick_labels=[DISPLAY.get(m, m) for m in models], showfliers=False,
                    patch_artist=True, medianprops=dict(color="black"), widths=0.6)
    for i, m in enumerate(models):
        bp["boxes"][i].set_facecolor("#4C72B0" if m == "gatedfusion" else "#DDDDDD")
        bp["boxes"][i].set_alpha(0.85)
    maj = S["references"]["majority_accuracy_mean"]
    ax.axhline(maj, color="crimson", ls="--", lw=1.4, label=f"majority class ({maj:.3f})")
    ax.axhline(0.5, color="grey", ls=":", lw=1.0)
    ax.set_ylabel("accuracy per stock")
    ax.set_ylim(0.35, 0.72)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.figs, "accuracy_dist.png"), dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------- cost sensitivity
    pj_path = os.path.join(args.dir, "portfolio.json")
    if os.path.exists(pj_path):
        with open(pj_path, encoding="utf-8") as f:
            pj = json.load(f)
        levels = ["zero", "low_6bp", "base_10bp", "high_15bp"]
        labels = ["0", "6 bp", "10 bp", "15 bp"]
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        width = 0.8 / (len(models) + 1)
        for i, m in enumerate(models):
            vals = [pj["portfolio"][m][lv]["annual_return"] * 100 for lv in levels]
            ax.bar(np.arange(len(levels)) + i * width, vals, width, label=DISPLAY.get(m, m))
        bh = [pj["buy_hold_equal_weight"]["annual_return"] * 100] * len(levels)
        ax.plot(np.arange(len(levels)) + 0.4, bh, color="black", ls="--", lw=1.6,
                label="buy and hold")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(np.arange(len(levels)) + 0.4)
        ax.set_xticklabels(labels)
        ax.set_xlabel("one-way cost per position change")
        ax.set_ylabel("annualised return (\\%)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(args.figs, "cost_curve.png"), dpi=200)
        plt.close(fig)

    # ------------------------------------------------------------------------- by year
    by_path = os.path.join(args.dir, "by_year.json")
    if os.path.exists(by_path):
        with open(by_path, encoding="utf-8") as f:
            byy = json.load(f)
        years = sorted(byy)
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        for m in ["gatedfusion", "patchtst", "mlp", "dlinear"]:
            vals = [byy[y]["models"].get(m, np.nan) for y in years]
            ax.plot(years, vals, marker="o", ms=3, lw=1.3, label=DISPLAY.get(m, m))
        ax.plot(years, [byy[y]["majority"] for y in years], color="crimson", ls="--", lw=1.4,
                label="majority class")
        ax.set_xlabel("test year")
        ax.set_ylabel("accuracy")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.figs, "by_year.png"), dpi=200)
        plt.close(fig)

    # ------------------------------------------------------------------------ ablation
    ab = [m for m in ["attention+conv", "ssm-only", "gatedfusion", "gatedfusion-guided"]
          if m in set(ps.model)]
    if ab:
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        means = [S["models"][m]["accuracy_mean"] for m in ab]
        sds = [S["models"][m]["accuracy_std"] for m in ab]
        ax.bar(range(len(ab)), means, yerr=sds, capsize=3,
               color=["#999999", "#999999", "#4C72B0", "#DD8452"])
        ax.axhline(S["references"]["majority_accuracy_mean"], color="crimson", ls="--", lw=1.3,
                   label="majority class")
        ax.set_xticks(range(len(ab)))
        ax.set_xticklabels([DISPLAY.get(m, m) for m in ab], rotation=20, fontsize=8)
        ax.set_ylabel("accuracy")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.figs, "ablation.png"), dpi=200)
        plt.close(fig)

    # ------------------------------------------------------------------ sharpe vs drawdown
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m in models:
        sub = ps[ps.model == m]
        ax.scatter(sub.max_drawdown * 100, sub.sharpe, s=6, alpha=0.45,
                   label=DISPLAY.get(m, m) if m in
                   ("gatedfusion", "patchtst", "mlp", "dlinear") else None)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("maximum drawdown (\\%)")
    ax.set_ylabel("Sharpe ratio")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.figs, "sharpe_scatter.png"), dpi=200)
    plt.close(fig)

    print("figures written to", args.figs)


if __name__ == "__main__":
    main()
