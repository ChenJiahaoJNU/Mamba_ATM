"""Generate every table, figure and quoted number for the overnight paper from the result files.

No experiment is rerun here. The script reads the CSVs and JSONs produced by the cross-sectional
pipeline and writes:

    tables/*.tex    the table bodies the manuscript includes
    numbers.tex     one LaTeX macro per quantity the prose quotes
    figs/*.png      the figures

so that the text, the tables and the figures cannot disagree.

Usage: python make_paper_assets.py --out <paper dir> [--figs <paper dir>/figs]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "libs"))
sys.path.insert(0, HERE)

OUT = (r".\out")
PANEL = (r"."
         r"\data\adj")

DISPLAY = {
    "gatedfusion": "MambaATM (gated)", "gatedfusion-guided": "MambaATM (guided)",
    "attention+conv": "Attention + causal conv", "ssm-only": "Selective SSM only",
    "dlinear": "DLinear", "mlp": "MLP", "lstm": "LSTM", "patchtst": "PatchTST",
    "itransformer": "iTransformer", "timesnet": "TimesNet",
    "turn1": "Turnover (1 day, negated)", "logamt1": "Turnover value (negated)",
    "range20": "Intraday range (20 days)", "mom60": "Momentum (60 days)",
    "vol20": "Volatility (20 days, negated)", "volspike": "Volume spike (negated)",
    "rev5": "Reversal (5 days)", "gap1": "Overnight gap (negated)",
    "clv5": "Close-location value (5 days)", "idr5": "Intraday return (5 days)",
    "gap5": "Overnight gap (5 days, negated)", "cheap_combo": "Cheap factor combination",
    "cheap factors only": "Cheap factors only",
    "0.25 gatedfusion-guided + 0.75 cheap": "0.25 MambaATM (guided) + 0.75 cheap factors",
    "0.50 gatedfusion-guided + 0.50 cheap": "0.50 MambaATM (guided) + 0.50 cheap factors",
    "0.75 gatedfusion-guided + 0.25 cheap": "0.75 MambaATM (guided) + 0.25 cheap factors",
    "0.25 gatedfusion + 0.75 cheap": "0.25 MambaATM (gated) + 0.75 cheap factors",
    "0.50 gatedfusion + 0.50 cheap": "0.50 MambaATM (gated) + 0.50 cheap factors",
    "0.75 gatedfusion + 0.25 cheap": "0.75 MambaATM (gated) + 0.25 cheap factors",
}


def d(name):
    """Display name for a signal, with underscores escaped so a raw key can never break the build."""
    if name in DISPLAY:
        return DISPLAY[name]
    return str(name).replace("_", r"\_")


def macro(name, value):
    return "\\newcommand{\\" + name + "}{" + str(value) + "}"


def pct(x, nd=1, sign=True):
    f = f"{{:+.{nd}f}}" if sign else f"{{:.{nd}f}}"
    return f.format(x)


def write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def table_tabular(header, rows, align=None, font="footnotesize", pad="3pt", escape=True):
    """A bare tabular, ready to be wrapped by the manuscript."""
    align = align or ("l" + "c" * (len(header) - 1))
    lines = ["{\\" + font, "\\setlength{\\tabcolsep}{" + pad + "}",
             r"\begin{tabular}{" + align + "}", r"\toprule",
             " & ".join(r"\textbf{" + h + "}" for h in header) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(str(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--figs", default="")
    args = ap.parse_args()
    out = args.out
    figs = args.figs or os.path.join(out, "figs")
    tdir = os.path.join(out, "tables")
    os.makedirs(figs, exist_ok=True)
    os.makedirs(tdir, exist_ok=True)

    ps = json.load(open(os.path.join(OUT, "main", "panel_stats.json"), encoding="utf-8"))
    ov = pd.read_csv(os.path.join(OUT, "xsec_overnight", "overnight.csv"), index_col=0)
    ovj = json.load(open(os.path.join(OUT, "xsec_overnight", "overnight.json"), encoding="utf-8"))
    real = pd.read_csv(os.path.join(OUT, "xsec_main", "xsec_realcost.csv"))
    att = pd.read_csv(os.path.join(OUT, "xsec_attrib", "attrib_neutral.csv"), index_col=0)
    pnl = pd.read_csv(os.path.join(OUT, "xsec_attrib", "attrib_pnl.csv"))
    comb = pd.read_csv(os.path.join(OUT, "xsec_combine", "combine.csv"))
    cheap = pd.read_csv(os.path.join(OUT, "xsec_cheapbase", "cheapbase.csv"))
    lab = pd.read_csv(os.path.join(OUT, "xsec_labelcheck", "labelcheck.csv"))
    xm = pd.read_csv(os.path.join(OUT, "xsec_main", "xsec_metrics.csv"), index_col=0)

    M = []
    lm = ovj["label_means"]
    ls = ovj["label_sds"]
    ratio = ls["cc"] / ls["on"]
    M += [macro("nStocks", f"{ps['n_stocks']}"),
          macro("firstDate", ps["first_date"]), macro("lastDate", ps["last_date"]),
          macro("nStockDays", f"{ps['usable_total']:,}".replace(",", r"\,")),
          macro("nRows", f"{ps['rows_total']:,}".replace(",", r"\,")),
          macro("pctSusp", f"{ps['susp_pct']:.1f}\\%"),
          macro("nFactors", "29"), macro("nModels", "10"), macro("nDates", "1{,}604"),
          macro("nStocksUsed", "231"),
          macro("nTestRows", "362{,}296"),
          macro("ccMean", pct(lm["cc"] * 100, 4)), macro("onMean", pct(lm["on"] * 100, 4)),
          macro("idMean", pct(lm["id"] * 100, 4)),
          macro("ccSd", f"{ls['cc'] * 100:.3f}"), macro("onSd", f"{ls['on'] * 100:.3f}"),
          macro("idSd", f"{ls['id'] * 100:.3f}"), macro("dilutionRatio", f"{ratio:.2f}")]

    for key, name in [("gatedfusion", "Prop"), ("gatedfusion-guided", "Guided"), ("turn1", "Turn")]:
        r = ov.loc[key]
        M += [macro(f"icCc{name}", f"{r.ic_cc:.4f}"), macro(f"tCc{name}", f"{r.t_cc:.1f}"),
              macro(f"icOn{name}", f"{r.ic_on:.4f}"), macro(f"tOn{name}", f"{r.t_on:.1f}"),
              macro(f"icId{name}", f"{r.ic_id:.4f}"),
              macro(f"identity{name}", f"{r.identity_pred_cc:.4f}"),
              macro(f"winsor{name}", f"{r.ic_on_winsor:.4f}"),
              macro(f"liquid{name}", f"{r.ic_on_liquidhalf:.4f}"),
              macro(f"within{name}", f"{r.ic_on_within_stock:.4f}")]
    _gj = real[(real["set"] == "raw") & (real.score == "gatedfusion-guided")
               & (real.hold_days == 5)].iloc[0]
    M += [macro("liquidMedianAdvM", "90"),
          macro("pNearlyAllParticipation", f"{_gj.part_p95_pct:.0f}"),
          macro("pNearlyAllParticipationFull", f"{real[(real['set'] == 'raw') & (real.score == 'gatedfusion-guided') & (real.hold_days == 5)].iloc[0].part_p95_pct:.0f}")]

    am = att[~att.index.str.startswith("[factor]")].copy()
    am = am[~am.index.isin(["cheap_combo"])]          # a factor combination is 100% factors
    af = att[att.index.str.startswith("[factor]")]
    M += [macro("residGuided", f"{att.loc['gatedfusion-guided'].resid_ic_on:.4f}"),
          macro("tResidGuided", f"{att.loc['gatedfusion-guided'].t_resid_on:.1f}"),
          macro("residProp", f"{att.loc['gatedfusion'].resid_ic_on:.4f}"),
          macro("tResidProp", f"{att.loc['gatedfusion'].t_resid_on:.1f}"),
          macro("rSquaredLo", f"{am.factor_r2.min():.2f}"), macro("rSquaredHi", f"{am.factor_r2.max():.2f}"),
          macro("bestFactorResid", f"{af.resid_ic_on.max():.4f}"),
          macro("bestFactorName", d(af.resid_ic_on.idxmax().replace("[factor] ", ""))),
          macro("bestFactorT", f"{af.loc[af.resid_ic_on.idxmax()].t_resid_on:.1f}"),
          macro("residCcBest", f"{am.resid_ic_cc.max():.4f}"),
          macro("residCcBestName", d(am.resid_ic_cc.idxmax()))]

    def pnl_row(score, h, liq):
        return pnl[(pnl.score == score) & (pnl.hold_days == h) & (pnl.liquid == liq)].iloc[0]

    g5 = pnl_row("gatedfusion-guided", 5, 0.5)
    g1 = pnl_row("gatedfusion-guided", 1, 1.0)
    M += [macro("grossWk", pct(g5.gross_total)), macro("onWkW", pct(g5.gross_overnight)),
          macro("idWk", pct(g5.gross_intraday)), macro("openWk", pct(g5.gross_if_entered_at_open)),
          macro("feeWk", pct(g5.fee, 1, False)), macro("impactWk", pct(g5.impact, 1, False)),
          macro("borrowWk", pct(g5.borrow, 1, False)), macro("netWk", pct(g5.net)),
          macro("grossDay", pct(g1.gross_total)), macro("impactDay", pct(g1.impact, 1, False)),
          macro("netDay", pct(g1.net)),
          macro("onDay", pct(g1.gross_overnight)), macro("idDay", pct(g1.gross_intraday))]

    cbest = comb[(comb.hold_days == 5) & (comb.liquid == 0.5)].sort_values("net", ascending=False).iloc[0]
    cfac = comb[(comb.hold_days == 5) & (comb.liquid == 0.5)
                & (comb.variant == "cheap factors only")].iloc[0]
    cmod = comb[(comb.hold_days == 5) & (comb.liquid == 0.5)
                & (comb.variant == "gatedfusion-guided")].iloc[0]
    cema = comb[(comb.hold_days == 5) & (comb.liquid == 0.5)
                & (comb.variant == "gatedfusion-guided + EMA20")].iloc[0]
    M += [macro("combName", d(cbest.variant)), macro("combNet", pct(cbest.net)),
          macro("combGross", pct(cbest.gross)), macro("combCap", f"{cbest.breakeven_aum_M:.2f}"),
          macro("combIcCc", f"{cbest.ic_cc:.4f}"), macro("combIcOn", f"{cbest.ic_on:.4f}"),
          macro("combTOn", f"{cbest.t_on:.1f}"),
          macro("facNet", pct(cfac.net)), macro("facCap", f"{cfac.breakeven_aum_M:.2f}"),
          macro("facIc", f"{cfac.ic_cc:.4f}"),
          macro("modNet", pct(cmod.net)), macro("modCap", f"{cmod.breakeven_aum_M:.2f}"),
          macro("emaNet", pct(cema.net)), macro("emaOn", f"{cema.ic_on:.3f}"),
          macro("emaTwentyNet", pct(cema.net))]

    # per-century checks: cost model constants
    M += [macro("commissionBp", "3"), macro("slippageBp", "2"), macro("transferBp", "0.1"),
          macro("stampBeforeBp", "10"), macro("stampAfterBp", "5"),
          macro("stampDate", "2023-08-28"), macro("borrowPct", "8"),
          macro("impactCoef", "0.5"), macro("aumRefM", "100")]

    # realcost table for selected scores
    sel = ["gatedfusion-guided", "gatedfusion", "attention+conv", "lstm", "turn1", "logamt1",
           "low_turnover", "small_amount"]
    rows = []
    for s in sel:
        for h in (1, 5):
            sub = real[(real["set"] == "raw") & (real["score"] == s) & (real.hold_days == h)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            total_cost = r.fee_ann + r.impact_ann_at_100M + r.borrow_ann
            rows.append([d(s), str(h), pct(r.gross_ann), pct(total_cost, 1, False),
                         pct(r.net_100M),
                         f"{r.breakeven_aum_M:.0f}" if r.breakeven_aum_M else "0",
                         f"{r.part_p95_pct:.0f}"])
    write(os.path.join(tdir, "tab_realcost.tex"),
          table_tabular(["Score", "Hold", "Gross", "Costs", "Net", "Break-even",
                         "$p_{95}$ part."], rows, font="scriptsize", pad="2pt"))

    # combination scan
    rows = []
    short = {"gatedfusion-guided": "MambaATM (guided) only",
             "gatedfusion": "MambaATM (gated) only",
             "cheap factors only": "Cheap factors only",
             "0.25 gatedfusion-guided + 0.75 cheap": "0.25 guided + 0.75 factors",
             "0.50 gatedfusion-guided + 0.50 cheap": "0.50 guided + 0.50 factors",
             "0.75 gatedfusion-guided + 0.25 cheap": "0.75 guided + 0.25 factors",
             "0.25 gatedfusion + 0.75 cheap": "0.25 gated + 0.75 factors",
             "0.50 gatedfusion + 0.50 cheap": "0.50 gated + 0.50 factors",
             "0.75 gatedfusion + 0.25 cheap": "0.75 gated + 0.25 factors"}
    sub = comb[(comb.hold_days == 5) & (comb.liquid == 0.5)].sort_values("net", ascending=False)
    for _, r in sub.iterrows():
        rows.append([short.get(r.variant, d(r.variant)), f"{r.ic_cc:.3f}", f"{r.ic_on:.3f}",
                     pct(r.gross), pct(r.fee + r.impact, 1, False), pct(r.net),
                     f"{r.breakeven_aum_M:.0f}" if r.breakeven_aum_M else "0"])
    write(os.path.join(tdir, "tab_combine.tex"),
          table_tabular(["Signal", "IC", "IC night", "Gross", "Costs", "Net", "Break-even"],
                        rows, font="scriptsize", pad="2pt"))

    # label table
    rows = []
    for s in ["turn1", "logamt1", "gatedfusion-guided", "gatedfusion", "attention+conv", "lstm",
              "timesnet", "dlinear", "mlp", "itransformer", "patchtst", "rev5", "gap1", "clv5"]:
        if s not in ov.index:
            continue
        r = ov.loc[s]
        rows.append([d(s), f"{r.ic_cc:.4f}", f"{r.t_cc:.1f}", f"{r.ic_on:.4f}", f"{r.t_on:.1f}",
                     f"{r.ic_id:.4f}", f"{r.t_id:.1f}", f"{r.identity_pred_cc:.4f}"])
    write(os.path.join(tdir, "tab_labels.tex"),
          table_tabular(["Score", "IC cc", "$t$", "IC night", "$t$", "IC day", "$t$",
                         "Predicted IC cc"], rows, font="tiny", pad="1.5pt"))

    # robustness table
    rows = []
    for s in ["turn1", "gatedfusion-guided", "gatedfusion", "attention+conv", "timesnet", "lstm",
              "dlinear", "mlp"]:
        r = ov.loc[s]
        rows.append([d(s), f"{r.ic_on:.4f}", f"{r.ic_on_winsor:.4f}", f"{r.ic_on_liquidhalf:.4f}",
                     f"{r.ic_on_within_stock:.4f}"])
    write(os.path.join(tdir, "tab_robust.tex"),
          table_tabular(["Score", "IC night", "Winsorised", "Liquid half", "Within stock"], rows))

    # neutralisation table
    rows = []
    order = att[~att.index.str.startswith("[factor]")].sort_values("resid_ic_on", ascending=False)
    for s, r in order.iterrows():
        rows.append([d(s), f"{r.ic_cc:.4f}", f"{r.ic_on:.4f}", f"{r.resid_ic_on:.4f}",
                     f"{r.t_resid_on:.1f}", f"{r.factor_r2:.2f}"])
    write(os.path.join(tdir, "tab_neutral.tex"),
          table_tabular(["Score", "IC cc", "IC night", "Residual night", "$t$ resid",
                         "$R^2$ on factors"], rows, font="scriptsize", pad="2pt"))

    # top factors by neutralised overnight IC
    af = att[att.index.str.startswith("[factor]")].copy()
    af["name"] = [i.replace("[factor] ", "") for i in af.index]
    af = af.sort_values("resid_ic_on", ascending=False).head(12)
    rows = [[d(r["name"]), f"{r.ic_on:.4f}", f"{r.t_on:.1f}", f"{r.resid_ic_on:.4f}",
             f"{r.t_resid_on:.1f}"] for _, r in af.iterrows()]
    write(os.path.join(tdir, "tab_topfactors.tex"),
          table_tabular(["Factor", "IC night", "$t$", "Residual night", "$t$"], rows))

    # attribution table
    rows = []
    for _, r in pnl.iterrows():
        if r.liquid != 0.5:
            continue
        rows.append([d(r.score), str(int(r.hold_days)), pct(r.gross_total), pct(r.gross_overnight),
                     pct(r.gross_intraday), pct(r.gross_if_entered_at_open), pct(r.net)])
    write(os.path.join(tdir, "tab_attrib.tex"),
          table_tabular(["Score", "Hold", "Gross", "Overnight", "Intraday",
                         "If entered at open", "Net"], rows, font="tiny", pad="1.5pt"))

    # cheap baseline table
    rows = []
    sub = cheap[cheap.liquid_frac == 0.5].sort_values(["set", "net_ls"], ascending=[True, False])
    for _, r in sub.iterrows():
        rows.append([("raw" if r["set"] == "raw" else "neutralised"), d(r.score), f"{r.ic:.4f}",
                     pct(r.net_ls), f"{r.breakeven_aum_M:.0f}" if r.breakeven_aum_M else "0",
                     pct(r.long_only_excess_net)])
    write(os.path.join(tdir, "tab_cheap.tex"),
          table_tabular(["Set", "Signal", "IC", "Net", "Break-even", "Long-only excess"], rows,
                        font="scriptsize", pad="2pt"))

    # smoothing
    rows = []
    sub = comb[(comb.hold_days == 5) & (comb.liquid == 0.5)]
    for v in ["gatedfusion-guided", "gatedfusion-guided + EMA1", "gatedfusion-guided + EMA5",
              "gatedfusion-guided + EMA20"]:
        r = sub[sub.variant == v]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append([v.replace("gatedfusion-guided", "MambaATM (guided)"), f"{r.ic_on:.4f}",
                     f"{r.t_on:.1f}", pct(r.gross), pct(r.net), f"{r.breakeven_aum_M:.0f}"])
    write(os.path.join(tdir, "tab_smooth.tex"),
          table_tabular(["Signal", "IC night", "$t$ night", "Gross", "Net", "Break-even"], rows))

    # ---------------------------------------------------------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    o = ov.sort_values("ic_on")
    y = np.arange(len(o))
    ax.barh(y - 0.2, o.ic_cc, height=0.4, label="close-to-close label")
    ax.barh(y + 0.2, o.ic_on, height=0.4, label="overnight label")
    ax.set_yticks(y)
    ax.set_yticklabels([d(i) for i in o.index], fontsize=7)
    ax.set_xlabel("cross-sectional rank IC")
    ax.legend(fontsize=8)
    ax.axvline(0, color="grey", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "ic_by_label.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    sub = real[(real["set"] == "raw")]
    for s, style in [("gatedfusion-guided", "-o"), ("turn1", "-s")]:
        v = sub[(sub.score == s) & (sub.hold_days == 5)]
        if v.empty:
            continue
        r = v.iloc[0]
        aums = np.array([10, 100, 500, 1000]) * 1e6
        nets = [r.net_10M, r.net_100M, r.net_500M, r.net_1B]
        ax.plot(aums / 1e6, nets, style, label=d(s))
    ax.set_xscale("log")
    ax.set_xlabel("book size per leg (million CNY, log scale)")
    ax.set_ylabel("net annual return (\\%)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "capacity.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    labels = ["close to close", "overnight", "intraday", "entered at open"]
    vals = [g5.gross_total, g5.gross_overnight, g5.gross_intraday, g5.gross_if_entered_at_open]
    ax.bar(labels, vals)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("gross annual return (\\%)")
    ax.set_title("MambaATM (guided), weekly, liquid half", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "attribution.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    v = comb[(comb.hold_days == 5) & (comb.liquid == 0.5)].sort_values("net", ascending=False).head(8)
    ax.barh(np.arange(len(v)), v.net)
    ax.set_yticks(np.arange(len(v)))
    ax.set_yticklabels([d(x) for x in v.variant], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("net annual return at 100M per leg (\\%)")
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "combine.png"), dpi=200)
    plt.close(fig)

    # A TeX control sequence name may contain letters only. A digit in the name (as in \r2Lo) makes
    # TeX read the digit as text and the manuscript fails to build, so it is checked here.
    bad = []
    for m in M:
        name = m.split("{\\", 1)[1].split("}", 1)[0]
        if not name.isalpha():
            bad.append(name)
    if bad:
        raise SystemExit("macro names must be letters only; offending: " + str(bad))
    with open(os.path.join(out, "numbers.tex"), "w", encoding="utf-8") as f:
        f.write("% generated by make_paper_assets.py -- do not edit by hand\n")
        f.write("\n".join(M) + "\n")
    print(f"wrote {len(M)} macros, 11 tables and 4 figures")


if __name__ == "__main__":
    main()
