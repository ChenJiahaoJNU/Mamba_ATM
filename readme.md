# MambaATM: correction and rework

This repository carries the rework of a manuscript rejected by *Evolving Systems*
(Ms. No. EVOS-D-26-02015). It exists because the submitted version reported results that its own
code and data do not support. Everything here was recomputed from the raw panel rather than edited,
and the twelve tables, the six figures and the response to the reviewers are all generated from one
set of result files in `results/`.

The manuscript itself is not in this repository. It is under review, and its text is not needed to
reproduce anything: every number it quotes is in `results/` or in the generated table bodies, and the
table below maps each of them to the script that writes it. The text files that only exist to typeset
the manuscript, which are the source, the compiled PDF, the class, the bibliography style files and
the macro file, are deliberately absent for the same reason.

## What was wrong

Three defects were found in the submitted version and each is measured in `EVIDENCE.md`.

The submitted dataset held four unnamed stocks with 2,435 rows each, of which 248 of 258 feature
columns were constant, and its quoted return column was uncorrelated with its own price columns
(between $-0.15$ and $+0.03$). The rework rebuilds the panel from the source instead.

The submitted model class constructed a residual projection inside `forward`, so that projection was
never trained, and the audit that counts trainable parameters returned zero. It also used symmetric
padding in a depthwise convolution, which let position $t$ read $t+1$. The causality regression test
in `code/verify_zoo.py` fails the submitted class with a change of $4\times10^{-3}$ to
$7\times10^{-3}$ and passes the models used here with a change of exactly zero.

The accuracies reported in the submitted version, between 0.60 and 0.85, cannot be reproduced on data
that passes the integrity checks. `CORRECTION.md` states what was corrected and `EVIDENCE.md` records
how each defect was measured.

## What the rework finds

The manuscript reports two results on one panel, and the second is what explains the first.

On the absolute daily direction of a single stock, with 232 named A-share stocks covering 2015-01-06 to
2025-12-31, 587,751 usable stock-days and five expanding-window walk-forward folds:

* Ten models, which are MLP, LSTM, attention with a causal convolution, DLinear, PatchTST,
  iTransformer, TimesNet, the selective state-space branch on its own and the two variants of the
  proposed model, span 1.4 points of accuracy, from 0.507 to 0.521, against a majority-class rate of
  0.5285. **Not one of them reaches that rate.** DLinear is the most accurate at 0.5213 and MambaATM
  is fifth of ten on mean rank at 0.5159.
* The guided state update, which the earlier version described as the defining mechanism of the
  model, is implemented here as a separate variant. It changes accuracy by $+0.001$ points and Sharpe
  by $-0.03$, so the architectural claim that motivated the earlier version is not supported.
* The majority-class rate is above the accuracy of every learned model in eight of the nine test
  years, which is what the reported accuracies were measuring.
* At the portfolio level the equal-weight portfolio loses money at every cost level tested, including
  zero cost, so transaction costs are not the cause.

On the cross-sectional rank of the same panel, with the same features and the same models:

* Splitting the next-day return into an overnight leg and an intraday leg shows where the signal is.
  The rank information coefficient on the overnight leg is 2.04 times the close-to-close one for every
  model and factor tested, from 0.067 to 0.124 for the proposed model, and the identity relating the
  two coefficients holds to within a tenth for every signal, so the small close-to-close numbers are
  the overnight signal diluted by intraday noise rather than a weak signal.
* A twenty-nine-factor library nevertheless explains 0.69 to 0.89 of the cross-sectional variance of
  the ten models. The residual overnight coefficient of the proposed model is 0.0047, against 0.0242
  for the best single factor, intraday range over twenty days, so the architecture adds little once
  the library is accounted for.
* With commission, time-varying stamp duty, square-root impact and borrow charged explicitly, the only
  configuration that survives is weekly rebalancing on the liquid half of the panel. The best
  combination nets 33.9 per cent a year at a capacity near 526 million CNY per leg, and three quarters
  of its weight is on cheap factors rather than on the model. Daily rebalancing of the same signal
  loses 102.4 per cent a year, because impact alone is 169 per cent against a gross of 115 per cent.

So the target, and not the architecture, decides the answer, and neither target makes the architecture
the source of the return. Both halves are reported as measured.

## Scope

In this repository. The code that builds and screens the panel, the ten model implementations, the
walk-forward pipeline with its four audits, the cross-sectional target and the twenty-nine-factor
library, the cost model with impact and borrow, every result file those produce, and the manuscript
with its tables and figures. `VERIFICATION.md` records each check and the command behind it.

Not in this repository. The market data, which is 123 MB of CSV and is hosted as a Hugging Face
dataset (see `data/README.md`). The trained model weights, which are not needed, because the
predictions they produce are in `results/main/preds.npz` and every table and figure is generated from
that file rather than from a re-training run. And the archived versions of the earlier, withdrawn
manuscript, except where `CORRECTION.md` and `EVIDENCE.md` quote them to document what was wrong.

## Layout

```
code/               all analysis code; code/panel/ builds and screens the panel
code/smoke_test.py  one command that checks the whole chain on a few stocks
results/            every result file the tables and figures are generated from
data/               the panel itself is not stored here, see data/README.md
tables/             the twelve LaTeX table bodies
figs/               the six figures
CORRECTION.md       what was wrong in the submitted version and what changed
EVIDENCE.md         the measurements behind every statement in CORRECTION.md
VERIFICATION.md     every check, with the command that produced it
MERGE_NOTES.md      how the two halves of the study were combined, and the corrections made then
```

`VERIFICATION.md` is the place to start if you want to check a claim. Each row of its tables names
the command that produces the number, and every one of those commands is in `code/`.

## Paper to code map

Every table and figure of the manuscript, and the script that writes it.

| manuscript element | produced by |
|---|---|
| Table 1, the panel | `code/make_tables.py` from `results/main/panel_stats.json` |
| Table 2, the claim against the implementation | hand-written, `tables/tab_claim.tex` |
| Table 3, model size and wall-clock time | `code/make_tables.py` from `results/main/sizes.json` |
| Table 4, the main absolute-direction result | `code/make_tables.py` from `results/main/summary.json` |
| Table 5, accuracy fold by fold | `code/make_tables.py` from `results/main/per_fold.csv` |
| Table 6, the beats and loses counts | `code/make_tables.py` from `results/main/summary.json` |
| Table 7, the equal-weight portfolio at four cost levels | `code/make_tables.py` from `results/main/portfolio.json` |
| Table 8, Friedman and Holm-corrected Wilcoxon | `code/make_tables.py` from `results/main/stats_tests.json` |
| Table 9, the gate and the guidance ablation | `code/make_tables.py` from `results/main/summary.json` |
| Table 10, window length, seed and per-year stability | `code/make_tables.py` from `results/seq20`, `seq120`, `seed0sub`, `seed7`, `seed13` |
| Table 11, the trading metrics of the long-flat rule | `code/make_tables.py` from `results/main/summary.json` |
| Table 12, the three labels | `code/make_paper_assets.py` from the factor and score archives |
| Table 13, the robustness checks on the overnight coefficient | `code/make_paper_assets.py` from `xsec_overnight` |
| Table 14, neutralisation against the factor library | `code/make_paper_assets.py` from `xsec_main/xsec_neutralised.csv` |
| Table 15, the strongest individual factors | `code/make_paper_assets.py` from `xsec_attrib/attrib_neutral.csv` |
| Table 16, weekly net returns against cheap factors | `code/make_paper_assets.py` from `xsec_cheapbase/cheapbase.csv` |
| Table 17, the cost decomposition and break-even size | `code/make_paper_assets.py` from `xsec_main/xsec_realcost.csv` |
| Table 18, where the return comes from | `code/make_paper_assets.py` from `xsec_attrib/attrib_pnl.csv` |
| Table 19, smoothing the signal | `code/make_paper_assets.py` from `xsec_combine/combine.csv` |
| Table 20, mixing the model with cheap factors | `code/make_paper_assets.py` from `xsec_combine/combine.csv` |
| Fig. 1, the task setting | hand-drawn, `figs/task.png` |
| Fig. 2, the architecture | hand-drawn, `figs/framework.png` |
| Fig. 3, accuracy per stock against the reference | `code/make_figs.py` |
| Fig. 4, the portfolio against the cost level | `code/make_figs.py` |
| Fig. 5, the equity curves | `code/portfolio.py` |
| Fig. 6, accuracy by test year | `code/make_figs.py` |
| Fig. 7, the gate and the guidance with dispersion | `code/make_figs.py` |
| Fig. 8, information coefficient by label | `code/make_paper_assets.py` |
| Fig. 9, where the weekly return comes from | `code/make_paper_assets.py` |
| Fig. 10, capacity against book size | `code/make_paper_assets.py` |
| Fig. 11, the combination weight scan | `code/make_paper_assets.py` |

Three items are not generated from the result files, and the manuscript says so where it uses them.
`tables/tab_claim.tex` is maintained by hand, so if the architecture changes, that table has to be
edited rather than regenerated, and the four rows whose facts a script does check are the causality
test, the parameter count, the forward-pass module check and the standardiser check, all in
`code/verify_zoo.py`. `figs/task.png` and `figs/framework.png` are schematics drawn by hand.

The key results and the script that produces each.

| result | produced by |
|---|---|
| the four protocol audits | `code/verify_zoo.py` |
| panel integrity, return against price | `code/panel_audit.py`, `code/panel_stats.py` |
| ten models on five walk-forward folds, absolute direction | `code/run_panel.py`, 144 minutes on one RTX 3060 |
| the equal-weight portfolio and its cost sensitivity | `code/portfolio.py` |
| Friedman, mean ranks, Holm-corrected Wilcoxon | `code/stats_tests.py` |
| accuracy by test year against the majority class | `code/by_year.py` |
| the ten models on the cross-sectional target | `code/run_xsec.py`, about 2.5 hours |
| the overnight and intraday decomposition, and its robustness | `code/xsec_overnight.py` |
| the twenty-nine-factor library | `code/factors.py` |
| neutralisation of every score against the library | `code/xsec_neutralize.py` |
| attribution of each score to the library | `code/xsec_attrib.py` |
| the cheap-factor baseline, raw and neutralised | `code/xsec_cheapbase.py` |
| the combination weight scan | `code/xsec_combine.py` |
| the cost decomposition, break-even and capacity | `code/xsec_realcost.py` |
| every number the prose quotes | `code/emit_numbers.py` and `code/make_paper_assets.py` |
| the manuscript's format, anonymity and archive checks | `code/evos_checklist.py` |

## Data

The panel is hosted separately, as a Hugging Face dataset, because it is 123 MB of CSV.

    https://huggingface.co/datasets/Changahou/Mamba_ATM

It holds two directories. `adj/` is the screened panel the paper uses, which is 232 stocks with
610,988 rows. `adj_unscreened/` is the same panel before screening, which is 240 stocks, so the
screen can be re-run. Each file is one stock, named by its exchange code such as `000001.SZ.csv`, and
is UTF-8 with a byte-order mark and the exchange's own Chinese column headers.

To reproduce, place those two directories under `data/`:

```
data/adj/*.csv
data/adj_unscreened/*.csv
```

The screening rule is in `code/panel/screen_panel.py` and is stated in the paper. A stock is excluded
entirely if its adjusted series contains any daily move beyond 25 per cent, which happens where a
suspension meets the delisting process or where the adjustment factor breaks. Excluding the whole
stock rather than the offending row is deliberate, because dropping an interior row leaves the next
return spanning a two-day gap and manufactures exactly the spurious return that made the previous
panel unusable.

Note the survivorship screen. Stocks whose adjusted series contain an impossible move are removed,
and stocks that delisted during the sample are absent, which flatters every long-only rule including
buy and hold. The paper says so in its conclusion.

## Reproducing the paper

The environment is a conda environment with PyTorch. The direct dependencies are `torch`, `numpy`,
`pandas`, `scipy`, `matplotlib` and, for the panel builder only, `baostock`; the longer
`code/requirements.txt` is the frozen list of the environment the runs were made in. The rework was
run on one RTX 3060, and the state-space branch uses a pure-PyTorch stand-in for the fused kernels, so
no Triton and no CUDA toolkit are required to run it on a CPU.

Start with the smoke test. It reads a few stocks, builds the windows, makes the walk-forward splits,
trains one small model for one epoch and prints the accuracy next to the majority-class rate, so a
broken checkout fails in about a minute rather than three hours into the main run.

```bash
python code/smoke_test.py
```

It expects the panel under `data/adj`, and it says so if the panel is missing. Then the full chain:

```powershell
$py = "python"

# 1. the four protocol audits: shapes, parameter counts, causality, standardiser leakage
$env:VERIFY_OUT = "out\main\audit.json"; $py code\verify_zoo.py

# 2. panel statistics, which the panel table and the prose both read
$py code\panel_stats.py --panel data\adj --out out\main\panel_stats.json

# 3. the main run: 232 stocks x 10 models x 5 folds, about 144 minutes on one RTX 3060
$py code\run_panel.py --seq 60 --folds 5 --epochs 30 --seed 0 --workers 6 --out out\main

# 4. everything derived from the saved predictions
$py code\portfolio.py --preds out\main\preds.npz --out out\main --figs figs
$py code\stats_tests.py out\main\per_stock.csv --out out\main
$py code\by_year.py --preds out\main\preds.npz --out out\main\by_year.json

# 5. the numbers and the table bodies
$py code\emit_numbers.py  --dir out\main --panel data\adj --out numbers.tex
$py code\make_tables.py   --dir out\main --tables tables
$py code\make_figs.py     --dir out\main --figs figs
```

The cross-sectional half of the paper starts from step 3's saved scores:

```powershell
$py code\run_xsec.py --panel data\adj --seq 60 --out out\xsec_main
$py code\xsec_overnight.py --scores out\xsec_main\xsec_scores_all.npz --out out\xsec_overnight
$py code\xsec_neutralize.py --scores out\xsec_main\xsec_scores_all.npz --out out\xsec_main
$py code\xsec_attrib.py    --scores out\xsec_main\xsec_scores_all.npz --out out\xsec_attrib
$py code\xsec_cheapbase.py --scores out\xsec_main\xsec_scores_all.npz --rebalance 5 --out out\xsec_cheapbase
$py code\xsec_combine.py   --scores out\xsec_main\xsec_scores_all.npz --out out\xsec_combine
$py code\xsec_realcost.py  --scores out\xsec_main\xsec_scores_all.npz --panel data\adj `
    --neutral-scores out\xsec_main\xsec_neutral_scores.npz --out out\xsec_main
$py code\make_paper_assets.py --out out
```

`code\run_analysis.ps1` runs the analysis half in one go. `code\evos_checklist.py` checks the
manuscript against the journal's format rules and anonymity rules, builds it in a clean directory and
verifies the submission archive.

The six figures are not all generated the same way. Four are result figures drawn from `results/`
by `make_figs.py`, `portfolio.py` and `make_paper_assets.py`. One, `figs/sharpe_scatter.png`, is drawn
by `make_figs.py` but is not used in the manuscript. The remaining two, `figs/task.png` and
`figs/framework.png`, are schematics drawn by hand and are therefore not reproducible from the result
files.

The architecture schematic is also the one figure that has to be read with care. It is included as
the architecture *as it was described* in the earlier version of the work. It shows a
price-regression output and stop-loss and take-profit rules, none of which the evaluation reported
here uses, and the caption says so and points to the claim table that lists every difference between
that description and the implementation.

## Work that is in the repository but not in this manuscript

`code/` also holds a cross-sectional line of work, in `xsection.py`, `factors.py`, `costs.py` and the
`xsec_*.py` scripts. It studies the same panel under a different target and its results are not in
`results/`, so those scripts cannot be run from this repository alone. They are kept because they are
part of the project record.

## Licence

MIT, see `LICENSE`. The market data is retrieved from the public `baostock` interface and carries
whatever terms that source imposes.
