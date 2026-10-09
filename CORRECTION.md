# Correction of the earlier version of this work

This repository holds the reworked version of the MambaATM study. The earlier version, submitted to
*Evolving Systems* as Ms. No. EVOS-D-26-02015, reported results that the code and the data do not
support. The review identified the discrepancy; this note records it in full so that anyone who
encounters the earlier release knows what changed and why.

## The architecture that was described was not the architecture that ran

The manuscript described attention *guiding the state update* of the state-space branch. The code
computed a sigmoid gate over the two branch outputs and mixed them. Worse, in the submitted
configuration the sequence-model slot was filled by `test_correctness(q, k, v, n)` from the
authors' `ssd_minimal.py` — a numerical correctness check whose discretisation step and state
transition are drawn from a fixed seed on every forward pass — so the proposed model held **zero**
learnable parameters, while its own baselines instantiated a real Mamba2 block. The proposed model
was structurally weaker than the models it was compared against.

The rework implements both mechanisms: the gate that the code actually used, and the guided
modulation that the text claimed, with zero-initialised projections so that the guided variant
starts as the plain branch. Section 4 of the reworked manuscript and Table 1 list the differences.

## The causality test did not pass

The manuscript stated that every model passed a causality test with a change of exactly zero before
the cutoff. Measured on the submitted class, the change was 0.004361 at the submitted configuration
and 0.006600 in the configuration `main.py` used. The cause is `nn.Conv1d(kernel_size=3, padding=1)`
in both attention classes: symmetric padding lets position *t* read *t+1*. The reworked models pass
the same test with a change of exactly zero.

## The dataset that was released does not match the dataset that was described

`Stock 1.csv` to `Stock 4.csv` on the linked data repository hold 500 rows, not the 2,435 described;
258 columns, of which 248 (`extra_feature_*`) are identically zero in every row and were dropped by
the submitted loader only for `日期` and `涨跌幅`; the period is 2023-01-10 to 2025-02-07, not
2015 to 2025; the prices are per-column z-scores over the whole sample, not values in $[0, 1]$, so
test-period statistics entered the training inputs through the standardisation; and the return column
does not agree with the price columns of the same file —
corr(quoted return, price-implied return) is $-0.041$, $+0.032$, $-0.148$ and $-0.115$ for the four
stocks, with mean absolute differences of 0.42, 0.49, 0.17 and 1.39 against returns whose own scale
is about 0.02. The four assets are not named, so the experiment cannot be repeated on them.

Those four files are byte-identical to `000001`, `000006`, `000007` and `000008` of the processed
folder of the project, which is how the defect was traced. The reworked study uses the panel those
files were derived from: 232 named A-share stocks, forward-adjusted prices, 2015-01-06 to
2025-12-31, screened, with the return column agreeing with the price columns to machine precision
(mean absolute difference $5\times10^{-17}$).

## What the reworked experiments find

On the rebuilt panel the nine learned models, including DLinear, PatchTST, iTransformer and
TimesNet, lie within about two percentage points of accuracy around a majority-class rate of 0.53.
MambaATM is the most accurate of them, the difference is smaller than the cross-sectional dispersion
across the 232 stocks, it does not survive a Holm-corrected Wilcoxon test against the stronger
baselines, and its equal-weight portfolio return is dominated by turnover and cost. The
attention-guided variant is not better than the plain gate.

The earlier reported accuracies of 0.60 to 0.85 are not reproducible on data that passes the
integrity checks, and the reworked manuscript states the negative result instead.

## Files

* `code/` — the models, the walk-forward pipeline, the trading simulator, the four audit scripts and
  the scripts that generate every table, figure and number in the manuscript.
* `code/panel/` — the panel builder and the screening rule, so the data can be rebuilt from source.
* `data/` — the screened panel and the same panel before screening.
* `EVIDENCE.md` — the measurements behind every claim on this page, with the script that produces it.
* `results/` — the per-stock and per-fold metrics, the saved predictions and the run summaries.

The earlier code is kept in `code/legacy_amt.py`, verbatim, so that the behaviour described above
stays reproducible.
