# Evidence: why the MambaATM manuscript cannot be made consistent with its data

Collected in `_rework/` during the rework attempt. Every number below is reproduced by a script in
`_rework/code/`, run against the shipped dataset and the shipped repository.

## 1. The model that produced the reported numbers has no state-space component

`main.py` comments `AdvancedMambaAMT` out (lines 200-207) and evaluates
`("Attention-MambaAMT", "MambaAMT")` through `SelfAttentionMamba` instead. In that class:

* `__init__` sets `self.model = None` for `model_type="MambaAMT"`, because the value matches none
  of the `mamba` / `s4` / `mamba2_original` branches;
* `forward` (line 476) calls `test_correctness(q, k, v, n)` for the second branch. That function
  lives in `ssd_minimal.py`, is copyright Albert Gu and Tri Dao, and is the authors' correctness
  check for their SSD listing. It re-seeds the global RNG (`torch.manual_seed(42)`), draws its
  discretisation step and state transition from that seed on every forward pass, and takes its
  state inputs from the attention block's own q, k and v.

Measured: the sequence-model slot holds **0 learnable parameters**, the "second branch" holds
**0 learnable parameters**, while the corrected branch holds **30,094**.

By contrast the baselines `MLP-mamba2`, `Attention-mamba2` and `LSTM-mamba2` instantiate the real
`Mamba2Simple(d_state=32, expand=8, headdim=32)`. The proposed model was therefore structurally
weaker than its own baselines, which is why `MLP-mamba2` beats it in the submitted Tables 1 and 2.

## 2. The model fails the causality test the manuscript says it passes

Same test, same randomisation, perturbing inputs at positions >= 60 and checking positions < 60:

| model | max change before the cutoff | verdict |
|---|---|---|
| submitted `SelfAttentionMamba(MambaAMT)`, dim_q == dim_v | 0.004361 | FAILS |
| submitted, the configuration `main.py` uses | 0.006600 | FAILS |
| corrected `AdvancedMambaAMT` | 0.000000 | PASS |

Cause: `nn.Conv1d(kernel_size=3, padding=1)` in both classes, symmetric padding, so position t
reads t+1. The manuscript states that every model passes with a change of exactly zero.

Also: `x_proj = nn.Linear(...)` is constructed inside `forward`, so those weights are never
trained and are redrawn every call, including in eval mode.

## 3. The shipped dataset cannot support the experiment

`Stock 1.csv` to `Stock 4.csv`, from the Hugging Face link in the manuscript:

| property | manuscript | file |
|---|---|---|
| samples per stock | 2435 daily | **500** |
| features | 6 (open, close, high, low, volume, amount) | **258 columns** |
| period | 2015 to 2025 | **2023-01-10 to 2025-02-07** |
| prices | normalised to [0,1] | **per-column z-scores, range -1.9 to 3.1** |

Further:

* **248 of the 258 columns are `extra_feature_*` and are identically zero in every row.** The
  submitted loader drops only `日期` and `涨跌幅`, so 248 of its 256 input dimensions were a
  constant zero.
* The return column is **inconsistent with the price columns in the same file**:
  corr(close-to-close return implied by the price columns, `涨跌幅`) = -0.041, +0.032, -0.148,
  -0.115 for the four stocks, with mean absolute differences of 0.42, 0.49, 0.17 and 1.39 against
  returns whose own scale is about 0.02. Per-column z-scoring destroyed the arithmetic.
* The z-scores were computed over the whole 500-day sample, so test-period statistics enter the
  training inputs whichever split is used.
* **The four stocks are unnamed.** The reviewer noted this. It means the experiment cannot be
  repeated on the original assets even if the file were repaired.

Consequence, measured: no model beats the trivial references. On Stock 1, walk-forward with five
folds, pooled test blocks of 45 days, majority-class rate 0.5263 and buy-and-hold +19.22 percent
at a Sharpe of 0.90:

| model | accuracy | F1 | total return | Sharpe |
|---|---|---|---|---|
| MLP | 0.4561 | 0.496 | -4.45% | -0.26 |
| DLinear | 0.4912 | 0.529 | +1.12% | +0.16 |
| LSTM | 0.5351 | 0.036 | - | - |
| attention | 0.5044 | 0.017 | - | - |
| gated fusion with the real branch | 0.5351 | 0.086 | +2.32% | +0.90 |

The accuracies near the majority rate and the near-zero F1 come from the models collapsing onto
one class. A hyper-parameter sweep (hidden 32/64/128, 30 or 80 epochs, two learning rates) does
not change this: every MLP configuration lands between 0.4518 and 0.4956.

The single-feature rank correlations with the next-day return are -0.107, -0.094, -0.089 and
-0.131 at their largest, and the daily return autocorrelations are near zero except for Stock 3 at
+0.2518, which is implausibly high for a daily equity return and is itself a data warning sign.

## 4. What this means for the objective

The objective was to rewrite the manuscript so that it agrees with the code and with the data.
The code can be repaired, and has been: `_rework/code/ssm_branch.py` is a selective state-space
branch with input-dependent parameters, verified causal; `model_zoo.py` implements the gated
fusion honestly; the baselines run under pure-PyTorch stand-ins for the missing Triton kernels.

The data cannot be repaired. Its label column does not match its feature columns, most of its
columns are constant, its stated dimensions are wrong, its assets are unnamed, and under a
leak-free protocol nothing beats the majority class or buy-and-hold. No rewriting of the
manuscript can make its reported accuracies consistent with this dataset, and the experiments
cannot be repeated because the assets cannot be identified.

Resolving this needs a decision that is not mine to take:

1. supply the original raw panel and the stock codes, and the experiments can be redone properly;
2. authorise rebuilding the benchmark from a public source such as baostock, in which case the
   paper becomes a new study and overlaps the IJMLC submission;
3. treat this manuscript as unpublishable and report the code and data defects, either as part of
   the IJMLC submission or as a separate critical note.

## 5. Resolution (added after the search for the original data)

The raw material was never missing. It was in the project tree the whole time, and the shipped
dataset was a *processed* derivative of it:

* `代码\data_panel\` and its byte-identical copy `代码\data_panel_rawprice_backup\` hold 240 named
  A-share stocks, 2015-01-06 to 2025-12-30, 2,670 rows each, with the return column consistent with
  the price columns to machine precision. Turnover rate is identically zero in this source.
* `代码\data_panel_adj\` (232 stocks, screened) and `代码\data_panel_adj_unscreened\` (240 stocks) hold
  the same universe retrieved through baostock with forward-adjusted prices and a real turnover rate.
* `MambaATM\pre_data\00000X_processed_data.csv` is the source of the shipped dataset: `Stock 1.csv`
  to `Stock 4.csv` are byte-identical (MD5) to `000001`, `000006`, `000007` and `000008`. The 500-row,
  258-column, 248-zero-column, per-column z-scored files that blocked the rework are the output of a
  processing step applied to this panel, not the panel itself.

So option 1 of the list above was satisfied by the existing project data, and the rework continued on
`data_panel_adj`, which passes every check the shipped derivative fails (measured: correlation
between the quoted and the price-implied return 1.000, mean absolute difference $5\times10^{-17}$,
previous close equal to the previous close for every row, no constant feature column, largest
adjusted daily move 11.5 per cent).

The earlier conclusion of this file -- that the shipped dataset cannot support the experiment --
stands. What changed is that the data needed to run the experiment properly exists, so the
manuscript was rewritten on it rather than withdrawn.

