# MambaATM: A Predictive Decision-Making Framework for Automated Trading Using Mamba2 and Causal Attention

Code and results for the manuscript of the same name, submitted to *Evolving Systems*.

MambaATM couples Mamba2 (a selective state-space model) with causal attention and a
prediction-driven decision layer, so that long-range dependencies in financial time series are
captured at a cost that stays near linear in the sequence length. The repository contains the
model implementations, the training and evaluation pipeline over several random repeats, the
paired significance tests, and the trading simulator.

## Contents

Everything sits in one directory. The modules import each other by bare module name, so please
keep the flat layout when you run them.

| file | role |
|---|---|
| `main.py` | **entry point for the experiments.** Trains every model on every stock in `./pre_data` over `REPEAT_TIMES` repeats and writes one workbook per run |
| `models.py` | all model definitions: `MLP`, `CombinedMLP`, `SelfAttention`, `SelfAttentionMamba`, `LSTMModel`, `CombinedLSTM`, `AdvancedMambaAMT` |
| `trainer.py` | training loop, per-epoch logging, metric computation, workbook output |
| `data_utils.py` | panel loading, standardisation, binary and raw label construction, run logs |
| `stats_test.py` | paired t-test and related significance routines |
| `trading_simulator.py` | trading engine: costs, position limits, stop-loss and take-profit |
| `Main_trade.py` | standalone demonstration of the simulator on synthetic prices, see the warning below |
| `s4.py`, `mamba_simple.py`, `mamba2_simple_original.py`, `ssd_minimal.py` | the sequence cores and a numerical check of the SSD scan |
| `mamba2.py` | an alternative Mamba2 implementation kept for reference; not imported by the pipeline |
| `check_excel_structure.py` | aggregates `./out` into `./analysis/analysis_<timestamp>/`, builds the model comparison figures and a written report |
| `heatplot.py` | pairwise significance heatmaps |
| `metrics_summary.py` | collects every `*metrics*.xlsx` in a folder into one table |
| `results/` | the figures and workbooks reported in the manuscript |
| `requirements.txt` | the exact environment the results were produced in |

## Environment

The results were produced under the following environment. `requirements.txt` pins it exactly.

| component | version |
|---|---|
| Python | 3.10.14 (conda) |
| PyTorch | 2.1.1, CUDA 11.8 build |
| torchvision / torchaudio | 0.16.1 / 2.1.1 |
| mamba-ssm | 2.2.2, built against torch 2.1 + cu118 |
| causal-conv1d | 1.4.0, built against torch 2.1 + cu118 |
| triton | 2.1.0 |
| pytorch-lightning | 2.0.4 |
| einops | 0.8.0 |
| numpy / pandas / scipy / scikit-learn | 1.26.3 / 2.2.3 / 1.14.1 / 1.6.0 |
| matplotlib / openpyxl | 3.9.3 / 3.1.5 |
| wandb | 0.19.0 |
| GPU | NVIDIA RTX 3060, 12 GB |

`mamba-ssm` and `causal-conv1d` have no prebuilt wheels for this combination. Build them against
torch 2.1.1 with CUDA 11.8, for example

```
conda create -n mamba2 python=3.10.14 -y
conda activate mamba2
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install causal-conv1d==1.4.0
pip install mamba-ssm==2.2.2
pip install -r requirements.txt
```

An earlier version of these notes pointed at locally built wheel files and at an absolute path on
the machine that produced the results. Neither exists in this repository, and the commands above
replace them.

## Data

The panel used in the paper is openly available at
<https://huggingface.co/datasets/Changahou/Mamba_ATM>, as four CSV files named `Stock 1.csv` to
`Stock 4.csv`.

`main.py` reads whatever is in `./pre_data` and expects each file to be named
`<code>_processed_data.csv`, so put the four files there under those names:

```
mkdir pre_data
# download the four CSVs from the link above, then
cp "Stock 1.csv" pre_data/000001_processed_data.csv
cp "Stock 2.csv" pre_data/000002_processed_data.csv
cp "Stock 3.csv" pre_data/000003_processed_data.csv
cp "Stock 4.csv" pre_data/000004_processed_data.csv
```

Each CSV must carry the columns `日期` (date) and `涨跌幅` (return); both are required and the
loader stops on a file that lacks either. Every remaining column is used as a feature, which is why
the files ship with the price and volume columns plus a block of `extra_feature_*` columns. Files
are read as UTF-8 with a byte-order mark.

> **The loader will not tell you if the data is missing.** If `./pre_data` does not exist,
> `data_utils.py` silently creates the directory, fills it with random synthetic series and trains
> on those. The run finishes without an error and the numbers look plausible. Check that
> `./pre_data` holds your four files before you start, and check the row count in the log.

## Reproducing the results

```bash
python main.py                   # train and evaluate every model, writes ./out and ./logs
python check_excel_structure.py  # aggregate ./out into ./analysis/analysis_<timestamp>/
python heatplot.py               # pairwise significance heatmaps
python metrics_summary.py        # one table from every *metrics*.xlsx in a folder
```

`main.py` is configured at the top of the file. The settings behind the reported results are
`SEED = 42`, `REPEAT_TIMES = 5`, `TEST_SIZE = 0.25`, `TIME_SERIES_SPLIT = True`, and 100 epochs at
a learning rate of 0.001 with batch size 32. The split is chronological rather than random when
`TIME_SERIES_SPLIT` is set, which is what keeps the evaluation out of the future.

Outputs:

* `./out/<model>_repeat_<k>_<stock>.xlsx` — one workbook per model, repeat and stock, holding the
  test predictions and the metrics for that run.
* `./logs/experiment_log_<timestamp>.txt` — the run log, including data shapes and per-epoch
  losses.
* `./analysis/analysis_<timestamp>/` — aggregated tables, the comparison figures, a copy of the raw
  workbooks and a written `analysis_report.md`.

## About `Main_trade.py`

`Main_trade.py` generates its own synthetic price series under a fixed seed and two synthetic
prediction sets, then runs them through the trading simulator to show what the engine does with a
good and a bad signal. It is a demonstration of `trading_simulator.py`, not the backtest behind the
manuscript's trading figures, and it reads nothing from `./pre_data`. Use `main.py` for the
experiments.

## Results

`results/` holds the reported output: `accuracy_comparison.png` and `rmse_comparison.png` for the
prediction metrics, `heatmap_wilcoxon.png` for the pairwise tests, `all_models_metrics_summary.xlsx`
and `Return_Rate_results.xlsx` for the tables, and `trade_points/` with one trade-by-trade plot per
model and stock. The workbook of the same name at the top level is an earlier summary kept for
comparison; `results/all_models_metrics_summary.xlsx` is the one to read.

## Licence

MIT, see `LICENSE`. The dataset is redistributed through the Hugging Face link above.
