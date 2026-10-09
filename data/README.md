# The panel is not stored in this repository

It is 123 MB of CSV, so it is hosted as a Hugging Face dataset instead:

    https://huggingface.co/datasets/Changahou/MATM

Download it and place the two directories here, so that the paths the scripts expect are restored.

```
data/adj/*.csv              232 stocks, 610,988 rows, the screened panel the paper uses
data/adj_unscreened/*.csv   240 stocks, the same panel before screening, so the screen can be re-run
```

Each file is one stock, named by its exchange code such as `000001.SZ.csv`, encoded as UTF-8 with a
byte-order mark, with the exchange's own Chinese column headers:

| column | meaning |
|---|---|
| 日期 | trading date |
| 当日开盘 | open |
| 收盘价 | close, forward adjusted for splits and distributions |
| 最高价 | high |
| 最低价 | low |
| 成交量 | volume, in shares |
| 成交额 | turnover value, in yuan |
| 涨跌幅 | quoted close-to-close return |
| 换手率 | turnover rate |
| 昨收 | previous close |

The eight columns used as features by the models are open, close, high, low, volume, turnover value,
turnover rate and previous close. The quoted return is used to build the label, which is the sign of
the next trading day's close-to-close return.

The screening rule is in `../code/panel/screen_panel.py`. A stock is excluded entirely if its
adjusted series contains any daily move beyond 25 per cent. `adj_unscreened/` is kept so that the
screen can be re-run and re-checked: running `screen_panel.py` on it reproduces `adj/` exactly.

Two things to know before reusing the panel. Suspension days are still present as rows with zero
volume, and they are removed inside the pipeline rather than in the files, so 22,798 of the 610,988
rows are untradable days. And the screen is survivorship-biased, because stocks that delisted during
the sample are absent, which the paper states in its conclusion.
