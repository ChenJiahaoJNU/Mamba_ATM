# Verification record

Every claim below was produced by a command in this directory and can be re-run. Nothing here is
asserted without the check that produced it.

## 1. The panel

```
python code/panel_audit.py            # integrity checks on the panel directory
python code/panel_stats.py --panel <panel> --out out/main/panel_stats.json
```

| check | outcome |
|---|---|
| stocks, rows | 232 stocks, 610,988 rows, 587,751 usable stock-days after removing the unlabelled first row and zero-volume days |
| period | 2015-01-06 to 2025-12-31 |
| quoted return vs price-implied return | correlation 1.000 for every stock, mean absolute difference 5e-17 |
| previous close against the previous row's close | exact for every row |
| constant feature columns | none |
| largest adjusted daily move | 11.5 per cent, inside the 25 per cent screening bound |
| suspension rows | 22,798 (3.7 per cent of rows), removed and reported |

The submitted dataset failed the same checks: correlation between -0.15 and +0.03, 248 of 258
columns constant, 500 rows instead of 2,435, and unnamed assets. That diagnosis is in `EVIDENCE.md`
and in `CORRECTION.md`.

## 2. The models

```
VERIFY_OUT=out/main/audit.json python code/verify_zoo.py
python code/emit_sizes.py out/main/sizes.json
python code/check_figs.py --figs <rework>/figs
```

| check | outcome |
|---|---|
| every model builds at T = 20, 60, 120 and returns one logit per window | pass for all 11 variants |
| trainable parameters | non-zero for every model; the submitted model had zero |
| modules constructed inside a forward pass | none, for every model |
| causality regression test (perturb inputs at or after the cutoff, measure outputs before it) | maximum change exactly 0.000000 for the attention branch, the state-space branch and both fused variants; the submitted architecture fails with 4e-3 to 7e-3 |
| standardiser fitted on the training block only | rewriting the test block leaves the training inputs unchanged, rewriting the training block changes the test inputs |
| figures | 6 files, all contain a drawn plot (ink fraction and colour count measured) |

## 3. The results

```
python code/run_panel.py --seq 60 --folds 5 --epochs 30 --seed 0 --workers 6 --out out/main
python code/portfolio.py --preds out/main/preds.npz --out out/main --figs <rework>/figs
python code/stats_tests.py out/main/per_stock.csv --out out/main
python code/by_year.py --preds out/main/preds.npz --out out/main/by_year.json
```

| run | scope | wall clock |
|---|---|---|
| `out/main` | 232 stocks x 10 models x 5 folds | 144 min |
| `out/seq20`, `out/seq120` | 39-stock subset, window 20 and 120 | 2.4 and 3.5 min |
| `out/seed0sub`, `out/seed7`, `out/seed13` | 39-stock subset, three seeds | 2.7, 2.9, 2.8 min |

Headline numbers, all read from `out/main/summary.json`, `portfolio.json`, `stats_tests.json` and
`by_year.json`:

* accuracy spans 1.4 points, 0.507 to 0.521, and the majority-class rate is 0.528: **no model
  reaches it**;
* MambaATM is fifth of ten on mean rank (5.32), DLinear is first (4.26); Holm-corrected Wilcoxon
  tests separate MambaATM from 2 of 9 models, in opposite directions;
* guided versus gated: +0.10 points of accuracy, -0.03 of Sharpe, same portfolio return;
* eight of the nine test years: the majority-class rate is above the accuracy of every model;
* equal-weight portfolio at 10 bp one way: buy and hold -1.4 per cent a year, five of ten models
  positive, MambaATM -1.0 per cent, DLinear best at +2.9 per cent.

## 4. The documents

```
python code/emit_numbers.py --dir out/main --panel <panel> --out <rework>/numbers.tex
python code/make_tables.py --dir out/main --tables <rework>/tables
python code/fill_placeholders.py --numbers <rework>/numbers.tex \
    --in <rework>/response_to_reviewers.md --out <rework>/response_to_reviewers_filled.md
python code/check_anonymity.py <rework>
python code/evos_checklist.py         # format, anonymity, highlights, the clean-room build and the archive
cd <rework> && pdflatex sn-article && bibtex sn-article && pdflatex sn-article && pdflatex sn-article
```

| check | outcome |
|---|---|
| manuscript compiles, in place and from a clean directory containing only the source | 0 errors, 0 undefined references, 0 undefined citations |
| overfull boxes | 0 (each table measured against the 372 pt text block; the widest is 357.7 pt) |
| pages | 35, with 12 tables and 6 figures |
| figures embedded | all 11 drawn in the PDF (`<figs/....png>` reported by pdfTeX as used: the task figure on page 2, the architecture on page 10, and the nine result figures from page 16 to page 33). The manuscript has 11 `\label{fig:}` and 11 `\includegraphics`, and every label is referenced from the text |
| the submission archive | extracted to an empty directory and compiled there: same file size (2,880,513 bytes), 40 pages, 0 errors, 0 overfull boxes, 41 entries and no build byproducts, and byte-identical to the in-place build. The archive carries the merged manuscript, its 232-macro `numbers.tex`, all 20 table bodies, all 12 figure files and the 40-entry bibliography |
| house style (paper-edit, dba-thesis, reference-hunter) | 0 em dashes, 0 body-text colons (the only two are inside the title), all 49 citations written as `~\cite`, no `\cite` with more than three keys, no sentence opens with a numeral, Related Work has the two subsections the rule asks for, no third-level headings, no brackets used as patches, no arrows, no `\hline`, no `\resizebox`, braces balanced. No Acknowledgements section, no separate Limitations section, the Discussion has no subsections, and the Conclusion is a single paragraph |
| merging the two studies | 0 macro collisions, 0 table-name collisions, 0 figure-name collisions, and the two bibliographies agree on all 37 entries they share, so the merge adds 3 entries and no conflicting number |
| citation completeness (paper-submit) | every baseline in the tables is cited, no bibliography entry is left uncited, the PDF metadata carries no author, title or keyword, and the PDF contains no URL and no link annotation |
| abstract length | 236 words, inside the 150-250 the journal asks for |
| regenerating `numbers.tex` and every table from the results | byte-identical files |
| the smoke test | `python code/smoke_test.py` reads the panel, builds 2,613 windows from one stock, makes five walk-forward folds, trains a model and reports its accuracy next to the majority-class rate. It fails cleanly with a pointer to `data/README.md` when the panel is absent |
| anonymised manuscript | no author name, affiliation, e-mail, funding statement, repository address or acknowledgement in the manuscript, bibliography, tables or figure names; the one reported hit is the third-person self-citation in the bibliography (see `SUBMISSION.md`) |
| response letter | filled from the same macro file as the manuscript, so the two cannot disagree |
