# Assemble the release that accompanies the MambaATM resubmission.
#
# Layout:
#   README.md, LICENSE, .gitignore
#   code/panel/          the panel builder and the screening rule
#   code/                the rework: models, protocol, audits, statistics, tables, figures
#   data/adj/            the screened panel the paper uses (232 stocks)
#   data/adj_unscreened/ the same panel before screening (240 stocks), so the screen is reproducible
#   results/             every file the paper's tables and figures are generated from
#
# The release deliberately contains no author-identifying information in the manuscript-related files
# except the title page, which lives outside this package.

$ErrorActionPreference = "Stop"
$root = "D:\研究生学习资料\暨南大学-博士学习资料\论文\重投\MambaATM"
$w = "$root\_rework"
$rel = "$root\_release_MambaATM"

Remove-Item $rel -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path "$rel\code\panel", "$rel\data", "$rel\results" | Out-Null

Copy-Item "$w\code\*.py" "$rel\code\" -Force
Copy-Item "$w\code\*.ps1" "$rel\code\" -Force
Copy-Item "$w\README.md" "$rel\README.md" -Force
Copy-Item "$w\EVIDENCE.md" "$rel\EVIDENCE.md" -Force

Copy-Item "$root\_release_IJMLC\code\panel\download_panel_baostock.py" "$rel\code\panel\" -Force
Copy-Item "$root\_release_IJMLC\code\panel\screen_panel.py" "$rel\code\panel\" -Force
Copy-Item "$root\_release_IJMLC\code\panel\download_matched_universe.py" "$rel\code\panel\" -Force
Copy-Item "$root\_release_IJMLC\LICENSE" "$rel\LICENSE" -Force

Write-Output "copying the panel (about 130 MB, this takes a minute)"
Copy-Item "$root\代码\data_panel_adj" "$rel\data\adj" -Recurse -Force
Copy-Item "$root\代码\data_panel_adj_unscreened" "$rel\data\adj_unscreened" -Recurse -Force

foreach ($d in @("main", "seq20", "seq120", "seed7", "seed13")) {
    if (Test-Path "$w\out\$d") {
        New-Item -ItemType Directory -Force -Path "$rel\results\$d" | Out-Null
        Copy-Item "$w\out\$d\*" "$rel\results\$d\" -Recurse -Force -Exclude "preds.npz"
        Write-Output "  results/$d copied"
    }
}
Copy-Item "$w\out\main\preds.npz" "$rel\results\main\preds.npz" -Force -ErrorAction SilentlyContinue

# the manuscript-side artifacts, so a reader can see exactly what was submitted
foreach ($f in @("sn-article.tex", "numbers.tex", "sn-bibliography.bib", "sn-jnl.cls")) {
    if (Test-Path "$root\_mambaatm_snjnl\rework\$f") {
        Copy-Item "$root\_mambaatm_snjnl\rework\$f" "$rel\" -Force
    }
}
Copy-Item "$root\_mambaatm_snjnl\rework\tables" "$rel\tables" -Recurse -Force
Copy-Item "$root\_mambaatm_snjnl\rework\figs" "$rel\figs" -Recurse -Force

$size = (Get-ChildItem $rel -Recurse -File | Measure-Object Length -Sum).Sum
$files = (Get-ChildItem $rel -Recurse -File).Count
Write-Output ("release assembled: {0} files, {1:N0} MB" -f $files, ($size / 1MB))
