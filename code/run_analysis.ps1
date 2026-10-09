# Run the whole analysis and document pipeline from the finished results.
#
# Assumes the experiments have already been run into _rework\out\<tag>. Produces:
#   _rework\out\main\{audit.json,panel_stats.json,sizes.json,portfolio.json,stats_tests.json,by_year.json,tables\}
#   _mambaatm_snjnl\rework\{numbers.tex,tables\*,figs\*}
#   _mambaatm_snjnl\rework\response_to_reviewers_filled.md

$ErrorActionPreference = "Stop"
$root = "."
$w = "$root"
$m = "$root\_mambaatm_snjnl\rework"
$panel = "$root\data\adj"
$py = "C:\ProgramData\anaconda3\envs\rl1\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "$w\libs"

function Step($name) { Write-Output ""; Write-Output "########## $name ##########" }

Step "panel statistics"
& $py "$w\code\panel_stats.py" --panel $panel --out "$w\out\main\panel_stats.json"

Step "parameter counts"
& $py "$w\code\emit_sizes.py" "$w\out\main\sizes.json"

Step "protocol audits"
$env:VERIFY_OUT = "$w\out\main\audit.json"
& $py "$w\code\verify_zoo.py"
Remove-Item Env:\VERIFY_OUT

Step "portfolio and cost sensitivity"
& $py "$w\code\portfolio.py" --preds "$w\out\main\preds.npz" --out "$w\out\main" --figs "$m\figs"

Step "statistical tests"
& $py "$w\code\stats_tests.py" "$w\out\main\per_stock.csv" --out "$w\out\main"

Step "accuracy by year"
& $py "$w\code\by_year.py" --preds "$w\out\main\preds.npz" --out "$w\out\main\by_year.json"

Step "numbers.tex"
& $py "$w\code\emit_numbers.py" --dir "$w\out\main" --panel $panel --out "$m\numbers.tex"

Step "tables"
& $py "$w\code\make_tables.py" --dir "$w\out\main" --tables "$m\tables"

Step "figures"
& $py "$w\code\make_figs.py" --dir "$w\out\main" --figs "$m\figs"

Step "response letter placeholders"
& $py "$w\code\fill_placeholders.py" --numbers "$m\numbers.tex" `
    --in "$m\response_to_reviewers.md" --out "$m\response_to_reviewers_filled.md"

Step "anonymity check on the manuscript package"
& $py "$w\code\check_anonymity.py" "$m"

Write-Output ""
Write-Output "pipeline complete"
