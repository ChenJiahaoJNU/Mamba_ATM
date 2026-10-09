# Sensitivity experiments, one configuration at a time.
#
# Only the proposed model is run for the window-length and seed checks, because the claims the paper
# makes about them concern the proposed model against the majority-class rate of the same subset.
# The subset is every sixth stock of the sorted panel, so the runs stay short and the configurations
# are directly comparable with each other.

$ErrorActionPreference = "Continue"
$root = "."
$w = "$root"
$panel = "$root\data\adj"
$py = "C:\ProgramData\anaconda3\envs\rl1\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "$w\libs"

$all = (Get-ChildItem $panel -File -Filter *.csv | Sort-Object Name).BaseName
$codes = @()
for ($i = 0; $i -lt $all.Count; $i += 6) { $codes += $all[$i] }
$list = ($codes -join ",")
Write-Output "subset: $($codes.Count) stocks; first $($codes[0]), last $($codes[-1])"

function Run-One($tag, $seq, $seed) {
    Write-Output "=== $tag (seq=$seq seed=$seed) ==="
    & $py -u "$w\code\run_panel.py" --panel $panel --models "gatedfusion" --seq $seq --seed $seed `
        --stocks $list --folds 5 --epochs 30 --workers 4 --out "$w\out\$tag" 2>&1 |
        ForEach-Object { "$_" }
    Write-Output "=== $tag done (exit $LASTEXITCODE) ==="
}

Run-One "seq20" 20 0
Run-One "seq120" 120 0
Run-One "seed0sub" 60 0
Run-One "seed7" 60 7
Run-One "seed13" 60 13

Write-Output "sensitivity runs complete"
