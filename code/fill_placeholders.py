"""Substitute {{MACRO}} placeholders in a text file with the values from numbers.tex.

The response letter quotes the same quantities as the manuscript. Rather than typing them twice, the
letter carries placeholders and this script fills them from the generated macro file, so the two
documents cannot disagree.

Usage: python fill_placeholders.py --numbers <rework>/numbers.tex --in response_to_reviewers.md \
                                   --out response_to_reviewers_filled.md
"""
import argparse
import re

MAP = {
    "ACC_PROP": "accProp", "ACC_PROP_SD": "accPropSd", "ACC_GUIDED": "accGuided",
    "ACC_GUIDED_DIFF": "accGuidedMinusPlain", "SH_GUIDED": "shGuided", "SH_PROP": "shProp",
    "ACC_MAJ": "accMaj", "ACC_LONG": "accLong", "ACC_SECOND": "accSecond",
    "ACC_SECOND_NAME": "accSecondName", "ACC_RANGE": "rangeAcc", "BEAT_MAJ": "beatMajProp",
    "BEAT_BH": "beatBHProp", "N_SIG": "nSignificantVsProp", "N_COMPARED": "nCompared",
    "ACC_DLINEAR": "accDLinear", "ACC_PATCH": "accPatch", "ACC_ITRANS": "accITrans",
    "ACC_TIMES": "accTimes", "ACC_MLP": "accMLP", "ACC_LSTM": "accLSTM", "ACC_ATT": "accAtt",
    "ACC_SSM": "accSSM", "RET_BH": "retBH", "N_STOCKS": "nStocks", "FIRST_DATE": "firstDate",
    "LAST_DATE": "lastDate", "N_STOCK_DAYS": "nStockDays", "SUSP_PCT": "pctSusp",
    "ACC_BEST": "accBest", "SH_BEST": "shBest", "SH_BEST_NAME": "shBestName",
    "PORT_ANN_PROP": "portAnnProp", "PORT_ANN_BH": "portAnnBH", "PORT_SH_PROP": "portShProp",
    "PORT_SH_BH": "portShBH", "PORT_DD_PROP": "portDDProp", "PORT_DD_BH": "portDDBH",
    "RANK_PROP_ACC": "rankPropAcc", "FRIEDMAN_ACC": "friedmanAcc", "FRIEDMAN_ACC_P": "friedmanAccP",
    "N_MODELS": "nModels", "N_FOLDS": "nFolds", "SEQ_LEN": "seqLen",
    "ACC_WORST": "accWorst", "ACC_BEST": "accBest", "ACC_BEST_NAME": "accBestName",
    "RANK_PROP_ACC": "rankPropAcc", "N_PORT_BEAT_BH": "nPortBeatBH",
    "PORT_ANN_BEST": "portAnnBest", "PORT_ANN_PROP": "portAnnProp", "PORT_ANN_BH": "portAnnBH",
    "RUN_MINUTES": "runMinutes", "ACC_DLINEAR": "accDLinear", "TURN_PROP": "turnProp",
    "ACC_RANGE": "rangeAcc", "ACC_PROP_SD": "accPropSd",
    "N_PORT_PROFIT": "nPortProfit", "PORT_PROFIT_NAMES": "portProfitNames",
    "PORT_SH_PROP": "portShProp",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers", required=True)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    macros = {}
    with open(args.numbers, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}", line.strip())
            if m:
                macros[m.group(1)] = m.group(2)
    with open(args.src, encoding="utf-8") as f:
        text = f.read()

    missing = set()

    def sub(match):
        key = match.group(1)
        name = MAP.get(key, key)
        if name in macros:
            return macros[name]
        missing.add(key)
        return match.group(0)

    filled = re.sub(r"\{\{(\w+)\}\}", sub, text)
    # numbers.tex escapes the percent sign for LaTeX; the letter is Markdown
    filled = filled.replace(r"\%", "%").replace(r"\,", ",")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(filled)

    print(f"wrote {args.out} ({len(macros)} macros available)")
    left = re.findall(r"\{\{(\w+)\}\}", filled)
    if left:
        print("unfilled placeholders:", sorted(set(left)))
    if missing:
        print("unknown macros:", sorted(missing))


if __name__ == "__main__":
    main()
