"""Checks from the paper-submit skill that apply to this manuscript.

Run against the rework directory:  python check_submit_rules.py <rework dir>
"""
import os
import re
import sys

from pypdf import PdfReader

# abbreviations the manuscript introduces; the skill requires the long form to appear once only
ABBREV = {
    "MLP": "multilayer perceptron",
    "LSTM": "long short-term memory",
    "SSM": "state-space model",
    "A-share": "A-share",
    "RMSE": "root mean squared error",
    "IC": "information coefficient",
    "bp": "basis point",
    "ADV": "average daily volume",
    "EMA": "exponential moving average",
    "F1": "F1",
}

# every model compared in the tables has to be cited somewhere in the manuscript
BASELINES = ["zeng2023dlinear", "nie2023patchtst", "liu2024itransformer", "wu2023timesnet",
             "gu2024linear", "Mamba2", "vaswani2017attention", "LSTM", "MLP"]


def main(root):
    tex = open(os.path.join(root, "sn-article.tex"), encoding="utf-8").read()
    bib = open(os.path.join(root, "sn-bibliography.bib"), encoding="utf-8").read()
    print("== PDF metadata and links ==")
    pdf = PdfReader(os.path.join(root, "sn-article.pdf"))
    meta = {k: v for k, v in (pdf.metadata or {}).items()}
    print(f"  metadata: {meta}")
    hits = [k for k in ("/Author", "/Subject", "/Keywords", "/Title") if meta.get(k)]
    print("  identity-bearing metadata fields: " + (", ".join(hits) if hits else "none"))
    names = re.findall(r"Chen|Jiahao|Jinan|暨南|Liu|Pan", str(meta))
    print("  author-name fragments in metadata: " + (", ".join(sorted(set(names))) if names else "none"))
    raw = open(os.path.join(root, "sn-article.pdf"), "rb").read()
    urls = re.findall(rb"https?://[^\s\)\]<>]{4,80}", raw)
    print(f"  url-like strings in the PDF: {len(urls)}")
    for u in sorted(set(urls))[:10]:
        print("     " + u.decode("latin-1"))

    print("\n== abbreviation consistency: long form should appear once ==")
    for short, long_form in ABBREV.items():
        n = len(re.findall(re.escape(long_form), tex, re.I))
        if n:
            print(f"  {short:8s} long form {n} time(s): {long_form}")

    print("\n== citation completeness for the baselines ==")
    for key in BASELINES:
        in_body = f"{{{key}}}" in tex or f",{key}" in tex or f"{key}," in tex
        in_bib = re.search(r"@\w+\{" + re.escape(key) + r",", bib) is not None
        print(f"  {key:22s} cited in text: {'yes' if in_body else 'NO '}   in bib: "
              f"{'yes' if in_bib else 'NO'}")

    print("\n== bib venue casing ==")
    venues = re.findall(r"(?:journal|booktitle)\s*=\s*[{\"]([^}\"]{5,120})", bib)
    lower = [v for v in venues if v != v.title() and v[0].islower()]
    print(f"  venue fields: {len(venues)}; starting with a lower-case letter: {len(lower)}")
    for v in lower[:10]:
        print("     " + v)

    print("\n== table captions with parenthetical explanations ==")
    for m in re.finditer(r"\\caption\{([^}]{0,400})", tex):
        cap = m.group(1)
        if "(" in cap and ")" in cap:
            frag = [s for s in re.findall(r"\([^)]{0,80}\)", cap)]
            print(f"  {frag}")

    print("\n== style counters ==")
    em = len(re.findall("---", tex))
    tilde_cite = len(re.findall(r"~\\cite", tex))
    bare_cite = len(re.findall(r"(?<!~)\\cite[pt]?\{", tex))
    colons = len(re.findall(r": ", tex))
    print(f"  em dash '---'          : {em}")
    print(f"  '\\cite' with '~'      : {tilde_cite}")
    print(f"  '\\cite' without '~'   : {bare_cite}")
    print(f"  colon + space          : {colons}")


if __name__ == "__main__":
    main(sys.argv[1])
