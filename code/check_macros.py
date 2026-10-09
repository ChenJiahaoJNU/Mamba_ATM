"""Report macros used in the manuscript that numbers.tex does not define.

The manuscript must never contain a number that is not generated from the results, and it must not
fail to compile because a macro is missing. This check compares the two files statically.

Usage: python check_macros.py --tex <rework>/sn-article.tex --numbers <rework>/numbers.tex
"""
import argparse
import re

# Commands that come from the class, the packages or the preamble, not from numbers.tex.
KNOWN = set("""
documentclass PassOptionsToPackage usepackage crefname newtheorem theoremstyle begin end title
author affil orgdiv orgname email abstract keywords maketitle section subsection subsubsection
paragraph input includegraphics caption label ref cref Cref citep cite textbf textit emph texttt
emph mathbf mathrm mathcal mathbb mathsf mathit frac sqrt sum prod int partial infty cdot cdots
odot otimes top rule midrule bottomrule addlinespace toprule multicolumn multirow textwidth
linewidth columnwidth textheight hline centering resizebox small footnotesize scriptsize tiny
Require Ensure State Comment If Else EndIf Return Procedure For EndFor While EndWhile itemize
item enumerate appendices backmatter bibliography newpage raggedbottom linumbers algorithm
algorithmic linewidth si hline cline phantom vspace hspace quad qquad left right big Big bigg
Bigg text subscript superscript textsuperscript url href newline noindent par textbf textit
thmstyleone thmstyletwo thmstylethree proof assumption Assumption theorem Theorem proposition
Proposition Lemma lemma remark Remark definition Definition example Example cancel textunderscore
xurl graphicspath DeclareMathOperator operatorname mathop limits displaystyle textstyle scriptstyle
mathrm mathnormal ensuremath hfill hrulefill raisebox makecell tabularx toprule bottomrule
""".split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", required=True)
    ap.add_argument("--numbers", required=True)
    args = ap.parse_args()

    with open(args.tex, encoding="utf-8") as f:
        body = f.read()
    body = re.sub(r"%.*", "", body)                      # drop comments

    used = set(re.findall(r"\\([A-Za-z]+)", body))
    defined = set()
    with open(args.numbers, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\\newcommand\{\\(\w+)\}", line.strip())
            if m:
                defined.add(m.group(1))

    missing = sorted(n for n in used - KNOWN if n not in defined and not n.startswith(("M", "S"))
                     and n not in ("geq", "leq", "neq", "approx", "times", "propto", "sim",
                                   "rightarrow", "Leftrightarrow", "rangle", "langle", "forall",
                                   "exists", "log", "exp", "min", "max", "tanh", "odot", "in", "to",
                                   "pm", "ll", "gg", "quad", "qquad", "infty", "Delta", "Delta",
                                   "top", "bot", "pm", "times", "cdot", "approx", "alpha", "beta",
                                   "gamma", "sigma", "mu", "theta", "rho", "epsilon", "tau", "pi")
                     )
    print(f"macros used: {len(used)}   defined in numbers.tex: {len(defined)}")
    if missing:
        print("used but not defined in numbers.tex:")
        for m in missing:
            print("   ", m)
    else:
        print("no undefined macros")
    unused = sorted(defined - used)
    if unused:
        print(f"defined but unused ({len(unused)}):")
        for m in unused[:40]:
            print("   ", m)


if __name__ == "__main__":
    main()
