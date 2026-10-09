"""Mechanical anonymity check for the manuscript that is uploaded for double-blind review.

The previous submission failed this check: one of the files in the submission package carried the
author block, the two e-mail addresses, the affiliation and a repository address that names the
authors. The check is now a script so that it cannot be skipped or done by eye.

It scans the manuscript source, the bibliography, the generated tables and the figure file names for
author names, e-mail addresses, affiliation strings, ORCID identifiers and personal repository
addresses.

Usage: python check_anonymity.py <rework directory>
"""
import glob
import os
import re
import sys

PATTERNS = [
    (r"Jiahao|Xiangdong", "author given name"),
    (r"Chen|Liu", "author family name (check the match by eye; these are common words)"),
    (r"Jinan University|暨南", "affiliation"),
    (r"jnu\.edu\.cn|stu2024", "e-mail address"),
    (r"@[A-Za-z0-9.-]+\.(edu|com|cn|org)", "e-mail address"),
    (r"ORCID|0000-0003-4673-1311", "ORCID"),
    (r"ChenJiahaoJNU|Changahou|changahou", "personal repository account"),
    # third-party repositories are not identity leaks, so only personal-looking accounts are flagged
    (r"github\.com/(?!state-spaces|huggingface|pytorch|matplotlib|scipy|numpy)[A-Za-z0-9_.-]+",
     "repository URL with a personal account"),
    (r"huggingface\.co/(?!datasets/state)[A-Za-z0-9_.-]+", "repository URL with an account name"),
    (r"Outstanding Innovative Talents|2025CXY", "funding acknowledgement"),
    (r"School of Economics", "affiliation"),
]

TEXT_EXT = (".tex", ".bib", ".txt", ".csv", ".json")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    targets = []
    for ext in TEXT_EXT:
        targets += glob.glob(os.path.join(root, "**", "*" + ext), recursive=True)
    # figure and table names can also carry a name
    targets += glob.glob(os.path.join(root, "figs", "*"))
    targets += glob.glob(os.path.join(root, "tables", "*"))
    targets = [t for t in targets if os.path.isfile(t)]

    # files that legitimately carry author details
    allowed = {"TitlePage_MambaATM.tex", "cover_letter.md", "response_to_reviewers.md",
               "response_to_reviewers_filled.md"}

    findings = []
    for path in sorted(targets):
        base = os.path.basename(path)
        if base in allowed or base.startswith("response_to_reviewers"):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        for line_no, line in enumerate(text.splitlines(), 1):
            for pattern, label in PATTERNS:
                if re.search(pattern, line):
                    # the manuscript refers to "Chen et al." style citations, allow those
                    if label.startswith("author family name") and re.search(
                            r"\\cite|Chen et al\.|Liu et al\.", line):
                        continue
                    findings.append((path, line_no, label, line.strip()[:120]))

    name_hits = [f for f in findings if not f[2].startswith("author family name")]
    if not findings:
        print("anonymity check: PASS (no identifying string found)")
        return 0
    print(f"anonymity check: {len(findings)} hits, {len(name_hits)} of them hard")
    for path, line_no, label, text in findings:
        rel = os.path.relpath(path, root)
        print(f"  {rel}:{line_no}  [{label}]  {text}")
    return 1 if name_hits else 0


if __name__ == "__main__":
    sys.exit(main())
