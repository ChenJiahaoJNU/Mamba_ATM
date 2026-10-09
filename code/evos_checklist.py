"""Submission checklist for the EVOS manuscript: format rules, anonymity, a clean-room build and the
submission archive."""
import glob
import os
import re
import subprocess
import sys

REWORK = (r"D:\研究生学习资料\暨南大学-博士学习资料\论文\重投\MambaATM\_mambaatm_snjnl\rework")
TMP = os.path.join(os.environ["TEMP"], "evos_check")


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    tex = read(os.path.join(REWORK, "sn-article.tex"))
    print("== format ==")
    abstract = tex[tex.index("\\abstract{") + 10: tex.index("\\keywords")]
    words = [w for w in re.sub(r"\s+", " ", re.sub(r"\\[A-Za-z]+\{?|[{}$\\]", " ", abstract)).strip().split(" ") if w.strip()]
    print(f"  abstract words: {len(words)}  (Evolving Systems asks for 150-250)")
    for label, pat in [("keywords", r"\\keywords"), ("data availability inside the manuscript",
                                                    r"\\section\*\{Data availability\}"),
                       ("code availability inside the manuscript",
                        r"\\section\*\{Code availability\}")]:
        print(f"  {label}: " + ("present" if re.search(pat, tex) else "ABSENT"))
    print(f"  tables: {len(re.findall(r'label.tab:', tex))}   figures: {len(re.findall(r'label.fig:', tex))}")

    print("\n== anonymity of the manuscript ==")
    hard = []
    for label, pat in [("author block", r"\\author\*?\["), ("e-mail", r"[A-Za-z0-9._%-]+@[A-Za-z0-9.-]+"),
                       ("affiliation", r"Jinan University|暨南"), ("funding", r"Outstanding Innovative|2025CXY"),
                       ("personal repository", r"ChenJiahaoJNU|Changahou"),
                       ("ORCID", r"ORCID|0000-0003")]:
        hit = re.search(pat, tex)
        print(f"  {label}: " + ("FOUND - fix" if hit else "absent"))
        if hit:
            hard.append(label)
    bib = read(os.path.join(REWORK, "sn-bibliography.bib"))
    selfcite = re.search(r"author=\{Chen, Jiahao", bib)
    print("  self-citation in the bibliography: " + ("present (third person in the text)" if selfcite
                                                     else "absent"))

    print("\n== highlights ==")
    hl = read(os.path.join(REWORK, "MambaATM_highlights.txt"))
    bullets = [b for b in hl.splitlines() if b.startswith("-")]
    print(f"  {len(bullets)} bullets (Springer asks for 3-5)")
    for b in bullets:
        flag = "OK " if len(b) <= 85 else "TOO LONG"
        print(f"  {flag} {len(b):3d} chars  {b[2:]}")

    print("\n== clean-room build ==")
    os.makedirs(TMP, exist_ok=True)
    for f in ["sn-article.tex", "numbers.tex", "sn-bibliography.bib", "sn-jnl.cls",
              "sn-mathphys-ay.bst", "sn-mathphys-num.bst", "sn-basic.bst"]:
        src = os.path.join(REWORK, f)
        if os.path.exists(src):
            import shutil
            shutil.copy(src, TMP)
    for sub in ("tables", "figs"):
        src = os.path.join(REWORK, sub)
        dst = os.path.join(TMP, sub)
        if os.path.isdir(dst):
            import shutil
            shutil.rmtree(dst)
        import shutil
        shutil.copy(src, dst) if os.path.isfile(src) else shutil.copytree(src, dst)
    for cmd in (["pdflatex", "-interaction=batchmode", "sn-article.tex"],
                ["bibtex", "sn-article"],
                ["pdflatex", "-interaction=batchmode", "sn-article.tex"],
                ["pdflatex", "-interaction=batchmode", "sn-article.tex"]):
        subprocess.run(cmd, cwd=TMP, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = read(os.path.join(TMP, "sn-article.log"))
    pdf = os.path.join(TMP, "sn-article.pdf")
    print(f"  pdf produced: {os.path.getsize(pdf):,} bytes" if os.path.exists(pdf) else "  NO PDF")
    print(f"  errors: {len(re.findall(r'^! ', log, re.M))}")
    print(f"  overfull boxes: {len(re.findall(r'Overfull .hbox', log))}")
    undef = set(re.findall(r"(?:Reference|Citation) `([^']+)' (?:on page|undefined)", log))
    print(f"  undefined references: {sorted(undef) if undef else 'none'}")
    pages = re.search(r"Output written on sn-article.pdf \((\d+) pages", log)
    print(f"  pages: {pages.group(1) if pages else '?'}")

    print("\n== submission archive ==")
    import shutil
    import zipfile
    archive = os.path.join(REWORK, "EvolvingSystems_MambaATM_source.zip")
    # the manuscript-side files only: the class, the bibliography, the macros, the tables, the
    # figures, the compiled bibliography and a reference PDF. No .aux/.log/.out and no strays.
    members = ["sn-article.tex", "numbers.tex", "sn-bibliography.bib", "sn-jnl.cls",
               "sn-mathphys-ay.bst", "sn-mathphys-num.bst", "sn-basic.bst", "sn-article.bbl",
               "sn-article.pdf"]
    members += [f"tables/{os.path.basename(p)}" for p in glob.glob(os.path.join(TMP, "tables", "*.tex"))]
    members += [f"figs/{os.path.basename(p)}" for p in glob.glob(os.path.join(TMP, "figs", "*.png"))]
    if os.path.exists(archive):
        os.remove(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name in members:
            z.write(os.path.join(TMP, name.replace("/", os.sep)), name)
    print(f"  {len(members)} entries, {os.path.getsize(archive):,} bytes")
    junk = [n for n in members if os.path.splitext(n)[1].lower() in (".aux", ".log", ".out", ".blg")]
    print("  build byproducts in the archive: " + (", ".join(junk) if junk else "none"))

    unpack = os.path.join(TMP, "_unpacked")
    if os.path.isdir(unpack):
        shutil.rmtree(unpack)
    with zipfile.ZipFile(archive) as z:
        z.extractall(unpack)
    for cmd in (["pdflatex", "-interaction=batchmode", "sn-article.tex"],
                ["bibtex", "sn-article"],
                ["pdflatex", "-interaction=batchmode", "sn-article.tex"],
                ["pdflatex", "-interaction=batchmode", "sn-article.tex"]):
        subprocess.run(cmd, cwd=unpack, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ulog = read(os.path.join(unpack, "sn-article.log"))
    print(f"  extracted and compiled: errors {len(re.findall(r'^! ', ulog, re.M))}, "
          f"overfull {len(re.findall(r'Overfull .hbox', ulog))}")
    same_size = os.path.getsize(pdf) == os.path.getsize(os.path.join(unpack, "sn-article.pdf"))
    print(f"  same size as the clean-room build: {same_size}")
    print("\n  note: the self-citation and the repository address are the two items that need a"
          "\n  decision; see SUBMISSION.md.")


if __name__ == "__main__":
    sys.exit(main())
