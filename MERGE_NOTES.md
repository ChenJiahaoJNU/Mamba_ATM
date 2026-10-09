# Merging the two manuscripts into one

The merged paper is the resubmission of Ms. No. EVOS-D-26-02015. It carries the negative result on the
absolute daily direction of one stock and the cross-sectional study that explains it, on one panel and
one set of models.

## What was merged

`code/merge_check.py` checked the two papers for namespace collisions before anything was combined, and
`code/merge_splice.py` moved the cross-sectional prose across by block.

| namespace | result |
|---|---|
| macros | 0 collisions. The two files share 6 macros with identical values. The merged `numbers.tex` defines 232 macros, each exactly once, and loses none |
| table files | 0 collisions, 20 in the merge |
| figure files | 0 collisions, 12 in the merge |
| bibliography | EVOS's 37 entries are all present in the other file with identical text, so the merge adds only 3 entries, for 40 |
| preamble | the two files load the same 22 packages |
| section labels | 6 collisions, resolved by dropping the cross-sectional paper's own section headings during the splice |

Blocks moved, with the heading demoted to `\subsection` so they sit inside Experiments:

| block | source lines | becomes |
|---|---|---|
| related work | 46 | Related Work 2.3 to 2.5 |
| the three labels | 17 | Methods 4.5 |
| the overnight leg | 63 | Experiments 5.6 |
| the factor library | 60 | Experiments 5.7 |
| costs, capacity and execution | 131 | Experiments 5.8 |

Written by hand, because these are where two studies have to become one argument: the abstract, the
introduction's `However` and `Although` paragraphs and its contribution list, the discussion, and the
conclusion.

## Style

The paper-edit rules were applied to the spliced text as well, not only to the original. The merged
file has 0 em dashes, its only colons are the two inside the title, and all 49 citations are written
as `~\cite`.

## Corrections carried in

* The panel table's removal lines now sum to the usable total, which they did not before.
* The combination table's cost column now includes the borrow fee, so gross less costs equals net.
* Every economics table is on one cost basis. The tradability filter that the cost-decomposition table
  applied is now applied by the combination and attribution runs too, which is what made the four
  result files agree. Before that change the same signal had four different net returns, produced with
  different universes and reference book sizes that no table disclosed.
* The two cheap-factor constructs, one with two factors and one with four, are now labelled apart.

## Status

40 pages, 0 errors, 0 undefined references, 0 overfull boxes, 0 LaTeX warnings. 20 tables and 11
figures. The Experiments section runs from page 13 to page 35, which is the part that would be trimmed
if the paper has to be shorter.

## Skill compliance

Checked against `paper-edit`, `dba-thesis` and `reference-hunter` in
`H:\修改两文章\.claude\skills`.

| rule | source | state |
|---|---|---|
| no em dashes in the body | paper-edit, dba-thesis 4 | 0 |
| no colons in the body | paper-edit | 0, the only two are inside the title |
| no sentence opening with a numeral | paper-edit | 0 |
| every citation written as `~\cite` | paper-edit | 49 of 49 |
| at most 3 keys in one `\cite` | reference-hunter | 2 citations had 5 and 4 keys, both split |
| Introduction, six stages | paper-edit | followed, with two deliberate exceptions below |
| Related Work has two subsections | paper-edit | reduced from five to two |
| no third-level headings | dba-thesis 1 | 9 `\subsubsection` converted to bold run-in leads |
| no brackets as patches, no arrows | dba-thesis 6, 7 | 0 |
| `\toprule` not `\hline` | dba-thesis 12 | 0 uses of `\hline` |
| no `\resizebox` | dba-thesis 15 | 0 |
| braces balanced | dba-thesis 29 | 825 open, 825 close |
| no DOI or URL in a citation | reference-hunter | the PDF contains no URL and no link annotation |

Two deliberate exceptions on the Introduction. It reports four contributions rather than three,
because the merged paper reports two studies and the fourth contribution carries the cross-sectional
half. And it keeps a short paragraph stating the measured result between the contribution list and the
section roadmap, because on a paper whose main result is negative the reader needs the result before
the roadmap. The six stages themselves, background, landscape, `However`, `Although`, contributions,
remainder, are all present and in order, the background is three sentences, and each contribution is
two sentences.

Two rules were not applied. `dba-thesis` wants the Conclusion in three paragraphs, and `paper-edit`
wants it in one, so the two skills disagree and the journal rule was followed. `reference-hunter`
forbids DOIs in the bibliography, which applies to hand-written `\bibitem` entries, whereas this paper
uses BibTeX with the journal's own style file, where DOIs are integral to the entry.

## A correction to this file's own process

The Related Work restructure was first done by inserting two blocks in the wrong place, which left
three copies of the cross-sectional prose in the file. The range was verified to contain only
duplicates before being deleted, and the file was rebuilt and checked afterwards. The lesson is the
same one the other skills state, that structural edits belong in single deliberate moves rather than
in successive approximations.

## What is not done

* Trimming to the length the journal expects, by moving detail tables to an appendix.
* The response letter, which still answers the reviewers on the pre-merge scope.
* The cover letter, title page and highlights, which still describe the earlier title.
* `code-supplement` requires the repository README to carry a table mapping each paper component to
  the code file that produces it, and a one-line smoke test. The README written earlier has neither.
* Regenerating the submission archive, the release and the two publication folders.
