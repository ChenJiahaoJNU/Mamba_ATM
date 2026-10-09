"""Insert the non-breaking space before every citation that lacks one.

The paper-edit convention is `~\\cite{}` so a citation never starts a line. The pattern below is
anchored on the literal backslash sequence, so it cannot touch any other construct. It reports the
before and after counts and the citation-key set, so the change is verifiable rather than assumed.
"""
import re
import sys

PATH = sys.argv[1]

with open(PATH, encoding="utf-8") as f:
    tex = f.read()

keys_before = sorted(set(re.findall(r"\\cite[pt]?\{([^}]*)\}", tex)))
bare_before = re.findall(r"(?<!~)\\cite[pt]?\{", tex)
tilde_before = re.findall(r"~\\cite[pt]?\{", tex)

fixed = re.sub(r"(?<!~)(\\cite[pt]?\{)", r"~\1", tex)

bare_after = re.findall(r"(?<!~)\\cite[pt]?\{", fixed)
tilde_after = re.findall(r"~\\cite[pt]?\{", fixed)
double = re.findall(r"~~\\cite", fixed)
keys_after = sorted(set(re.findall(r"\\cite[pt]?\{([^}]*)\}", fixed)))

print(f"  bare before      : {len(bare_before)}")
print(f"  tilde before     : {len(tilde_before)}")
print(f"  bare after       : {len(bare_after)}")
print(f"  tilde after      : {len(tilde_after)}")
print(f"  doubled tildes   : {len(double)}")
print(f"  citation keys same: {keys_before == keys_after}  ({len(keys_before)} keys)")

if double or bare_after or keys_before != keys_after:
    print("  REFUSING TO WRITE")
    sys.exit(1)

with open(PATH, "w", encoding="utf-8", newline="") as f:
    f.write(fixed)
print("  written")
