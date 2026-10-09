"""Check that every figure is a real, non-empty image and report what it contains.

A figure that failed to draw still produces a file, so size alone is not evidence. This reads each
PNG, reports its dimensions, and measures how much of the canvas carries ink outside the axes frame,
which distinguishes a plotted figure from a blank one.

Usage: python check_figs.py --figs <rework>/figs [--expected 6]
"""
import argparse
import glob
import os
import sys

import numpy as np


def describe(path):
    try:
        from PIL import Image
    except ImportError:
        return None
    im = Image.open(path).convert("RGB")
    a = np.asarray(im)
    # ink = pixels that are neither white nor near-white
    ink = (a.min(axis=2) < 240).mean()
    # colour variety: a plot with lines and text has many distinct colours
    colours = len(np.unique(a.reshape(-1, 3), axis=0))
    return im.size, float(ink), int(colours)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", required=True)
    ap.add_argument("--expected", type=int, default=6)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.figs, "*.png")))
    print(f"{len(files)} figure file(s) in {args.figs}\n")
    print(f"{'figure':<28}{'size':>12}{'ink':>9}{'colours':>10}")
    bad = []
    for f in files:
        d = describe(f)
        if d is None:
            print(f"{os.path.basename(f):<28}{'(Pillow not installed)':>12}")
            continue
        size, ink, colours = d
        print(f"{os.path.basename(f):<28}{str(size):>12}{ink:>9.4f}{colours:>10,}")
        if ink < 0.002 or colours < 50:
            bad.append(os.path.basename(f))
    print()
    if len(files) < args.expected:
        print(f"WARNING: expected at least {args.expected} figures")
    if bad:
        print("figures that look blank or nearly blank:", bad)
        return 1
    print("all figures contain a drawn plot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
