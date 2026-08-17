#!/usr/bin/env python3
"""Trim whitespace margins from diagram PNGs so each figure fills its canvas.

Handles opaque backgrounds (including subtle gradients) with a tolerance-based
bounding box, and transparent backgrounds via the alpha channel.
"""
import os
from PIL import Image
import numpy as np

d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diagrams')
TOL = 30  # sum of |R-G-B| channel diffs above which a pixel counts as content


def trim_bbox(im):
    """Return (l, t, r, b) content bbox or None."""
    rgba = np.asarray(im.convert('RGBA'), dtype=np.int16)
    alpha = rgba[..., 3]
    if (alpha.min() < 250).any():  # has transparency
        rows = alpha.max(axis=1) > 16
        cols = alpha.max(axis=0) > 16
    else:
        rgb = rgba[..., :3]
        bg = rgb[2, rgb.shape[1] - 2].astype(np.int16)
        diff = np.abs(rgb - bg).sum(axis=2)
        rows = diff.max(axis=1) > TOL
        cols = diff.max(axis=0) > TOL
    if not rows.any():
        return None
    ys = np.where(rows)[0]
    xs = np.where(cols)[0]
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


for f in sorted(os.listdir(d)):
    if not f.endswith('.png'):
        continue
    p = os.path.join(d, f)
    im = Image.open(p)
    w, h = im.size
    bbox = trim_bbox(im)
    if bbox is None:
        print(f, 'no content, skipped')
        continue
    l, t, r, b = bbox
    if r - l >= w - 2 and b - t >= h - 2:
        print(f, 'already full canvas')
        continue
    pad = 14
    l = max(0, l - pad)
    t = max(0, t - pad)
    r = min(w, r + pad)
    b = min(h, b + pad)
    im.crop((l, t, r, b)).save(p)
    print(f, f'trimmed -> {r - l}x{b - t} (was {w}x{h})')
