"""Paths and the signal channel shared by every step.

The datasets are read from ./data by default; set ATLAS_DATA to use another folder.
"""
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("ATLAS_DATA", ROOT / "data")).resolve()
OUT = ROOT / "out"        # intermediate results (masks, predictions, tables)
ATLAS = ROOT / "atlas"    # the page: index.html, atlas.json, img/


def signal(img, dataset):
    """One channel, normalised like Cellpose: 1st percentile -> 0, 99th -> 1.

    Official test uses channel 1 (cytoplasm), glioma the mean of RGB, the others their only channel.
    """
    a = img.astype(np.float32)
    if a.ndim == 3:
        a = a[..., 1] if dataset == "official_test" else a.mean(-1)
    lo, hi = np.percentile(a, [1, 99])
    return (a - lo) / max(hi - lo, 1e-6)
