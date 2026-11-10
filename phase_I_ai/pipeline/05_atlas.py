"""Build the page data: two UMAP layouts, cell outcomes per run, and one overlay image per run.

Layouts use input features only, never IoU, cell probability or flow agreement.
  allcells: one point per annotated cell, all datasets.
  glioma:   one point per annotated glioma cell.
Hallucinated masks are left out; only annotated cells are analysed here.

Reads out/ (steps 01-04, including the rescaled glioma runs); writes atlas/atlas.json and atlas/img/.
    python 05_atlas.py              # everything
    python 05_atlas.py --overlays   # re-render the images only
"""
import json
import shutil
import sys

import numpy as np
import pandas as pd
import umap
from cellpose import io
from PIL import Image
from skimage.segmentation import find_boundaries
from sklearn.preprocessing import StandardScaler

from common import ATLAS, DATA, OUT, signal

IMG = ATLAS / "img"
SEED = 42
DATASETS = ["official_test", "breast", "flow", "glioma"]
NAMES = {"official_test": "Official test", "breast": "Breast", "flow": "Flow chamber", "glioma": "Glioma"}
OUTCOMES = ["correct", "inaccurate", "merged", "split", "missed"]
RUNS = ["60", "80", "100"]  # assumed cell diameters of the rescaled glioma runs (02_infer.py --diameter)
INPUTS = {"diameter": ("Cell size", "px across"), "aspect_ratio": ("Elongation", "length / width"), "solidity": ("Compactness", "1 = no indentations"),
          "touching_frac": ("Touching neighbours", "share of outline"), "contrast": ("Contrast", "inside minus surroundings"),
          "edge_strength": ("Edge strength", "sharper is higher"), "border_dist": ("Distance to image border", "px")}
CELL_LAYOUT = ["diameter", "aspect_ratio", "solidity", "touching_frac", "gap_rel", "contrast", "edge_strength", "internal_std", "border_dist"]
LOG_COLS = {"diameter", "aspect_ratio", "border_dist"}
N_BINS = 5
IMAGE_PX = 1000  # longest side of the overlay images; the drawer zooms up to about 2.5x on wide screens
# Outline colours: brighter versions of the page's outcome colours, so thin lines stay visible on dark images.
MASK_COLORS = {"correct": (40, 225, 140), "inaccurate": (191, 0, 255), "merge": (255, 80, 75),
               "fragment": (38, 165, 255)}  # keyed by the mask's own outcome; masks over background are not drawn
MISSED = (255, 190, 0)  # annotated cell with no mask


def project(X):
    """Median fill, standardise, UMAP to 2D, scale each axis to 0..1."""
    X = pd.DataFrame(X).apply(lambda s: s.fillna(s.median()))
    Z = StandardScaler().fit_transform(X.to_numpy(float))
    u = umap.UMAP(n_neighbors=15, min_dist=0.2, random_state=SEED).fit_transform(Z)
    return np.round((u - u.min(0)) / (u.max(0) - u.min(0)), 4)


def cell_layout(g):
    X = g[CELL_LAYOUT].copy()
    for c in LOG_COLS:
        X[c] = np.log1p(X[c])
    return project(X)


def cells():
    """Every annotated cell with its input features and its outcome in each run."""
    o = pd.read_parquet(OUT / "objects.parquet")
    f = pd.read_parquet(OUT / "object_features.parquet")
    g = o[o.kind == "gt"].merge(f[f.kind == "gt"], on=["image_id", "kind", "label"]).reset_index(drop=True)
    for run in RUNS:
        r = pd.read_parquet(OUT / "runs" / f"glioma_d{run}" / "objects.parquet")
        r = r[(r.kind == "gt")][["image_id", "label", "outcome"]].rename(columns={"outcome": f"o{run}"})
        g = g.merge(r, on=["image_id", "label"], how="left")
    return g


def bins(g):
    """Five equal-count input groups, computed within each dataset."""
    out, text = {}, {}
    for f in INPUTS:
        col = np.full(len(g), -1, int)
        text[f] = {}
        digits = 0 if f in ("diameter", "border_dist") else 2
        for ds, h in g.groupby("dataset"):
            s = h[f].dropna()
            b = pd.qcut(s.rank(method="first"), N_BINS, labels=False)
            col[s.index.to_numpy()] = b.to_numpy()
            text[f][ds] = [f"{s[b == i].min():.{digits}f} to {s[b == i].max():.{digits}f}" for i in range(N_BINS)]
        out[f] = col.tolist()
    return out, text


def boundaries(mask):
    return find_boundaries(mask, mode="inner")


def render_overlays(manifest):
    """One image per run. Each mask outline gets the colour of its own outcome; missed cells are outlined in amber.

    Masks lying mostly on background (no annotated cell there) are left out.
    """
    if IMG.exists():
        shutil.rmtree(IMG)
    IMG.mkdir(parents=True)
    sources = {"default": OUT, **{r: OUT / "runs" / f"glioma_d{r}" for r in RUNS}}
    objects = {name: pd.read_parquet(d / "objects.parquet") for name, d in sources.items()}
    for image_id, m in manifest.iterrows():
        v = signal(io.imread(str(DATA / m.image_path)), m.dataset)
        gt = np.load(OUT / "gt" / f"{image_id}.npz")["gt"].astype(np.int64)
        scale = min(1.0, IMAGE_PX / max(gt.shape))  # longest side at most IMAGE_PX, nearest-pixel resampling
        rows = np.minimum((np.arange(round(gt.shape[0] * scale)) / scale).astype(int), gt.shape[0] - 1)
        cols = np.minimum((np.arange(round(gt.shape[1] * scale)) / scale).astype(int), gt.shape[1] - 1)
        small = lambda a: a[np.ix_(rows, cols)]
        small_gt = small(gt)
        for name in (["default"] + RUNS if m.dataset == "glioma" else ["default"]):
            o = objects[name]
            o = o[o.image_id == image_id]
            pred = small(np.load(sources[name] / "raw" / f"{image_id}.npz")["pred"].astype(np.int64))
            missed = o[(o.kind == "gt") & (o.outcome == "missed")].label.astype(int).to_numpy()
            n = int(max(pred.max(), o.label.max() if len(o) else 0)) + 1
            lut, drawn = np.zeros((n, 3), np.uint8), np.zeros(n, bool)
            for outcome, color in MASK_COLORS.items():
                labels = o[(o.kind == "pred") & (o.outcome == outcome)].label.astype(int).to_numpy()
                lut[labels], drawn[labels] = color, True
            rgb = np.dstack([(np.clip(small(v), 0, 1) * 255).astype(np.uint8)] * 3)
            edge = boundaries(pred)
            rgb[boundaries(small_gt) & np.isin(small_gt, missed)] = MISSED
            edge &= drawn[pred]
            rgb[edge] = lut[pred[edge]]
            Image.fromarray(rgb).save(IMG / f"{image_id}.overlay.{name}.jpg", quality=78)


def main():
    manifest = pd.read_parquet(OUT / "manifest.parquet").set_index("image_id")
    if sys.argv[1:] == ["--overlays"]:
        render_overlays(manifest)
        print("overlays done")
        return

    g = cells()
    index = {k: i for i, k in enumerate(manifest.index)}
    code = {o: i for i, o in enumerate(OUTCOMES)}
    ds_idx = {d: i for i, d in enumerate(DATASETS)}

    images = [dict(id=image_id, ds=ds_idx[m.dataset], h=int(m.height), w=int(m.width)) for image_id, m in manifest.iterrows()]

    gb, text = bins(g)
    gl = g[g.dataset == "glioma"]
    gpos = cell_layout(gl)
    gid = np.full(len(g), -1, int)
    gid[gl.index.to_numpy()] = np.arange(len(gl))
    apos = cell_layout(g)
    box = lambda h: [[int(a), int(b), int(c), int(d)] for a, b, c, d in zip(h.y0, h.x0, h.y1, h.x1)]

    data = dict(
        datasets=[dict(id=d, name=NAMES[d]) for d in DATASETS], outcomes=OUTCOMES, runs=RUNS,
        inputs=[dict(key=k, label=v[0], unit=v[1], text=text[k]) for k, v in INPUTS.items()],
        images=images,
        cells=dict(img=[index[i] for i in g.image_id], ds=[ds_idx[d] for d in g.dataset], outcome=[code[o] for o in g.outcome],
                   glioma=gid.tolist(), bins=gb),
        glioma=dict(x=gpos[:, 0].tolist(), y=gpos[:, 1].tolist(), cell=gl.index.tolist(), label=gl.label.astype(int).tolist(),
                    box=box(gl), runs={r: [code[v] for v in gl[f"o{r}"]] for r in RUNS}),
        allcells=dict(x=apos[:, 0].tolist(), y=apos[:, 1].tolist(), box=box(g)),
    )
    ATLAS.mkdir(exist_ok=True)
    (ATLAS / "atlas.json").write_text(json.dumps(data, allow_nan=False, separators=(",", ":")))
    render_overlays(manifest)
    print("images", len(images), "cells", len(g), "glioma cells", len(gl))


if __name__ == "__main__":
    main()
