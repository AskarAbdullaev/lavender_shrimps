"""Match predictions to ground truth and give every object exactly one outcome.

Outputs in out/:
  objects.parquet  one row per GT cell (kind="gt") and per predicted mask (kind="pred")
  pairs.parquet    every overlapping GT/pred pair, for tracing merges and splits
  images.parquet   per-image counts and AP

"correct" uses the official Cellpose assignment at IoU >= MATCH_IOU. Anything else gets the first
rule that applies, using cov_gt = overlap/GT area and cov_pred = overlap/pred area.

GT cells:   merged      a mask covers >= MAJORITY of this cell and of another GT cell
            split       >= 2 masks, each covering >= SPLIT_COV of it and lying mostly inside it
            inaccurate  best IoU >= WEAK_IOU, or a mask covers >= MAJORITY of it
            missed      otherwise
Pred masks: merge       covers >= MAJORITY of two or more GT cells
            spurious    >= MAJORITY of the mask is background
            fragment    >= MAJORITY of it lies in one GT cell, but it covers < MAJORITY of that cell
            inaccurate  otherwise
"""
from pathlib import Path

import argparse

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops_table

from common import OUT

RUN = OUT  # where predictions are read and results written; --run points it at out/runs/<run>

MATCH_IOU = 0.5
MAJORITY = 0.5
SPLIT_COV = 0.2
WEAK_IOU = 0.2
MIN_SIZE = 15  # Cellpose removes smaller masks, so smaller GT cells cannot be matched
GT_OUTCOMES = ["correct", "merged", "split", "inaccurate", "missed"]
PRED_OUTCOMES = ["correct", "merge", "spurious", "fragment", "inaccurate"]


def assign(iou, thr):
    """Official Cellpose assignment: maximise matches at thr, break ties by IoU."""
    if 0 in iou.shape:
        return np.array([], int), np.array([], int)
    cost = -(iou >= thr).astype(float) - iou / (2 * min(iou.shape))
    r, c = linear_sum_assignment(cost)
    keep = iou[r, c] >= thr
    return r[keep], c[keep]


def pair_table(gt, pred, gt_area, pred_area):
    fg = (gt > 0) & (pred > 0)
    k = pred.max() + 1 if pred.size else 1
    codes, inter = np.unique(gt[fg].astype(np.uint64) * int(k) + pred[fg], return_counts=True)
    p = pd.DataFrame({"gt_label": (codes // int(k)).astype(int), "pred_label": (codes % int(k)).astype(int), "inter": inter})
    p["cov_gt"] = p.inter / p.gt_label.map(gt_area)
    p["cov_pred"] = p.inter / p.pred_label.map(pred_area)
    p["iou"] = p.inter / (p.gt_label.map(gt_area) + p.pred_label.map(pred_area) - p.inter)
    return p


def props(mask, kind):
    t = pd.DataFrame(regionprops_table(mask.astype(np.int32), properties=("label", "area", "centroid", "bbox")))
    t = t.rename(columns={"centroid-0": "cy", "centroid-1": "cx", "bbox-0": "y0", "bbox-1": "x0",
                          "bbox-2": "y1", "bbox-3": "x1"})
    h, w = mask.shape
    t["touches_border"] = (t.y0 == 0) | (t.x0 == 0) | (t.y1 == h) | (t.x1 == w)
    t["kind"] = kind
    return t


def n_pieces(mask, labels):
    slices = ndi.find_objects(mask.astype(np.int32))
    return [ndi.label(mask[slices[int(i) - 1]] == i)[1] for i in labels]


def gt_outcome(g, ps, by_pred):
    if ps.empty:
        return "missed"
    for row in ps[ps.cov_gt >= MAJORITY].itertuples():
        others = by_pred.get(row.pred_label)
        if ((others.gt_label != g) & (others.cov_gt >= MAJORITY)).any():
            return "merged"
    if ((ps.cov_gt >= SPLIT_COV) & (ps.cov_pred >= MAJORITY)).sum() >= 2:
        return "split"
    if ps.iou.max() >= WEAK_IOU or ps.cov_gt.max() >= MAJORITY:
        return "inaccurate"
    return "missed"


def pred_outcome(ps, area):
    if ps.empty or 1 - ps.inter.sum() / area >= MAJORITY:
        return "merge" if (not ps.empty and (ps.cov_gt >= MAJORITY).sum() >= 2) else "spurious"
    if (ps.cov_gt >= MAJORITY).sum() >= 2:
        return "merge"
    if ((ps.cov_pred >= MAJORITY) & (ps.cov_gt < MAJORITY)).any():
        return "fragment"
    return "inaccurate"


def process(r):
    gt = np.load(OUT / "gt" / f"{r.image_id}.npz")["gt"].astype(np.int64)
    pred = np.load(RUN / "raw" / f"{r.image_id}.npz")["pred"].astype(np.int64)
    g, p = props(gt, "gt"), props(pred, "pred")
    gt_area, pred_area = dict(zip(g.label, g.area)), dict(zip(p.label, p.area))
    pairs = pair_table(gt, pred, gt_area, pred_area)

    # Dense IoU matrix for the assignment
    gi, pi = {v: i for i, v in enumerate(g.label)}, {v: i for i, v in enumerate(p.label)}
    iou = np.zeros((len(g), len(p)))
    iou[pairs.gt_label.map(gi).to_numpy(int), pairs.pred_label.map(pi).to_numpy(int)] = pairs.iou
    a, b = assign(iou, MATCH_IOU)
    gt_match = {int(g.label[i]): int(p.label[j]) for i, j in zip(a, b)}
    pred_match = {v: k for k, v in gt_match.items()}
    pairs["matched"] = [gt_match.get(x) == y for x, y in zip(pairs.gt_label, pairs.pred_label)]

    by_gt, by_pred = dict(tuple(pairs.groupby("gt_label"))), dict(tuple(pairs.groupby("pred_label")))
    empty = pairs.iloc[0:0]

    def fill(tab, match, group, other, outcome_fn):
        out, best_iou, best, n = [], [], [], []
        for lab, area in zip(tab.label, tab.area):
            ps = group.get(lab, empty)
            out.append("correct" if lab in match else outcome_fn(lab, area, ps))
            best_iou.append(float(ps.iou.max()) if len(ps) else 0.0)
            best.append(int(ps.loc[ps.iou.idxmax(), other]) if len(ps) else -1)
            n.append(len(ps))
        tab["outcome"], tab["best_iou"], tab["best_other"], tab["n_overlaps"] = out, best_iou, best, n
        tab["matched_other"] = [match.get(x, -1) for x in tab.label]
        tab["bg_frac"] = [1 - group.get(l, empty).inter.sum() / a for l, a in zip(tab.label, tab.area)]

    fill(g, gt_match, by_gt, "pred_label", lambda lab, _, ps: gt_outcome(lab, ps, by_pred))
    fill(p, pred_match, by_pred, "gt_label", lambda _, area, ps: pred_outcome(ps, area))
    g["n_pieces"], p["n_pieces"] = n_pieces(gt, g.label), 1
    g["below_min_size"], p["below_min_size"] = g.area < MIN_SIZE, False

    objects = pd.concat([g, p], ignore_index=True)
    for c in ("image_id", "dataset", "subset", "group"):
        objects[c] = getattr(r, c)
    pairs.insert(0, "image_id", r.image_id)

    row = dict(image_id=r.image_id, dataset=r.dataset, subset=r.subset, group=r.group,
               n_gt=len(g), n_pred=len(p), tp=len(a), fp=len(p) - len(a), fn=len(g) - len(a))
    for t in (0.5, 0.75, 0.9):
        tp = len(assign(iou, t)[0])
        row[f"ap{int(t * 100)}"] = tp / (len(g) + len(p) - tp) if len(g) + len(p) - tp else np.nan
    for o in GT_OUTCOMES:
        row[f"gt_{o}"] = int((g.outcome == o).sum())
    for o in PRED_OUTCOMES:
        row[f"pred_{o}"] = int((p.outcome == o).sum())
    return objects, pairs, row


def main():
    global RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="match the experiment in out/runs/<run>/ instead of the base run")
    args = ap.parse_args()
    if args.run:
        RUN = OUT / "runs" / args.run
    m = pd.read_parquet(OUT / "manifest.parquet")
    m = m[[(RUN / "raw" / f"{i}.npz").exists() for i in m.image_id]]
    res = [process(r) for r in m.itertuples()]
    objects = pd.concat([x[0] for x in res], ignore_index=True)
    pairs = pd.concat([x[1] for x in res], ignore_index=True)
    images = pd.DataFrame([x[2] for x in res])
    objects.to_parquet(RUN / "objects.parquet")
    pairs.to_parquet(RUN / "pairs.parquet")
    images.to_parquet(RUN / "images.parquet")

    pd.set_option("display.width", 200, "display.max_columns", 40)
    for kind in ("gt", "pred"):
        o = objects[objects.kind == kind]
        print(f"\n{kind}: outcome share by dataset")
        t = pd.crosstab(o.dataset, o.outcome, normalize="index").round(3)
        t["n"] = o.groupby("dataset").size()
        print(t)
    print(images.groupby("dataset")[["ap50", "ap75", "ap90"]].mean().round(3))


if __name__ == "__main__":
    main()
