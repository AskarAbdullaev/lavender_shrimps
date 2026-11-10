"""Compute descriptive features for every object (true cell or predicted mask) and image.

Outputs in out/, joinable on (image_id, kind, label) and image_id:
  object_features.parquet
  image_features.parquet

Input features use only the image and the object's own outline. For both kinds, neighbours are
objects of the same kind. flow_cos and cellprob_mean are model outputs: they describe an error
but must not be used to explain it.

Appearance uses one signal channel, percentile-normalised like Cellpose (1st -> 0, 99th -> 1):
official test = channel 1 (the cytoplasm channel; channel 2 is always empty and channel 0 only
sometimes used), glioma = mean of RGB, others = their only channel.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from cellpose import dynamics, io
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential

from common import DATA, OUT, signal

DEVICE = torch.device("cuda")

TOUCH_DIST = 3     # outline pixel counts as touching if a neighbour is within this many px (gap <= 2)
GAP_CAP_DIAMS = 2  # nearest-neighbour search radius, in cell diameters
RING_FRAC = 0.25   # background ring width, as a share of the diameter (at least RING_MIN px)
RING_MIN = 3
SMOOTH_SIGMA = 1.0  # smoothing before measuring edges, so edge strength is less noise-driven



def clip_box(y0, x0, y1, x1, pad, shape):
    return max(y0 - pad, 0), max(x0 - pad, 0), min(y1 + pad, shape[0]), min(x1 + pad, shape[1])


def flow_cosine(mask, dP):
    """Agreement between the network's flow and the ideal flow of the given mask, per pixel."""
    seq = relabel_sequential(mask)[0]
    ideal = np.asarray(dynamics.masks_to_flows_gpu(seq, device=DEVICE)[0], np.float32)
    denom = np.sqrt((ideal ** 2).sum(0) * (dP ** 2).sum(0))
    return (ideal * dP).sum(0) / np.maximum(denom, 1e-6)


def object_rows(mask, kind, v, grad, cellprob, cos, only=None):
    h, w = mask.shape
    rows = []
    for r in regionprops(mask):
        if only is not None and r.label not in only:
            continue
        y0, x0, y1, x1 = r.bbox
        diam = float(np.sqrt(4 * r.area / np.pi))

        # Crowding: distance from this object's outline to the nearest neighbouring object
        cap = int(np.ceil(max(GAP_CAP_DIAMS * diam, 10)))
        a0, b0, a1, b1 = clip_box(y0, x0, y1, x1, cap, mask.shape)
        crop = mask[a0:a1, b0:b1]
        obj = crop == r.label
        others = (crop > 0) & ~obj
        outline = obj & ~ndi.binary_erosion(obj)
        if others.any():
            d = ndi.distance_transform_edt(~others)[outline]
            touch, gap = float((d <= TOUCH_DIST).mean()), min(float(d.min()) - 1, cap)
        else:
            touch, gap = 0.0, cap

        # Appearance: inside versus a ring of unlabelled pixels just outside the object
        ring_w = max(RING_MIN, int(round(RING_FRAC * diam)))
        a0, b0, a1, b1 = clip_box(y0, x0, y1, x1, ring_w, mask.shape)
        crop = mask[a0:a1, b0:b1]
        obj = crop == r.label
        ring = (ndi.distance_transform_edt(~obj) <= ring_w) & (crop == 0)
        pix = v[a0:a1, b0:b1]
        outline = obj & ~ndi.binary_erosion(obj)
        inside = pix[obj]
        minor = r.axis_minor_length

        rows.append(dict(
            kind=kind, label=r.label, diameter=diam,
            aspect_ratio=r.axis_major_length / minor if minor > 0 else np.nan,
            solidity=r.solidity,
            touching_frac=touch,
            gap_rel=gap / diam,
            border_dist=min(y0, x0, h - y1, w - x1),
            contrast=float(inside.mean() - pix[ring].mean()) if ring.sum() >= 5 else np.nan,
            edge_strength=float(grad[a0:a1, b0:b1][outline].mean()),
            internal_std=float(inside.std()),
            cellprob_mean=float(cellprob[a0:a1, b0:b1][obj].mean()),
            flow_cos=float(cos[a0:a1, b0:b1][obj].mean()),
        ))
    return rows


def process(r):
    img = io.imread(str(DATA / r.image_path))
    gt = np.load(OUT / "gt" / f"{r.image_id}.npz")["gt"].astype(np.int32)
    z = np.load(OUT / "raw" / f"{r.image_id}.npz")
    pred, dP, cellprob = z["pred"].astype(np.int32), z["dP"], z["cellprob"]

    v = signal(img, r.dataset)
    grad = sobel(ndi.gaussian_filter(v, SMOOTH_SIGMA))
    rows = (object_rows(gt, "gt", v, grad, cellprob, flow_cosine(gt, dP))
            + object_rows(pred, "pred", v, grad, cellprob, flow_cosine(pred, dP)))
    for row in rows:
        row["image_id"] = r.image_id

    area = gt[gt > 0]
    areas = np.bincount(area)[np.unique(area)]
    image = dict(
        image_id=r.image_id,
        density_per_mpx=r.n_gt / (r.height * r.width) * 1e6,
        foreground=float((gt > 0).mean()),
        median_diameter=float(np.median(np.sqrt(4 * areas / np.pi))),
        area_cv=float(areas.std() / areas.mean()),
        has_second_channel=bool(r.dataset == "official_test" and img[..., 0].max() > 0),
        image_std=float(v.std()),
        fg_bg_contrast=float(v[gt > 0].mean() - v[gt == 0].mean()),
    )
    return rows, image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="one image per dataset, print, save nothing")
    args = ap.parse_args()

    m = pd.read_parquet(OUT / "manifest.parquet")
    if args.smoke:
        m = m.groupby("dataset").head(1)
    objects, images = [], []
    start = time.time()
    for i, r in enumerate(m.itertuples(), 1):
        rows, image = process(r)
        objects += rows
        images.append(image)
        print(f"[{i}/{len(m)}] {r.image_id} {len(rows)} objects, {time.time() - start:.0f}s", flush=True)

    obj, img = pd.DataFrame(objects), pd.DataFrame(images)
    if args.smoke:
        pd.set_option("display.width", 220, "display.max_columns", 40)
        print(obj.describe().T.round(3))
        return
    obj.to_parquet(OUT / "object_features.parquet")
    img.to_parquet(OUT / "image_features.parquet")
    print(f"done: {len(obj)} objects, {len(img)} images")


if __name__ == "__main__":
    main()
