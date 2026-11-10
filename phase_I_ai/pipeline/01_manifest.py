"""Index every usable image, write ground truth as uniform label masks, record input QC.

The official *train* split is excluded on purpose (likely model training data).
Image paths are stored relative to the data folder.
Outputs: out/manifest.parquet, out/gt/<image_id>.npz (gt, and cat for per-label class).
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from cellpose import io
from scipy import ndimage as ndi
from skimage.draw import polygon

from common import DATA, OUT

GT = OUT / "gt"

MIN_SIZE = 15  # Cellpose drops predicted masks below this, so smaller GT can never match


def slug(*parts):
    return "__".join(re.sub(r"[^A-Za-z0-9._-]+", "_", p).strip("_") for p in parts)


def paired_tif(root, dataset, group_fn):
    src, tgt = root / "Training_source", root / "Training_target"
    for p in sorted(src.glob("*.tif")):
        mask = tgt / p.name
        if not mask.exists():
            print(f"WARNING no mask for {p.name}")
            continue
        yield dict(image_id=slug(dataset, p.stem), dataset=dataset, subset="all",
                   group=group_fn(p.stem), image_path=p, mask_path=mask)


def breast_group(stem):
    m = re.match(r"(.*?) - Position (\d+)_", stem)
    return f"{m[1]}|pos{m[2]}" if m else stem


def flow_group(stem):
    m = re.search(r"FOV(\d+)", stem)
    return f"FOV{m[1]}" if m else stem


def official_test():
    d = DATA / "official_dataset" / "test"
    for p in sorted(d.glob("*_img.png")):
        stem = p.stem.replace("_img", "")
        yield dict(image_id=slug("official_test", stem), dataset="official_test", subset="test",
                   group=stem, image_path=p, mask_path=d / f"{stem}_masks.png")


def glioma():
    base = DATA / "glioma_c6_dataset"
    for sub, d, anno in [("gen", base / "gen", base / "gen" / "anno_gen.json"),
                         *[(f"spec_{s}", base / "spec" / s, base / "spec" / s / f"anno_{s}.json")
                           for s in ("train", "valid", "test")]]:
        coco = json.loads(anno.read_text())
        by_img = {}
        for a in coco["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        for info in coco["images"]:
            stem = Path(info["file_name"]).stem
            yield dict(image_id=slug("glioma", stem), dataset="glioma", subset=sub, group=stem,
                       image_path=d / "images" / info["file_name"], coco_anns=by_img.get(info["id"], []),
                       coco_cats={c["id"]: c["name"] for c in coco["categories"]},
                       shape=(info["height"], info["width"]))


def rasterize(anns, cats, shape):
    """Polygons -> labels. Nuclei are dropped; larger cells paint first so small ones stay visible."""
    cells = [a for a in anns if "nucleus" not in cats[a["category_id"]].lower()]
    n_nuclei = len(anns) - len(cells)
    shapes = []
    for a in cells:
        rr, cc = [], []
        for poly in a["segmentation"]:
            xy = np.asarray(poly, float).reshape(-1, 2)
            r, c = polygon(xy[:, 1], xy[:, 0], shape)
            rr.append(r), cc.append(c)
        rr, cc = np.concatenate(rr), np.concatenate(cc)
        shapes.append((len(rr), rr, cc, cats[a["category_id"]]))
    shapes.sort(key=lambda s: -s[0])
    gt = np.zeros(shape, np.uint32)
    painted = 0
    for i, (n, rr, cc, _) in enumerate(shapes, 1):
        gt[rr, cc] = i
        painted += n
    return gt, np.array([s[3] for s in shapes]), n_nuclei, painted - int((gt > 0).sum())


def qc_labels(gt):
    ids, areas = np.unique(gt[gt > 0], return_counts=True)
    slices = ndi.find_objects(gt)
    split = 0
    for i in ids:
        s = slices[int(i) - 1]
        if ndi.label(gt[s] == i)[1] > 1:
            split += 1
    return dict(n_gt=len(ids), n_gt_below_min=int((areas < MIN_SIZE).sum()), n_gt_disconnected=split,
                gt_median_area=float(np.median(areas)) if len(ids) else np.nan,
                foreground=float((gt > 0).mean()))


def main():
    GT.mkdir(parents=True, exist_ok=True)
    items = [*official_test(),
             *paired_tif(DATA / "Breast cancer cell dataset" / "Training dataset", "breast", breast_group),
             *paired_tif(DATA / "Flow chamber dataset" / "Training dataset", "flow", flow_group),
             *glioma()]
    dup = [k for k, v in Counter(i["image_id"] for i in items).items() if v > 1]
    assert not dup, dup

    rows = []
    for it in items:
        img = io.imread(str(it["image_path"]))
        extra = dict(n_nuclei_dropped=0, overlap_px=0)
        cat = None
        if "coco_anns" in it:
            gt, cat, nuc, over = rasterize(it["coco_anns"], it["coco_cats"], it["shape"])
            extra = dict(n_nuclei_dropped=nuc, overlap_px=over)
        else:
            gt = io.imread(str(it["mask_path"]))
            assert gt.ndim == 2, (it["image_id"], gt.shape)
            gt = gt.astype(np.uint32)
        assert img.shape[:2] == gt.shape, (it["image_id"], img.shape, gt.shape)

        np.savez_compressed(GT / f"{it['image_id']}.npz", gt=gt, **({} if cat is None else {"cat": cat}))
        channels = 1 if img.ndim == 2 else img.shape[-1]
        rows.append(dict(
            image_id=it["image_id"], dataset=it["dataset"], subset=it["subset"], group=it["group"],
            image_path=str(it["image_path"].relative_to(DATA)), height=img.shape[0], width=img.shape[1],
            channels=channels, dtype=str(img.dtype), img_min=float(img.min()), img_max=float(img.max()),
            **extra, **qc_labels(gt)))
        print(rows[-1]["image_id"], rows[-1]["n_gt"], flush=True)

    m = pd.DataFrame(rows)
    m.to_parquet(OUT / "manifest.parquet")
    pd.set_option("display.width", 200, "display.max_columns", 40)
    print(m.groupby(["dataset", "subset"]).agg(
        images=("image_id", "size"), groups=("group", "nunique"), cells=("n_gt", "sum"),
        below_min=("n_gt_below_min", "sum"), disconnected=("n_gt_disconnected", "sum"),
        nuclei_dropped=("n_nuclei_dropped", "sum"), overlap_px=("overlap_px", "sum"),
        median_area=("gt_median_area", "median")))
    print(m.groupby("dataset")[["height", "width"]].agg(["min", "max"]))
    print(m.groupby(["dataset", "dtype", "channels"]).size())
    print("empty GT images:", m.loc[m.n_gt == 0, "image_id"].tolist())


if __name__ == "__main__":
    main()
