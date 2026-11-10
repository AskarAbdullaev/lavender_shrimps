"""Run cpsam_v2 once per manifest image and keep masks, flows and cell probability.

Default Cellpose settings, so results describe out-of-the-box behaviour. Finished images are
skipped, so the run can be resumed.

    python 02_infer.py                 # everything
    python 02_infer.py --check-repeat  # run one image per dataset twice, compare, save nothing
    python 02_infer.py --dataset glioma --diameter 80 --run glioma_d80
                                       # separate experiment under out/runs/<run>/; base run untouched
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import cellpose
import numpy as np
import pandas as pd
import torch
from cellpose import io, models

from common import DATA, OUT

MODEL = "cpsam_v2"


def predict(model, path, diameter=None):
    img = io.imread(str(DATA / path))
    masks, flows, _ = model.eval(img, diameter=diameter)
    return masks.astype(np.uint32), flows[1].astype(np.float32), flows[2].astype(np.float32)


def write_meta(model, out, diameter):
    ckpt = Path(model.pretrained_model)
    meta = dict(model=MODEL, checkpoint=str(ckpt), checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                cellpose=cellpose.version, torch=torch.__version__, platform=platform.platform(),
                device=torch.cuda.get_device_name(0),
                diameter=diameter, diam_mean=float(model.net.diam_mean.item()),
                params="CellposeModel.eval defaults (flow_threshold=0.4, cellprob_threshold=0, min_size=15)")
    (out / "inference_meta.json").write_text(json.dumps(meta, indent=2))


def check_repeat(model, m):
    for _, r in m.groupby("dataset").head(1).iterrows():
        a = predict(model, r.image_path)
        b = predict(model, r.image_path)
        print(r.dataset, "n_masks", int(a[0].max()), "masks identical", bool((a[0] == b[0]).all()),
              "max |dP diff|", float(np.abs(a[1] - b[1]).max()),
              "max |cellprob diff|", float(np.abs(a[2] - b[2]).max()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-repeat", action="store_true")
    ap.add_argument("--dataset", help="only this dataset")
    ap.add_argument("--diameter", type=float, help="assumed cell diameter in px (default: no rescaling)")
    ap.add_argument("--run", help="experiment name; writes to out/runs/<run>/ (required with --diameter/--dataset)")
    args = ap.parse_args()
    if (args.diameter or args.dataset) and not args.run:
        ap.error("--diameter/--dataset need --run, so the base run is never overwritten")

    assert torch.cuda.is_available(), "no GPU"
    m = pd.read_parquet(OUT / "manifest.parquet")
    if args.dataset:
        m = m[m.dataset == args.dataset]
        assert len(m), args.dataset
    out = OUT / "runs" / args.run if args.run else OUT
    raw = out / "raw"
    model = models.CellposeModel(gpu=True, pretrained_model=MODEL)
    if args.check_repeat:
        return check_repeat(model, m)

    raw.mkdir(parents=True, exist_ok=True)
    write_meta(model, out, args.diameter)
    times, failed = [], []
    for i, r in enumerate(m.itertuples(), 1):
        dest = raw / f"{r.image_id}.npz"
        if dest.exists():
            continue
        t = time.time()
        try:
            pred, dP, cellprob = predict(model, r.image_path, args.diameter)
            tmp = dest.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, pred=pred, dP=dP, cellprob=cellprob)
            tmp.rename(dest)  # an interrupted run never leaves a half-written result
        except Exception as e:
            failed.append(dict(image_id=r.image_id, error=repr(e)))
            print("FAILED", r.image_id, repr(e), flush=True)
            continue
        times.append(dict(image_id=r.image_id, seconds=time.time() - t, n_pred=int(pred.max())))
        print(f"[{i}/{len(m)}] {r.image_id} masks={int(pred.max())} {times[-1]['seconds']:.1f}s", flush=True)

    if times:
        pd.DataFrame(times).to_csv(out / "inference_times.csv", index=False)
    pd.DataFrame(failed, columns=["image_id", "error"]).to_csv(out / "inference_failed.csv", index=False)
    print(f"done, new={len(times)} failed={len(failed)}")


if __name__ == "__main__":
    main()
