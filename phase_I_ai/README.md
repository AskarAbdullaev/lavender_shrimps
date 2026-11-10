# CellposeSAM Atlas

An interactive page that shows how CellposeSAM (`cpsam_v2`, Cellpose 4.2.1.1) behaves on 47,428 annotated cells from four microscopy datasets. Each cell sits on a UMAP map of input features measured without the model. The right card shows what happened to the selected cells: correct, inaccurate boundary, merged, split or missed. Clicking a cell opens its image with the model's masks drawn on.

This folder holds only what is needed to rebuild that page from the raw datasets.

## Contents

| Path | What it does |
|---|---|
| `pipeline/common.py` | Paths and the shared signal channel |
| `pipeline/01_manifest.py` | Lists every image and writes the annotations as label masks |
| `pipeline/02_infer.py` | Runs `cpsam_v2` with default settings; `--diameter` adds the rescaled glioma runs |
| `pipeline/03_match.py` | Matches masks to annotations and gives each annotated cell one outcome |
| `pipeline/04_features.py` | Measures the input features of each cell (size, elongation, contrast, …) |
| `pipeline/05_atlas.py` | Fits the two UMAP maps and writes `atlas/atlas.json` and the overlay images in `atlas/img/` |
| `atlas/index.html` | The page itself; loads `atlas.json` and `img/` from its own folder |
| `run_all.sh` | Runs steps 01 to 05 in order |

## Data

Put the four datasets in `data/` (or point `ATLAS_DATA` at the folder that holds them), with this layout:

```
data/
  official_dataset/test/            Cellpose dataset, test split (*_img.png, *_masks.png)
  Breast cancer cell dataset/Training dataset/{Training_source,Training_target}/*.tif
  Flow chamber dataset/Training dataset/{Training_source,Training_target}/*.tif
  glioma_c6_dataset/{gen,spec/train,spec/valid,spec/test}/   images/ and anno_*.json (COCO)
```

The official training split is deliberately left out, because the model was probably trained on it.

Sources: the Cellpose dataset (cellpose.org, download terms apply), the breast cancer cell and flow chamber datasets from Fazeli et al. 2020 (Zenodo 4034976 and 4034939, CC BY 4.0), and Glioma C6 by Malashin et al. 2025 (Zenodo 15083188).

## Rebuild

Needs Python 3.12 and an NVIDIA GPU.

```bash
pip install -r requirements.txt
./run_all.sh
```

This takes about 45 minutes, mostly inference. It writes about 15 GB of intermediate files to `out/` and the page data to `atlas/`. Inference is deterministic: running an image twice gives identical masks on the same machine.

## View

The page loads its data with `fetch`, so it needs a local web server rather than opening the file directly:

```bash
python -m http.server 8767 --bind 127.0.0.1 --directory atlas
```

Then open http://localhost:8767. To share it, upload the whole `atlas/` folder (index.html, atlas.json, img/) to any static host.
