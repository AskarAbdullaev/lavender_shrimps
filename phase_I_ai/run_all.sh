#!/usr/bin/env bash
# Rebuild the atlas from the raw datasets. Takes about 45 minutes on one GPU, mostly inference.
set -euo pipefail
cd "$(dirname "$0")/pipeline"

python 01_manifest.py                      # image list, annotation masks
python 02_infer.py                         # cpsam_v2 at default settings
for d in 60 80 100; do                     # glioma again with an assumed cell size
  python 02_infer.py --dataset glioma --diameter "$d" --run "glioma_d$d"
done
python 03_match.py                         # outcome per annotated cell
for d in 60 80 100; do python 03_match.py --run "glioma_d$d"; done
python 04_features.py                      # input features per cell
python 05_atlas.py                         # atlas/atlas.json and atlas/img/
