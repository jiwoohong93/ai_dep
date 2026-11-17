#!/bin/sh
# Download the trained weights (baseline_vitb.pth, fairfundusnet.pth) into ./release/
# requires gdown (included in requirements.txt)
cd "$(dirname "$0")" && python -m gdown --folder "https://drive.google.com/drive/folders/1GIr2s0Wj4X_9Mz5KdDaHFXeths3KG97T" -O release
