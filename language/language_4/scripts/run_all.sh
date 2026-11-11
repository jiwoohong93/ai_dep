#!/bin/bash
# Full pipeline, in order. Large artifacts go to paths.{hf_home,acts,steer}. Uses GPUs G0 and G1 (default 0, 1).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
CFG=${CFG:-configs/default.yaml}
G0=${G0:-0}; G1=${G1:-1}
$PY scripts/01_prepare_data.py --config $CFG
CUDA_VISIBLE_DEVICES=$G0 $PY scripts/02_extract.py --config $CFG --sets kobbq_trainval,kmmlu_val &
CUDA_VISIBLE_DEVICES=$G1 $PY scripts/02_extract.py --config $CFG --sets benign,caa_unknown,caa_stereo &
wait
CUDA_VISIBLE_DEVICES=$G0 $PY scripts/03_baseline.py --config $CFG
$PY scripts/04_directions.py --config $CFG
CUDA_VISIBLE_DEVICES=$G0 $PY scripts/05_alphasteer.py --config $CFG
$PY -m pytest -q tests
$PY scripts/06_layer_select.py --config $CFG
for st in screen1 screen2 full; do
  CUDA_VISIBLE_DEVICES=$G0 $PY scripts/07_sweep.py --config $CFG --stage $st --worker 0 --n_workers 2 &
  CUDA_VISIBLE_DEVICES=$G1 $PY scripts/07_sweep.py --config $CFG --stage $st --worker 1 --n_workers 2 &
  wait
done
$PY scripts/07_sweep.py --config $CFG --stage select
CUDA_VISIBLE_DEVICES=$G0 $PY scripts/08_final_eval.py --config $CFG
