#!/bin/bash
# Fine-tuning on UAV-PDD2023 tiles (view-separated split); waits for ALLDONE in runs/queue.log (written by 04_train_queue.sh)
# and writes FTDONE to runs/queue_ft.log when finished.
cd "$(dirname "$0")/.."
mkdir -p runs results/runs
until grep -q ALLDONE runs/queue.log 2>/dev/null; do sleep 120; done
export YOLO_CACHE=False YOLO_WORKERS=6 YOLO_BATCH=16
run(){ name=$1; model=$2; data=$3; seed=$4
  if [ -f runs/$name/meta.json ]; then echo "skip $name"; return; fi
  echo "=== $name start $(date)"; python code/03_train.py $model $data $seed $name 30 > runs/$name.log 2>&1; echo "=== $name end $(date)"; }
run ft_uav_highrpd_s42 runs/y11s_source_s42/weights/best.pt data/uavpdd_tiles/uavpdd_tiles.yaml 42
run ft_uav_coco_s42    yolo11s.pt                            data/uavpdd_tiles/uavpdd_tiles.yaml 42
echo FTDONE >> runs/queue_ft.log; echo FTDONE
