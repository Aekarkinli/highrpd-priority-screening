#!/bin/bash
# Sequential training queue (GPU-bound). Epochs=50, patience=25, batch=16. Writes ALLDONE to runs/queue.log when finished.
cd "$(dirname "$0")/.."
mkdir -p runs results/runs
export YOLO_CACHE=False YOLO_WORKERS=6 YOLO_BATCH=16
run(){ name=$1; model=$2; data=$3; seed=$4
  if [ -f runs/$name/meta.json ]; then echo "skip $name"; return; fi
  echo "=== $name start $(date)"; python code/03_train.py $model $data $seed $name 50 > runs/$name.log 2>&1; echo "=== $name end $(date)"; }
run y11s_source_s42   yolo11s.pt data/splits/highrpd_source.yaml 42
run y8s_source_s42    yolov8s.pt data/splits/highrpd_source.yaml 42
run y11s_random_s42   yolo11s.pt data/splits/highrpd_random.yaml 42
run y8s_random_s42    yolov8s.pt data/splits/highrpd_random.yaml 42
run y11s_pit_s42      yolo11s.pt data/splits/highrpd_source_pitx3.yaml 42
run y11s_source_s123  yolo11s.pt data/splits/highrpd_source.yaml 123
run y8s_source_s123   yolov8s.pt data/splits/highrpd_source.yaml 123
run y11s_pit_s123     yolo11s.pt data/splits/highrpd_source_pitx3.yaml 123
run y11s_source_s3407 yolo11s.pt data/splits/highrpd_source.yaml 3407
run y8s_source_s3407  yolov8s.pt data/splits/highrpd_source.yaml 3407
run y11s_pit_s3407    yolo11s.pt data/splits/highrpd_source_pitx3.yaml 3407
echo ALLDONE >> runs/queue.log; echo ALLDONE
