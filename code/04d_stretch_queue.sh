#!/bin/bash
# Training queue for the revised protocol: stretch-separated partition (three seeds, two detectors), additional seeds
# of the random partition, pit oversampling on the stretch partition and a 150-epoch convergence run.
# Waits while a game holds the GPU; evaluates each run right after training; appends STRETCHDONE to runs/queue_stretch.log.
cd "$(dirname "$0")/.."
mkdir -p runs results/runs
export YOLO_CACHE=False YOLO_WORKERS=6 YOLO_BATCH=16
LOG=runs/queue_stretch.log
waitgpu(){ while tasklist 2>/dev/null | grep -qi "cs2.exe"; do echo "waiting for GPU $(date)" >> $LOG; sleep 600; done; }
run(){ name=$1; model=$2; data=$3; seed=$4; ep=${5:-50}; pat=${6:-25}
  if [ ! -f runs/$name/meta.json ]; then
    waitgpu; echo "=== $name start $(date)" >> $LOG
    YOLO_PATIENCE=$pat python code/03_train.py $model $data $seed $name $ep > runs/$name.log 2>&1; echo "=== $name end $(date)" >> $LOG
  fi
  if [ -f runs/$name/meta.json ] && [ ! -f results/runs/$name/summary.json ]; then
    python code/07_evaluate.py $name > results/runs_eval_$name.log 2>&1; python code/07b_decompose.py $name >> results/runs_eval_$name.log 2>&1
    echo "=== eval $name $(date)" >> $LOG
  fi; }
# re-evaluation of the earlier runs with the reference scales of the stretch-separated training subset
for r in y11s_source_s42 y8s_source_s42 y11s_random_s42 y8s_random_s42 y11s_source_s123 y8s_source_s123 y11s_source_s3407 y8s_source_s3407; do waitgpu; run $r; done
run y11s_stretch_s42    yolo11s.pt data/splits/highrpd_stretch.yaml 42
run y8s_stretch_s42     yolov8s.pt data/splits/highrpd_stretch.yaml 42
run y11s_stretch_s123   yolo11s.pt data/splits/highrpd_stretch.yaml 123
run y8s_stretch_s123    yolov8s.pt data/splits/highrpd_stretch.yaml 123
run y11s_stretch_s3407  yolo11s.pt data/splits/highrpd_stretch.yaml 3407
run y8s_stretch_s3407   yolov8s.pt data/splits/highrpd_stretch.yaml 3407
run y11s_random_s123    yolo11s.pt data/splits/highrpd_random.yaml 123
run y8s_random_s123     yolov8s.pt data/splits/highrpd_random.yaml 123
run y11s_random_s3407   yolo11s.pt data/splits/highrpd_random.yaml 3407
run y8s_random_s3407    yolov8s.pt data/splits/highrpd_random.yaml 3407
run y11s_stretch_e150_s42 yolo11s.pt data/splits/highrpd_stretch.yaml 42 150 50
run y11s_stretchpit_s42   yolo11s.pt data/splits/highrpd_stretch_pitx3.yaml 42
run y11s_stretchpit_s123  yolo11s.pt data/splits/highrpd_stretch_pitx3.yaml 123
run y11s_stretchpit_s3407 yolo11s.pt data/splits/highrpd_stretch_pitx3.yaml 3407
echo STRETCHDONE >> $LOG
