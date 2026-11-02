#!/bin/bash
# Control experiments: YOLO11s on the size- and class-matched frame-separated training subset (three seeds), and
# fine-tuning on UAV-PDD2023 from the stretch-separated HighRPD weights and from COCO weights with three seeds each.
# Appends CONTROLDONE to runs/queue_control.log.
cd "$(dirname "$0")/.."
mkdir -p runs results/runs
export YOLO_CACHE=False YOLO_WORKERS=6 YOLO_BATCH=16
LOG=runs/queue_control.log
run(){ name=$1; model=$2; data=$3; seed=$4; ep=${5:-50}
  if [ ! -f runs/$name/meta.json ]; then
    echo "=== $name start $(date)" >> $LOG
    YOLO_PATIENCE=25 python code/03_train.py $model $data $seed $name $ep > runs/$name.log 2>&1; echo "=== $name end $(date)" >> $LOG
  fi; }
for s in 42 123 3407; do run y11s_sourcematched_s$s yolo11s.pt data/splits/highrpd_sourcematched.yaml $s; done
for s in 42 123 3407; do run ft_uav_stretch_s$s runs/y11s_stretch_s$s/weights/best.pt data/uavpdd_tiles/uavpdd_tiles.yaml $s 30; done
for s in 123 3407; do run ft_uav_coco_s$s yolo11s.pt data/uavpdd_tiles/uavpdd_tiles.yaml $s 30; done
python code/07d_eval_ft.py ft_uav_stretch_s42 ft_uav_stretch_s123 ft_uav_stretch_s3407 ft_uav_coco_s42 ft_uav_coco_s123 ft_uav_coco_s3407 >> $LOG 2>&1
echo CONTROLDONE >> $LOG
