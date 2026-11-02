#!/bin/bash
# Fine-tuning on the sequence-separated UAV-PDD2023 partition (code/33_uavpdd_seq_split.py) from the stretch-separated
# HighRPD weights of the same seed and from COCO weights, three seeds each. Appends SEQDONE to runs/queue_seq.log.
cd "$(dirname "$0")/.."
mkdir -p runs results/runs
export YOLO_CACHE=False YOLO_WORKERS=6 YOLO_BATCH=16
LOG=runs/queue_seq.log
D=data/uavpdd_tiles_seq/uavpdd_tiles_seq.yaml
run(){ name=$1; model=$2; seed=$3
  if [ ! -f runs/$name/meta.json ]; then
    echo "=== $name start $(date)" >> $LOG
    YOLO_PATIENCE=25 python code/03_train.py $model $D $seed $name 30 > runs/$name.log 2>&1; echo "=== $name end $(date)" >> $LOG
  fi; }
for s in 42 123 3407; do run ft_uav_seq_stretch_s$s runs/y11s_stretch_s$s/weights/best.pt $s; done
for s in 42 123 3407; do run ft_uav_seq_coco_s$s yolo11s.pt $s; done
python code/07d_eval_ft.py ft_uav_seq_stretch_s42 ft_uav_seq_stretch_s123 ft_uav_seq_stretch_s3407 ft_uav_seq_coco_s42 ft_uav_seq_coco_s123 ft_uav_seq_coco_s3407 >> $LOG 2>&1
echo SEQDONE >> $LOG
