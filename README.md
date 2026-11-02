# highrpd-priority-screening

Code, partitions and results for the article

> M. H. Kesikoglu, A. E. Karkınlı, *Propagation of detection errors into image-derived pavement inspection priority rankings from UAV imagery*, submitted to the International Journal of Pavement Engineering (2026).

The study converts bounding-box detections of line cracks, block cracks and pits in high-altitude (50 m) UAV
imagery into class-specific extent proxies, a 0-100 tile priority score and frame-level summaries, applies the same
rules to the annotations, and measures how detection errors propagate into the priority ranking. Consecutive frames
of the HighRPD survey overlap on the ground, so the main experiments use a stretch-separated partition with a buffer,
which is compared with the usual random and frame-separated partitions. UAV-PDD2023 serves as external data.

## Contents

| Folder | Content |
|---|---|
| `code/` | Numbered scripts of the pipeline (see below) and `inventory.py`, which defines the proxies, the score, the screening bands and the frame aggregation |
| `data/splits/` | Random, frame-separated and stretch-separated partition lists of HighRPD, the pit-oversampled and size-matched training lists, the common contrast tiles and the Ultralytics dataset files |
| `results/` | Tile index, annotation-based inventories, frame registration and overlap audit results, capture blocks, run inventory, proxy check outputs, and under `results/revision/` the analyses of the article |
| `results/runs/` | Evaluation summaries and tile inventories of every training run |
| `figures/` | Figures of the article (PDF and PNG) |

Trained checkpoints are distributed as a release asset (`highrpd-priority-screening-weights.zip`,
https://github.com/Aekarkinli/highrpd-priority-screening/releases). The archive unpacks into the repository root as
`runs/<name>/weights/best.pt` and `runs/<name>/meta.json`, which is all that the evaluation scripts need to reproduce
the reported results without training. `results/run_inventory.csv` lists every run with its data, seed, epochs,
best epoch and the optimiser chosen by the automatic selection of Ultralytics.

## Data

* **HighRPD**: He et al., Data in Brief 59 (2025) 111377, https://doi.org/10.17632/sywswj7djj.1 (CC BY 4.0).
  Extract to `data/HighRPD/images` and `data/HighRPD/labels`.
* **UAV-PDD2023**: Yan and Zhang, Data in Brief 51 (2023) 109692, https://doi.org/10.5281/zenodo.8429208 (CC BY 4.0).
  Extract to `data/external` (folders `Annotations` and `JPEGImages`).

The datasets are not redistributed. `code/00_prepare_index.py` clips the fifty-five HighRPD label files whose
coordinates lie marginally outside the tile and builds the tile index. The shipped partition lists document the
partitions with paths relative to the repository root. Ultralytics requires an absolute `path` entry in the YAML
files, so replace the placeholder in `data/splits/*.yaml` (or rerun `01_make_splits.py` and
`17_make_stretch_split.py`) before training.

## Pipeline

Run every script from the repository root.

| Step | Script | Main output |
|---|---|---|
| Label clipping and tile index | `00_prepare_index.py` | `results/tiles_index.csv` |
| Random and frame-separated partitions | `01_make_splits.py` | `data/splits/{random,source}_*.txt` |
| Frame registration of sequence neighbours | `16_frame_overlap.py` | `results/frame_overlap_pairs.csv` |
| Stretch-separated partition with buffer | `17_make_stretch_split.py` | `data/splits/stretch_*.txt`, `results/stretch_blocks.csv` |
| Annotation-based inventory and reference scales | `02_gt_inventory.py` | `results/tiles_gt_inventory.csv`, `results/reference_scale_p95.json` |
| Exhaustive overlap audit of the final partition | `24_overlap_audit.py`, `28_val_overlap_check.py` | `results/overlap_audit_pairs.csv`, `results/revision/overlap_audit.json` |
| Detector training | `03_train.py`, `04_train_queue.sh`, `04d_stretch_queue.sh` | `runs/<name>/` |
| Size- and class-matched control and fine-tuning seeds | `23_matched_split.py`, `04e_control_queue.sh` | `data/splits/sourcematched_train.txt`, `runs/<name>/` |
| Detection accuracy of every run | `07_evaluate.py`, `07b_decompose.py` | `results/runs/<name>/summary.json` |
| Pixel-level pseudo-reference for the proxies | `05_sam_proxy.py` | `results/sam_*` |
| UAV-PDD2023 conversion and tilings | `06_uavpdd_convert.py`, `06b_uavpdd_tiles.py`, `06c_uavpdd_tiles_gsd.py` | `data/uavpdd_tiles*/` |
| Threshold selection, agreement, baselines, error decomposition, external data | `18_revision_analyses.py` | `results/revision/<tag>/revision.json` |
| Controlled contrast on common tiles | `22_leakage_contrast.py` | `results/revision/leakage_contrast.json` |
| Score design, paired baselines, leave-one-block-out | `25_score_robustness.py` | `results/revision/score_robustness.*` |
| Fine-tuned models and external tiling details | `07d_eval_ft.py`, `26_external_details.py` | `results/runs/ft_*/summary_ft.json`, `results/revision/external_details.json` |
| Run inventory | `27_run_inventory.py` | `results/run_inventory.csv` |
| Exact error decomposition, checkpoint check, block sensitivity, UAV-PDD2023 view audit | `29_error_decomposition.py`, `30_checkpoint_check.py`, `31_block_sensitivity.py`, `32_uavpdd_view_audit.py` | `results/revision/*.json` |
| Sequence-separated UAV-PDD2023 partition and its fine-tuning | `33_uavpdd_seq_split.py`, `04f_uavpdd_seq_queue.sh` | `data/uavpdd_tiles_seq/`, `runs/ft_uav_seq_*` |
| Figures and tables | `09_figures_gt.py`, `19_fig_overlap.py`, `21_figures_ijpe.py`, `20_tables_ijpe.py` | `figures/`, `paper_ijpe/tables/` |

A complete reproduction from the released checkpoints is

```bash
python code/00_prepare_index.py && python code/01_make_splits.py
python code/16_frame_overlap.py && python code/17_make_stretch_split.py && python code/02_gt_inventory.py
python code/24_overlap_audit.py
python code/06_uavpdd_convert.py && python code/06b_uavpdd_tiles.py && python code/06c_uavpdd_tiles_gsd.py
python code/18_revision_analyses.py y11s_stretch y11s_stretch_s42,y11s_stretch_s123,y11s_stretch_s3407 --ext
python code/18_revision_analyses.py y8s_stretch y8s_stretch_s42,y8s_stretch_s123,y8s_stretch_s3407 --ext
python code/18_revision_analyses.py y11s_stretchpit y11s_stretchpit_s42,y11s_stretchpit_s123,y11s_stretchpit_s3407
python code/28_val_overlap_check.py && python code/22_leakage_contrast.py && python code/25_score_robustness.py
python code/07d_eval_ft.py ft_uav_stretch_s42 ft_uav_stretch_s123 ft_uav_stretch_s3407 ft_uav_coco_s42 ft_uav_coco_s123 ft_uav_coco_s3407
python code/33_uavpdd_seq_split.py && python code/07d_eval_ft.py ft_uav_seq_stretch_s42 ft_uav_seq_stretch_s123 ft_uav_seq_stretch_s3407 ft_uav_seq_coco_s42 ft_uav_seq_coco_s123 ft_uav_seq_coco_s3407
python code/26_external_details.py && python code/27_run_inventory.py
python code/29_error_decomposition.py && python code/30_checkpoint_check.py && python code/31_block_sensitivity.py && python code/32_uavpdd_view_audit.py
python code/05_sam_proxy.py && python code/09_figures_gt.py && python code/19_fig_overlap.py && python code/21_figures_ijpe.py && python code/20_tables_ijpe.py
```

Training from scratch runs `bash code/04d_stretch_queue.sh`, `bash code/04e_control_queue.sh` and `bash code/04f_uavpdd_seq_queue.sh` first. Detection
accuracy is computed by the Ultralytics validation routine (confidence 0.001), independently of the operating
threshold used for the inventories.

## Environment

Python 3.14, PyTorch 2.12 (CUDA 12.6), Ultralytics 8.3.161 and the packages in `requirements.txt`.
Training used one 8 GB GPU, and a 50-epoch run takes about 90 minutes.

## Citation

See `CITATION.cff`. Please cite the article when using the partitions, the scoring rules or the results.

## Licence

MIT (see `LICENSE`). The datasets keep their own CC BY 4.0 licences.
