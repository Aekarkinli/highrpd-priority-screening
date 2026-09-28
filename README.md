# highrpd-priority-screening

Code, partitions and results for the article

> M. H. Kesikoglu, A. E. Karkınlı, *Maintenance priority screening of pavement distress from high-altitude UAV imagery*, submitted to Measurement (2026).

The study converts bounding-box detections of line cracks, block cracks and pits in high-altitude (50 m) UAV
imagery into class-specific distress extent proxies, a visual maintenance priority score (VMPS) and section-level
priorities, and measures how detection errors propagate through that chain. It uses the open HighRPD dataset
with a partition by source group, compares it with the usual random tile partition, and applies the same
scoring rules to the UAV-PDD2023 dataset.

## Contents

| Folder | Content |
|---|---|
| `code/` | Numbered scripts of the pipeline (see below) and `inventory.py`, which defines the proxies, the score, the screening bands and the section aggregation |
| `data/splits/` | Random and source-separated partition lists of HighRPD (70/15/15), the pit-oversampled training list and the Ultralytics dataset files |
| `results/` | Tile index, source-group table (Supplementary Table S1), annotation-based inventories, sensitivity analyses, proxy validation outputs and, under `results/runs/`, the evaluation summaries of every training run |
| `figures/` | Figures of the article (PDF and PNG) |

Trained checkpoints are distributed separately as a release asset (`highrpd-priority-screening-weights.zip`,
about 260 MB). It unpacks into the repository root as `runs/<name>/weights/best.pt` and `runs/<name>/meta.json`,
which is all that the evaluation scripts need to reproduce the reported results without training.

## Data

* **HighRPD**: He et al., Data in Brief 59 (2025) 111377, https://doi.org/10.17632/sywswj7djj.1 (CC BY 4.0).
  Extract to `data/HighRPD/images` and `data/HighRPD/labels`.
* **UAV-PDD2023**: Yan and Zhang, Data in Brief 51 (2023) 109692, https://doi.org/10.5281/zenodo.8429208 (CC BY 4.0).
  Extract to `data/external` (folders `Annotations` and `JPEGImages`).

The datasets are not redistributed. `code/00_prepare_index.py` clips the fifty-five HighRPD label files whose
coordinates lie marginally outside the tile and builds the tile index; `code/01_make_splits.py` then reproduces
the partitions of the article exactly (fixed seeds) and rewrites the list and YAML files in `data/splits/` with
absolute paths for the local machine. The shipped copies of these files document the partitions; Ultralytics
requires an absolute `path` entry in the YAML files, so run `01_make_splits.py` before training.

## Pipeline

Run every script from the repository root.

| Step | Script | Main output |
|---|---|---|
| Label clipping and tile index | `00_prepare_index.py` | `results/tiles_index.csv`, `results/clipped_label_files.txt` |
| Tile adjacency check | `00_tile_adjacency_check.py` | `results/tile_adjacency_check.csv` |
| Partitions, oversampled list, partition tables | `01_make_splits.py` | `data/splits/`, `results/table_splits.csv`, `results/supplementary_table_S1_source_groups.csv` |
| Annotation-based inventory, VMPS, section aggregation, sensitivity | `02_gt_inventory.py` | `results/tiles_gt_inventory.csv`, `results/sections_gt_inventory.csv`, `results/reference_scale_p95.json` |
| Detector training | `03_train.py`, `04_train_queue.sh` | `runs/<name>/weights/best.pt`, `runs/<name>/meta.json` |
| Pixel-level pseudo-reference for the proxies (SAM box prompts) | `05_sam_proxy.py` | `results/sam_*.csv`, `results/sam_masks.npz` |
| UAV-PDD2023 conversion and tiling | `06_uavpdd_convert.py`, `06b_uavpdd_tiles.py` | `data/uavpdd_tiles/`, `results/uavpdd_tiles_index.csv` |
| Evaluation of every run | `07_evaluate.py`, `07b_decompose.py`, `07c_eval_queue.sh` | `results/runs/<name>/summary.json`, `decomposition.json` |
| Fine-tuning on UAV-PDD2023 | `04b_finetune_queue.sh`, `07d_eval_ft.py` | `results/runs/ft_*/summary_ft.json` |
| Figures and tables | `08_characterize.py`, `09_figures_gt.py`, `13_figures_revised.py`, `10_tables.py` | `figures/`, `paper/tables/` |

The training queue runs the eleven HighRPD configurations of the article sequentially (50 epochs, batch 16,
early-stopping patience 25, seeds 42, 123 and 3407) and appends `ALLDONE` to `runs/queue.log`; the fine-tuning
queue waits for that marker and appends `FTDONE` to `runs/queue_ft.log`; the evaluation queue evaluates finished
runs as they appear and exits after `FTDONE`. A complete run therefore is

```bash
python code/00_prepare_index.py && python code/01_make_splits.py && python code/02_gt_inventory.py
python code/06_uavpdd_convert.py && python code/06b_uavpdd_tiles.py
bash code/04_train_queue.sh &
bash code/07c_eval_queue.sh &
bash code/04b_finetune_queue.sh &
# after FTDONE:
python code/07d_eval_ft.py ft_uav_highrpd_s42 ft_uav_coco_s42
python code/05_sam_proxy.py && python code/08_characterize.py && python code/09_figures_gt.py && python code/13_figures_revised.py && python code/10_tables.py
```

With the released checkpoints in place, the training and fine-tuning queues can be skipped and the evaluation
scripts called directly, for example `python code/07_evaluate.py y11s_source_s42`.

## Environment

Python 3.14, PyTorch 2.12 (CUDA 12.6), Ultralytics 8.3.161 and the packages in `requirements.txt`.
Training used one 8 GB GPU; a 50-epoch run takes about 90 minutes.

## Citation

See `CITATION.cff`. Please cite the article when using the partitions, the scoring rules or the results.

## Licence

MIT (see `LICENSE`). The datasets keep their own CC BY 4.0 licences.
