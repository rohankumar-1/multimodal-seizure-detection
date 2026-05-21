# Multimodal seizure detection (SeizeIT2)

This repository implements multimodal models for wearable seizure monitoring on the [SeizeIT2](https://arxiv.org/abs/2502.01224) dataset: 2 s windows of EEG (bitemporal) and ECG, bandpassed and resampled to 128 Hz, with labels for seizure vs non-seizure and (among seizures) hemispheric **lateralization** (left vs right). We compare unimodal baselines (SVM, ChronoNet, matrix profile) against two fusion approaches built on shared 1D CNN encoders in [`lib/`](lib/):

- **SupFusion** — supervised fusion with additive, bilinear, or concatenation heads for **binary detection**, and a geo-task variant for **lateralization** / localization (`supfusion.py`, `supfusion_lat_loc.py`).
- **FactorCL** — supervised [FactorCL](https://github.com/pliang279/FactorCL)-style training with InfoNCE / CLUB critics and a joint classifier (`factorcl.py`), using the same encoders and data pipeline for a fair comparison.

Training supports official train/val/test splits and pooled or k-fold evaluation when cohort label balance differs (especially lateralization). Table numbers from the course report are frozen under [`results/report/`](results/report/).

**Authors** (MIT 6.S985, Spring 2025):

- **Rohan Kumar** — Operations Research Center — [`roku@mit.edu`](mailto:roku@mit.edu) — [code](https://github.com/rohankumar-1/mmai)
- **Natalie Barnouw** — Computer Science & Biology — [`nbarnouw@mit.edu`](mailto:nbarnouw@mit.edu) — [code](https://github.com/nbarnouw/multimodal-seizure-detection-nb)
- **Marlene Moerig** — Visiting Researcher, Laboratory for Computational Physiology; Charité – Universitätsmedizin Berlin — [`moerig@mit.edu`](mailto:moerig@mit.edu)

## Repository layout

| Path | Purpose |
|------|---------|
| `preprocess.py` | Bandpass, resample, z-score; build `train/val/test.npz` |
| `chrononet.py`, `svm.py`, `matrix_profile.py` | Unimodal baselines |
| `lib/` | `splits`, `metrics`, `cli`; `supfusion/` and `factorcl/` runners |
| `supfusion.py` | Thin CLI → binary SupFusion (`lib/supfusion/binary.py`) |
| `supfusion_lat_loc.py` | Thin CLI → geo SupFusion (`lib/supfusion/geo.py`) |
| `factorcl.py` | Thin CLI → FactorCL (`lib/factorcl/runner.py`) |
| `supfusion_lib/` | Deprecated shim re-exporting `lib` (backward compatibility) |
| `metrics.py` | Shared evaluation helpers |
| `scripts/eval_dropout.py` | Modality-dropout robustness (FactorCL vs SupFusion) |
| `scripts/eval_critics.py` | FactorCL bucket critic summary |
| `scripts/1_preprocess.ipynb`, `scripts/2_preprocess.ipynb` | Exploratory preprocessing notebooks (see also `preprocess.py`) |
| `results/report/` | JSON snapshots used in the paper tables |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

SeizeIT2 data under `data/` is not shipped in this repo (paths are gitignored). Model code lives in `lib/` (no external `multimodal` package). `import supfusion_lib` still works via a thin re-export shim.

## Preprocessing

```bash
python preprocess.py \
  --train_npz data/train_data_2sec.npz \
  --val_npz data/val_data_2sec.npz \
  --test_npz data/test_data_2sec_run_only.npz \
  --out_dir data/processed_2s_128Hz \
  --desired_fs 128
```

## Training (examples)

```bash
# Unimodal baselines
python chrononet.py
python svm.py
python matrix_profile.py

# SupFusion binary detection (three fusion ablations)
python supfusion.py --fusion add
python supfusion.py --fusion bil
python supfusion.py --fusion cat

# Lateralization: bilinear joint EEG (paper table row)
python supfusion_lat_loc.py --task lateralization --mode kfold --fusion-type bilinear

# FactorCL
python factorcl.py --label binary_label --mode kfold
python factorcl.py --label lateralization --mode pooled
```

Checkpoints are written under `runs/` (gitignored). Copy the best checkpoint paths into the eval commands below.

## Paper metrics (reproduce tables)

```bash
# Modality dropout → Table robustness
PYTHONPATH=. python scripts/eval_dropout.py \
  --label binary_label \
  --factorcl-ckpt runs/factorcl_binary_label_kfold_seed0_k5/fold0/factorcl_best_val_auc.pt \
  --supfusion-ckpt runs/supfusion/supfusion_bil.pt

PYTHONPATH=. python scripts/eval_dropout.py \
  --label lateralization --keep-classes 2,3 \
  --factorcl-ckpt runs/factorcl_lateralization_pooled_seed0_v10_t10/factorcl_best_val_auc.pt \
  --supfusion-ckpt runs/supfusion_lat_loc/lateralization/kfold_seed0_k5_bilinear_joint/fold0/best.pt

# FactorCL critics → critic summary table
PYTHONPATH=. python scripts/eval_critics.py --task binary_label \
  --factorcl-ckpt runs/factorcl_binary_label_kfold_seed0_k5/fold0/factorcl_best_val_auc.pt

PYTHONPATH=. python scripts/eval_critics.py --task lateralization --keep-classes 2,3 \
  --factorcl-ckpt runs/factorcl_lateralization_kfold_seed0_k5_joint/fold0/factorcl_best_val_auc.pt
```

Outputs go to `results/report/robustness/` and `results/report/critics/`. See [`results/report/README.md`](results/report/README.md) for which JSON maps to each table in `doc.tex`.

## Citation

If you use this code, cite the SeizeIT2 dataset paper and FactorCL (`references.bib` in this repo).
