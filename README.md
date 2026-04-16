# multimodal-seizure-detection
Multimodal approaches to seizure detection on SeizeIT2 dataset

## Layout (minimal)

| File | Role |
|------|------|
| `preprocess.py` | Bandpass, notch, resample, z-score from train; `SupervisedMultimodalDataset`; optional `python preprocess.py --train_npz ...` |
| `metrics.py` | NN / SVM / matrix-profile metrics |
| `chrononet.py` | ChronoNet train + test (`python chrononet.py`) |
| `svm.py` | Multimodal SVM on flattened windows (`python svm.py`) |
| `matrix_profile.py` | ECG matrix profile + eval (`python matrix_profile.py`) |

From the repo root, edit paths inside each script’s `main()`, then run it. Preprocess once:

```bash
python preprocess.py --train_npz data/train_data_2sec.npz --val_npz data/val_data_2sec.npz \
  --test_npz data/test_data_2sec_run_only.npz --out_dir data/processed_fs128 --desired_fs 128
```

## TODO:

For the midterm proposal, we need to reimplement current SOTA models. First, we should do baseline unimodal models. 

- https://arxiv.org/abs/2502.01224 (original dataset paper)
    - [ ] implement ChronoNet (unimodal)
    - [ ] implement SVM (multimodal)

> unsure on actual AUCs of the models in the paper


- https://www.mdpi.com/1424-8220/25/24/7687 (new paper comparing multiple ECG-onlymodels)
    - [ ] implement MatrixProfile (unimodal, ECG)
    - [ ] implement MADRID model (unimodal, ECG)
    - [ ] implement TimeVQVAE-AD (unimodal, ECG)

> preprocessing was 0.5-40Hz butter band filtering, downsampling to 8Hz, z-score normalization
> also did postprocessing to remove artifacts (temporal clustering, anomaly merges)
> hyperparameter tuning over window size (2s to 900s, log-scale)




### Other

[Survey of multimodal approaches to seizure detection](https://arxiv.org/pdf/2601.05095)

