# multimodal-seizure-detection
Multimodal approaches to seizure detection on SeizeIT2 dataset


## TODO:

For the midterm proposal, we need to reimplement current SOTA models. First, we should do baseline unimodal models. 

- https://arxiv.org/abs/2502.01224 (original dataset paper)
    - [ ] implement ChronoNet (multimodal)
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

