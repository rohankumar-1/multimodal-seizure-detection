# Report metrics (frozen JSON)

These files are the numbers referenced in [`doc.tex`](../../doc.tex). Re-running training may differ slightly; use these for the submitted report.

## Binary detection (`tab:binary_seizure`)

| File | Model |
|------|--------|
| `baselines/matrix_profile_ecg.json` | Matrix Profile (ECG) |
| `baselines/svm_eeg.json` | SVM EEG-only |
| `baselines/chrononet_eeg.json` | ChronoNet v2 EEG |
| `baselines/svm_eeg_ecg.json` | SVM EEG+ECG |
| `supfusion_binary_additive.json` | SupFusion Φ_add |
| `supfusion_binary_bilinear.json` | SupFusion Φ_bil |
| `supfusion_binary_concat.json` | SupFusion Φ_cat |
| `factorcl_binary_kfold_fold0.json` | FactorCL (k-fold fold 0 **validation** AUC 0.903) |

## Lateralization (`tab:lateralization`, `tab:factorcl_views_fold0`)

| File | Notes |
|------|--------|
| `supfusion_lateralization_bilinear_kfold.json` | 5-fold summary; bilinear joint EEG |
| `factorcl_lateralization_pooled.json` | FactorCL M=2, pooled split, **test** AUC 0.963 |
| `factorcl_lateralization_m2_kfold_fold0.json` | M=2 k-fold fold 0 val (view ablation) |
| `factorcl_lateralization_m3_kfold_fold0.json` | M=3 k-fold fold 0 val (view ablation) |

## Robustness (`tab:robustness`)

| File | Task |
|------|------|
| `robustness/binary_label.json` | Detection dropout |
| `robustness/lateralization.json` | Lateralization dropout |

## FactorCL critics (`tab:factorcl_critic_summary`)

| File | Task |
|------|------|
| `critics/binary_label.json` | Detection |
| `critics/lateralization.json` | Lateralization |

Key fields: `mi_critic_batch_mean` (`infonce_xy_eeg`, `infonce_xy_ecg`, `club_x1x2_cond`, …).
