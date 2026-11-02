# M4 — Road Damage Object Detection (YOLO11n)

## Scope
M4 uses the M1 preprocessing outputs for four-class road-damage object detection (D00, D10, D20, D40). This module is separate from M2/M3 classification.

## Data protocol
- Training: 23,016 images.
- Validation: 7,207 images.
- Independent country holdout: Norway, 8,161 test images, 11,229 annotated damage instances.
- Norway is excluded from model training, validation, and hyperparameter selection.

## Baseline
Ultralytics YOLO11n, 30 epochs, image size 640, batch size 16, seed 42, deterministic training, Tesla T4 GPU. Model checkpoints and dataset files reside outside GitHub in the M4 Google Drive project folder.

## Reported independent Norway evaluation
| Metric | Result |
|---|---:|
| Precision | 0.2410 |
| Recall | 0.1235 |
| mAP@50 | 0.0792 |
| mAP@50–95 | 0.0319 |

Per-class mAP@50: D00 0.131; D10 0.0656; D20 0.0907; D40 0.0297.

These are recorded baseline results, not recalculated by this documentation commit. Cross-country generalization is weak, especially for D40. The Norway holdout must not be used to select a revised model.

## Files and reproducibility
The cleaned notebook is stored separately as `M4_Object_Detection_Cleaned.ipynb` in the Google Drive M4 folder. **The notebook is not yet included in this GitHub commit.** It depends on Google Drive paths, a prepared YOLO dataset, and saved checkpoints. Do not use `Run all` without reviewing the expensive extraction, copying, training, and test-evaluation cells.

The complete dataset, model checkpoints, and large training outputs should not be committed to GitHub. The final reports and YAML configuration should be added separately once available for upload.

## Limitations
- Significant Norway domain shift and class imbalance.
- Images without supported bounding-box labels are not necessarily confirmed negatives.
- Google Drive absolute paths and temporary Colab local storage reduce portability.
- The notebook's outputs were not embedded in the uploaded copy.
