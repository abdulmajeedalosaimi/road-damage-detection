# Road Damage Classification — M2 and M3

## Overview

This folder contains the completed M2–M3 classification experiments for four RDD2022 damage classes:

| Label | Damage type |
|---|---|
| D00 | Longitudinal crack |
| D10 | Transverse crack |
| D20 | Alligator crack |
| D40 | Pothole |

The main notebook is **`Road_Damage_M2_M3_.ipynb`**. It contains the training code, saved outputs, learning curves, model comparisons, validation error analysis, a proposed data-improvement plan, and final test interpretation.

Three models were trained and evaluated:

1. **Scratch:** a small CNN trained from random initialization.
2. **Frozen:** an ImageNet-pretrained ResNet18 with a frozen backbone and a trainable classification head. Backbone BatchNorm statistics remain fixed.
3. **Finetune:** an independently initialized ImageNet-pretrained ResNet18 with all layers trainable.

The fine-tuned ResNet18 checkpoint from **epoch 4** was selected using validation macro F1 before final testing.

## Files and workflow

| File | Purpose |
|---|---|
| `Road_Damage_M2_M3_.ipynb` | Main executable notebook with completed results and interpretation |
| `requirements.txt` | Package versions recorded for the completed experiment |
| `requirements-colab.txt` | Additional dependencies for Colab setup |
| `M1_validation_report.json` | Earlier checks of the supplied M1 manifests |
| `VERIFICATION.md` | Verification notes and limitations |

The `m2m3_*.py` files and `sync_notebook.py` belong to an earlier modular implementation. The completed notebook is self-contained and does not import them. Use the notebook to reproduce the reported results; do not use `sync_notebook.py` to update it.

## Data and evaluation protocol

The experiments use the existing M1 classification crops and split assignments. No new split was created.

| Split | Crops | Countries |
|---|---:|---|
| Training | 12,000 | India, Japan, United States |
| Validation | 9,463 | China, Czech |
| Test | 11,229 | Norway |

Training contains 3,000 crops per class. Validation and test retain their existing class distributions. The notebook checks required manifest fields, valid labels, unique crop paths, and separation of parent images across splits.

The required input files are:

- `classification_manifest.csv`
- `classification_crops.zip`

The original M1 annotations and split manifest remain supporting provenance records.

Each model’s checkpoint is selected by its highest validation macro F1, retaining the earlier epoch on ties. The preferred model is also selected using validation macro F1. All three fixed checkpoints are then evaluated on the held-out Norway test set. Test results were not used for model selection or subsequent tuning.

## Training configuration

| Setting | Value |
|---|---|
| Random seed | 42 |
| Epochs per model | 10 |
| Batch size | 32 |
| Input resolution | 224 × 224 RGB |
| Optimizer | AdamW |
| Weight decay | 0.0001 |
| Scratch learning rate | 0.001 |
| Frozen and finetune learning rate | 0.0003 |
| Loss | Cross-entropy |
| Hardware | Google Colab, NVIDIA Tesla T4 |

Training images are resized and randomly flipped horizontally, then normalized using ImageNet statistics. Validation and test use resizing and normalization without random augmentation.

Predictions use the class with the highest softmax probability. No confidence-rejection threshold is applied. Top-2 accuracy measures whether the true class is among the two highest-scoring classes.

ECE uses ten equal-width confidence bins and the original softmax probabilities. Temperature scaling was not applied.

Python, NumPy and PyTorch seeds are fixed. These results represent one training seed; identical results across different hardware or software environments are not guaranteed.

## Results

All results below use the fixed M1 splits and 224 × 224 inputs. Accuracy columns are percentages; macro F1 and ECE are on a 0–1 scale.

| Model | Best epoch | Validation macro F1 | Test accuracy | Test top-2 accuracy | Test macro F1 | Test ECE |
|---|---:|---:|---:|---:|---:|---:|
| Scratch | 8 | 0.63004 | 54.55% | 86.89% | 0.42669 | 0.08334 |
| Frozen | 10 | 0.72635 | 57.32% | 90.28% | 0.48508 | 0.15102 |
| Finetune | 4 | **0.78494** | **78.45%** | **94.88%** | **0.61935** | **0.06534** |

The selected fine-tuned model achieved **85.65% validation accuracy**. Its test-accuracy 95% confidence interval was **77.46%–79.44%**, calculated using 1,000 bootstrap resamples of parent images, keeping crops from the same parent together. This interval does not measure variation across training seeds or countries.

Per-class precision, recall and F1, confusion matrices, reliability diagrams, losses and learning curves are available in the saved reports and W&B evaluation runs. The notebook also displays the selected model’s test per-class metrics and diagnostic plots.

## Error analysis and limitations

The most frequent validation confusion was **D00 → D20**, with 428 crops. The largest test confusion was **D00 → D40**, with 1,167 crops.

On the test set, D40 recall was 0.8373 but precision was 0.2240, indicating many false-positive pothole predictions. D20 and D40 had lower F1 scores than D00 and D10.

The notebook’s validation gallery deliberately shows eight highly confident mistakes. Their displayed confidence rounds to 1.00; this selected gallery is not representative of all predictions. Several crops appear blurry or have low contrast, and some contain road markings. These observations do not establish that their labels are incorrect.

The proposed training-data improvements are:

- Add diverse examples of easily confused crack types, including road markings and textured backgrounds.
- Review ambiguous training labels against the original images.
- Include varied lighting and sharpness, and test mild blur or contrast augmentation that preserves the damage label.
- Assess changes using validation macro F1, per-class metrics and calibration.

These are proposed future experiments, not improvements already implemented. Validation and test labels were not changed.

Country differences, class imbalance and detail lost during crop resizing may contribute to the validation-to-test performance gap. D00 represents approximately 76.32% of test crops, so accuracy must be interpreted alongside macro F1 and per-class results.

The M1 visual-audit records require reconciliation by image identity before reporting a verified label-error percentage. The manifest checks do not establish visual label correctness.

## Experiment tracking

- [W&B project](https://wandb.ai/alharby6626-a/rdd2022-road-damage)
- Group: `alharby6626_simple_v1`
- [Validation comparison](https://wandb.ai/alharby6626-a/rdd2022-road-damage/runs/21x5c025)
- [Test comparison](https://wandb.ai/alharby6626-a/rdd2022-road-damage/runs/upe9cn2v)

Training, validation evaluation and test evaluation were logged as separate runs for each model. Evaluation runs include per-class tables, confusion matrices, reliability diagrams and learning curves.

## Saved results and checkpoints

The completed experiment was saved under:

`/content/drive/MyDrive/AI Project /Results/m2_m3/alharby6626/simple_v1`

The folder name `AI Project ` contains a trailing space. Teammates can use different locations by changing the notebook’s path settings.

Each model folder contains:

- `best.pt`
- `config.json`
- `history.csv`
- `training_complete.txt`
- `learning_curves.png`
- `wandb_url.txt`
- `val/` and `test/` reports

Evaluation reports include metrics, per-class results, predictions, confusion matrices and calibration outputs. The experiment folder also contains validation and test comparison tables, the validation error list and error gallery.

The selected checkpoint is `finetune/best.pt`. Access checkpoints through the shared project Drive folder; access permissions must allow the intended reviewer or teammate to retrieve them. Datasets and model checkpoints are kept outside GitHub.

## Reviewing or reproducing the experiment

To review the completed work, open the notebook’s saved outputs and follow the comparison links above. Retraining is not required to inspect the results.

For a fresh reproduction:

1. Open `Road_Damage_M2_M3_.ipynb` in a Colab GPU runtime.
2. Use the recorded package versions as the environment reference. The notebook setup pins `wandb==0.22.3` and `protobuf==6.33.5`; the completed run used `torch==2.11.0+cu130` and `torchvision==0.26.0+cu130`.
3. Mount Drive and set `DATA_FOLDER` and `RESULTS_FOLDER` for your account.
4. Choose a new `EXPERIMENT` name to preserve the completed run.
5. Set `FINAL_TEST_READY = False` before running the notebook.
6. Run the cells in order and train all three models using the recorded settings.
7. Review validation results and fix model choices before enabling final testing.
8. Save the notebook with outputs and retain the generated reports, configuration and package versions.

The simple notebook does not resume an interrupted optimizer. It prevents training over an existing checkpoint and prevents repeating final testing in an experiment folder that already contains the test comparison. After reconnecting, completed models can be reviewed without rerunning their training cells.

A teammate reproduction has not yet been independently verified.
