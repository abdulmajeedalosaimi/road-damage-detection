# RDD2022 Road-Damage Classification

This project implements the Phase 1 deliverables for the RDD2022 road-damage dataset.

## 1. Put the raw data here

Extract the **RDD2022 ZIP**. The extracted root must contain country folders such as
`Japan`, `India`, `Czech`, `Norway`, `United_States`, `China_MotorBike`, and `China_Drone`.
You can copy or move the folder to:

```text
rdd2022-road-damage/data/raw/RDD2022/
```

To avoid copying a large dataset, each team member should keep it in their chosen Drive/local folder.
Copy `config.local.example.yaml` to `config.local.yaml` and set `paths.raw_dir` to the shared dataset root.
`config.local.yaml` is ignored by Git, so every team member can use a different local path.

For example:

```text
C:/your-shared-data-folder/Images
```

Example expected files:

```text
data/raw/RDD2022/Japan/train/images/Japan_000001.jpg
data/raw/RDD2022/Japan/train/annotations/xmls/Japan_000001.xml
```

The text files are useful references, but this project needs the **JPG images** and matching **XML labels**.

## 2. Create an environment

Use Python 3.12 or 3.13 for the most reliable PyTorch installation.

```powershell
cd "<your-local-project-folder>\road-damage-detection"
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Select `.venv` in VS Code. Run `wandb login` once if you will log experiments online.

## 3. Run M1: data foundation

```powershell
python scripts/prepare_m1.py --config config.local.yaml
```

It uses labelled **training** folders only; official RDD2022 test labels are withheld.
It holds out exactly one source group as the test set. China Drone and China MotorBike remain together
as the China source group, preventing capture-source leakage.
It creates: an audit report, corrupt-file and duplicate tables, source-disjoint split manifest,
verified annotations, class-balance tables/figure, and a dataset card.

## 4. Build M2/M3 classification crops

```powershell
python scripts/make_classification_crops.py --config config.local.yaml
```

Each ground-truth damage box is converted into a 224x224 crop. Crops inherit their parent image split,
so one image cannot leak into training and testing.

## 5. Train M2 and M3

```powershell
# CNN trained from scratch (M2)
python scripts/train_classifier.py --config config.local.yaml --model scratch

# ImageNet ResNet-18 with frozen backbone (M3)
python scripts/train_classifier.py --config config.local.yaml --model resnet18 --stage frozen

# ImageNet ResNet-18 with full fine-tuning (M3)
python scripts/train_classifier.py --config config.local.yaml --model resnet18 --stage finetune
```

Each run saves the exact configuration, learning curves, best checkpoint, metrics,
confusion matrix, calibration error, and an error gallery in `runs/`. The validation split
selects the best checkpoint; the test split is evaluated only once afterward.

## 6. What to submit for Phase 1

1. `reports/DATASET_CARD.md`, `reports/m1/audit_report.json`, split manifest, and class-balance figure/table (M1).
2. The scratch-CNN and pretrained-model run folders, including W&B run links or reports (M2–M3).
3. `requirements.txt`, `config.yaml`, and the code repository for reproducibility.
