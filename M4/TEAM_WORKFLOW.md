# Team workflow

## What belongs where

| Location | Store |
|---|---|
| GitHub | Code, `README.md`, configurations, `split_manifest.csv`, audits, small report figures, and dataset card |
| Shared Drive | Raw RDD2022 images/XML files, classification crops, pothole videos/masks, model checkpoints, and large outputs |
| Each member's computer | A cloned GitHub repository and a local `config.local.yaml` pointing to their data location |

Never commit raw images, videos, checkpoints, `wandb/`, or `runs/` to GitHub.

## Fixed leakage-safe split

1. The M1 owner runs the audit on RDD2022 labelled training data only.
2. M1 publishes the generated `data/processed/split_manifest.csv`, audit summary, class-balance figure, and dataset card.
3. Every later phase uses exactly this manifest. Do not create independent random splits.
4. One country/source group is held out for test. China Drone and China MotorBike are one China group and must stay in the same split.
5. For M5, split Pothole Videos by whole clip before extracting frames; frames from one clip must never cross splits.

## Branches

Create a branch before editing shared code:

```text
m1-data-foundation
m2-scratch-classifier
m3-transfer-learning
m4-object-detection
m5-segmentation
```

Open a pull request for review before merging into `main`.
