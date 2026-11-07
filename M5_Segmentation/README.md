# M5: Pothole Segmentation

**Contributor:** Abdulrahman Alsalmi, 225493  
**Branch:** `feature/m5-segmentation`  
**Experiment:** `m5_yolo11n_seg_v1`

This folder contains the executed M5 notebook, inference code, configuration,
reports, figures and evaluation evidence. YOLO11n-seg predicts pothole masks and
estimates the fraction of the input image covered by potholes. The additional
instance-segmentation laboratory analyses are adapted to this project and are
documented separately from the original evaluation.

## Start here

- [Executed notebook](M5_Pothole_Segmentation.ipynb): code and saved outputs.
- [Main M5 report](M5_Report.md): method, results, failure cases and limitations.
- [Laboratory supplement](lab_extension/M5_Lab_Supplement.md): additional comparisons.
- [Lab mapping](lab_extension/lab_alignment.csv): application adaptations and substitutions.
- [Inference handoff](HANDOFF.md): input/output contract and M4 class mapping.
- [Completion checklist](completion_checklist.csv): recorded deliverables and evidence.

The notebook already contains results. Training does not need to run again to
inspect this submission. Sections 1-19 contain the original experiment; Sections
20-29 contain the supplementary analyses and their export.

## Data and evaluation

The Pothole Videos Dataset v3 contains 619 paired RGB/mask clips: 372 training,
124 validation and 123 test. We use 12 fixed frames per clip, giving 4,464 training,
1,488 validation and 1,476 test frames. Whole clips remain in their published
splits. Video IDs do not overlap; this alone does not prove independence of every
physical pothole or location.

The saved one-class YOLO11n-seg model was fine-tuned for 30 epochs. Checkpoint
selection uses validation fitness. Confidence 0.05 was selected using validation
foreground Dice and kept fixed for the test evaluation.

| Test measure | Result |
|---|---:|
| Mask mAP50-95 | 0.68555 |
| Box mAP50-95 | 0.71902 |
| Mean IoU | 0.90102 |
| Foreground IoU | 0.83675 |
| Foreground Dice | 0.90409 |
| Area MAE | 1.84448 percentage points |

Mean IoU averages foreground and background IoU per image, then averages images.
The source provides binary semantic masks, so instance AP uses derived
8-connected foreground regions. Area percentage is an image-area proxy, not
pothole depth or a validated engineering severity category.

On the fixed 100-image validation subset with identical retained detections,
YOLO mask AP is 0.65856 and SAM mask AP is 0.35349. This conditional comparison
is separate from unrestricted headline AP. The original model and operating
threshold remain unchanged. The initialization baseline is reconstructed;
the empty-mask control is synthetic, not a second training run.

## Experiment records

- [Original model experiment](https://wandb.ai/alharby6626-a/rdd2022-road-damage/runs/65y07alg)
- [Additional laboratory analyses](https://wandb.ai/alharby6626-a/rdd2022-road-damage/runs/1bbce280)

These W&B links require the appropriate project access. Access from another
account must be checked using that account.

## Model weights and complete archive

Following the repository's TEAM_WORKFLOW.md, model checkpoints, raw videos,
extracted dataset frames and large outputs stay on the shared Google Drive.
They are intentionally not tracked in this GitHub folder.

Within the shared **AI Project / M5_Segmentation** folder:

- Selected model and complete evidence: `exports/m5_yolo11n_seg_v1_with_lab/`
- Selected checkpoint: `exports/m5_yolo11n_seg_v1_with_lab/best.pt`
- Complete archive: `exports/m5_yolo11n_seg_v1_submission_with_lab.zip`
- The archive also contains the reconstructed initialization and full COCO
  reference/prediction JSON files omitted from this lighter GitHub copy.

The original selected model is also recorded as a model artifact in the original
W&B experiment. Its SHA-256 is:

`5fb5f6925489060792d160f4a04bc3620c0b59a5de5aba013f2e6f5523254d44`

`github_manifest.json` lists the files in this GitHub copy. The complete archive
has its own `artifact_manifest.json` for all submitted artifacts.

## Run inference

Obtain the selected `best.pt` from Drive or the original W&B model artifact and
place a local copy in this folder. Checkpoints are ignored by this folder's
`.gitignore`. Install a hardware-compatible PyTorch/torchvision pair, then install
the packages recorded in `requirements.txt`. The original environment is recorded
in `environment.json`.

From this folder:

```python
from m5_inference import PotholeSegmenter

segmenter = PotholeSegmenter('best.pt', 'inference_config.json', device='cpu')
result = segmenter.predict('example.jpg')
print(result['area_percent'])
```

The result contains the union mask, individual masks and confidence scores, and
the mapping M5 class 0 -> M4 class 3 / D40. Masks return at the input image's
original resolution. M4-to-M5 integration and deployment latency have not been
validated by this work.

## Reproduce the notebook

Open the notebook in Colab with a GPU. The saved experiment uses the shared
`AI Project` folder in MyDrive and `pothole_video.zip`. Keep the original dataset
and complete exported experiment in their recorded Drive locations. Sections
20-29 can reconnect to that saved experiment without retraining. They require
the saved original exports, including `best.pt`, configuration and selection
records; this GitHub copy alone does not contain those model files.

## Dataset attribution

[Pothole Videos Dataset, version 3](https://data.mendeley.com/datasets/5bwfg4v4cd/3),
Ihsan, Harjoko and Amrizal, CC BY 4.0. See the main report for references and
limitations. Displayed dataset examples are retained as evaluation/report figures.
