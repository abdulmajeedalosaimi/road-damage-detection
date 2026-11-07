# M5 Pothole Segmentation Report

**Contributor:** Abdulrahman Alsalmi, 225493
**Experiment:** m5_yolo11n_seg_v1
**Prepared from actual execution:** 2026-11-05T17:39:39.113779+00:00

## Task and scope
M5 segments pothole pixels, evaluates the masks and measures error in an area-based severity proxy.
The project guidelines require SAM-generated/refined and visually verified masks, a fine-tuned segmentation
model, mask mAP50-95, mean IoU, Dice and at least one analyzed failure. This work does not change M4.
The Day 6 instance-segmentation lab has now been reviewed. Application-adapted analyses and disclosed architecture-specific substitutions are documented in lab_extension/M5_Lab_Supplement.md.

## Data and preparation
Pothole Videos Dataset version 3 contains 619 RGB/mask pairs: 372 train, 124 validation and 123 test.
All 48 frames of every pair underwent technical decoding checks. The modeling benchmark uses the fixed
12 indices [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47]: 4,464 training, 1,488 validation and 1,476 test frames.
These are sampled-frame results, not an evaluation of all 29,712 source frames.
All frames from one clip stay in its published split. ID, encoded-file and representative-frame checks
are recorded; the similarity screen cannot rule out every repeated pothole identity or location.
Reference foreground is grayscale > 127 before nearest-neighbor resizing to 640 x 640. RGB uses area interpolation.
Raw videos/reference annotations were not edited. The original compressed masks and resizing introduce limitations.

## SAM annotation exercise
SAM-b generated six proposals on training frame 24 using boxes derived from training reference masks.
This is assisted annotation, not an automatic detection test. Proposals are compared with the RGB image
and reference, then visually reviewed. Original reference masks are retained; no unverified proposal is
silently used as ground truth. Review status: **Completed and recorded**.
SAM/reference overlap on these prompted training examples is not model test performance.

## Model and selection
YOLO11n-seg was fine-tuned for up to 30 epochs with batch 8, input 640 x 640,
seed 42, AdamW and recorded augmentation settings. The pretrained model is adapted to one class: pothole.
The best checkpoint is selected by Ultralytics validation fitness (box mAP50-95 + mask mAP50-95 in the pinned version).
Those internal training metrics use polygon targets. Final reported mask metrics below use the reference binary rasters.
Operating confidence was selected from validation by maximum mean foreground Dice (ties prefer a higher threshold):
**0.05**. NMS IoU is 0.7; inference has at most 100 predictions per image.
AP uses confidence floor 0.001 and IoU thresholds 0.50:0.05:0.95, independently of the operating threshold.
Checkpoint fingerprint: `5fb5f6925489060792d160f4a04bc3620c0b59a5de5aba013f2e6f5523254d44`. The protocol/selection are locked before test inference.

## Results and definitions
| Split | Frames | Videos | Mask mAP50-95 | Mean IoU | FG IoU | Dice | Area MAE (pp) | Area RMSE (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 1488 | 124 | 0.6649 | 0.8824 | 0.8146 | 0.8855 | 2.936 | 8.072 |
| test | 1476 | 123 | 0.6856 | 0.9010 | 0.8367 | 0.9041 | 1.844 | 2.745 |

- **Mean IoU:** per-image average of foreground IoU and background IoU, then averaged over frames.
- **Foreground IoU and Dice:** foreground overlap scores, averaged over frames. Every video contributes 12 frames.
- **Mask AP:** COCO segmentation AP against 8-connected reference regions, without area filtering.
  The dataset provides semantic masks, not native instance IDs. This is a derived-region AP evaluation;
  disconnected pieces can be counted separately, and it is not directly interchangeable with a native-instance benchmark.
- **Severity proxy:** foreground union area / total image area, expressed as a percentage.
  MAE/RMSE compare predicted and reference percentages, so errors are in **percentage points**.
  This is not measured road-surface area, pothole depth or a validated engineering severity category.

## Failure analysis
### val
Video 0066, frame 0: foreground IoU 0.0000, Dice 0.0000. There are 10,339 missed foreground pixels (100.00% of the reference area) and 3,266 extra foreground pixels; missed foreground dominates. True area is 2.52% of the image and predicted area is 0.80%, a signed error of -1.73 percentage points. The error map locates the mismatch. Possible explanations to check visually include ambiguous boundaries, water/texture variation and detail lost during resizing; these are hypotheses, not established causes. A future training-only experiment could add reviewed examples with similar appearance or increase input resolution. The current model and threshold remain fixed; test failures are reporting evidence only.

### test
Video 0488, frame 47: foreground IoU 0.0000, Dice 0.0000. There are 34,601 missed foreground pixels (100.00% of the reference area) and 39,069 extra foreground pixels; extra predicted foreground dominates. True area is 8.45% of the image and predicted area is 9.54%, a signed error of 1.09 percentage points. The error map locates the mismatch. Possible explanations to check visually include ambiguous boundaries, water/texture variation and detail lost during resizing; these are hypotheses, not established causes. A future training-only experiment could add reviewed examples with similar appearance or increase input resolution. The current model and threshold remain fixed; test failures are reporting evidence only.


## Limitations and next experiments
This is one seed and a fixed sampled-frame benchmark. Nearby frames are correlated, and splitting by video
does not prove independence by physical pothole/location. The dataset uses top-down views from one region;
generalization to M4 vehicle-mounted road images remains untested. Binary semantic labels do not provide
native instance identities, and polygon conversion can change holes/boundaries. The conversion audit is saved.
Possible future improvements must use training/validation evidence. Test results are for reporting only.
No numerical minimum score is promised; performance must be assessed from the executed results.

## Handoff and references
The package contains the checkpoint, inference helper, fixed threshold, environment, manifests, metrics,
review records and figures. M5 class 0 maps to M4 D40/class 3. See HANDOFF.md for the inference contract.
Raw dataset videos/frames are excluded. Share source data under its license and attribution requirements.

- Project guidelines: Capstone_Project_Guidelines_UPDATED.pdf, M5 requirements (pages 6-7 and 32-33).
- [Pothole Videos Dataset v3](https://data.mendeley.com/datasets/5bwfg4v4cd/3), Ihsan, Harjoko and Amrizal, CC BY 4.0.
- [YOLO11](https://docs.ultralytics.com/models/yolo11/), [segmentation](https://docs.ultralytics.com/tasks/segment/),
  [SAM](https://docs.ultralytics.com/models/sam/), [COCO evaluation implementation](https://github.com/cocodataset/cocoapi).

## Additional laboratory analyses
See lab_extension/M5_Lab_Supplement.md and lab_extension/lab_alignment.csv. The original checkpoint, threshold and primary test metrics are unchanged.
Supplementary W&B run: https://wandb.ai/alharby6626-a/rdd2022-road-damage/runs/1bbce280
