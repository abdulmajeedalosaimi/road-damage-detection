# M2 + M3 — Road Damage Classification

This package builds on the reviewed M1 files. **It contains a training workflow, not completed project training results.**

## Getting started

Open Road_Damage_M2_M3_Colab.ipynb in Google Colab. The notebook is self-contained and includes the complete code and instructions in English. Separate Python file uploads are not required.

1. Select a GPU runtime.
2. Mount Drive and set DATA_FOLDER to your data directory.
3. Confirm that the directory contains:
   - annotations.csv
   - classification_manifest.csv
   - split_manifest.csv
   - classification_crops.zip
4. Complete manifest and image validation, then sign in to W&B.
5. Run scratch, frozen and finetune in that order.
6. Review validation results and error images. Enable FINAL_TEST_READY only after completing this review and fixing the experiments.

The default data path is an example that must be updated. Results are saved to a separate folder in Drive. If Colab disconnects, rerun the setup cells and the relevant training cell using the same configuration and output folder to resume from the last completed epoch.

Training time depends on the available hardware. If memory runs out, start a new experiment with a new output folder and a smaller batch_size, and record the change.

## Package contents

| File | Purpose |
|---|---|
| Road_Damage_M2_M3_Colab.ipynb | Self-contained Colab workflow with English instructions |
| m2m3_data.py | Manifest checks, Windows path normalization and image verification |
| m2m3_models.py | SmallCNN, ResNet18 and correct backbone freezing |
| m2m3_metrics.py | Metrics, figures, error samples and a provisional improvement plan |
| m2m3_train.py | Training, checkpoints, resuming, W&B tracking and final evaluation |
| sync_notebook.py | Refreshes the embedded code after editing the Python modules |
| M1_validation_report.json | Results and file hashes from validation of the actual M1 CSV files |
| VERIFICATION.md | Completed software checks and their limitations |
| requirements-colab.txt | Additional W&B dependency; actual runtime versions are recorded during execution |

After editing the Python modules, run python sync_notebook.py and upload the updated notebook. Edit the modules rather than the long embedded source-code cell.

## Verified M1 manifest properties

- The split manifest contains 38,384 original images; there are 55,002 annotations and 32,692 crops.
- Crop counts: train = 12,000; validation = 9,463; test = 11,229.
- Every crop matches a source image_id/object_index, class and split.
- Validation and test crops exactly cover the annotations in their respective splits.
- The training subset contains 3,000 crops per class.
- Training sources: India, Japan and United_States. Validation sources: China and Czech. Test source: Norway.
- No parent image crosses splits in the manifests, and no recorded SHA256 value is duplicated.
- Japan_Japan_003187, object 0, is excluded from the crop manifest; its original annotation remains present.
- All annotation bounding boxes are within their parent-image dimensions.

These are manifest checks. The notebook verifies the actual project images in Colab when classification_crops.zip is available. The existing M1 crop-verification report states that the images were checked previously; that project-image check was not repeated during package preparation.

## Label-audit issue requiring reconciliation

Some M1 audit decisions appear to have been transferred by row number between different samples. Rows 27 and 44 in the crop-audit CSV refer to different images from those associated with the decisions in the original image sample:

- Crop sample: row 27 = China_MotorBike_China_MotorBike_001424; row 44 = Norway-002_Norway_007278.
- Original sample described in the notebook: 027 = China_Drone_China_Drone_001616; 044 = Japan_Japan_003187.

A shared row number does not establish that a decision applies to both samples. Reconcile decisions with the M1 author using image_id/object_index before reporting a label-error rate or changing labels. This package does not use the disputed decisions to remove images or change M1 splits.

## Experiment design

SmallCNN retains the architecture from the existing training script. ResNet18 uses ImageNet1K_V1 weights. The frozen and finetune experiments independently start from the same ImageNet weights; fine-tuning does not resume the frozen-head experiment.

Training transforms resize images to 224 × 224, apply horizontal flipping and rotation of up to 10 degrees, and use ImageNet normalization. Evaluation uses resizing and normalization only. RandomResizedCrop was removed from this training workflow to reduce the risk of cutting damage out of an existing crop. This does not undo distortion or detail loss introduced when the M1 crops were created.

Select each checkpoint using the highest validation macro F1, retaining the earlier epoch on ties. The test split is not used for training or checkpoint selection. After fixing the experiments, evaluate all three models on Norway. Training and final evaluation use separate W&B runs within the same group, with an additional combined comparison.

Each run records the configuration, seed, data and code hashes, and package versions. This practical comparison does not isolate pretraining while holding architecture fixed. Additional seeds can measure training variability if resources permit; no multi-seed project experiments have been completed here.

## Interpreting outputs

- Macro F1 gives equal weight to the four classes. Accuracy can appear high because D00 dominates the Norway test set.
- Confusion matrices show both raw counts and normalization by the true class.
- ECE uses 10 equal-width confidence bins. The reliability diagram compares accuracy with confidence.
- Probabilities are raw softmax outputs; this version does not apply temperature scaling.
- Multiclass Brier score is the mean of the summed squared probability errors across classes.
- The accuracy confidence interval uses 2,000 bootstrap draws over parent images, keeping all crops from a sampled image together. It does not measure variability across training seeds or new countries.
- Top-2 accuracy uses stable ordering when probabilities tie.
- Country-level macro metrics use the same four-class label set; support is documented in the reports.
- Training metrics are measured during optimization on augmented images; validation metrics use evaluation mode.

The generated data-improvement plan uses validation statistics to prioritize classes and confusion pairs. It is **provisional**: complete error_review.csv with visual observations, then link proposed actions to verified causes. Metrics alone do not establish why an error occurred.

## Saving and submission

The notebook saves checkpoints, learning curves, metric JSON files, per-class probabilities for every crop, confusion matrices, reliability diagrams, an error-review sample and an improvement plan to Drive. It also saves the actual environment's requirements.lock.txt and a copy of the code.

Upload the code and notebook to the team's working branch for review. Keep images, large checkpoints and account credentials outside the GitHub repository. Package preparation did not modify M1 source files or upload anything to GitHub.

## Implementation references

- [PyTorch ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [PyTorch BatchNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
- [W&B logging](https://docs.wandb.ai/models/track/log)
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)

