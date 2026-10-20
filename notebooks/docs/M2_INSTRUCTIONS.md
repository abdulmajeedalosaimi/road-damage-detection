# M2 – Model Development and Training Instructions

**Project:** Road Damage Detection (RDD2022)  
**Repository:** https://github.com/abdulmajeedalosaimi/road-damage-detection  
**Assigned module:** M2 – Model Development, Training and Evaluation  
**Previous module:** M1 – Data Preprocessing and Verification

## 1. GitHub Repository and Collaboration

### 1.1 Clone and update the repository

Clone the repository to your own computer:

```bash
git clone https://github.com/abdulmajeedalosaimi/road-damage-detection.git
cd road-damage-detection
```

If the repository is already cloned, retrieve the latest approved version:

```bash
git switch main
git pull origin main
```

Confirm that the latest M1 preprocessing work is available.

### 1.2 Create a dedicated M2 branch

Do not develop directly on the main branch.

```bash
git switch -c feature/m2-model-training
```

All M2 changes should initially remain within this branch.

### 1.3 Protect completed M1 work

Do not modify or delete:

- `notebooks/M1_Preprocessing_Overview.ipynb`
- Existing M1 preprocessing scripts
- Original raw data
- M1 audit reports and decisions
- M1 train/validation/test assignments
- The approved configuration without coordination

Create new M2 files instead of overwriting M1 files.

### 1.4 Git workflow

Commit changes regularly using descriptive messages.

Example:

```bash
git add notebooks/M2_Model_Training.ipynb
git commit -m "M2: Add initial model training notebook"
git push -u origin feature/m2-model-training
```

Open a pull request when the work is ready for review.

Do not merge into `main` without approval.

## 2. Dataset Preparation and Utilization

### 2.1 Use verified M1 outputs

The M1 preprocessing stage produced the following reported outputs:

| Dataset component              |  Count |
| ------------------------------ | -----: |
| Original annotations           | 55,002 |
| Generated classification crops | 32,692 |
| Training crops                 | 12,000 |
| Validation crops               |  9,463 |
| Testing crops                  | 11,229 |

The difference between original annotations and generated crops is primarily explained by intentional training-set sampling.

The training set was limited to 12,000 crops, while the validation and test crops were retained without the same sampling cap.

### 2.2 Required M1 data files

Use the existing project outputs, including:

- `data/processed/classification_manifest.csv`
- `data/processed/classification_crops/`
- `data/processed/split_manifest.csv`
- `data/processed/annotations.csv`

These files may need to be transferred separately or regenerated using the approved preprocessing pipeline because processed datasets are generally excluded from Git tracking.

Do not assume that cloning GitHub automatically downloads all processed images.

### 2.3 Never create a new random split

The existing M1 split must be preserved.

| Split      | Purpose          | Allowed use                                             |
| ---------- | ---------------- | ------------------------------------------------------- |
| Train      | Model learning   | Fit weights and training-only transformations           |
| Validation | Model selection  | Hyperparameter tuning, early stopping, model comparison |
| Test       | Final evaluation | Evaluate the finalized model                            |

Do not combine training, validation, and testing data.

### 2.4 Norway country holdout

M1 verified that:

- Norway original images: 8,161
- Norway images in training: 0
- Norway images in validation: 0
- Norway images in testing: 8,161

Norway must remain exclusively in the test split.

This requirement evaluates the model's ability to generalize to a country not represented during training.

Do not use Norway data to fit model weights, tune hyperparameters, select checkpoints, or make model-development decisions.

### 2.5 Data leakage prevention

Before training, verify:

1. No overlapping image IDs across train, validation, and test.
2. No identical crop paths across different splits.
3. No exact duplicate image content across splits.
4. No unresolved near-duplicate or source-image overlap.
5. No data transformations fitted on validation or test data.
6. No augmentation of validation or test samples.
7. No test-set-driven model selection.

Use the existing leakage-audit results and rerun relevant checks when required.

Any detected leakage must be reported and resolved before model training.

## 3. Image Preprocessing and Augmentation

### 3.1 Training transformations

Follow the verified M1 configuration:

- Resize to 224 × 224 pixels.
- Random horizontal flip with probability 0.5.
- Random rotation between −10° and +10°.
- Convert images to tensors.
- Normalize using the configured channel statistics.

Mean:

```python
[0.485, 0.456, 0.406]
```

Standard deviation:

```python
[0.229, 0.224, 0.225]
```

### 3.2 Validation and testing transformations

Apply deterministic preprocessing only:

- Resize to 224 × 224.
- Convert to tensor.
- Apply the same normalization.

Do not apply random horizontal flipping, random rotation, or other stochastic augmentation during validation or testing.

### 3.3 Avoid double preprocessing

The classification crops have already been generated and resized in M1.

Confirm whether the data loader is applying additional resizing or normalization. Avoid unintentionally normalizing images twice.

## 4. Model Development

### 4.1 Confirm the assigned M2 model

Before implementing the model, verify the official assignment requirements.

Determine whether M2 requires:

- Image classification using the generated damage crops.
- Object detection using full images and bounding boxes.
- A particular architecture or baseline.
- Multiple models for comparison.

Do not substitute a classification model for an object detection requirement.

### 4.2 Classification labels

The current classification crop dataset includes four damage classes:

| Label | Damage category    |
| ----- | ------------------ |
| D00   | Longitudinal crack |
| D10   | Transverse crack   |
| D20   | Alligator crack    |
| D40   | Pothole            |

Confirm that the label-to-index mapping is identical across training, validation, testing, and saved model checkpoints.

### 4.3 Model architecture

If the assigned task is crop classification, an appropriate approach may include:

- A simple CNN baseline.
- A pretrained ResNet architecture.
- Another approved image classification architecture.

If the assigned task is object detection, use an approved detector and the full-image annotations rather than treating the classification crops as detection data.

Record the selected architecture and justification.

## 5. Training Configuration

### 5.1 Required training parameters

Document the following:

| Parameter            | Requirement                                      |
| -------------------- | ------------------------------------------------ |
| Batch size           | Start with 16 if suitable for available hardware |
| Epochs               | Define and document                              |
| Learning rate        | Define and document                              |
| Optimizer            | Record optimizer and settings                    |
| Loss function        | Appropriate for the assigned model               |
| Random seed          | Use the project configuration                    |
| Image size           | Follow approved M1 configuration                 |
| Hardware             | Record CPU/GPU and device                        |
| Early stopping       | Use validation performance if applicable         |
| Checkpoint selection | Based on validation, never test                  |

The batch size of 16 is a starting configuration, not a mandatory project requirement unless explicitly stated in the assignment.

### 5.2 Training procedure

For each epoch:

1. Train using the training split.
2. Calculate training loss and relevant training metrics.
3. Evaluate on the validation split.
4. Record the results.
5. Save the best checkpoint using a predefined validation criterion.
6. Apply early stopping if configured.

Do not repeatedly evaluate the test set during training.

### 5.3 Reproducibility

Set random seeds for Python, NumPy, and the machine learning framework.

Record:

- Python version.
- Framework and package versions.
- Random seed.
- Model architecture.
- Training configuration.
- Hardware specifications.
- Dataset version and split manifest.
- Git commit hash.

Document any sources of nondeterminism.

## 6. Model Evaluation

### 6.1 Classification metrics

If M2 is a classification task, report:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Macro-averaged F1.
- Per-class performance.
- Confusion matrix.
- Training and validation loss curves.

Consider class imbalance when interpreting overall accuracy.

### 6.2 Object detection metrics

If M2 is an object detection task, report the assignment-required detection metrics, potentially including:

- Precision and recall.
- mAP@0.50.
- mAP@0.50:0.95.
- Per-class average precision.
- Inference time, if required.

Use the evaluation protocol specified in the assignment.

### 6.3 Final test evaluation

After finalizing model selection and hyperparameters using validation data, evaluate the selected model on the held-out test set.

Report the Norway holdout performance separately where appropriate.

Do not use test results to retrospectively tune the model.

## 7. Coding Standards and File Organization

### 7.1 Separate M2 development files

Recommended structure:

```text
road-damage-detection/
├── notebooks/
│   ├── M1_Preprocessing_Overview.ipynb
│   └── M2_Model_Training.ipynb
├── scripts/
│   ├── train_m2.py
│   ├── evaluate_m2.py
│   └── m2_dataset.py
├── reports/
│   ├── m1/
│   └── m2/
├── models/
│   └── m2/
├── data/
│   ├── raw/
│   └── processed/
├── config.yaml
└── README.md
```

This is a proposed organization; adapt it to the repository's existing conventions.

### 7.2 Coding requirements

- Use Python with clearly structured functions and modules.
- Avoid duplicating preprocessing logic already implemented in M1.
- Avoid hard-coded absolute file paths.
- Use `pathlib.Path` for file management.
- Read relevant settings from configuration files.
- Use meaningful variable and function names.
- Add Markdown explanations to notebooks.
- Avoid unnecessary repeated imports and unused variables.
- Include error handling for missing data or invalid labels.
- Validate inputs before training.
- Avoid installing packages from inside production scripts.
- Keep reusable training and evaluation logic in Python modules where practical.

### 7.3 Notebook documentation

Each major section should contain a Markdown explanation describing:

- The objective.
- The methodology.
- The code being executed.
- The expected output.
- The interpretation of results.

The notebook should be executable in order from a clean kernel using the documented environment.

## 8. Output Management

### 8.1 Required deliverables

Prepare the following, as applicable:

- M2 model development notebook.
- Training Python script.
- Evaluation Python script.
- Model configuration.
- Training and validation results.
- Final performance metrics.
- Confusion matrix or detection evaluation figures.
- Model checkpoint or instructions for retrieving it.
- Reproducibility information.
- M2 technical summary.

### 8.2 Large files and GitHub

Do not commit:

- Raw image datasets.
- Entire generated crop directories.
- Virtual environments.
- Temporary files.
- Large model checkpoints without an approved storage approach.
- Sensitive information, tokens, or passwords.

Use an agreed shared storage location or artifact repository for large datasets and checkpoints.

Document how teammates can obtain these files.

### 8.3 Reports

Save appropriate lightweight reports in `reports/m2/`.

Use clear filenames, for example:

```text
m2_training_history.csv
m2_classification_report.csv
m2_confusion_matrix.png
m2_model_configuration.yaml
m2_evaluation_summary.md
```

## 9. GitHub Handover and Integration

Before requesting a merge:

1. Save and run the M2 notebook.
2. Confirm the data loader uses the correct splits.
3. Verify Norway is excluded from training and validation.
4. Confirm no unresolved data leakage.
5. Check that model training and evaluation complete successfully.
6. Verify that outputs are reproducible.
7. Confirm that no M1 files were unintentionally modified.
8. Review Git changes and ensure large datasets are not staged.
9. Commit and push the M2 branch.
10. Open a pull request for review.

Do not force-push to the shared main branch.

## 10. Important M1 Findings to Preserve

The following findings must be reflected in the M2 documentation:

**Visual label audit**

- Samples reviewed: 100.
- Correct: 98.
- Incorrect: 1.
- Uncertain: 1.
- Definitive reviews: 99.
- Observed error rate: 1.01%.
- 95% Wilson confidence interval: 0.18%–5.50%.

The confidence interval reflects the audited definitive sample and assumes the sampling method supports generalization. It is not a guarantee of the error rate for every class or country.

**Annotation limitations**

- One confirmed problematic annotation was identified for exclusion in the crop-generation workflow.
- One uncertain review requires further assessment.
- The audit sample does not establish that all annotations are correct.

**Sampling limitations**

- Training crops were capped by the preprocessing configuration.
- The final classification crop dataset contains 32,692 crops.
- Training class balance and potential sampling bias should be assessed.

**Country holdout**

- Norway is exclusively assigned to the test split.
- The Norway holdout must remain untouched throughout model development.

## 11. Final Acceptance Criteria

M2 is ready for submission when:

- The approved model has been implemented.
- The verified M1 dataset has been used correctly.
- The original split assignments are preserved.
- The Norway holdout remains isolated.
- Data leakage checks have passed.
- Training and validation results are documented.
- Final test evaluation follows the approved protocol.
- Required metrics and figures are generated.
- Reproducibility information is recorded.
- The code is committed to the M2 branch.
- A pull request has been submitted for review.

**The colleague is responsible for M2 only. Completed M1 work must remain intact, traceable, and reproducible.**
