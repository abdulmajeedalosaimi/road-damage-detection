# Phase 1 — Reproducibility Verification

## Purpose

Independently verify the team's prepared classification data and reproduce evaluation of the saved fine-tuned ResNet18 checkpoint.

This notebook checks data integrity, evaluates the checkpoint locally, compares metrics with the original results, and saves verification evidence.

## Scope

- Validate class labels and parent-image split assignments.
- Verify that Norway is reserved for testing.
- Check that all 32,692 referenced crops exist and can be decoded.
- Evaluate the fine-tuned checkpoint on 11,229 test crops.
- Compare accuracy, macro-F1, top-2 accuracy, and ECE using an absolute tolerance of `1e-6`.
- Save predictions, reports, environment details, and input fingerprints.

This work reproduces evaluation of an existing checkpoint. It does not reproduce training or independently evaluate the scratch and frozen checkpoints.

## Required Files

Obtain the prepared data and original model results from the team.

The data search folder must contain exactly one copy of each file:

- `split_manifest.csv`
- `annotations.csv`
- `classification_manifest.csv`
- `classification_crops.zip`

The same search folder must contain exactly one original `test_comparison.csv`. Its parent folder must contain the fine-tuned model files in this structure:

```text
original_results/
├── test_comparison.csv
└── finetune/
    ├── best.pt
    └── config.json
```

The data files may be located in separate subfolders because the notebook searches recursively.

Keep `classification_crops.zip` compressed; extraction is not required.

## Project Location

Keep this folder inside the full project repository:

```text
project_root/
├── scripts/
├── road_damage_m2_m3_colab/
└── Phase1_Reproducibility/
    ├── Phase1_Reproducibility.ipynb
    ├── README.md
    └── results/
```

The notebook locates the project root using the `scripts` and `road_damage_m2_m3_colab` folders. Copying only the notebook to an unrelated folder is not sufficient.

## Python Environment

Select a Python environment containing:

- ipykernel
- numpy
- pandas
- matplotlib
- Pillow
- scikit-learn
- torch
- torchvision
- wandb

The successful verification used Python 3.14.6 on Windows with CPU execution. A GPU is not required.

Each saved run includes `requirements_local.txt`, which records the versions of the listed packages. `verification_summary.json` also records the Python version and PyTorch/Torchvision runtime versions.

These files document the tested environment; they are not a complete dependency lockfile.

## How to Run

1. Open the full project folder in VS Code.
2. Open `Phase1_Reproducibility.ipynb`.
3. Select the Python kernel containing the required packages.
4. Find the cell defining `DATA_SEARCH_ROOT`.
5. Change that path to the folder containing your prepared data and original results.

Example:

```python
DATA_SEARCH_ROOT = Path(r"D:\RoadDamageInputs")
```

Use your own local path. Do not keep another teammate's computer path.

6. Save the notebook.
7. Restart the kernel.
8. Select **Run All** and wait for all cells to finish.
9. Confirm that all four metric comparisons are `True`.
10. Save the notebook again to retain the new outputs.

The notebook reads the seed, image size, batch size, and class order from the selected checkpoint's `config.json`.

## Verified Results

The local evaluation produced:

| Metric | Local result |
|---|---:|
| Accuracy | 0.784487 |
| Macro-F1 | 0.619348 |
| Top-2 accuracy | 0.948793 |
| ECE | 0.065339 |

Accuracy, macro-F1, and top-2 accuracy matched the original values exactly in the recorded comparison. The ECE absolute difference was approximately `3.65e-8`, below the `1e-6` tolerance.

## Saved Outputs

Each complete run creates a new timestamped folder under `results/`.

| File | Contents |
|---|---|
| `finetune_comparison.csv` | Original/local metrics, differences, and tolerance checks |
| `finetune_test_predictions.csv` | Test predictions |
| `finetune_probabilities.npz` | Saved probability arrays |
| `finetune_classification_report.csv` | Per-class precision, recall, and F1 |
| `finetune_confusion_matrix.csv` | Confusion matrix counts |
| `finetune_confusion_matrix.png` | Confusion matrix figure |
| `verification_summary.json` | Evaluation settings, metrics, environment, and pass status |
| `requirements_local.txt` | Recorded versions of the listed packages |
| `input_fingerprints.json` | Fingerprints of six input files |

Fingerprints cover the checkpoint, its configuration, the original comparison CSV, and the three input CSV files. The crop ZIP is not fingerprinted; its referenced images are checked for presence and readability.

## Interpretation and Limitations

Performance varies across classes. D00 and D10 have higher F1 scores than D20 and D40.

The largest confusion is 1,167 D00 crops predicted as D40. D40 has high recall but low precision, indicating many false-positive predictions.

The test set is imbalanced, so accuracy should be interpreted alongside macro-F1 and per-class metrics.

Passing the comparison verifies the reported metrics for this saved checkpoint and test set. It does not establish reproducibility of model training.

## Troubleshooting

- **Missing input:** Check `DATA_SEARCH_ROOT` and confirm that the required file exists beneath it.
- **Multiple matches:** Narrow the search folder so it contains only one intended copy of each required input and one original `test_comparison.csv`.
- **Project root not found:** Keep this notebook inside the full project structure shown above.
- **Import failure:** Confirm that VS Code is using the intended Python environment.
- **Unexpected kernel state:** Restart the kernel and run the notebook from the beginning.
- **Metric mismatch:** Check the input files, checkpoint, configuration, class order, and preprocessing against the original run.

## Repository Contents

Include this notebook, this README, and the selected verification evidence in the repository.

Keep raw data archives and model checkpoints outside Git. Teammates need access to those inputs separately to run the verification.