# Verification — 24 October 2026

## Passed

- All 27 notebook cells are valid notebook JSON; every Python code cell compiles.
- Embedded modules are byte-for-byte equivalent to the delivered Python sources (UTF-8 BOM normalized).
- Actual M1 CSVs: all 32,692 crops join to unique source annotation objects; class, split and parent metadata match.
- Validation/test crops exactly cover their source annotations. Training retains 3,000 crops per class.
- Source-image dimensions and bounding-box constraints checked across all 55,002 annotations.
- Recorded SHA256 uniqueness and country/parent-image separation checked across 38,384 parent records.
- Windows crop paths normalize correctly; absolute and traversal paths rejected.
- Safe ZIP extraction, complete-cache reuse and image decoding checked on a synthetic archive.
- Metrics checked against analytically known perfect, completely wrong and uniform predictions, including confidence exactly 1.0 and stable top-2 ties.
- Parent-image bootstrap confidence interval checked on a known perfect example.
- Real CPU forward/backward operations verified frozen ResNet weights AND BatchNorm buffers stay fixed while the classifier updates.
- Full-training mode verified backbone gradients and BatchNorm updates.
- Synthetic SmallCNN training interrupted after a saved epoch, resumed, and compared with uninterrupted training: final CPU weights matched exactly.
- DataLoader checked with two worker processes on Windows; Colab worker behavior still depends on its runtime.
- Synthetic scratch/frozen/finetune workflows completed two epochs each, generated validation reports and selected checkpoints.
- Training succeeded before any synthetic test-image files existed, confirming that training does not read test pixels.
- Explicit final-test gate, locked checkpoint comparison, final three-model metrics and reuse of completed test results checked.
- A changed configuration was rejected for an existing run directory.
- W&B 0.19.8 real SDK tested in offline mode: init, scalar logging, summary, Image, Table, evaluation artifact and finish.
- Synthetic confusion, calibration and learning-curve figures visually inspected for readable labels/layout.

## Scope and limits

These checks are software checks, not RDD2022 performance results. Synthetic images were colored squares. ResNet smoke tests used randomly initialized weights to avoid downloading pretrained parameters during verification; the delivered notebook requests ImageNet1K_V1 for actual transfer-learning runs.

The actual classification_crops.zip was not available locally, so project-image decode, actual crop archive layout and visual labels were not rechecked in this session. The notebook checks these files before training.

No actual M2/M3 project training, GPU execution, Colab Drive mount, online W&B authentication/upload, ImageNet weight download, manual error analysis or final project evaluation was performed. These remain user-run steps in Colab.

The two M1 audit samples still need reconciliation by image ID. Do not use the copied row-number decisions to claim a verified label-error rate.

Training tests used Python 3.13, torch 2.13.0+cpu, torchvision 0.28.0+cpu and pandas 2.2.3. Actual M1 table validation also passed with bundled Python 3.12/pandas 3.0.1. Local testing dependencies were installed only inside the task's work directory; the user's Python installation and source M1 files were not modified.

Colab keeps its supplied compatible torch/torchvision pair and adds wandb==0.19.8. Each real experiment records its actual environment and pip freeze to requirements.lock.txt; this file, not an assumed Colab version, is the reproduction record.

## English-language revision

All notebook instructions, form labels, messages and the README are in English. The ten deliverable files and decoded notebook/JSON strings were checked for Arabic Unicode characters; none remain. Every Python module and notebook code cell compiles, and the embedded modules match the delivered source files.

This revision changes documentation and user-facing text only. The training modules, experiment configuration, data hashes and notebook execution logic are unchanged. The previously recorded model tests were not rerun for this text-only revision.
