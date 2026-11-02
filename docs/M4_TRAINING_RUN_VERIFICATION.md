# M4 training run verification (2026-11-02)

## Source
User-uploaded archive `yolo11n_baseline-20261102T120207Z-1-001.zip`, containing the original Ultralytics training run. Review was static; no retraining or evaluation occurred.

## Verified
- `args.yaml`: pretrained YOLO11n detector, 30 epochs, batch 16, image size 640, GPU 0, seed 42, deterministic True, validation split `val`, 10-epoch patience, and plots enabled.
- `results.csv`: 30 completed epochs. At epoch 30: precision 0.31275, recall 0.26893, mAP50 0.20963, mAP50-95 0.08277 on the **training-run validation split**, not Norway.
- Highest validation mAP50 in the logged epochs: 0.22392 (epoch 25). This is not the Norway result.
- Separate previously committed `reports/m4/m4_norway_final_test_summary.csv`: Norway independent test mAP50 0.0791904 and mAP50-95 0.0319246.
- Original run ZIP includes: `results.png`, `confusion_matrix.png`, `confusion_matrix_normalized.png`, `BoxPR_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`, `BoxF1_curve.png`, `labels.jpg`, training and validation batch examples, `args.yaml`, `results.csv`, and checkpoint files.
- The ZIP contains `best.pt`, `last.pt`, and periodic checkpoint files. They were not committed because binary checkpoints are large and may not be needed in GitHub.

## Remaining repository gaps
- Original PNG/JPG plots are present in the uploaded archive, **not yet committed to GitHub**. The GitHub text-file connector cannot directly transfer local binary files.
- The complete original `args.yaml` and `results.csv` remain in the uploaded archive; `configs/m4_run_settings.yaml` below is a selected-key transcription for reference, **not** a byte-for-byte copy.
- No end-to-end execution of the cleaned notebook has been performed. Exact environment and dependencies (including Ultralytics version), dataset availability, and run reproducibility remain unverified.
- To include images, upload the original figures using GitHub's **Add file → Upload files** to `reports/m4/figures/` on `feature/m4-object-detection`. Do not upload model weights unless specifically required.
- Do not use the independent Norway test for hyperparameter selection.

## Conclusion
The existing training artifacts have been inspected, and the training history and holdout results are documented. The figures are available in the uploaded ZIP, but final repository figure inclusion and executable reproducibility remain unverified.
