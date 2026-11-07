# M4 GitHub Submission Inventory

Checked against the `feature/m4-object-detection` branch on 2026-11-02.

## Present in repository
- `notebooks/M4_Object_Detection_Cleaned.ipynb` — uploaded cleaned notebook (106 code cells, 17 Markdown cells; no saved outputs).
- `docs/M4_OBJECT_DETECTION.md` — baseline configuration, dataset protocol, Norway holdout metrics and limitations.

## Not present in repository at time of check
- M4-specific evaluation reports (including `m4_norway_final_test_summary.csv`).
- YOLO `data.yaml` configuration for M4.
- Training and validation figures, including confusion matrix / precision-recall or learning curves.
- Model checkpoint `best.pt` (large binary; **not required in GitHub** if retained and referenced in shared storage).

The `reports/` folder currently contains only `reports/m1/`; M4 reports are **not** committed. The source notebook refers to artifacts in Google Drive under `AI Project /M4_Object_Detection/`, but those Drive artifacts have not been inspected or verified by this GitHub-only audit.

## Final verification before submission
1. Export small M4 report CSVs and key PNG figures from Google Drive and commit them under `reports/m4/`.
2. Add a portable copy of M4 `data.yaml` (replace absolute paths with an example or clearly document required path edits).
3. Keep large image data, local Colab copies, checkpoints and training caches out of GitHub.
4. Confirm the committed notebook opens and that all references to required artifacts are correct.
5. Avoid rerunning the Norway final test for cleanup; it is a held-out evaluation, not a tuning dataset.

This inventory is documentation only; no data, model, or evaluation was modified.
