# M4 final static review — 2026-11-02

## Scope and verification level
Reviewed the uploaded `M4_Object_Detection_Cleaned.ipynb` locally and the user-supplied M4 reports ZIP. This is a **static** review: no Google Colab execution, Google Drive inspection, retraining, or Norway reevaluation occurred.

## Verified
- Notebook JSON loads; 106 code cells, 17 Markdown cells; all code cells pass basic Python parsing after removing Colab line magics. This does **not** guarantee runtime success.
- No saved cell outputs or execution counts in the cleaned notebook.
- The training YAML and training config use `images/train` and `images/val`, with four classes D00/D10/D20/D40, seed 42, 30 epochs, imgsz 640, batch 16, deterministic True.
- Norway is not in the training YAML; the independent Norway test uses `images/test` and is evaluated by a separate final evaluation cell.
- The uploaded ZIP includes the Norway final summary, validation-source comparison, Czech confidence comparison, Czech error summary and Czech size-recall summary. These five CSVs were verified under `reports/m4/` on the branch.
- The uploaded ZIP contains **no PNG/JPG training plots or evaluation figures**.

## Remaining items / important caveats
1. **Figures:** The notebook references `runs/yolo11n_baseline/results.png`, confusion matrices, PR curves and other plots, but the ZIP does not include them. Export the key existing PNGs from Google Drive to `reports/m4/figures/` if figures are required; do not regenerate by retraining.
2. **Runtime reproducibility unverified:** This notebook depends on Google Colab, a mounted Drive directory named `AI Project ` (trailing space), M1 manifests, source archives, a prepared YOLO dataset, `ultralytics`, a CUDA GPU, and a saved `best.pt`. None of these external runtime assets was verified in this review.
3. **Rerun safety:** The notebook contains an explicit one-epoch smoke test, a 30-epoch baseline training call, and a final Norway evaluation call. **Do not run all cells** merely to inspect or reproduce saved results. The training config has `exist_ok=False`, so a rerun may create additional numbered run directories; subsequent cells still refer to `yolo11n_baseline`. Open existing artifacts/read-only reporting cells instead.
4. **No proof of end-to-end execution:** The cleaned notebook has no stored outputs; do not claim its end-to-end execution has been validated.
5. **Path portability:** `configs/m4_data.yaml` uses `/content/m4_local/yolo_dataset`. This path must be populated in Colab before use and is not portable to a different machine unchanged.
6. **Reproducibility metadata:** For stronger reproducibility, archive the original `runs/yolo11n_baseline/args.yaml`, `results.csv`, and exact Ultralytics version alongside the existing reports, if allowed and available.
7. **Holdout discipline:** Preserve Norway as an untouched independent test; do not use its metrics for parameter tuning.

## Status
**Code organization and main report upload: complete.**
**Full submission readiness: pending figure availability and runtime artifact checks.**
This review did not modify M1/M2/M3, data splits, labels, models, or existing reports.
