# M8 Submission Evidence Index

Contributor: Ghassan Saad AlGhaidani — 225495.

Experiment: `development_20261107T084317_589129Z`

Use the service archive selected below.
Keep relative paths intact when transferring the M8 directory.
Detailed experiment evidence is retained beside the listed summaries.

## Notebook

[Open file](M8.ipynb)

Main notebook; save after running this cell.

## Demo recording

[Open file](Screen%20Recording%202026-11-07%20145250.mp4)

Recorded fallback. This cell does not assess video content.

## Updated service ZIP

[Open file](results/development_20261107T084317_589129Z/service_package_v3_20261107T113714_763360Z/M8_service_bundle_v3.zip)

Selected schema 1.2 service delivery archive.

## Package verification

[Open file](results/development_20261107T084317_589129Z/service_package_v3_20261107T113714_763360Z/package_verification.json)

Archive integrity and fingerprint checks.

## Model card

[Open file](results/development_20261107T084317_589129Z/model_card_v3.md)

Intended use, recorded results, and limitations.

## Operating point

[Open file](results/development_20261107T084317_589129Z/provisional_operating_point.json)

Development-selected detector settings.

## Threshold comparison

[Open file](results/development_20261107T084317_589129Z/threshold_sweep_20261107T085057_136070Z/operating_point_comparison.csv)

Detection metrics across development operating points.

## Updated pipeline edge cases

[Open file](results/development_20261107T084317_589129Z/pipeline_v3_edge_case_tests.json)

Pipeline logic, severity-proxy, and timing checks.

## Updated HTTP validation

[Open file](results/development_20261107T084317_589129Z/service_bundle_v3_20261107T105413_863679Z/http_input_validation.json)

Updated service input validation evidence.

## Updated HTTP benchmark

[Open file](results/development_20261107T084317_589129Z/http_benchmark_v3_20261107T105731_616795Z/summary.json)

PyTorch HTTP latency, throughput, and memory.

## Isolated backend benchmark

[Open file](results/development_20261107T084317_589129Z/isolated_backend_benchmark_20261107T110252_891074Z/summary.json)

Fresh-process direct-pipeline measurements.

## Updated fresh-process verification

[Open file](results/development_20261107T084317_589129Z/fresh_process_v3_20261107T112241_965708Z/verification.json)

Reproduction in the existing installed environment.

## Extracted archive rehearsal

[Open file](results/development_20261107T084317_589129Z/delivery_rehearsal_20261107T113958_987260Z/verification.json)

HTTP rehearsal of the selected delivery archive.

## Failure gallery presentation

[Open file](results/development_20261107T084317_589129Z/failure_gallery_20261107T111151_749321Z/presentation_20261107T111619_645388Z/README.md)

Presentation copies; visual review applies to the inspected version.

## M4 ONNX

[Open file](results/development_20261107T084317_589129Z/onnx_export_20261107T093155_840251Z/m4/m4.onnx)

Exported detector.

## M5 ONNX

[Open file](results/development_20261107T084317_589129Z/onnx_export_20261107T093155_840251Z/m5/m5.onnx)

Exported segmenter.

## ONNX export manifest

[Open file](results/development_20261107T084317_589129Z/onnx_export_20261107T093155_840251Z/export_manifest.json)

Export settings and artifact records.

## Raw numerical parity

[Open file](results/development_20261107T084317_589129Z/onnx_export_20261107T093155_840251Z/raw_numerical_parity.json)

Raw-output agreement on the recorded sample.

## Pipeline parity

[Open file](results/development_20261107T084317_589129Z/onnx_export_20261107T093155_840251Z/pipeline_parity.json)

Agreement within its recorded version and sample scope.

## Crop-mask parity

[Open file](results/development_20261107T084317_589129Z/onnx_export_20261107T093155_840251Z/crop_mask_parity.json)

Union-mask agreement on the recorded crops.

## External Dependencies and Scope

The service ZIP excludes checkpoints and datasets.
Follow its README and set M8_PROJECT_ROOT explicitly.

Functional service checks do not establish prediction accuracy.
Clean-environment installation and full fresh-kernel notebook replay
remain unverified. Integrated mask and severity-proxy accuracy have
not been established on the development cohorts.

No independent final-test performance or production approval is claimed.
Quantitative acceptance criteria were not predetermined for the
recorded benchmarks.

The service does not automatically anonymize uploaded images.
This index validates file integrity, not video content or image privacy.

