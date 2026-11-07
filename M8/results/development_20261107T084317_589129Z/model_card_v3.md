# M8 Integrated Road Damage System — Model Card

## System and Intended Use

Contributor: Ghassan Saad AlGhaidani — 225495.

This educational prototype combines an M4 YOLO11n road-damage detector
with an M5 YOLO11n segmentation model. The selected local HTTP backend
is PyTorch on CPU, with report schema version 1.2.

The intended use is a supervised demonstration of road-damage detection,
pothole segmentation, image-area reporting, and inference timing.
Outputs require human interpretation.

The system is not approved for autonomous maintenance prioritization,
road-safety certification, or engineering severity assessment.

## Models and Processing

M4 detects D00, D10, D20, and D40. Only predicted D40 regions trigger M5,
whose class 0 represents potholes. Other damage detections are retained.

M4 uses image size 640, confidence 0.10, NMS IoU 0.50, and max_det 300.
M5 uses image size 640, confidence 0.05, NMS IoU 0.70, and max_det 100.

M5 crops are resized to a 640-by-640 square using INTER_AREA.
Binary masks are restored to crop dimensions using nearest-neighbor
interpolation and placed in original-image coordinates.
Overlapping masks are unioned without double-counting pixels.

M8 integrates existing checkpoints; it does not retrain the models.
Training provenance and original model evaluations belong to the
respective M4 and M5 handoffs.

## Outputs and Severity Interpretation

The HTTP response includes a prediction report, model fingerprints,
an annotated PNG, and a binary union-mask PNG.

Reports include detection classes, confidence scores, bounding boxes,
segmentation status, predicted mask areas, and CPU stage timings.

For a D40 detection with a nonempty predicted mask, the severity proxy is:

100 × predicted crop-mask pixels / full-image pixels.

This is an image-area proxy. It is not physical area, depth, or an
engineering severity category. It depends on camera position,
perspective, framing, and segmentation quality.

Missing masks produce unavailable severity rather than zero severity.
Crack severity is explicitly unsupported. Per-detection mask areas can
overlap; the overall union area is calculated separately.

## Development Evaluation

The general subset contains 90 validation images: 30 each from
China Drone, China MotorBike, and Czech. It contains 21 verified
background images and 132 reference boxes, including five D40 boxes.

A separate D40-enriched diagnostic cohort contains 90 additional images
and 122 D40 reference boxes. It is not representative of deployment
class prevalence.

At the selected operating point and same-class matching IoU >= 0.50:

| Cohort | Micro-F1 | D40 TP | D40 FP | D40 FN | D40 precision | D40 recall |
|---|---:|---:|---:|---:|---:|---:|
| General | 0.350 | 3 | 8 | 2 | 0.273 | 0.600 |
| D40 diagnostic | 0.258 | 26 | 21 | 96 | 0.553 | 0.213 |

These data informed operating-point selection. Results are development
measurements, not unbiased final-test performance or AP/mAP.
No Norway final-test data was used in the documented M8 experiments.

No reference segmentation masks or severity labels were used to measure
accuracy of the integrated pipeline on these cohorts.

## Functional Verification and ONNX

The updated pipeline passed 27 controlled edge-case checks.
The updated HTTP service passed its recorded smoke checks and six input
validation checks.

Both models were exported to ONNX. Earlier checks passed 18 raw-output
comparisons on six images, pipeline comparisons on seven images, and
union-mask comparisons on 47 fixed crops.

Those checks have their recorded version and sample scope. They do not
prove universal backend equivalence, model accuracy, or complete parity
of every field introduced by later reporting changes.

## Performance

The updated local PyTorch HTTP service completed 540 measured requests
with concurrency 1 after ten warm-up requests.

| Cohort | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|
| General | 95.63 | 193.88 | 312.61 |
| D40 diagnostic | 120.06 | 308.97 | 412.41 |

Successful requests per measurement-loop second: 7.054.
Sampled peak server RSS: 518.46 MiB.

A separate fresh-process direct-pipeline comparison measured:

| Backend | Calls | p50 ms | p95 ms | p99 ms | Calls/loop second | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch | 540 | 64.11 | 235.90 | 354.67 | 10.548 | 638.95 |
| ONNX Runtime | 540 | 119.33 | 616.84 | 750.14 | 5.483 | 787.49 |

HTTP and direct-pipeline measurements have different scopes.
The backend comparison used an artificial 50/50 cohort mixture,
fixed backend order, and unnormalized runtime thread defaults.
Worker RSS includes libraries, models, and preloaded images.
Sampling may miss short-lived memory peaks.

Results apply to the recorded Intel Core Ultra 7 155H CPU environment.
Concurrent-load capacity and production performance were not established.

## Limitations and Failure Modes

- Low D40 recall can prevent M5 from seeing actual potholes.
- False-positive D40 detections can trigger segmentation of unrelated regions.
- Partial or duplicate detections can fail one-to-one box matching.
- A nonempty predicted mask does not confirm a true pothole.
- Image-area proxies are not calibrated across camera viewpoints.
- Clean-environment dependency installation remains unverified.
- The local HTTP prototype has not been validated for public deployment.
- Uploaded images are not automatically anonymized.

The failure gallery documents selected development examples.
Its presentation redaction applies only to those prepared images.

## Acceptance Status

PyTorch is retained for the local supervised demonstration.
This is not production approval.

Predetermined quantitative acceptance targets were not available for the
recorded benchmarks; retrospective measurements must not be presented
as passing criteria defined in advance.

Operational deployment remains NO-GO pending an agreed prospective
evaluation protocol, independent accuracy assessment, appropriate
severity validation, and deployment-specific reliability requirements.
