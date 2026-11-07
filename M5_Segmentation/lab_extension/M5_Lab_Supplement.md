# M5 laboratory adaptation and additional evaluation

Contributor: Abdulrahman Alsalmi, 225493.

## Scope and chronology
The reference is Day6_Lab_Instance_segmentation_v2_student.ipynb. Its experiments are adapted to potholes and
YOLO11n-seg. The project guide asks for application-adapted laboratory deliverables; this is not a claim that
the COCO/Detectron2 classroom notebook was reproduced verbatim. Architecture-specific substitutions are listed below.
These analyses were added after the original test evaluation. They do not select a new model or operating point.
The original model fingerprint is 5fb5f6925489060792d160f4a04bc3620c0b59a5de5aba013f2e6f5523254d44; its threshold stays 0.05.
The original Dice/IoU/area results remain the primary benchmark. The additional pass with this fixed model collects
actual detector boxes for box AP and object-level reporting; it is not a fresh test used for model selection.
Reference masks are binary semantics. Every instance ID in this supplement is an 8-connected region, not a native
physical-pothole identity. Crowds are not annotated. All size bins use the 640x640 reference-region area.

## Data representations
Training objects, polygon simplification and the RLE check are recorded in data_exercises.json and
polygon_simplification.csv. External contours can fill holes or lose tiny regions; this exercise does not change
the reference annotations. Column-major manual RLE was verified against pycocotools.

## Initialization and trained model
The original initial checkpoint was not saved. A one-class initialization is reconstructed from the recorded
pretrained YOLO11n-seg source and seed. Shared box/mask weights transfer; the pothole classifier is newly adapted.
This is a retrospective initialization control, not proof of the exact original initial weight values.
It is evaluated only on the same 100 fixed validation frames as the trained model, one frame from each of 100 videos.
The lab's 80-class COCO zoo model is not presented as a pothole classifier.

| model | split | task | AP | AP50 | AP75 | AP_small | AP_medium | AP_large |
|---|---|---|---|---|---|---|---|---|
| reconstructed_initialization | val100 | bbox | 0.0034 | 0.0075 | 0.0025 | 0.0000 | 0.0002 | 0.0197 |
| reconstructed_initialization | val100 | segm | 0.0020 | 0.0052 | 0.0010 | 0.0000 | 0.0002 | 0.0141 |
| finetuned | val100 | bbox | 0.6720 | 0.9368 | 0.7310 | 0.0000 | 0.4857 | 0.7269 |
| finetuned | val100 | segm | 0.6647 | 0.9251 | 0.7854 | 0.0000 | 0.4302 | 0.7184 |

Training losses and internal validation AP come from the original 30-epoch history. Those internal AP numbers
use polygon labels. YOLO's mask loss is not the lab's isolated Mask R-CNN BCE, so the ln(2) reference is not imposed.
The empty-mask diagnostic deliberately changes masks while keeping detections fixed. It checks metric behavior;
it is not the lab's second stalled-head training run and must not be described as one.

## Box versus mask AP
| model | split | task | AP | AP50 | AP75 | AP_small | AP_medium | AP_large |
|---|---|---|---|---|---|---|---|---|
| finetuned | val | bbox | 0.6772 | 0.9387 | 0.7649 | 0.0003 | 0.3964 | 0.7115 |
| finetuned | val | segm | 0.6649 | 0.9321 | 0.7944 | 0.0000 | 0.3553 | 0.6990 |
| finetuned | test | bbox | 0.7190 | 0.9332 | 0.8112 | 0.0000 | 0.5134 | 0.7327 |
| finetuned | test | segm | 0.6856 | 0.9488 | 0.7670 | 0.0000 | 0.4687 | 0.7005 |

AP uses the original low confidence floor 0.001, not the deployment threshold.
Undefined size bins remain N/A. The repeated fixed-model mask AP is compared with the original in fixed_model_recheck.json.

## Object-level precision, recall and F1
One-to-one, class-aware mask matching uses IoU >= 0.5. These are object-level metrics, distinct from pixel Dice.
The diagnostic maximum validation F1 is 0.9079 at confidence 0.6.
The deployed confidence remains 0.05, selected earlier by validation foreground Dice.
At this original threshold, validation precision/recall/F1 are 0.8184/
0.9348/0.8727; test values are 0.8552/
0.9628/0.9058. The test threshold is not swept or changed.
Manual mask AP50 (0.925057) agrees with COCOeval (0.925057) on the 100-frame subset.

## SAM prompts and detector-box comparison
The one-click exercise produces three masks with measured reference IoU [0.3063308903789178, 0.43602263899678173, 0.9189978107516419].
The reference-box prompt reaches 0.9212. This prompted exercise uses annotation information
and is not automatic test performance.

The main SAM comparison instead uses predicted boxes only. It retains the detector's classes, scores and detection
count. Both methods are evaluated on the same 100 validation images at the original retained-detection threshold.
This conditional AP must not replace the unrestricted model headline AP.

| method | AP | AP50 | AP75 | AP_small | AP_medium | AP_large |
|---|---|---|---|---|---|---|
| YOLO masks (kept detections) | 0.6586 | 0.9121 | 0.7741 | 0.0000 | 0.4022 | 0.7184 |
| SAM masks (same detections) | 0.3535 | 0.7688 | 0.2946 | 0.0000 | 0.2349 | 0.3850 |

SAM minus YOLO mask AP on these same detections: -0.3051.
- small: no matched regions; no conclusion for this size bin.
- medium: 11 paired regions; model/SAM median IoU 0.8105/0.7791; SAM wins on 18.2%.
- large: 94 paired regions; model/SAM median IoU 0.8893/0.7371; SAM wins on 8.5%.

SAM has 93,735,472 parameters; YOLO has 2,857,651.
Measured YOLO pipeline time is 37.47 ms/image, with SAM adding
519.36 ms/image. Their sequential sum is
556.83 ms/image. Warm-up is excluded; model timing includes
image loading and mask serialization, whereas SAM timing begins after image loading. These are local comparative
measurements, not HTTP service latency or a production benchmark. Cached timings may come from an earlier session;
device counts are saved in finetuned_subset_timing.json and sam_details.json.
SAM still inherits missed objects, false detections and class/score errors from its prompts. This analysis does not
replace the selected deployable model. SAM remains an annotation aid unless a separate validation-only deployment
experiment justifies its added cost. Three predetermined comparisons are saved in sam_comparison_gallery.png.

## Laboratory-to-project mapping
| lab_task | topic | evidence | application_adaptation |
|---|---|---|---|
| 1 | Training annotation summary | data_exercises.json | One pothole class; derived-region counts and size bins; no source crowd labels. |
| 2 | Polygon simplification | polygon_simplification.csv | Up to 150 training regions per size bin; eps 0,1,2,4 at evaluation resolution. |
| 3 | RLE encoding/decoding | data_exercises.json | Manual column-major round trip checked against pycocotools; wrong-order IoU recorded. |
| 4 | Experiment configuration | ../protocol.json; extension_protocol.json | Retain the original YOLO recipe and frozen operating threshold. |
| 5 | Transferred and newly adapted weights | weight_transfer.csv; initialization_details.json | YOLO class adaptation instead of a new Mask R-CNN mask head; initialization reconstructed retrospectively. |
| 6 | COCO prediction format | coco_val100_reference.json; coco_val100_finetuned.json | Explicit detector boxes, masks, confidence and one pothole category. |
| 7 | Initialization baseline AP | initialization_vs_finetuned.csv | Same fixed 100 validation images; COCO classes are not mislabeled as potholes. |
| 8 | Manual mask AP50 | object_operating_point.json | Score-ordered one-to-one matching and 101-point interpolation agree with COCOeval. |
| 9 | Training curves | training_component_curves.png | Original successful training only. Mask R-CNN ln(2) stalled-head training is not repeated for YOLO. |
| 10 | Box/mask AP overall and by size | box_mask_ap_by_size.csv; initialization_vs_finetuned.csv | Mask AP uses mask areas. COCO size bins refer to 640x640 derived regions. |
| 11 | Diagnosing a mask-specific failure | mask_failure_sanity_check.csv | Synthetic empty-mask control: same detector boxes, deliberately corrupted masks. Not a second trained run. |
| 12 | Object-level threshold analysis | validation_object_thresholds.csv | Validation-only PR/F1 diagnostic; does not replace the threshold already chosen by Dice. |
| 13 | Test metrics, gallery and handoff | box_mask_ap_by_size.csv; ../figures/test_qualitative_gallery.png; ../HANDOFF.md | Frozen checkpoint/threshold; additional test box reporting is explicitly post-hoc; original test results stay primary. |
| 14 | SAM point versus box | sam_point_vs_box.png; sam_point_vs_box.json | Three point interpretations and one reference-box prompt on a validation pothole. |
| 15 | SAM from predicted boxes | sam_vs_model_ap.csv; sam_paired_by_size.csv; sam_details.json | 100 distinct validation videos; same detections/scores/classes; paired IoU by size and measured timing. |
| 16 | Comparison gallery | sam_comparison_gallery.png | Three predetermined validation images: reference, YOLO and SAM. |

## References
- Project guidelines, M5 rubric and road-damage project requirements, pages 6-7 and 32-33.
- Course handout: Day6_Lab_Instance_segmentation_v2_student.ipynb.
- SAM predictor API: https://docs.ultralytics.com/reference/models/sam/predict/
- Segmentation model initialization: https://docs.ultralytics.com/reference/models/yolo/segment/train/
