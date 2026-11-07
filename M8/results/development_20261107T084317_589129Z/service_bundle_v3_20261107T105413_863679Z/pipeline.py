"""M8 integrated road-damage inference pipeline."""

import json
import math
import time
import cv2
import numpy as np

def run_integrated_inference(
    image_bgr,
    detector,
    segmenter,
    detector_settings,
    segmenter_settings,
    device="cpu",
):
    if str(device) != "cpu":
        raise ValueError("Stage timing currently supports CPU only.")

    timing_start = time.perf_counter()
    timing_last = timing_start
    stage_ms = dict.fromkeys([
        "input_validation",
        "m4_predict",
        "detection_postprocess",
        "crop_preparation",
        "m5_predict",
        "mask_restoration",
        "severity_calculation",
        "annotation",
        "report_creation",
    ], 0.0)

    def mark_stage(name):
        nonlocal timing_last
        now = time.perf_counter()
        stage_ms[name] += (now - timing_last) * 1000.0
        timing_last = now

    # Validate the input before calling either model.
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("Input must be a NumPy BGR image.")

    if (
        image_bgr.dtype != np.uint8
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or min(image_bgr.shape[:2]) == 0
    ):
        raise ValueError("Expected a non-empty uint8 image with 3 BGR channels.")

    if detector.task != "detect" or segmenter.task != "segment":
        raise ValueError("Expected a detection model and a segmentation model.")

    expected_names = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}

    if dict(detector.names) != expected_names:
        raise ValueError("Unexpected M4 class mapping.")

    if dict(segmenter.names) != {0: "pothole"}:
        raise ValueError("Unexpected M5 class mapping.")

    source = np.ascontiguousarray(image_bgr.copy())
    height, width = source.shape[:2]
    detection_config = dict(detector_settings)
    segmentation_config = dict(segmenter_settings)

    size = segmentation_config["imgsz"]
    if not isinstance(size, int) or size <= 0:
        raise ValueError("M5 imgsz must be a positive integer.")

    mark_stage("input_validation")
    result = detector.predict(
        source=source,
        device=device,
        **detection_config,
        augment=False,
        save=False,
        verbose=False,
    )[0]

    mark_stage("m4_predict")
    if tuple(result.orig_shape) != (height, width):
        raise ValueError("Unexpected detector output dimensions.")

    detections = []
    union_mask = np.zeros((height, width), dtype=bool)
    segmentation_calls = 0
    segmentation_limit_hits = 0

    if result.boxes is not None:
        coordinates = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, confidence, class_value in zip(
            coordinates, confidences, classes
        ):
            if not float(class_value).is_integer():
                raise ValueError("Non-integer detector class.")

            class_id = int(class_value)
            confidence = float(confidence)
            box = [float(value) for value in box]

            if (
                class_id not in expected_names
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
                or not all(math.isfinite(value) for value in box)
            ):
                raise ValueError("Invalid detector output.")

            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("Invalid detector box coordinates.")

            detections.append({
                "class_id": class_id,
                "damage_code": expected_names[class_id],
                "confidence": confidence,
                "bbox_xyxy": box,
                "segmentation": {
                    "status": "not_applicable",
                    "mask_pixels": None,
                    "mask_percent_of_crop": None,
                    "mask_percent_of_full_image": None,
                },
                "severity": None,
                "severity_status": (
                    "unavailable_no_predicted_mask"
                    if class_id == 3
                    else "unsupported_damage_class"
                ),
            })

    detections.sort(key=lambda item: item["confidence"], reverse=True)

    mark_stage("detection_postprocess")
    for detection in detections:
        if detection["class_id"] != 3:
            continue

        x1, y1, x2, y2 = detection["bbox_xyxy"]
        left, top = int(np.floor(x1)), int(np.floor(y1))
        right, bottom = int(np.ceil(x2)), int(np.ceil(y2))

        crop = source[top:bottom, left:right].copy()
        crop_height, crop_width = crop.shape[:2]

        model_input = cv2.resize(
            crop, (size, size), interpolation=cv2.INTER_AREA
        )

        mark_stage("crop_preparation")
        seg_result = segmenter.predict(
            source=model_input,
            device=device,
            **segmentation_config,
            retina_masks=True,
            augment=False,
            save=False,
            verbose=False,
        )[0]

        mark_stage("m5_predict")
        segmentation_calls += 1

        if tuple(seg_result.orig_shape) != (size, size):
            raise ValueError("Unexpected segmentation output dimensions.")

        crop_mask = np.zeros((crop_height, crop_width), dtype=bool)
        instance_count = (
            len(seg_result.boxes) if seg_result.boxes is not None else 0
        )

        limit_reached = instance_count >= segmentation_config["max_det"]
        segmentation_limit_hits += int(limit_reached)

        if seg_result.masks is None:
            if instance_count:
                raise ValueError("M5 returned boxes without masks.")
        else:
            if seg_result.boxes is None:
                raise ValueError("M5 returned masks without boxes.")

            masks = seg_result.masks.data.cpu().numpy()
            mask_classes = seg_result.boxes.cls.cpu().numpy()

            if len(masks) != instance_count or not np.isfinite(masks).all():
                raise ValueError("Invalid M5 mask output.")

            for mask, mask_class in zip(masks, mask_classes):
                if float(mask_class) != 0:
                    raise ValueError("Unexpected M5 mask class.")

                restored = cv2.resize(
                    (mask > 0.5).astype(np.uint8),
                    (crop_width, crop_height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

                crop_mask |= restored

        union_mask[top:bottom, left:right] |= crop_mask
        pixels = int(crop_mask.sum())

        detection["segmentation"] = {
            "status": "mask_predicted" if pixels else "no_mask_predicted",
            "crop_bounds_xyxy": [left, top, right, bottom],
            "predicted_instances": instance_count,
            "max_det_reached": bool(limit_reached),
            "mask_pixels": pixels,
            "mask_percent_of_crop": (
                100.0 * pixels / (crop_width * crop_height)
            ),
            "mask_percent_of_full_image": 100.0 * pixels / (width * height),
        }

        mark_stage("mask_restoration")

    for detection in detections:
        if detection["class_id"] != 3:
            continue

        segmentation = detection["segmentation"]
        if segmentation["mask_pixels"] > 0:
            detection["severity"] = {
                "measure": "predicted_pothole_area_percent_of_full_image",
                "value": segmentation["mask_percent_of_full_image"],
                "units": "percent",
                "mask_pixels": segmentation["mask_pixels"],
                "denominator_pixels": int(width * height),
                "interpretation": "image_area_proxy_only",
                "physical_area_estimated": False,
                "depth_estimated": False,
                "engineering_category": None,
            }
            detection["severity_status"] = "area_proxy_estimated"

    union_pixels = int(union_mask.sum())

    if segmentation_calls == 0:
        pipeline_status = "no_d40_detected"
    elif union_pixels == 0:
        pipeline_status = "d40_detected_no_mask_predicted"
    else:
        pipeline_status = "d40_detected_with_mask_predictions"

    mark_stage("severity_calculation")
    annotated = source.copy()
    annotated[union_mask] = (
        0.6 * source[union_mask].astype(np.float32)
        + 0.4 * np.array([0, 255, 0], dtype=np.float32)
    ).astype(np.uint8)

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        left, top = int(np.floor(x1)), int(np.floor(y1))
        right, bottom = int(np.ceil(x2)), int(np.ceil(y2))
        color = (0, 255, 255) if detection["class_id"] == 3 else (255, 200, 0)

        cv2.rectangle(
            annotated, (left, top), (right - 1, bottom - 1), color, 2
        )
        cv2.putText(
            annotated,
            f"{detection['damage_code']} {detection['confidence']:.2f}",
            (left, max(12, top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    mark_stage("annotation")
    report = {
        "schema_version": "1.2",
        "status": pipeline_status,
        "image_width": width,
        "image_height": height,
        "m4_settings": detection_config,
        "m5_settings": segmentation_config,
        "detections": detections,
        "detection_count": len(detections),
        "d40_detection_count": segmentation_calls,
        "segmentation_calls": segmentation_calls,
        "m4_max_det_reached": (
            len(detections) >= detection_config["max_det"]
        ),
        "m5_crops_reaching_max_det": segmentation_limit_hits,
        "union_mask_pixels": union_pixels,
        "union_mask_percent_of_full_image": (
            100.0 * union_pixels / (width * height)
        ),
        "area_units": "pixels_and_image_percent",
        "severity_status": (
            "area_proxy_estimated"
            if any(
                d["severity_status"] == "area_proxy_estimated"
                for d in detections
            )
            else (
                "unavailable_no_predicted_mask"
                if segmentation_calls
                else "no_d40_detected"
            )
        ),
    }

    # Ensure the report can be returned later by an HTTP service.
    json.dumps(report, allow_nan=False)

    mark_stage("report_creation")

    report["timing"] = {
        "device": "cpu",
        "units": "milliseconds",
        "stages": stage_ms,
        "total_ms": (timing_last - timing_start) * 1000.0,
        "scope": (
            "Input validation through prediction-report JSON validation"
        ),
        "excluded": [
            "model loading",
            "file decoding",
            "PNG encoding",
            "HTTP",
            "timing metadata assembly",
        ],
        "predict_scope": (
            "Model predict calls include their preprocessing "
            "and postprocessing"
        ),
    }

    return report, annotated, union_mask
