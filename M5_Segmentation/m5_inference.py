"""Inference-only M5 handoff. No reference masks or M4 model are required."""
import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


class PotholeSegmenter:
    def __init__(self, checkpoint, config, device="cpu"):
        self.model = YOLO(str(checkpoint))
        self.config = json.loads(Path(config).read_text(encoding="utf-8"))
        self.device = device

    def predict(self, image):
        """Accept an image path or a BGR uint8 array; return masks at the original image size."""
        bgr = cv2.imread(str(image)) if isinstance(image, (str, Path)) else np.asarray(image)
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("Expected a readable three-channel BGR image.")
        original_h, original_w = bgr.shape[:2]
        size = int(self.config["imgsz"])
        resized = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
        result = self.model.predict(source=resized, imgsz=size,
            conf=self.config["threshold"], iou=self.config["nms_iou"],
            max_det=self.config["max_det"], retina_masks=True, device=self.device,
            verbose=False, augment=False)[0]
        union = np.zeros((original_h, original_w), bool)
        instances = []
        if result.masks is not None:
            for mask, score, category in zip(result.masks.data.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                if int(category) != 0:
                    raise ValueError("The M5 checkpoint must use class 0 for pothole.")
                binary = cv2.resize((mask > 0.5).astype(np.uint8), (original_w, original_h),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                union |= binary
                instances.append({"mask": binary, "confidence": float(score),
                    "m5_class_id": 0, "m4_class_id": 3, "damage_code": "D40"})
        return {"mask": union, "area_pixels": int(union.sum()),
            "area_fraction": float(union.mean()), "area_percent": 100 * float(union.mean()),
            "original_size_hw": [original_h, original_w], "instances": instances}
