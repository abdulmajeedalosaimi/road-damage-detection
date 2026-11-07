
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import platform
import sys
import threading
import time

import cv2
import numpy as np
import psutil
from ultralytics import YOLO

protocol_path = Path(sys.argv[1])
backend = sys.argv[2]
output_path = Path(sys.argv[3])
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

bundle = Path(protocol["bundle"])
pipeline_file = bundle / "pipeline.py"
if hashlib.sha256(pipeline_file.read_bytes()).hexdigest() != protocol["pipeline_sha256"]:
    raise ValueError("Pipeline fingerprint mismatch.")

sys.path.insert(0, str(bundle))
from pipeline import run_integrated_inference

models = {}
for name, task in (("m4", "detect"), ("m5", "segment")):
    path = Path(protocol["models"][backend][name])
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != protocol["model_sha256"][backend][name]:
        raise ValueError(f"Model fingerprint mismatch: {name}")
    models[name] = YOLO(str(path), task=task)

images = []
for case in protocol["cases"]:
    raw = (Path(protocol["dataset_root"]) / case["image"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != case["image_sha256"]:
        raise ValueError("Input image fingerprint mismatch.")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unreadable input.")
    images.append(image)

def infer(index):
    return run_integrated_inference(
        images[index], models["m4"], models["m5"],
        protocol["m4_settings"], protocol["m5_settings"],
        device="cpu",
    )

for index in protocol["warmup_case_indices"]:
    output = infer(index)
    del output

process = psutil.Process()
baseline_rss = int(process.memory_info().rss)
samples = []
sampling_errors = []
stop = threading.Event()
origin = time.perf_counter()

def sample_rss():
    while not stop.is_set():
        try:
            samples.append({
                "elapsed_seconds": time.perf_counter() - origin,
                "rss_bytes": int(process.memory_info().rss),
            })
        except Exception as exc:
            sampling_errors.append(str(exc))
            return
        stop.wait(protocol["rss_sampling_interval_seconds"])

sampler = threading.Thread(target=sample_rss, daemon=True)
rows = []
sampler.start()
loop_start = time.perf_counter()

try:
    for number, item in enumerate(protocol["schedule"], 1):
        index = item["case_index"]
        case = protocol["cases"][index]
        row = {
            "repeat": item["repeat"],
            "image": case["image"],
            "cohort": case["cohort"],
            "success": False,
            "latency_ms": None,
            "error": None,
        }
        started = time.perf_counter()
        try:
            output = infer(index)
            row["latency_ms"] = (time.perf_counter() - started) * 1000
            report, overlay, mask = output
            row.update({
                "success": True,
                "timing": report["timing"],
                "detection_count": report["detection_count"],
                "segmentation_calls": report["segmentation_calls"],
                "union_mask_pixels": report["union_mask_pixels"],
            })
            del output, report, overlay, mask
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

        if number % 30 == 0:
            print(f"{backend}: {number}/{len(protocol['schedule'])}", flush=True)

finally:
    wall_seconds = time.perf_counter() - loop_start
    stop.set()
    sampler.join(timeout=5)

    result = {
        "backend": backend,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: metadata.version(name)
            for name in ("ultralytics", "torch", "onnxruntime", "numpy")
        },
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "pipeline_sha256": protocol["pipeline_sha256"],
        "model_sha256": protocol["model_sha256"][backend],
        "expected_calls": len(protocol["schedule"]),
        "successful_calls": sum(row["success"] for row in rows),
        "measurement_loop_wall_seconds": wall_seconds,
        "baseline_rss_bytes": baseline_rss,
        "sampled_peak_rss_bytes": max(
            [baseline_rss] + [sample["rss_bytes"] for sample in samples]
        ),
        "sampling_errors": sampling_errors,
        "rss_samples": samples,
        "requests": rows,
    }
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

if (
    result["successful_calls"] != result["expected_calls"]
    or sampling_errors
):
    raise RuntimeError("Incomplete or failed backend measurements.")

print(f"PASS: {backend} measurements saved.", flush=True)
