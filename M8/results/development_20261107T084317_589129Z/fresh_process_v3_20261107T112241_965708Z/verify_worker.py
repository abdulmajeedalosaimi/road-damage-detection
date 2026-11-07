
from pathlib import Path
import hashlib
import json
import math
import sys

import cv2
import numpy as np
from ultralytics import YOLO

work = Path(sys.argv[1]).resolve()
job = json.loads((work / "job.json").read_text(encoding="utf-8"))
bundle = Path(job["bundle"])
root = Path(job["project_root"])

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert sha(bundle / "pipeline.py") == job["pipeline_sha256"]
assert sha(bundle / "config.json") == job["config_sha256"]

for name, expected in job["reference_sha256"].items():
    assert sha(work / name) == expected, name

sys.path.insert(0, str(bundle))
from pipeline import run_integrated_inference

config = json.loads(
    (bundle / "config.json").read_text(encoding="utf-8")
)
assert config["schema_version"] == "1.2"
assert config["device"] == "cpu"

models = {}
for name in ("m4", "m5"):
    path = (root / config[name]["checkpoint"]).resolve()
    assert path.is_relative_to(root)
    assert path.is_file(), path
    assert sha(path) == config[name]["sha256"], name
    models[name] = YOLO(str(path))
    models[name].to("cpu")

def read_image(name, flags):
    data = np.frombuffer((work / name).read_bytes(), dtype=np.uint8)
    result = cv2.imdecode(data, flags)
    assert result is not None, name
    return result

image = read_image("input.png", cv2.IMREAD_COLOR)
unchanged_input = image.copy()

report, overlay, mask = run_integrated_inference(
    image_bgr=image,
    detector=models["m4"],
    segmenter=models["m5"],
    detector_settings=config["m4"]["settings"],
    segmenter_settings=config["m5"]["settings"],
    device="cpu",
)

reference = json.loads(
    (work / "reference_report.json").read_text(encoding="utf-8")
)
differences = []

def compare(old, new, path="report"):
    if isinstance(old, dict):
        if not isinstance(new, dict) or set(old) != set(new):
            differences.append(path + ": dictionary keys differ")
            return
        for key in old:
            compare(old[key], new[key], path + "." + key)
    elif isinstance(old, list):
        if not isinstance(new, list) or len(old) != len(new):
            differences.append(path + ": list length differs")
            return
        for index, (a, b) in enumerate(zip(old, new)):
            compare(a, b, f"{path}[{index}]")
    elif isinstance(old, (int, float)) and not isinstance(old, bool):
        if not isinstance(new, (int, float)) or isinstance(new, bool):
            differences.append(path + ": numeric type differs")
            return
        tolerance = (
            0.0001 if ".bbox_xyxy[" in path
            else 0.000001 if path.endswith(".confidence")
            else 0.0
        )
        if not math.isclose(old, new, rel_tol=0.0, abs_tol=tolerance):
            differences.append(f"{path}: {old!r} != {new!r}")
    elif old != new:
        differences.append(f"{path}: {old!r} != {new!r}")

compare(
    {key: value for key, value in reference.items() if key != "timing"},
    {key: value for key, value in report.items() if key != "timing"},
)

timing = report["timing"]
stages = timing["stages"]
timing_values = [*stages.values(), timing["total_ms"]]

timing_valid = (
    set(stages) == set(reference["timing"]["stages"])
    and timing["device"] == "cpu"
    and timing["units"] == "milliseconds"
    and all(math.isfinite(value) and value >= 0 for value in timing_values)
    and math.isclose(
        sum(stages.values()),
        timing["total_ms"],
        rel_tol=1e-9,
        abs_tol=1e-6,
    )
)

reference_mask = read_image("reference_mask.png", cv2.IMREAD_GRAYSCALE)
reference_overlay = read_image("reference_overlay.png", cv2.IMREAD_COLOR)

checks = {
    "schema_version_1_2": report["schema_version"] == "1.2",
    "prediction_report_within_declared_tolerances": not differences,
    "mask_exact_match": np.array_equal(
        mask.astype(np.uint8) * 255, reference_mask
    ),
    "overlay_exact_match": np.array_equal(overlay, reference_overlay),
    "timing_valid": timing_valid,
    "input_preserved": np.array_equal(image, unchanged_input),
}

(work / "prediction_report.json").write_text(
    json.dumps(report, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)

result = {
    "purpose": "fresh_process_v3_verification",
    "checks": {key: bool(value) for key, value in checks.items()},
    "differences": differences,
    "pipeline_sha256": job["pipeline_sha256"],
    "config_sha256": job["config_sha256"],
    "checkpoint_sha256": {
        name: config[name]["sha256"] for name in ("m4", "m5")
    },
    "reference_sha256": job["reference_sha256"],
    "python_executable": sys.executable,
    "python_version": sys.version,
    "bbox_absolute_tolerance_pixels": 0.0001,
    "confidence_absolute_tolerance": 0.000001,
    "timing_values_compared_for_equality": False,
    "clean_install_verified": False,
}

(work / "verification.json").write_text(
    json.dumps(result, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)

for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {name}", flush=True)

if differences:
    print(json.dumps(differences[:20], indent=2), flush=True)

if not all(checks.values()):
    raise AssertionError("Fresh-process verification failed.")

print("PASS: Updated pipeline reproduced in a fresh Python process.")
