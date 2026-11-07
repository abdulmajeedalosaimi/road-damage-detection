
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

bundle = Path(sys.argv[1]).resolve()
project_root = Path(sys.argv[2]).resolve()
input_path = Path(sys.argv[3]).resolve()
output_dir = Path(sys.argv[4]).resolve()

sys.path.insert(0, str(bundle))
from pipeline import run_integrated_inference

config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))

models = {}
verified_hashes = {}

for key in ("m4", "m5"):
    checkpoint = (project_root / config[key]["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(project_root):
        raise ValueError(f"Checkpoint outside project root: {key}")

    actual_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual_hash != config[key]["sha256"]:
        raise ValueError(f"Checkpoint fingerprint mismatch: {key}")

    verified_hashes[key] = actual_hash
    models[key] = YOLO(str(checkpoint)).to(config["device"])

image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
if image is None:
    raise ValueError("Cannot read verification input.")

report, overlay, mask = run_integrated_inference(
    image,
    models["m4"],
    models["m5"],
    config["m4"]["settings"],
    config["m5"]["settings"],
    device=config["device"],
)

(output_dir / "report.json").write_text(
    json.dumps(report, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
np.savez_compressed(output_dir / "outputs.npz", mask=mask, overlay=overlay)
(output_dir / "process_info.json").write_text(
    json.dumps({
        "python_executable": sys.executable,
        "python_version": sys.version,
        "verified_checkpoint_hashes": verified_hashes,
        "pipeline_sha256": hashlib.sha256(
            (bundle / "pipeline.py").read_bytes()
        ).hexdigest(),
        "config_sha256": hashlib.sha256(
            (bundle / "config.json").read_bytes()
        ).hexdigest(),
    }, indent=2) + "\n",
    encoding="utf-8",
)
print("Fresh-process inference completed.")
