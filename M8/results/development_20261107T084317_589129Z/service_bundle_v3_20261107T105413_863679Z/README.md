# M8 Road Damage HTTP Prototype

## Version and Scope

This bundle contains the PyTorch CPU pipeline with report schema 1.2,
including pothole image-area proxies and per-stage timings.

The FastAPI application version remains 0.1.0. Application version and
prediction-report schema version are separate identifiers.

Read MODEL_CARD.md for evaluation results, intended use, and limitations.
This bundle is intended for a supervised local demonstration.
Operational deployment is not approved.

## Required External Models

Models and datasets are not included in the service-only package.

Set M8_PROJECT_ROOT to the directory containing Drive_assests.
The checkpoint paths below are relative to that root.
Startup verifies their SHA-256 fingerprints.

### M4

Path: `Drive_assests/M4_Combined/runs/yolo11n_baseline/weights/best.pt`

SHA-256: `e034ca9b5610017d20b22e9014d99439ea3f98117d783faf09113c82a3d9098a`

### M5

Path: `Drive_assests/M5_Segmentation-20261106T135443Z-1-001/M5_Segmentation/exports/m5_yolo11n_seg_v1_with_lab/best.pt`

SHA-256: `5fb5f6925489060792d160f4a04bc3620c0b59a5de5aba013f2e6f5523254d44`


Preserve this directory structure when moving the bundle.
Set M8_PROJECT_ROOT explicitly after relocation. The launch.local.json
fallback describes the original local layout and may not remain valid.

## Dependencies

requirements-service.txt records observed direct service dependencies.
environment_inventory.json records the original inspected environment.

These records are not a complete dependency lock. Installation in a
clean environment has not been verified. PyTorch wheel availability
depends on the Python version, operating system, and package index.

The following is a proposed clean-environment setup, not a recorded
successful installation. Run it from the extracted bundle directory:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-service.txt
```

ONNX export and verification tools are optional for the PyTorch service:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-verification.txt
```

## Start the Service

From the bundle directory, set the project root and use the Python
interpreter containing the required dependencies:

```powershell
$env:M8_PROJECT_ROOT = "C:\path\to\phase2"
python -m uvicorn app:app --host 127.0.0.1 --port 8766 --workers 1
```

For a successfully prepared virtual environment, replace `python` with
`.\.venv\Scripts\python.exe`.

Interactive API: http://127.0.0.1:8766/docs

Stop a foreground server with Ctrl+C. Do not start a second server on
the same port. Keep the service bound to localhost for this prototype.

## Health Check

```powershell
curl.exe --fail-with-body http://127.0.0.1:8766/health
```

GET /health returns readiness, CPU device, model fingerprints,
and upload limits. Successful startup verifies model tasks and classes.

## Predict

```powershell
curl.exe --fail-with-body -X POST `
  "http://127.0.0.1:8766/predict" `
  -F "file=@C:\path\to\example.png" `
  -o response.json
```

The upload field must be named `file`.

Supported input: readable single-frame JPEG or PNG.
Maximum encoded upload: 10 MiB.
Maximum decoded image size: 12,000,000 pixels.

Expected validation responses:

| Condition | HTTP status |
|---|---:|
| Missing file field | 422 |
| Empty or invalid image | 400 |
| Unsupported format | 415 |
| Byte or pixel limit exceeded | 413 |

## Response Structure

The JSON response contains:

- `report`: detection, segmentation, area-proxy, and timing results.
- `model_fingerprints`: SHA-256 fingerprints for M4 and M5.
- `annotated_png_base64`: base64-encoded annotated PNG.
- `mask_png_base64`: base64-encoded binary PNG, using values 0 and 255.

Bounding boxes use original-image pixel coordinates.

M4 retains all four damage classes. M5 runs only on predicted D40 crops.
No D40 detection means segmentation is skipped; it does not prove that
the road contains no potholes.

For a nonempty D40 mask, the severity proxy is predicted mask pixels
divided by full-image pixels, multiplied by 100. It is not physical
area, depth, or an engineering severity category.

Missing masks produce unavailable severity, not zero severity.
Crack severity is unsupported. Global union area avoids counting
overlapping mask pixels more than once.

Stage timings are CPU wall-clock measurements within the pipeline.
They exclude model loading, upload/file decoding, PNG encoding,
and HTTP transport. They are not end-to-end HTTP latency.

## Decode the Returned Images

Run this Python snippet beside response.json:

```python
from pathlib import Path
import base64
import json

response = json.loads(
    Path("response.json").read_text(encoding="utf-8")
)

for key, filename in [
    ("annotated_png_base64", "annotated.png"),
    ("mask_png_base64", "mask.png"),
]:
    Path(filename).write_bytes(
        base64.b64decode(response[key], validate=True)
    )
```

## Evidence and Limitations

Development reports, benchmark measurements, ONNX artifacts, and the
failure gallery are maintained separately under the experiment directory.

Previous-version results retain their original scope. Consult the model
card and evidence manifests before attributing a result to this version.

Low D40 recall limits detector-gated segmentation. Integrated mask and
severity-proxy accuracy remain unverified on the development cohorts.
No independent final-test evaluation or clean-install verification is
claimed.

The service does not automatically hide faces or license plates.
Prepared gallery redactions apply only to those presentation images.
Review and anonymize images before external sharing.

The prototype has not been validated for public hosting or concurrent
production workloads.
