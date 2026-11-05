# M5 handoff

Load `best.pt` and `inference_config.json` with `m5_inference.py`:

```python
from m5_inference import PotholeSegmenter
segmenter = PotholeSegmenter("best.pt", "inference_config.json", device="cpu")
result = segmenter.predict("example.jpg")
print(result["area_percent"])
```

`mask` is a boolean array at the supplied image's original resolution.
`instances` contains masks, confidence and the mapping M5 class 0 -> M4 class 3 / D40.
Area uses the union of masks, so overlapping instances are not counted twice.
The notebook evaluates at 640 x 640; the helper also returns a mask resized to the caller's original image size.
For a crop, area_percent describes that crop, not the full road image. Mapping a crop mask back to a frame
requires the crop coordinates. This notebook does not claim M4-to-M5 integration has been tested.
The source videos are top-down pothole views; vehicle-mounted road images can have a domain gap.

No reference masks, ground-truth boxes or SAM prompts are used by this inference helper.
SAM in the notebook is a separate training-annotation review exercise.
HTTP serving, ONNX conversion and service latency belong to the later integration/deployment workstream.
Raw videos and extracted dataset frames are excluded from the submission package.
