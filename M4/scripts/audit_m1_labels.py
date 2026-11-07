from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "m1"
OUTPUT = REPORTS / "label_audit"
OUTPUT.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(ROOT / "data/processed/split_manifest.csv")
annotations = pd.read_csv(ROOT / "data/processed/annotations.csv")

sample_path = OUTPUT / "sample.csv"
decisions_path = OUTPUT / "decisions.csv"

# Preserve the same sample when running again.
if sample_path.exists():
    sample = pd.read_csv(sample_path)
else:
    sample = manifest.sample(
        n=min(100, len(manifest)), random_state=42
    ).copy()
    sample.insert(0, "audit_id", range(1, len(sample) + 1))
    sample.to_csv(sample_path, index=False)

colors = {
    "D00": "red",
    "D10": "cyan",
    "D20": "yellow",
    "D40": "magenta",
}

decisions = []

for _, row in sample.iterrows():
    audit_id = int(row["audit_id"])
    path = Path(row["image_path"])
    if not path.is_absolute():
        path = ROOT / path

    with Image.open(path) as source:
        image = source.convert("RGB")

    # Keep original pixel coordinates so boxes match the annotations.
    draw = ImageDraw.Draw(image)
    boxes = annotations[
        annotations["image_id"] == row["image_id"]
    ]

    for _, box in boxes.iterrows():
        label = box["class"]
        color = colors.get(label, "white")
        coordinates = [
            float(box["xmin"]),
            float(box["ymin"]),
            float(box["xmax"]),
            float(box["ymax"]),
        ]
        draw.rectangle(coordinates, outline=color, width=3)
        draw.text(
            (coordinates[0] + 3, coordinates[1] + 3),
            label,
            fill=color,
            stroke_width=1,
            stroke_fill="black",
        )

    filename = f"audit_{audit_id:03d}.jpg"
    image.save(OUTPUT / filename, quality=95)

    decisions.append({
        "audit_id": audit_id,
        "sheet": filename,
        "image_id": row["image_id"],
        "source": row["source"],
        "split": row["split"],
        "box_count": len(boxes),
        "status": "pending",
        "notes": "",
    })

if not decisions_path.exists():
    pd.DataFrame(decisions).to_csv(decisions_path, index=False)

print(f"Created {len(sample)} annotated images in: {OUTPUT}")
print(f"Review decisions: {decisions_path}")
print("Existing review decisions were preserved.")