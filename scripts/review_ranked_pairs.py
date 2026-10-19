from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "m1"
OUTPUT = REPORTS / "ranked_pair_review"
OUTPUT.mkdir(parents=True, exist_ok=True)

pairs = pd.read_csv(REPORTS / "duplicate_pairs_initial_review.csv")
decisions = []

for number, (_, row) in enumerate(pairs.iterrows(), start=1):
    sheet = Image.new("RGB", (1000, 600), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (15, 15),
        f"Pair {number:03d} | {row['selection']} | "
        f"Score: {row['ranking_score']:.3f} (not a probability)",
        fill="black",
    )

    for side, x in [("a", 10), ("b", 510)]:
        path = Path(row[f"path_{side}"])
        if not path.is_absolute():
            path = ROOT / path

        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            thumbnail = ImageOps.contain(image, (480, 440))
            sheet.paste(thumbnail, (x, 55))

        draw.text(
            (x, 510), f"Split: {row[f'split_{side}']}", fill="black"
        )
        draw.text((x, 535), path.name, fill="black")

    filename = f"pair_{number:03d}.jpg"
    sheet.save(OUTPUT / filename, quality=95)

    decisions.append({
        "pair_id": number,
        "sheet": filename,
        "image_a": row["image_a"],
        "image_b": row["image_b"],
        "selection": row["selection"],
        "status": "pending",
        "notes": "",
    })

decision_path = REPORTS / "ranked_pair_review_decisions.csv"
if not decision_path.exists():
    pd.DataFrame(decisions).to_csv(decision_path, index=False)
else:
    print("Existing decisions file preserved.")

print(f"Created {len(pairs)} review sheets in: {OUTPUT}")
print(f"Decisions file: {decision_path}")