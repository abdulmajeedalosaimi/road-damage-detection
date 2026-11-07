from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "m1"
OUTPUT = REPORTS / "duplicate_review"
OUTPUT.mkdir(parents=True, exist_ok=True)

rows = pd.read_csv(REPORTS / "cross_split_candidates.csv")

COLS = 4
PER_PAGE = 16
CELL_W, CELL_H = 300, 250

for group_id, group in rows.groupby("candidate_group"):
    records = group.to_dict("records")

    for start in range(0, len(records), PER_PAGE):
        page = records[start:start + PER_PAGE]
        grid_rows = (len(page) + COLS - 1) // COLS
        sheet = Image.new(
            "RGB", (COLS * CELL_W, 40 + grid_rows * CELL_H), "white"
        )
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (10, 10),
            f"Candidate group {group_id} | Page {start // PER_PAGE + 1}",
            fill="black",
        )

        for index, row in enumerate(page):
            x = (index % COLS) * CELL_W
            y = 40 + (index // COLS) * CELL_H

            path = Path(row["image_path"])
            if not path.is_absolute():
                path = ROOT / path

            with Image.open(path) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (290, 190))
                sheet.paste(thumb, (x + 5, y))

            draw.text(
                (x + 5, y + 195),
                f"{row['split']} | {row['source']}",
                fill="black",
            )
            draw.text((x + 5, y + 215), path.name, fill="black")

        name = f"group_{int(group_id):04d}_page_{start // PER_PAGE + 1:02d}.jpg"
        sheet.save(OUTPUT / name, quality=90)

print(f"Review sheets saved to: {OUTPUT}")
print("These are candidates, not confirmed duplicates.")