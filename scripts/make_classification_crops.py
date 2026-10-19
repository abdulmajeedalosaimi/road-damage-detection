"""Create image-classification crops from verified RDD2022 ground-truth boxes."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def local_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(local_path(args.config).read_text(encoding="utf-8"))
    processed_dir = local_path(cfg["paths"]["processed_dir"])
    annotations = pd.read_csv(processed_dir / "annotations.csv")
        # Exclude the confirmed misplaced D40 box found in audit image 044.
    bad_annotation = (
        (annotations["image_id"] == "Japan_Japan_003187")
        & (annotations["object_index"] == 0)
        & (annotations["class"] == "D40")
    )

    excluded = annotations.loc[bad_annotation].copy()
    excluded["exclusion_reason"] = (
        "Audit 044: D40 box above road; source XML coordinates "
        "match processed coordinates."
    )

    report_dir = local_path(cfg["paths"]["reports_dir"]) / "m1"
    report_dir.mkdir(parents=True, exist_ok=True)
    excluded.to_csv(
        report_dir / "excluded_crop_annotations.csv", index=False
    )

    annotations = annotations.loc[~bad_annotation].copy()
    print(f"Excluded {len(excluded)} confirmed bad annotation(s).")
    crop_root = processed_dir / "classification_crops"
    if crop_root.exists() and any(crop_root.iterdir()):
        raise RuntimeError(
            "classification_crops is not empty. Rename that folder "
            "to classification_crops_backup before generating fresh crops."
        )
    crop_root.mkdir(parents=True, exist_ok=True)
    samples = []
    for split, split_data in annotations.groupby("split"):
        for label, class_data in split_data.groupby("class"):
            shuffled = class_data.sample(frac=1, random_state=cfg["seed"])
            if split == "train":
                shuffled = shuffled.head(cfg["classification"]["max_train_per_class"])
            samples.append(shuffled)
    samples = pd.concat(samples, ignore_index=True).rename(columns={"class": "label"})
    padding = cfg["classification"]["crop_padding_fraction"]
    image_size = cfg["classification"]["image_size"]
    output_rows = []
    for row in tqdm(samples.itertuples(index=False), total=len(samples), desc="Writing object crops"):
        destination = crop_root / row.split / row.label / f"{row.image_id}_{row.object_index}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(ROOT / row.image_path) as source:
                image = source.convert("RGB")
                width, height = image.size
                pad_x = (row.xmax - row.xmin) * padding
                pad_y = (row.ymax - row.ymin) * padding
                bounds = (max(0, int(row.xmin - pad_x)), max(0, int(row.ymin - pad_y)),
                          min(width, int(row.xmax + pad_x)), min(height, int(row.ymax + pad_y)))
                image.crop(bounds).resize((image_size, image_size), Image.Resampling.LANCZOS).save(destination, quality=95)
            output_rows.append({"crop_path": str(destination.relative_to(ROOT)), "image_id": row.image_id,
                                "split": row.split, "class": row.label})
        except OSError as error:
            print(f"Skipped {row.image_path}: {error}")
    output = pd.DataFrame(output_rows)
    output.to_csv(processed_dir / "classification_manifest.csv", index=False)
    balance = pd.crosstab(output["class"], output["split"]).reindex(cfg["classes"], fill_value=0)
    balance.to_csv(local_path(cfg["paths"]["reports_dir"]) / "m1" / "classification_crop_balance.csv")
    print(f"Created {len(output):,} crops under {crop_root}")
    print(balance)


if __name__ == "__main__":
    main()
