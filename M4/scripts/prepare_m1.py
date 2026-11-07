"""Audit RDD2022 labels and create a leakage-safe country-disjoint split."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]


def local_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def stored_path(path: Path) -> str:
    """Store project-relative paths when possible; otherwise retain a valid absolute path."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def visual_hash(path: Path) -> str:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((8, 8)).getdata())
    mean = sum(pixels) / len(pixels)
    return "".join("1" if item >= mean else "0" for item in pixels)


def country_of(raw_dir: Path, path: Path) -> str:
    return path.relative_to(raw_dir).parts[0]


def parse_xml(path: Path, allowed: set[str]) -> tuple[list[dict], list[str]]:
    root = ET.parse(path).getroot()
    boxes, ignored = [], []
    for obj_index, obj in enumerate(root.findall("object")):
        label = (obj.findtext("name") or "").strip()
        if label not in allowed:
            if label:
                ignored.append(label)
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            values = {name: float(box.findtext(name)) for name in ("xmin", "ymin", "xmax", "ymax")}
        except (TypeError, ValueError):
            continue
        if values["xmax"] <= values["xmin"] or values["ymax"] <= values["ymin"]:
            continue
        boxes.append({"object_index": obj_index, "class": label, **values})
    return boxes, ignored


def choose_splits(source_counts: dict[str, Counter], fractions: list[float], hold_out_one: bool) -> tuple[dict[str, str], float]:
    """Choose a source-disjoint assignment closest to class-count targets."""
    sources = sorted(source_counts)
    if len(sources) < 3:
        raise RuntimeError("At least three source countries are needed for a country-disjoint train/val/test split.")
    classes = sorted({label for counts in source_counts.values() for label in counts})
    total = Counter()
    for counts in source_counts.values():
        total.update(counts)
    targets = [{label: total[label] * fraction for label in classes} for fraction in fractions]
    best, best_cost = None, float("inf")
    for allocation in itertools.product(range(3), repeat=len(sources)):
        if set(allocation) != {0, 1, 2}:
            continue
        if hold_out_one and allocation.count(2) != 1:
            continue
        observed = [Counter() for _ in range(3)]
        for source, split_index in zip(sources, allocation):
            observed[split_index].update(source_counts[source])
        cost = sum(
            ((observed[index][label] - targets[index][label]) / max(1.0, targets[index][label])) ** 2
            for index in range(3) for label in classes
        )
        if cost < best_cost:
            best, best_cost = allocation, cost
    names = ("train", "val", "test")
    return {source: names[index] for source, index in zip(sources, best)}, best_cost


def source_group_of(cfg: dict, country: str) -> str:
    """Map country folders to a leakage-safe collection-source group."""
    return cfg["split"].get("source_group_map", {}).get(country, country)


def write_card(path: Path, cfg: dict, report: dict, source_split: dict[str, str]) -> None:
    class_lines = ["| Class | Meaning | Verified objects |", "|---|---|---:|"]
    for label in cfg["classes"]:
        class_lines.append(f"| {label} | {cfg['class_descriptions'][label]} | {report['class_counts'][label]} |")
    source_lines = ["| Collection source | Split |", "|---|---|"]
    source_lines += [f"| {source} | {split} |" for source, split in sorted(source_split.items())]
    text = "\n".join([
        "# Dataset Card — RDD2022 Road-Damage Classification", "",
        "## Intended use", "This project classifies object crops produced from existing RDD2022 road-damage boxes.",
        "It is a research prototype and must not be used alone for road-safety decisions.", "",
        "## Audit", f"- Valid labelled images: {report['valid_images']}",
        f"- Valid D00/D10/D20/D40 objects: {report['valid_annotations']}",
        f"- Corrupt/unreadable images: {report['corrupt_images']}",
        f"- Exact duplicate files removed: {report['exact_duplicates']}",
        f"- Visual-duplicate candidate groups: {report['visual_duplicate_candidates']}", "",
        "## Classes", *class_lines, "",
        "## Leakage-safe split", "Each collection-source group belongs to exactly one split.",
        "China Drone and China MotorBike are kept together as the China source group.",
        "This avoids training and testing on the same source, capture setup, and road environment.",
        "It creates a realistic domain-shift test and is more reliable than an image-random split.", *source_lines, "",
        "## Preprocessing", f"Each crop includes {cfg['classification']['crop_padding_fraction']:.0%} context around a verified box",
        f"and is resized to {cfg['classification']['image_size']}x{cfg['classification']['image_size']}.",
        "Training augmentation is applied only to train crops; validation and test use deterministic transforms.", "",
        "## Limitations", "The classes are imbalanced; road conditions and camera viewpoints differ across countries.",
        "Official RDD2022 test labels are withheld, so this project builds validation and test splits only from labelled training data.",
        "RDD2022 provides boxes but no instance masks, so it is not ground truth for segmentation.",
    ])
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = read_config(local_path(args.config))
    raw_dir = local_path(cfg["paths"]["raw_dir"])
    processed_dir = local_path(cfg["paths"]["processed_dir"])
    report_dir = local_path(cfg["paths"]["reports_dir"]) / "m1"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data was not found at: {raw_dir}")
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    extensions = {item.lower() for item in cfg["audit"]["image_extensions"]}
    image_paths = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in extensions and "train" in p.parts)
    xml_index = {(country_of(raw_dir, p), p.stem): p for p in raw_dir.rglob("*.xml")}
    allowed = set(cfg["classes"])
    exact_seen, visual_groups = {}, defaultdict(list)
    images, boxes, corrupt, missing_xml, ignored = [], [], [], [], Counter()

    for number, image_path in enumerate(image_paths, 1):
        country = country_of(raw_dir, image_path)
        xml_path = xml_index.get((country, image_path.stem))
        if xml_path is None:
            missing_xml.append({"image_path": str(image_path), "reason": "missing matching XML"})
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
            selected_boxes, skipped = parse_xml(xml_path, allowed)
            sha = file_hash(image_path)
            phash = visual_hash(image_path) if cfg["audit"]["verify_perceptual_duplicates"] else ""
        except (UnidentifiedImageError, OSError, ET.ParseError, ValueError) as error:
            corrupt.append({"image_path": str(image_path), "error": str(error)})
            continue
        ignored.update(skipped)
        image_id = f"{country}_{image_path.stem}"
        source = source_group_of(cfg, country)
        row = {"image_id": image_id, "country": country, "source": source, "image_path": stored_path(image_path),
               "xml_path": stored_path(xml_path), "width": width, "height": height,
               "sha256": sha, "phash": phash, "status": "valid", "duplicate_of": ""}
        if sha in exact_seen:
            row["status"] = "exact_duplicate"
            row["duplicate_of"] = exact_seen[sha]
            images.append(row)
            continue
        exact_seen[sha] = image_id
        images.append(row)
        visual_groups[phash].append(image_id)
        for box in selected_boxes:
            xmin, xmax = max(0, box["xmin"]), min(width, box["xmax"])
            ymin, ymax = max(0, box["ymin"]), min(height, box["ymax"])
            if xmax > xmin and ymax > ymin:
                boxes.append({"image_id": image_id, "country": country, "source": source, "image_path": row["image_path"],
                              "class": box["class"], "object_index": box["object_index"],
                              "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
        if number % 500 == 0:
            print(f"Audited {number:,}/{len(image_paths):,} images")

    valid_images = [row for row in images if row["status"] == "valid"]
    if not boxes:
        raise RuntimeError("No valid D00/D10/D20/D40 boxes were found. Check the raw-data directory.")
    source_counts = defaultdict(Counter)
    for box in boxes:
        source_counts[box["source"]][box["class"]] += 1
    split_by_source, split_cost = choose_splits(
        source_counts,
        [cfg["split"]["train_fraction"], cfg["split"]["val_fraction"], cfg["split"]["test_fraction"]],
        cfg["split"].get("hold_out_exactly_one_source", True),
    )
    split_by_image = {row["image_id"]: split_by_source[row["source"]] for row in valid_images}
    for row in valid_images:
        row["split"] = split_by_image[row["image_id"]]
    for box in boxes:
        box["split"] = split_by_image[box["image_id"]]

    image_frame, box_frame = pd.DataFrame(images), pd.DataFrame(boxes)
    image_frame.to_csv(report_dir / "image_audit.csv", index=False)
    pd.DataFrame(valid_images).to_csv(processed_dir / "split_manifest.csv", index=False)
    box_frame.to_csv(processed_dir / "annotations.csv", index=False)
    pd.DataFrame(corrupt).to_csv(report_dir / "corrupt_files.csv", index=False)
    pd.DataFrame(missing_xml).to_csv(report_dir / "missing_xml.csv", index=False)
    image_frame[image_frame.status == "exact_duplicate"].to_csv(report_dir / "exact_duplicates.csv", index=False)
    visual = [{"phash": phash, "image_ids": "|".join(ids), "count": len(ids)} for phash, ids in visual_groups.items() if len(ids) > 1]
    pd.DataFrame(visual).to_csv(report_dir / "perceptual_duplicate_candidates.csv", index=False)

    balance = pd.crosstab(box_frame["class"], box_frame["split"]).reindex(cfg["classes"], fill_value=0)
    balance.to_csv(report_dir / "class_balance_by_split.csv")
    axis = balance.plot(kind="bar", figsize=(9, 5), title="RDD2022 object balance by split")
    axis.set_xlabel("Road-damage class"); axis.set_ylabel("Verified objects")
    plt.tight_layout(); plt.savefig(report_dir / "class_balance.png", dpi=180); plt.close()

    report = {"raw_train_images_seen": len(image_paths), "valid_images": len(valid_images),
              "valid_annotations": len(boxes), "corrupt_images": len(corrupt), "missing_xml": len(missing_xml),
              "exact_duplicates": int((image_frame.status == "exact_duplicate").sum()),
              "visual_duplicate_candidates": len(visual), "ignored_source_labels": dict(ignored),
              "class_counts": {label: int((box_frame["class"] == label).sum()) for label in cfg["classes"]},
              "source_split": split_by_source,
              "country_to_source_group": {country: source_group_of(cfg, country) for country in sorted({row["country"] for row in valid_images})},
              "split_balance_cost": split_cost}
    (report_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_card(local_path(cfg["paths"]["reports_dir"]) / "DATASET_CARD.md", cfg, report, split_by_source)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
