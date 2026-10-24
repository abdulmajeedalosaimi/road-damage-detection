"""Read-only checks of the fixed M1 manifests and safe crop extraction."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import shutil
import stat
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

CLASSES = ["D00", "D10", "D20", "D40"]
SPLITS = ["train", "val", "test"]
FILES = ["annotations.csv", "classification_manifest.csv", "split_manifest.csv"]
EXPECTED_COUNTS = {"train": 12000, "val": 9463, "test": 11229}
EXPECTED_SOURCES = {"India": "train", "Japan": "train", "United_States": "train",
                    "China": "val", "Czech": "val", "Norway": "test"}


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def crop_relative(value):
    """Resolve Windows separators without using absolute raw-data paths."""
    value = str(value).replace("\\", "/")
    parts = PurePosixPath(value).parts
    require(not value.startswith("/") and not any(":" in p for p in parts), "Absolute crop path rejected")
    require(".." not in parts and parts.count("classification_crops") == 1, "Unsafe or unrecognized crop path")
    tail = parts[parts.index("classification_crops") + 1:]
    require(len(tail) == 3 and tail[0] in SPLITS and tail[1] in CLASSES, "Expected split/class/file.jpg")
    require(tail[2].lower().endswith(".jpg"), "Expected JPG crop")
    return "/".join(tail)


def validate_manifests(data_dir, strict=True):
    """Validate tables, not pixels or visual correctness of damage labels."""
    data_dir = Path(data_dir)
    a = pd.read_csv(data_dir / FILES[0], keep_default_na=False)
    c = pd.read_csv(data_dir / FILES[1], keep_default_na=False)
    s = pd.read_csv(data_dir / FILES[2], dtype={"phash": str, "sha256": str}, keep_default_na=False)
    required = [
        (a, ["image_id", "object_index", "class", "country", "source", "image_path", "split", "xmin", "ymin", "xmax", "ymax"]),
        (c, ["image_id", "crop_path", "class", "split"]),
        (s, ["image_id", "country", "source", "image_path", "width", "height", "sha256", "split", "status"]),
    ]
    for frame, cols in required:
        require(set(cols) <= set(frame.columns), f"Missing columns: {set(cols) - set(frame.columns)}")
        require(len(frame) > 0 and not frame[cols].isna().any().any(), "Missing required values")
        require(not frame[cols].astype(str).apply(lambda col: col.str.strip().eq("")).any().any(), "Blank required values")
        require(frame["split"].isin(SPLITS).all(), "Unknown split")
    require(not s.image_id.duplicated().any(), "Duplicate parent image ID")
    require(s.status.eq("valid").all(), "Invalid parent images")
    require((s[["width", "height"]] > 0).all().all(), "Invalid parent dimensions")
    require(s.sha256.str.fullmatch(r"[a-fA-F0-9]{64}").all(), "Invalid recorded SHA256")
    require(not s.sha256.duplicated().any(), "Recorded exact duplicate images remain")
    require((s.groupby("source").split.nunique() == 1).all(), "Country/source crosses splits")
    require(a["class"].isin(CLASSES).all() and c["class"].isin(CLASSES).all(), "Unknown class")
    require(not a.duplicated(["image_id", "object_index"]).any(), "Duplicate annotation object key")
    indices = pd.to_numeric(a.object_index, errors="coerce")
    require(np.isfinite(indices).all() and indices.ge(0).all() and indices.eq(indices.round()).all(), "Invalid object index")
    require(not c.crop_path.duplicated().any(), "Duplicate crop path")
    parent = a.merge(s, on="image_id", how="left", suffixes=("", "_parent"), validate="many_to_one", indicator=True)
    require(parent._merge.eq("both").all(), "Annotation parent missing")
    for col in ["split", "country", "source", "image_path"]:
        require(parent[col].eq(parent[col + "_parent"]).all(), f"Annotation {col} differs from parent")
    boxes = parent[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
    require(np.isfinite(boxes).all(), "Non-finite bounding box")
    require(((boxes[:, 0] >= 0) & (boxes[:, 1] >= 0) & (boxes[:, 2] > boxes[:, 0]) &
             (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] <= parent.width) &
             (boxes[:, 3] <= parent.height)).all(), "Invalid or out-of-bounds bounding box")
    c = c.copy()
    c["relative_path"] = c.crop_path.map(crop_relative)
    require(not c.relative_path.duplicated().any(), "Duplicate normalized crop path")
    tails = c.relative_path.str.split("/")
    require(tails.str[0].eq(c.split).all() and tails.str[1].eq(c["class"]).all(), "Crop folder label/split mismatch")
    parsed = tails.str[2].str.extract(r"^(.*)_(\d+)\.jpg$")
    require(parsed.notna().all().all() and parsed[0].eq(c.image_id).all(), "Crop filename does not match parent")
    c["object_index"] = parsed[1].astype(int)
    require(not c.duplicated(["image_id", "object_index"]).any(), "Duplicate crop object key")
    c = c.merge(a, on=["image_id", "object_index"], how="left", suffixes=("", "_annotation"), validate="one_to_one", indicator=True)
    require(c._merge.eq("both").all(), "Crop has no source annotation")
    for col in ["class", "split"]:
        require(c[col].eq(c[col + "_annotation"]).all(), f"Crop {col} does not match annotation")
    require((c.groupby("image_id").split.nunique() == 1).all(), "Parent image leaks across splits")
    require(not ((c.image_id == "Japan_Japan_003187") & (c.object_index == 0)).any(), "Excluded Japan crop is present")
    for split in ["val", "test"]:
        required_keys = set(a.loc[a.split.eq(split), ["image_id", "object_index"]].itertuples(index=False, name=None))
        actual_keys = set(c.loc[c.split.eq(split), ["image_id", "object_index"]].itertuples(index=False, name=None))
        require(required_keys == actual_keys, f"{split} crops do not exactly cover source annotations")
    counts = c.split.value_counts().to_dict()
    if strict:
        require(len(a) == 55002 and len(s) == 38384 and counts == EXPECTED_COUNTS, "M1 snapshot counts changed; review before training")
        require(s.groupby("source").split.first().to_dict() == EXPECTED_SOURCES, "M1 country allocation changed")
        require(c.loc[c.split.eq("train")].groupby("class").size().to_dict() == dict.fromkeys(CLASSES, 3000), "M1 balanced training subset changed")
    balance = pd.crosstab(c["class"], c.split).reindex(index=CLASSES, columns=SPLITS, fill_value=0)
    report = {
        "status": "PASS", "scope": "Manifest consistency only; pixels and visual labels not checked here.",
        "original_images": len(s), "annotations": len(a), "crops": len(c),
        "crop_counts": {k: int(v) for k, v in counts.items()},
        "class_counts": {k: {sp: int(balance.loc[k, sp]) for sp in SPLITS} for k in CLASSES},
        "parent_counts": {k: int(v) for k, v in c.groupby("split").image_id.nunique().items()},
        "country_allocation": s.groupby("source").split.first().to_dict(),
        "excluded_japan_annotation_present": bool(((a.image_id == "Japan_Japan_003187") & (a.object_index == 0)).any()),
        "excluded_japan_crop_present": False,
        "recorded_sha256_duplicates": int(s.sha256.duplicated().sum()),
        "manifest_sha256": {name: sha256(data_dir / name) for name in FILES},
    }
    c = c.drop(columns=["_merge", "class_annotation", "split_annotation"])
    return c, report


def extract_crops(archive, destination):
    """Extract known-layout JPGs to a fresh local directory without overwriting."""
    archive, destination = Path(archive), Path(destination)
    digest = sha256(archive)
    marker = destination / ".archive.json"
    if marker.exists():
        require(json.loads(marker.read_text())["sha256"] == digest, "Crop cache belongs to another ZIP")
        return destination
    require(not destination.exists() or not any(destination.iterdir()),
            "Nonempty crop cache without completion marker. Choose a new local cache directory.")
    with zipfile.ZipFile(archive) as z:
        entries, targets = [], set()
        for info in z.infolist():
            parts = PurePosixPath(info.filename.replace("\\", "/")).parts
            require(".." not in parts and not info.filename.startswith(("/", "\\")) and not any(":" in p for p in parts), "Unsafe ZIP member")
            require(not stat.S_ISLNK(info.external_attr >> 16), "ZIP symlinks rejected")
            if info.is_dir() or "__MACOSX" in parts or not info.filename.lower().endswith(".jpg"):
                continue
            tail = parts[parts.index("classification_crops") + 1:] if "classification_crops" in parts else parts
            require(len(tail) == 3 and tail[0] in SPLITS and tail[1] in CLASSES, f"Unexpected ZIP layout: {info.filename}")
            relative = "/".join(tail)
            require(relative not in targets, "Duplicate ZIP image name")
            targets.add(relative)
            entries.append((info, relative))
        require(bool(entries), "No crop JPGs found")
        require(sum(info.file_size for info, _ in entries) < 10 * 1024**3, "Unexpectedly large archive")
        destination.mkdir(parents=True, exist_ok=True)
        for info, relative in entries:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
    save_json(marker, {"sha256": digest, "files": len(entries)})
    return destination


def verify_pixels(crops, crop_root, image_size=224):
    """Decode every image once and verify RGB, size and manifest/file agreement."""
    crop_root = Path(crop_root)
    expected = set(crops.relative_path)
    actual = {p.relative_to(crop_root).as_posix() for p in crop_root.rglob("*") if p.is_file() and p.suffix.lower() == ".jpg"}
    require(actual == expected, f"Crop files differ: {len(expected-actual)} missing, {len(actual-expected)} extra")
    for index, rel in enumerate(crops.relative_path, 1):
        with Image.open(crop_root / rel) as im:
            im.load()
            require(im.mode == "RGB" and im.size == (image_size, image_size), f"Unexpected crop mode/size: {rel}")
        if index % 5000 == 0:
            print(f"Verified {index:,}/{len(crops):,} images", flush=True)
    return {"status": "PASS", "images_decoded": len(crops), "mode": "RGB", "size": [image_size, image_size]}

