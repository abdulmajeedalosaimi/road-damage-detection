from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "m1"
candidates = pd.read_csv(REPORTS / "cross_split_candidates.csv")

cache = {}


def features(image_path):
    if image_path in cache:
        return cache[image_path]

    path = Path(image_path)
    if not path.is_absolute():
        path = ROOT / path

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

        # Compare local brightness changes in both directions.
        horizontal = np.asarray(
            image.convert("L").resize((17, 16)),
            dtype=np.float32,
        )
        vertical = np.asarray(
            image.convert("L").resize((16, 17)),
            dtype=np.float32,
        )
        edges = np.concatenate([
            (horizontal[:, 1:] > horizontal[:, :-1]).ravel(),
            (vertical[1:, :] > vertical[:-1, :]).ravel(),
        ])

        # Compare finer grayscale structure.
        gray = np.asarray(
            image.convert("L").resize((64, 64)),
            dtype=np.float32,
        ).ravel()
        gray -= gray.mean()
        norm = np.linalg.norm(gray)
        gray = gray / norm if norm > 0 else np.zeros_like(gray)

        # Compare spatial color patterns.
        color = np.asarray(
            image.resize((32, 32)), dtype=np.float32
        ) / 255.0

    cache[image_path] = edges, gray, color
    return cache[image_path]


results = []
groups = list(candidates.groupby("candidate_group"))

for position, (group_id, group) in enumerate(groups, start=1):
    for left, right in combinations(group.to_dict("records"), 2):
        if left["split"] == right["split"]:
            continue

        edge_a, gray_a, color_a = features(left["image_path"])
        edge_b, gray_b, color_b = features(right["image_path"])

        edge_similarity = float(np.mean(edge_a == edge_b))
        structure_similarity = float(
            np.clip(np.dot(gray_a, gray_b), 0, 1)
        )
        color_similarity = float(
            1 - np.mean(np.abs(color_a - color_b))
        )

        # Heuristic ranking score, NOT a duplicate probability.
        score = (
            0.45 * edge_similarity
            + 0.40 * structure_similarity
            + 0.15 * color_similarity
        )

        results.append({
            "candidate_group": group_id,
            "image_a": left["image_id"],
            "split_a": left["split"],
            "path_a": left["image_path"],
            "image_b": right["image_id"],
            "split_b": right["split"],
            "path_b": right["image_path"],
            "ranking_score": score,
            "edge_similarity": edge_similarity,
            "structure_similarity": structure_similarity,
            "color_similarity": color_similarity,
        })

    if position % 25 == 0:
        print(f"Processed {position}/{len(groups)} groups", flush=True)

if not results:
    print("No cross-split candidate pairs found.")
else:
    ranked = pd.DataFrame(results).sort_values(
        "ranking_score", ascending=False
    )
    output = REPORTS / "ranked_duplicate_pairs.csv"
    ranked.to_csv(output, index=False)

    # Initial review batch: strongest 30 pairs plus up to 20 others.
    strongest = ranked.head(30)
    remaining = ranked.iloc[30:]
    sample = remaining.sample(
        n=min(20, len(remaining)), random_state=42
    )
    review = pd.concat([
        strongest.assign(selection="highest_ranked"),
        sample.assign(selection="random_remaining"),
    ])
    review.to_csv(
        REPORTS / "duplicate_pairs_initial_review.csv", index=False
    )

    print(f"\nRanked {len(ranked)} candidate pairs.")
    print(f"Saved: {output}")
    print(f"Initial review batch: {len(review)} pairs.")
    print("Scores are not probabilities or confirmed duplicates.")