from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "m1"

manifest = pd.read_csv(ROOT / "data" / "processed" / "split_manifest.csv")
candidates = pd.read_csv(REPORTS / "perceptual_duplicate_candidates.csv")

# Each image and source must belong to exactly one split.
assert manifest["image_id"].is_unique, "Repeated image IDs found."
assert manifest["split"].isin(["train", "val", "test"]).all()
assert manifest.groupby("source")["split"].nunique().max() == 1, (
    "A source appears in multiple splits."
)

test_sources = manifest.loc[manifest["split"] == "test", "source"].unique()
assert len(test_sources) == 1, "Expected exactly one held-out test source."

# Identical file contents must not cross splits.
hash_splits = manifest.groupby("sha256")["split"].nunique()
cross_split_exact = hash_splits[hash_splits > 1]
assert cross_split_exact.empty, "Exact duplicate content crosses splits."

lookup = manifest.set_index("image_id")
review_rows = []

for group_id, group in candidates.iterrows():
    image_ids = str(group["image_ids"]).split("|")
    unknown = set(image_ids) - set(lookup.index)
    assert not unknown, f"Candidate IDs missing from manifest: {unknown}"

    members = lookup.loc[image_ids]
    if members["split"].nunique() <= 1:
        continue

    for image_id, row in members.iterrows():
        review_rows.append({
            "candidate_group": group_id,
            "image_id": image_id,
            "source": row["source"],
            "split": row["split"],
            "image_path": row["image_path"],
        })

review = pd.DataFrame(
    review_rows,
    columns=["candidate_group", "image_id", "source", "split", "image_path"],
)
output = REPORTS / "cross_split_candidates.csv"
review.to_csv(output, index=False)

print("PASS: image IDs are unique.")
print("PASS: source groups do not cross splits.")
print(f"PASS: exactly one test source: {test_sources[0]}")
print("PASS: no exact duplicate content crosses splits.")
print(f"Visual candidate groups needing review: "
      f"{review['candidate_group'].nunique()}")
print(f"Review list saved to: {output}")
print("NOTE: visual candidates are not confirmed duplicates.")