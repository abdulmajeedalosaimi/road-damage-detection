from pathlib import Path
import json
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "reports" / "m1" / "label_audit"

reviews = pd.read_csv(AUDIT / "decisions.csv")
sample = pd.read_csv(AUDIT / "sample.csv")

if not reviews["audit_id"].is_unique:
    raise ValueError("Duplicate audit IDs in decisions.csv.")

if set(reviews["audit_id"]) != set(sample["audit_id"]):
    raise ValueError("Decisions do not match the saved sample.")

reviews["status"] = reviews["status"].fillna("").str.strip().str.lower()

allowed = {"correct", "error", "needs_review", "needs_closer_review"}
invalid = reviews.loc[~reviews["status"].isin(allowed)]

if not invalid.empty:
    raise ValueError(
        "Finish or correct these review entries:\n"
        + invalid[["audit_id", "status"]].to_string(index=False)
    )

n = len(reviews)
if n == 0:
    raise ValueError("No review entries found.")

correct = int((reviews["status"] == "correct").sum())
errors = int((reviews["status"] == "error").sum())
uncertain = n - correct - errors
potential_errors = errors + uncertain


def binomial_cdf(k, total, probability):
    if probability <= 0:
        return 1.0
    if probability >= 1:
        return 1.0 if k == total else 0.0

    return sum(
        math.comb(total, j)
        * probability ** j
        * (1 - probability) ** (total - j)
        for j in range(k + 1)
    )


def upper_bound(k, total, alpha=0.05):
    """One-sided exact Clopper-Pearson binomial upper bound."""
    if k == total:
        return 1.0

    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2
        if binomial_cdf(k, total, middle) > alpha:
            low = middle
        else:
            high = middle
    return (low + high) / 2


upper = upper_bound(potential_errors, n)

summary = {
    "sampled_images": n,
    "correct": correct,
    "observed_errors": errors,
    "unresolved": uncertain,
    "observed_error_fraction": errors / n,
    "conservative_error_fraction": potential_errors / n,
    "one_sided_95_percent_upper_bound": upper,
    "method": "Exact Clopper-Pearson binomial upper bound",
    "unit": "Image with at least one target annotation error",
    "unresolved_handling": "Counted as errors for conservative bound",
}

(AUDIT / "summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

report = f"""# M1 label-audit summary

- Images reviewed: {n}
- Correct: {correct}
- Observed errors: {errors}
- Unresolved: {uncertain}
- Observed error fraction: {errors / n:.2%}
- Conservative error fraction: {potential_errors / n:.2%}
- One-sided 95% upper confidence bound: {upper:.2%}

## Interpretation

The unit is an image with at least one target annotation error,
not an individual bounding box. Unresolved images are counted as
errors only for the conservative calculation; their recorded
review statuses remain unchanged.

The bound uses a binomial model for the random image sample.
It depends on the accuracy and consistency of the visual review.
It does not establish that unreviewed annotations are correct.

Image 044's original XML coordinates match the processed coordinates.
The incorrect D40 annotation was excluded from classification crops,
while its original audit error record was preserved.
"""

(AUDIT / "summary.md").write_text(report, encoding="utf-8")

print(report)
print(f"Saved summary.json and summary.md in: {AUDIT}")