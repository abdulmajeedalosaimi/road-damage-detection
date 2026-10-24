"""Classification metrics, calibration, error evidence and a provisional improvement plan."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from m2m3_data import CLASSES, require, save_json


def classification_metrics(labels, probabilities, bins=10):
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    require(len(y) > 0 and p.shape == (len(y), len(CLASSES)), "Invalid prediction shape")
    require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), "Invalid probabilities")
    require(np.allclose(p.sum(axis=1), 1, atol=1e-5), "Probabilities must sum to one")
    require(((y >= 0) & (y < len(CLASSES))).all(), "Invalid label IDs")
    pred = p.argmax(axis=1)
    matrix = np.bincount(y * len(CLASSES) + pred, minlength=len(CLASSES)**2).reshape(len(CLASSES), len(CLASSES))
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    precision = np.divide(matrix.diagonal(), predicted, out=np.zeros(4), where=predicted > 0)
    recall = np.divide(matrix.diagonal(), support, out=np.zeros(4), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(4), where=(precision + recall) > 0)
    confidence, correct = p.max(axis=1), pred == y
    bin_ids = np.minimum((confidence * bins).astype(int), bins - 1)
    calibration, ece = [], 0.0
    for index in range(bins):
        mask = bin_ids == index
        count = int(mask.sum())
        acc = float(correct[mask].mean()) if count else None
        conf = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(acc - conf)
        calibration.append({"bin": index, "lower": index / bins, "upper": (index + 1) / bins,
                            "count": count, "accuracy": acc, "confidence": conf})
    metrics = {
        "accuracy": float(correct.mean()),
        "top2_accuracy": float((np.argsort(-p, axis=1, kind="stable")[:, :2] == y[:, None]).any(axis=1).mean()),
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()),
        "ece": float(ece), "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
        "brier": float(((p - np.eye(4)[y]) ** 2).sum(axis=1).mean()),
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                             "f1": float(f1[i]), "support": int(support[i])} for i, name in enumerate(CLASSES)},
        "calibration_bins": calibration,
    }
    return metrics, matrix


def parent_bootstrap_accuracy(labels, probabilities, parent_ids, draws=2000, seed=42):
    """Percentile CI resampling parent images; all their crops travel together."""
    correct = np.asarray(probabilities).argmax(axis=1) == np.asarray(labels)
    _, group = np.unique(np.asarray(parent_ids), return_inverse=True)
    n = group.max() + 1
    sizes = np.bincount(group)
    successes = np.bincount(group, weights=correct.astype(int))
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for i in range(draws):
        chosen = rng.integers(0, n, size=n)
        estimates[i] = successes[chosen].sum() / sizes[chosen].sum()
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return {"method": "parent-image cluster bootstrap, percentile 95%",
            "lower": float(lo), "upper": float(hi), "draws": draws, "parent_images": int(n),
            "limitation": "Conditional on this held-out country and trained model; not a multi-seed or cross-country CI."}


def prediction_frame(rows, labels, probabilities):
    out = rows[["relative_path", "image_id", "object_index", "country", "source", "split", "class"]].reset_index(drop=True).copy()
    require(np.array_equal(np.asarray(labels), out["class"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()), "Prediction order mismatch")
    pred = probabilities.argmax(axis=1)
    out["predicted"] = [CLASSES[i] for i in pred]
    out["confidence"] = probabilities.max(axis=1)
    out["correct"] = out["class"].eq(out.predicted)
    for i, name in enumerate(CLASSES):
        out[f"prob_{name}"] = probabilities[:, i]
    return out


def plot_confusion(matrix, path, split):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    denominators = matrix.sum(axis=1, keepdims=True)
    rates = np.divide(matrix, denominators, out=np.zeros_like(matrix, dtype=float), where=denominators > 0)
    for ax, values, title, fmt in zip(axes, [matrix, rates], ["Counts", "Recall by true class"], ["d", ".2f"]):
        im = ax.imshow(values, cmap="Blues", vmin=0)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel="Predicted", ylabel="True", title=f"{split}: {title}")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, format(values[i, j], fmt), ha="center", va="center",
                        color="white" if values[i, j] > values.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_calibration(metrics, path, split):
    bins = pd.DataFrame(metrics["calibration_bins"])
    populated = bins[bins["count"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axes[0].plot(populated.confidence, populated.accuracy, "o-", label="Raw softmax")
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy",
                title=f"{split}: ECE = {metrics['ece']:.4f}")
    axes[0].legend()
    axes[1].bar((bins.lower + bins.upper) / 2, bins["count"], width=0.09)
    axes[1].set(xlabel="Confidence", ylabel="Number of crops", title="Confidence distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_errors(frame, crop_root, path, split):
    wrong = frame[~frame.correct].sort_values("confidence", ascending=False).head(16)
    fig, axes = plt.subplots(4, 4, figsize=(11, 11))
    for ax in axes.ravel():
        ax.axis("off")
    if wrong.empty:
        fig.text(0.5, 0.5, "No errors in this evaluated set", ha="center")
    for ax, (_, row) in zip(axes.ravel(), wrong.iterrows()):
        with Image.open(Path(crop_root) / row.relative_path) as im:
            ax.imshow(im)
        ax.set_title(f"{row['class']} -> {row.predicted} ({row.confidence:.2f})\n{Path(row.relative_path).stem[-27:]}", fontsize=8)
    fig.suptitle(f"{split}: highest-confidence errors (not a random sample)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history, path):
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, title in zip(axes, ["loss", "accuracy", "macro_f1"], ["Loss", "Accuracy", "Macro F1"]):
        for split in ["train", "val"]:
            key = f"{split}/{metric}"
            if key in frame:
                ax.plot(frame.epoch, frame[key], label=split)
        ax.set(xlabel="Epoch", title=title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def improvement_plan(frame, matrix, path):
    """Quantified validation evidence, with explicit human-review tasks, not invented causes."""
    lines = [
        "# Validation-based data-improvement plan",
        "",
        "Status: provisional until the error-review CSV is completed by a person.",
        "Counts below come from validation predictions. They do not establish why a label was misclassified.",
        "Use training data or newly collected training examples for improvements. Keep validation and Norway test fixed.",
        "",
    ]
    priority = []
    for i, name in enumerate(CLASSES):
        support = int(matrix[i].sum())
        misses = support - int(matrix[i, i])
        recall = matrix[i, i] / support if support else 0
        priority.append((recall, name, support, misses))
    for rank, (recall, name, support, misses) in enumerate(sorted(priority), 1):
        lines.append(f"{rank}. {name}: validation recall {recall:.3f}, {misses}/{support} missed. "
                     "Review original-resolution evidence and matched training examples; if coverage is deficient, "
                     "add verified training examples and compare validation macro F1 and this class's recall.")
    pairs = [(int(matrix[i, j]), CLASSES[i], CLASSES[j]) for i in range(4) for j in range(4) if i != j and matrix[i, j]]
    lines += ["", "## Largest observed confusion pairs", ""]
    for count, truth, pred in sorted(pairs, reverse=True)[:5]:
        lines.append(f"- {truth} -> {pred}: {count} errors. Review both classes for ambiguous labels, narrow crops, lighting and road texture; record the observed cause before selecting a fix.")
    high = int(((~frame.correct) & (frame.confidence >= 0.9)).sum())
    lines += [
        "", f"High-confidence errors (confidence >= 0.90): {high}. Inspect these first, then a random error sample.",
        "", "## Review and follow-up", "",
        "- Complete error_review.csv: observed_cause, evidence_note, proposed_action.",
        "- Count reviewed causes and replace these hypotheses with actions supported by those counts.",
        "- For confirmed training-label errors: record image_id/object_index, obtain team agreement, and version corrected labels.",
        "- For crop/context problems: first inspect original images; any recropping experiment must use the same parent splits and a separately versioned dataset.",
        "- For domain coverage problems: collect licensed training images of the underperforming damage classes under the observed conditions.",
        "- Re-evaluate proposed fixes on validation before the final test. Test errors may inform future work, but do not retune and present the same test as untouched.",
        "",
        "Known M1 audit caveat: row-number-based decisions appear to refer to different image IDs across two audit samples. Resolve by image_id/object_index before claiming a label-error rate or relabeling.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_evaluation(rows, labels, probabilities, crop_root, output, split, seed=42, bootstrap_draws=2000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics, matrix = classification_metrics(labels, probabilities)
    metrics["accuracy_ci95"] = parent_bootstrap_accuracy(labels, probabilities, rows.image_id, bootstrap_draws, seed)
    metrics["calibration_definition"] = "Uncalibrated softmax; top-label ECE with 10 equal-width bins; multiclass Brier=sum squared class errors."
    frame = prediction_frame(rows, labels, probabilities)
    frame.to_csv(output / "predictions.csv", index=False)
    per_class = pd.DataFrame(metrics["per_class"]).T.rename_axis("class").reset_index()
    per_class.to_csv(output / "per_class_metrics.csv", index=False)
    pd.DataFrame(metrics["calibration_bins"]).to_csv(output / "calibration_bins.csv", index=False)
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(output / "confusion_matrix.csv", index_label="true_class")
    countries = []
    for country, part in frame.groupby("country", sort=True):
        y = part["class"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
        p = part[[f"prob_{c}" for c in CLASSES]].to_numpy()
        m, _ = classification_metrics(y, p)
        countries.append({"country": country, "crops": len(part), **{k: v for k, v in m.items() if isinstance(v, (int, float))}})
    pd.DataFrame(countries).to_csv(output / "country_metrics.csv", index=False)
    plot_confusion(matrix, output / "confusion_matrix.png", split)
    plot_calibration(metrics, output / "reliability.png", split)
    plot_errors(frame, crop_root, output / "error_gallery.png", split)
    wrong = frame[~frame.correct]
    top = wrong.sort_values("confidence", ascending=False).head(16).copy()
    top["review_selection"] = "highest_confidence"
    remaining = wrong.drop(index=top.index)
    random_errors = remaining.sample(n=min(32, len(remaining)), random_state=seed).copy()
    random_errors["review_selection"] = "random_remaining_errors"
    review = pd.concat([top, random_errors], ignore_index=True)
    for col in ["observed_cause", "evidence_note", "proposed_action"]:
        review[col] = ""
    # Never overwrite a human's completed review.
    if not (output / "error_review.csv").exists():
        review.to_csv(output / "error_review.csv", index=False)
    if split == "val":
        improvement_plan(frame, matrix, output / "data_improvement_plan.md")
    save_json(output / "metrics.json", metrics)
    return metrics


def flat_metrics(metrics, prefix):
    out = {f"{prefix}/{key}": value for key, value in metrics.items() if isinstance(value, (int, float))}
    for name, values in metrics["per_class"].items():
        out.update({f"{prefix}/{name}/{key}": value for key, value in values.items()})
    if "accuracy_ci95" in metrics:
        out[f"{prefix}/accuracy_ci95_lower"] = metrics["accuracy_ci95"]["lower"]
        out[f"{prefix}/accuracy_ci95_upper"] = metrics["accuracy_ci95"]["upper"]
    return out

