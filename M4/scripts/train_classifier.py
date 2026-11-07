"""M2/M3 training with fixed seeds, W&B tracking, calibration, and error artifacts."""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import wandb

ROOT = Path(__file__).resolve().parents[1]


def local_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class WithPath(Dataset):
    def __init__(self, folder: datasets.ImageFolder): self.folder = folder
    def __len__(self): return len(self.folder)
    def __getitem__(self, index):
        image, label = self.folder[index]
        return image, label, self.folder.samples[index][0]


class SmallCNN(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, classes))
    def forward(self, value): return self.classifier(self.features(value))


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1); correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0, 1, bins + 1); result = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if mask.any(): result += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(result)


def evaluate(model, loader, device, criterion):
    model.eval(); losses, labels, predictions, probabilities, paths = [], [], [], [], []
    with torch.no_grad():
        for images, target, names in loader:
            images, target = images.to(device), target.to(device)
            logits = model(images); losses.append(criterion(logits, target).item() * len(target))
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            probabilities.extend(prob); labels.extend(target.cpu().numpy()); predictions.extend(prob.argmax(axis=1)); paths.extend(names)
    labels, predictions, probabilities = np.array(labels), np.array(predictions), np.array(probabilities)
    return {"loss": sum(losses) / len(loader.dataset), "labels": labels, "predictions": predictions,
            "probabilities": probabilities, "paths": paths}


def metrics(result, class_names):
    labels, predictions, probabilities = result["labels"], result["predictions"], result["probabilities"]
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=range(len(class_names)), zero_division=0)
    values = {"accuracy": float(accuracy_score(labels, predictions)),
              "top2_accuracy": float(np.mean(np.any(np.argsort(probabilities, axis=1)[:, -2:] == labels[:, None], axis=1))),
              "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
              "macro_f1": float(f1.mean()), "expected_calibration_error": expected_calibration_error(probabilities, labels)}
    values["per_class"] = {name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, name in enumerate(class_names)}
    return values


def save_confusion(result, class_names, output: Path):
    matrix = confusion_matrix(result["labels"], result["predictions"], labels=range(len(class_names)))
    figure, axis = plt.subplots(figsize=(6, 5)); image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis); axis.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
        xticklabels=class_names, yticklabels=class_names, xlabel="Predicted", ylabel="True", title="Test confusion matrix")
    for row in range(len(class_names)):
        for column in range(len(class_names)): axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout(); figure.savefig(output, dpi=180); plt.close(figure)


def save_error_gallery(result, class_names, output: Path):
    wrong = np.where(result["labels"] != result["predictions"])[0][:16]
    if not len(wrong): return
    figure, axes = plt.subplots(4, 4, figsize=(10, 10)); axes = axes.ravel()
    for axis, index in zip(axes, wrong):
        image = plt.imread(result["paths"][index]); axis.imshow(image)
        axis.set_title(f"true={class_names[result['labels'][index]]}\npred={class_names[result['predictions'][index]]}", fontsize=8); axis.axis("off")
    for axis in axes[len(wrong):]: axis.axis("off")
    figure.tight_layout(); figure.savefig(output, dpi=180); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", choices=("scratch", "resnet18"), required=True)
    parser.add_argument("--stage", choices=("frozen", "finetune"), default="finetune")
    args = parser.parse_args()
    cfg = yaml.safe_load(local_path(args.config).read_text(encoding="utf-8")); seed_everything(cfg["seed"])
    crop_root = local_path(cfg["paths"]["processed_dir"]) / "classification_crops"
    run_name = f"{args.model}-{args.stage}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = local_path(cfg["paths"]["runs_dir"]) / run_name; run_dir.mkdir(parents=True)
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    size = cfg["classification"]["image_size"]
    train_transform = transforms.Compose([transforms.RandomResizedCrop(size, scale=(0.75, 1.0)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.15, 0.15, 0.1), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    eval_transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_folder = datasets.ImageFolder(crop_root / "train", transform=train_transform)
    val_folder = datasets.ImageFolder(crop_root / "val", transform=eval_transform)
    test_folder = datasets.ImageFolder(crop_root / "test", transform=eval_transform)
    if train_folder.classes != val_folder.classes or train_folder.classes != test_folder.classes:
        raise RuntimeError("Every split must contain the same classes. Inspect class_balance_by_split.csv and revise the country split.")
    classes = train_folder.classes
    loaders = {"train": DataLoader(WithPath(train_folder), batch_size=cfg["classification"]["batch_size"], shuffle=True, num_workers=cfg["classification"]["num_workers"]),
               "val": DataLoader(WithPath(val_folder), batch_size=cfg["classification"]["batch_size"], shuffle=False, num_workers=cfg["classification"]["num_workers"]),
               "test": DataLoader(WithPath(test_folder), batch_size=cfg["classification"]["batch_size"], shuffle=False, num_workers=cfg["classification"]["num_workers"])}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "scratch":
        model = SmallCNN(len(classes)); learning_rate = cfg["classification"]["scratch_learning_rate"]
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1); model.fc = nn.Linear(model.fc.in_features, len(classes)); learning_rate = cfg["classification"]["transfer_learning_rate"]
        if args.stage == "frozen":
            for parameter in model.parameters(): parameter.requires_grad = False
            for parameter in model.fc.parameters(): parameter.requires_grad = True
    model.to(device); criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=cfg["classification"]["weight_decay"])
    run = wandb.init(project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"], mode=cfg["wandb"]["mode"], name=run_name, config={**cfg, "model": args.model, "stage": args.stage, "classes": classes, "device": str(device)})
    best_f1, history = -1.0, []
    for epoch in range(1, cfg["classification"]["epochs"] + 1):
        model.train(); total_loss, total_count = 0.0, 0
        for images, target, _ in loaders["train"]:
            images, target = images.to(device), target.to(device); optimizer.zero_grad()
            loss = criterion(model(images), target); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(target); total_count += len(target)
        validation = evaluate(model, loaders["val"], device, criterion); validation_metrics = metrics(validation, classes)
        row = {"epoch": epoch, "train_loss": total_loss / total_count, "val_loss": validation["loss"], **{f"val_{key}": value for key, value in validation_metrics.items() if key != "per_class"}}
        history.append(row); wandb.log(row)
        if validation_metrics["macro_f1"] > best_f1:
            best_f1 = validation_metrics["macro_f1"]; torch.save({"model": model.state_dict(), "classes": classes, "epoch": epoch}, run_dir / "best_checkpoint.pt")
    pd.DataFrame(history).to_csv(run_dir / "learning_curves.csv", index=False)
    checkpoint = torch.load(run_dir / "best_checkpoint.pt", map_location=device, weights_only=True); model.load_state_dict(checkpoint["model"])
    test = evaluate(model, loaders["test"], device, criterion); test_metrics = metrics(test, classes)
    pd.DataFrame({"path": test["paths"], "true": [classes[x] for x in test["labels"]], "predicted": [classes[x] for x in test["predictions"]], "confidence": test["probabilities"].max(axis=1)}).to_csv(run_dir / "test_predictions.csv", index=False)
    save_confusion(test, classes, run_dir / "confusion_matrix.png"); save_error_gallery(test, classes, run_dir / "error_gallery.png")
    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    wandb.log({f"test_{key}": value for key, value in test_metrics.items() if key != "per_class"})
    wandb.log({"confusion_matrix": wandb.Image(str(run_dir / "confusion_matrix.png"))})
    run.finish(); print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
