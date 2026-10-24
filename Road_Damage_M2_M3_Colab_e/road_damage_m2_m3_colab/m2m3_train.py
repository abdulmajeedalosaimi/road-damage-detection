"""Colab training and final evaluation. No test inference is performed by train_experiment."""
import copy
import gc
import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from m2m3_data import CLASSES, require, save_json, sha256
from m2m3_models import STAGES, build_model, training_mode, seed_everything, seed_worker, image_transform, atomic_checkpoint
from m2m3_metrics import classification_metrics, flat_metrics, plot_history, save_evaluation

DEFAULT_CONFIG = {
    "seed": 42, "image_size": 224, "batch_size": 32, "num_workers": 2, "epochs": 15,
    "scratch_lr": 0.001, "transfer_lr": 0.0003, "weight_decay": 0.0001,
    "wandb_project": "rdd2022-road-damage", "wandb_entity": None, "wandb_mode": "online",
    "group": "m2m3-baseline-seed42", "bootstrap_draws": 2000,
    "augmentation": "resize224; horizontal_flip_p0.5; rotation_10deg; ImageNet_normalize",
    "checkpoint_selection": "maximum validation macro F1; earliest epoch on ties",
    "pretrained_weights": "ResNet18_Weights.IMAGENET1K_V1",
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_hashes():
    root = Path(__file__).parent
    return {p.name: sha256(p) for p in sorted(root.glob("m2m3_*.py"))}


def environment():
    versions = {}
    for name in ["torch", "torchvision", "numpy", "pandas", "Pillow", "matplotlib", "wandb"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return {"python": sys.version, "platform": platform.platform(), "packages": versions,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


class CropDataset(Dataset):
    def __init__(self, rows, crop_root, training, size=224):
        self.rows = rows.reset_index(drop=True)
        self.paths = [Path(crop_root) / p for p in self.rows.relative_path]
        self.labels = self.rows["class"].map({name: i for i, name in enumerate(CLASSES)}).to_numpy()
        self.transform = image_transform(training, size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(self.labels[index])


def make_loader(rows, crop_root, cfg, training=False, generator=None):
    return DataLoader(CropDataset(rows, crop_root, training, cfg["image_size"]),
                      batch_size=cfg["batch_size"], shuffle=training,
                      num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=generator, persistent_workers=False)


def evaluate(model, loader, device):
    model.eval()
    labels, probabilities, loss_sum = [], [], 0.0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            labels.append(y.cpu().numpy())
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
    return np.concatenate(labels), np.concatenate(probabilities), loss_sum / len(loader.dataset)


class Tracker:
    """Disable only for local smoke tests. Final submission requires online W&B runs."""
    def __init__(self, cfg, stage, directory, kind="train", run_id=None):
        self.run = None
        self.wandb = None
        if cfg["wandb_mode"] != "disabled":
            import wandb
            self.wandb = wandb
            self.run = wandb.init(project=cfg["wandb_project"], entity=cfg["wandb_entity"],
                                  mode=cfg["wandb_mode"], group=cfg["group"], job_type=kind,
                                  name=f"{stage}-{kind}-{cfg['seed']}", config={**cfg, "stage": stage},
                                  id=run_id, resume="allow" if run_id and cfg["wandb_mode"] == "online" else None,
                                  dir=str(directory))
            if kind == "train":
                self.run.define_metric("epoch")
                self.run.define_metric("train/*", step_metric="epoch")
                self.run.define_metric("val/*", step_metric="epoch")

    def log(self, values):
        if self.run:
            self.run.log(values)

    def summary(self, values):
        if self.run:
            self.run.summary.update(values)

    def reports(self, directory, prefix):
        if not self.run:
            return
        directory = Path(directory)
        self.log({f"{prefix}/confusion_matrix": self.wandb.Image(str(directory / "confusion_matrix.png")),
                  f"{prefix}/reliability": self.wandb.Image(str(directory / "reliability.png")),
                  f"{prefix}/error_gallery": self.wandb.Image(str(directory / "error_gallery.png")),
                  f"{prefix}/per_class": self.wandb.Table(dataframe=pd.read_csv(directory / "per_class_metrics.csv")),
                  f"{prefix}/by_country": self.wandb.Table(dataframe=pd.read_csv(directory / "country_metrics.csv"))})
        # Metrics, prediction probabilities and review tables only; no source-image dataset upload.
        artifact = self.wandb.Artifact(f"{self.run.id}-{prefix}-reports", type="evaluation")
        for p in directory.iterdir():
            if p.is_file():
                artifact.add_file(str(p), name=p.name)
        self.run.log_artifact(artifact)

    def finish(self, failed=False):
        if self.run:
            self.run.finish(exit_code=1 if failed else 0)


def save_rng(generator):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def move_optimizer(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def train_experiment(stage, crops, crop_root, output_root, config, manifest_report, crop_archive_sha256):
    require(stage in STAGES, "Unknown stage")
    cfg = copy.deepcopy(config)
    require(cfg["epochs"] >= 1 and cfg["batch_size"] >= 1 and cfg["num_workers"] >= 0, "Invalid training configuration")
    require(cfg["image_size"] == 224 and cfg["augmentation"] == DEFAULT_CONFIG["augmentation"], "Update transform code and protocol together")
    directory = Path(output_root) / stage
    directory.mkdir(parents=True, exist_ok=True)
    spec = {"config": cfg, "stage": stage, "classes": CLASSES,
            "manifest_sha256": manifest_report["manifest_sha256"], "archive_sha256": crop_archive_sha256,
            "code_sha256": source_hashes()}
    spec_hash = fingerprint(spec)
    spec_path = directory / "experiment.json"
    if spec_path.exists():
        require(fingerprint(json.loads(spec_path.read_text(encoding="utf-8"))) == spec_hash,
                "This run folder contains another configuration/data/code version. Use a new output folder.")
    else:
        save_json(spec_path, spec)
    if (directory / "complete.json").exists():
        print(f"{stage}: training already complete; using saved validation results.")
        return json.loads((directory / "val/metrics.json").read_text())
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = crops[crops.split.eq("train")].copy()
    val_rows = crops[crops.split.eq("val")].copy()
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader = make_loader(train_rows, crop_root, cfg, True, generator)
    val_loader = make_loader(val_rows, crop_root, cfg)
    last_path = directory / "last_checkpoint.pt"
    model = build_model(stage, pretrained=not last_path.exists()).to(device)
    lr = cfg["scratch_lr"] if stage == "scratch" else cfg["transfer_lr"]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr,
                                 weight_decay=cfg["weight_decay"])
    history, best_f1, best_epoch, start_epoch = [], -1.0, 0, 1
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        require(checkpoint["spec_hash"] == spec_hash, "Checkpoint/config mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer(optimizer, device)
        history, best_f1 = checkpoint["history"], checkpoint["best_f1"]
        best_epoch, start_epoch = checkpoint["best_epoch"], checkpoint["epoch"] + 1
        restore_rng(checkpoint["rng"], generator)
        print(f"{stage}: resuming after epoch {start_epoch - 1}.")
    meta_path = directory / "tracking.json"
    run_id = json.loads(meta_path.read_text())["run_id"] if meta_path.exists() else uuid.uuid4().hex[:12]
    save_json(meta_path, {"run_id": run_id, "mode": cfg["wandb_mode"]})
    env = environment()
    save_json(directory / "environment.json", env)
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (directory / "requirements.lock.txt").write_text(frozen.stdout, encoding="utf-8")
    for name in spec["code_sha256"]:
        shutil.copyfile(Path(__file__).parent / name, directory / name)
    tracker = Tracker({**cfg, "manifest_sha256": spec["manifest_sha256"], "archive_sha256": crop_archive_sha256,
                       "code_sha256": spec["code_sha256"], "environment": env}, stage, directory, run_id=run_id)
    save_json(meta_path, {"run_id": run_id, "mode": cfg["wandb_mode"],
                         "url": tracker.run.url if tracker.run else None})
    failed = True
    try:
        tracker.summary({"parameters_total": sum(p.numel() for p in model.parameters()),
                         "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)})
        for epoch in range(start_epoch, cfg["epochs"] + 1):
            tick = time.monotonic()
            training_mode(model, stage)
            loss_sum, ys, probs = 0.0, [], []
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
                require(torch.isfinite(loss).item(), "Non-finite training loss")
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(y)
                ys.append(y.detach().cpu().numpy())
                probs.append(logits.detach().softmax(dim=1).cpu().numpy())
            train_m, _ = classification_metrics(np.concatenate(ys), np.concatenate(probs))
            y_val, p_val, val_loss = evaluate(model, val_loader, device)
            val_m, _ = classification_metrics(y_val, p_val)
            row = {"epoch": epoch, "train/loss": loss_sum / len(train_rows), "val/loss": val_loss,
                   "train/accuracy": train_m["accuracy"], "train/macro_f1": train_m["macro_f1"],
                   **flat_metrics(val_m, "val"), "epoch_seconds": time.monotonic() - tick}
            history.append(row)
            if val_m["macro_f1"] > best_f1:
                best_f1, best_epoch = val_m["macro_f1"], epoch
                atomic_checkpoint(directory / "best_checkpoint.pt",
                                  {"model": model.state_dict(), "classes": CLASSES, "stage": stage,
                                   "epoch": epoch, "spec_hash": spec_hash})
            atomic_checkpoint(last_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                          "epoch": epoch, "best_epoch": best_epoch, "best_f1": best_f1,
                                          "history": history, "rng": save_rng(generator), "spec_hash": spec_hash})
            pd.DataFrame(history).to_csv(directory / "learning_curves.csv", index=False)
            tracker.log(row)
            print(f"{stage} {epoch}/{cfg['epochs']} | loss {row['train/loss']:.4f} | val F1 {val_m['macro_f1']:.4f}", flush=True)
        checkpoint = torch.load(directory / "best_checkpoint.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        y_val, p_val, _ = evaluate(model, val_loader, device)
        val_metrics = save_evaluation(val_rows, y_val, p_val, crop_root, directory / "val", "val",
                                      cfg["seed"], cfg["bootstrap_draws"])
        plot_history(history, directory / "learning_curves.png")
        tracker.summary({"best_epoch": best_epoch, "selection": "validation macro F1",
                         **flat_metrics(val_metrics, "best_val")})
        tracker.reports(directory / "val", "best_val")
        if tracker.run:
            tracker.log({"learning_curves": tracker.wandb.Image(str(directory / "learning_curves.png"))})
        failed = False
    finally:
        tracker.finish(failed)
        del model, optimizer, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    save_json(directory / "complete.json", {"spec_hash": spec_hash, "best_epoch": best_epoch,
                                           "best_val_macro_f1": best_f1})
    return val_metrics


def final_evaluation(crops, crop_root, output_root, config, manifest_report, crop_archive_sha256, confirm=False):
    require(confirm is True, "Finish validation review and freeze the three experiments before final test.")
    root = Path(output_root)
    specs = {}
    for stage in STAGES:
        directory = root / stage
        require((directory / "complete.json").exists(), f"{stage} training is incomplete")
        spec = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
        require(spec["config"] == config and spec["code_sha256"] == source_hashes(), "Code/config changed after training")
        require(spec["manifest_sha256"] == manifest_report["manifest_sha256"] and spec["archive_sha256"] == crop_archive_sha256,
                "Data changed since training")
        val_m = json.loads((directory / "val/metrics.json").read_text())
        specs[stage] = {"checkpoint_sha256": sha256(directory / "best_checkpoint.pt"),
                        "spec_hash": fingerprint(spec), "val_macro_f1": val_m["macro_f1"]}
    selected = max(STAGES, key=lambda stage: specs[stage]["val_macro_f1"])
    plan = {"experiments": specs, "selected_by_validation": selected, "selection_metric": "macro_f1"}
    lock = root / "final_test_plan.json"
    if lock.exists():
        require(json.loads(lock.read_text()) == plan, "Final test plan already locked to different checkpoints")
    else:
        save_json(lock, plan)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = crops[crops.split.eq("test")].copy()
    comparison = []
    for stage in STAGES:
        directory = root / stage
        out = directory / "test"
        if not (out / "complete.json").exists():
            model = build_model(stage, pretrained=False).to(device)
            checkpoint = torch.load(directory / "best_checkpoint.pt", map_location="cpu", weights_only=True)
            require(checkpoint["spec_hash"] == specs[stage]["spec_hash"], "Test checkpoint does not match locked experiment")
            model.load_state_dict(checkpoint["model"])
            loader = make_loader(rows, crop_root, config)
            y, p, _ = evaluate(model, loader, device)
            m = save_evaluation(rows, y, p, crop_root, out, "test", config["seed"], config["bootstrap_draws"])
            tracker = Tracker({**config, "final_test_plan": plan}, stage, directory, kind="final_test")
            try:
                tracker.summary({**flat_metrics(m, "test"), "best_epoch": checkpoint["epoch"],
                                 "selected_by_validation": stage == selected})
                tracker.reports(out, "test")
                url = tracker.run.url if tracker.run else None
            except BaseException:
                tracker.finish(True)
                raise
            tracker.finish()
            save_json(out / "complete.json", {"checkpoint_sha256": specs[stage]["checkpoint_sha256"], "wandb_url": url})
            del model, loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        require(json.loads((out / "complete.json").read_text())["checkpoint_sha256"] == specs[stage]["checkpoint_sha256"],
                "Saved test results came from a different checkpoint")
        m = json.loads((out / "metrics.json").read_text())
        comparison.append({"model": stage, "selected_by_validation": stage == selected,
                           "val_macro_f1": specs[stage]["val_macro_f1"],
                           **{f"test_{k}": v for k, v in m.items() if isinstance(v, (int, float))},
                           "test_accuracy_ci95_low": m["accuracy_ci95"]["lower"],
                           "test_accuracy_ci95_high": m["accuracy_ci95"]["upper"]})
    table = pd.DataFrame(comparison)
    table.to_csv(root / "comparison.csv", index=False)
    if config["wandb_mode"] != "disabled" and not (root / "comparison_logged.json").exists():
        tracker = Tracker(config, "comparison", root, kind="comparison")
        try:
            tracker.log({"model_comparison": tracker.wandb.Table(dataframe=table)})
        except BaseException:
            tracker.finish(True)
            raise
        tracker.finish()
        save_json(root / "comparison_logged.json", {"plan_hash": fingerprint(plan)})
    return table

