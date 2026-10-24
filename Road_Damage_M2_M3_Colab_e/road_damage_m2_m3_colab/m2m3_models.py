"""Model definitions: retain M1 SmallCNN; freeze ResNet buffers as well as weights."""
import random
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms

STAGES = ["scratch", "frozen", "finetune"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class SmallCNN(nn.Module):
    def __init__(self, classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, classes))

    def forward(self, value):
        return self.classifier(self.features(value))


def build_model(stage, pretrained=True):
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    if stage == "scratch":
        return SmallCNN()
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 4)
    if stage == "frozen":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
    return model


def training_mode(model, stage):
    if stage == "frozen":
        model.eval()  # Keep BatchNorm running statistics fixed.
        model.fc.train()
    else:
        model.train()


def image_transform(training, size=224):
    steps = [transforms.Resize((size, size))]
    if training:
        # Mild transforms retain the precomputed crop extent; no RandomResizedCrop.
        steps += [transforms.RandomHorizontalFlip(),
                  transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BILINEAR,
                                            fill=(124, 116, 104))]
    steps += [transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(steps)


def atomic_checkpoint(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)

