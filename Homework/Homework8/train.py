
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from model import HairModel


SEED = 42
BATCH_SIZE = 20
EPOCHS = 10
DEVICE = torch.device("cpu")


def set_seed(seed):
    """Configure deterministic behavior for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transform(augment=False):
    """Build image preprocessing with optional training augmentation."""
    operations = [
        transforms.Resize(
            (200, 200),
            interpolation=InterpolationMode.BILINEAR,
        ),
    ]

    if augment:
        operations.extend([
            transforms.RandomRotation(
                50,
                interpolation=InterpolationMode.NEAREST,
            ),
            transforms.RandomResizedCrop(
                200,
                scale=(0.9, 1.0),
                ratio=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
        ])

    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transforms.Compose(operations)


def create_dataloaders(augment=False):
    """Create training and evaluation data loaders."""
    train_dataset = datasets.ImageFolder(
        "data/train",
        transform=get_transform(augment=augment),
    )

    eval_dataset = datasets.ImageFolder(
        "data/test",
        transform=get_transform(augment=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, eval_loader


def run_epoch(model, loader, criterion, optimizer=None):
    """Calculate sample-weighted loss and accuracy for one epoch."""
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        predictions = (torch.sigmoid(logits) >= 0.5).float()
        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def train_phase(model, optimizer, criterion, augment, filename):
    """Train for ten epochs and save the phase metrics."""
    train_loader, eval_loader = create_dataloaders(augment=augment)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "evaluation_loss": [],
        "evaluation_accuracy": [],
    }

    phase = "Augmented" if augment else "Baseline"
    print(f"\nStarting {phase} training...", flush=True)

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer
        )
        eval_metrics = run_epoch(
            model, eval_loader, criterion
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["evaluation_loss"].append(eval_metrics["loss"])
        history["evaluation_accuracy"].append(eval_metrics["accuracy"])

        print(
            f"{phase} Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_accuracy={train_metrics['accuracy']:.4f} | "
            f"eval_loss={eval_metrics['loss']:.4f} | "
            f"eval_accuracy={eval_metrics['accuracy']:.4f}",
            flush=True,
        )

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    print(f"{phase} results saved to {filename}", flush=True)


def main():
    set_seed(SEED)

    model = HairModel().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.002,
        momentum=0.8,
    )

    # Train the baseline model for the first ten epochs.
    train_phase(
        model,
        optimizer,
        criterion,
        augment=False,
        filename="history_baseline.json",
    )

    # Continue training the same model and optimizer with augmentation.
    train_phase(
        model,
        optimizer,
        criterion,
        augment=True,
        filename="history_augmented.json",
    )

    print("\nAll 20 training epochs completed.", flush=True)


if __name__ == "__main__":
    main()
