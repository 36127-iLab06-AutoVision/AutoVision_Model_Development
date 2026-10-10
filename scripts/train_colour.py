import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    PROJECT_ROOT
    / "reports"
    / "final_dataset_preparation"
    / "final_colour_split.csv"
)

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "compcars_sv"
    / "extracted"
    / "sv_data"
)

MODEL_DIR = PROJECT_ROOT / "models" / "final_training" / "colour"
RESULT_DIR = PROJECT_ROOT / "reports" / "final_training" / "colour"


class ColourDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = IMAGE_ROOT / row["relative_path"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, int(row["class_id"])


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    targets_all = []
    predictions_all = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available(),
            ):
                outputs = model(images)
                loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct_top1 += (predictions == targets).sum().item()

            top5 = outputs.topk(5, dim=1).indices
            correct_top5 += (
                top5 == targets.unsqueeze(1)
            ).any(dim=1).sum().item()

            total += targets.size(0)

            targets_all.extend(targets.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())

    return {
        "loss": total_loss / total,
        "accuracy": correct_top1 / total,
        "top5_accuracy": correct_top5 / total,
        "macro_f1": f1_score(
            targets_all,
            predictions_all,
            average="macro",
            zero_division=0,
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--smoke-test", action="store_true")

    args = parser.parse_args()

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU memory GB:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / 1024**3,
                2,
            ),
        )

    df = pd.read_csv(CSV_PATH)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    num_classes = int(df["class_id"].nunique())

    print("Classes:", num_classes)
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Official test:", len(test_df))

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.80, 1.0),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.10,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = ColourDataset(train_df, train_transform)
    val_dataset = ColourDataset(val_df, eval_transform)
    test_dataset = ColourDataset(test_df, eval_transform)

    if args.smoke_test:
        train_dataset = Subset(
            train_dataset,
            range(min(1024, len(train_dataset))),
        )
        val_dataset = Subset(
            val_dataset,
            range(min(256, len(val_dataset))),
        )

        print("SMOKE TEST MODE")
        print("Smoke train:", len(train_dataset))
        print("Smoke val:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    counts = (
        train_df["class_id"]
        .value_counts()
        .reindex(range(num_classes))
    )

    class_weights = (
        len(train_df)
        / (num_classes * counts)
    ).values

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    print("Class weights:")
    print(class_weights.cpu().tolist())

    print("Loading ImageNet pretrained Swin-T...")

    model = models.swin_t(
        weights=models.Swin_T_Weights.DEFAULT
    )

    model.head = nn.Linear(
        model.head.in_features,
        num_classes,
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available(),
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = MODEL_DIR / (
        "swin_t_smoke_best.pt"
        if args.smoke_test
        else "swin_t_final_best.pt"
    )

    history = []

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\n===== EPOCH {epoch}/{args.epochs} =====")

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available(),
            ):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

            if batch_idx % 100 == 0:
                print(
                    f"Batch {batch_idx:4d}/{len(train_loader)} "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss = running_loss / total
        train_accuracy = correct / total

        val = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch} results")
        print(f"Learning rate:       {current_lr:.8f}")
        print(f"Train loss:          {train_loss:.4f}")
        print(f"Train accuracy:      {train_accuracy:.4f}")
        print(f"Validation loss:     {val['loss']:.4f}")
        print(f"Validation Top-1:    {val['accuracy']:.4f}")
        print(f"Validation Top-5:    {val['top5_accuracy']:.4f}")
        print(f"Validation Macro F1: {val['macro_f1']:.4f}")

        history.append(
            {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val["loss"],
                "val_accuracy": val["accuracy"],
                "val_top5_accuracy": val["top5_accuracy"],
                "val_macro_f1": val["macro_f1"],
            }
        )

        if val["macro_f1"] > best_val_f1:
            best_val_f1 = val["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val["accuracy"],
                    "val_top5_accuracy": val["top5_accuracy"],
                    "val_macro_f1": val["macro_f1"],
                    "num_classes": num_classes,
                    "architecture": "swin_t",
                    "image_size": 224,
                    "class_weights": class_weights.cpu(),
                },
                checkpoint_path,
            )

            print(
                "\nSaved new best checkpoint:",
                checkpoint_path,
            )

        else:
            epochs_without_improvement += 1

            print(
                "No Macro F1 improvement:",
                f"{epochs_without_improvement}/{args.patience}",
            )

        pd.DataFrame(history).to_csv(
            RESULT_DIR / (
                "swin_t_smoke_history.csv"
                if args.smoke_test
                else "swin_t_training_history.csv"
            ),
            index=False,
        )

        scheduler.step()

        if epochs_without_improvement >= args.patience:
            print("\nEarly stopping triggered.")
            break

    print("\n===== TRAINING COMPLETE =====")
    print("Best epoch:", best_epoch)
    print("Best validation Macro F1:", best_val_f1)

    if args.smoke_test:
        print("SMOKE TEST FINISHED SUCCESSFULLY")
        return

    print("\n===== OFFICIAL TEST EVALUATION =====")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print(f"Test loss:     {test['loss']:.4f}")
    print(f"Test Top-1:    {test['accuracy']:.4f}")
    print(f"Test Top-5:    {test['top5_accuracy']:.4f}")
    print(f"Test Macro F1: {test['macro_f1']:.4f}")

    results = {
        "architecture": "swin_t",
        "pretrained": True,
        "weighted_cross_entropy": True,
        "num_classes": num_classes,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_loss": test["loss"],
        "test_accuracy": test["accuracy"],
        "test_top5_accuracy": test["top5_accuracy"],
        "test_macro_f1": test["macro_f1"],
        "checkpoint": str(checkpoint_path),
    }

    with open(
        RESULT_DIR / "swin_t_final_metrics.json",
        "w",
    ) as f:
        json.dump(results, f, indent=4)

    print("\nFINAL COLOUR TRAINING FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
