import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
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

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_training"
    / "colour"
    / "swin_t_final_best.pt"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "final_training"
    / "colour"
)


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


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    df = pd.read_csv(CSV_PATH)
    test_df = df[df["split"] == "test"].copy()

    num_classes = int(df["class_id"].nunique())

    print("Classes:", num_classes)
    print("Official test images:", len(test_df))

    transform = transforms.Compose(
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

    test_dataset = ColourDataset(
        test_df,
        transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    print("Checkpoint epoch:", checkpoint["epoch"])
    print(
        "Checkpoint validation Macro F1:",
        checkpoint["val_macro_f1"],
    )

    model = models.swin_t(weights=None)

    model.head = nn.Linear(
        model.head.in_features,
        num_classes,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    targets_all = []
    predictions_all = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device,
                non_blocking=True,
            )

            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available(),
            ):
                outputs = model(images)
                loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct_top1 += (
                predictions == targets
            ).sum().item()

            top5 = outputs.topk(5, dim=1).indices

            correct_top5 += (
                top5 == targets.unsqueeze(1)
            ).any(dim=1).sum().item()

            total += targets.size(0)

            targets_all.extend(
                targets.cpu().tolist()
            )

            predictions_all.extend(
                predictions.cpu().tolist()
            )

            if batch_idx % 50 == 0:
                print(
                    f"Test batch {batch_idx}/{len(test_loader)}"
                )

    test_loss = total_loss / total
    test_accuracy = correct_top1 / total
    test_top5 = correct_top5 / total

    test_macro_f1 = f1_score(
        targets_all,
        predictions_all,
        average="macro",
        zero_division=0,
    )

    print("\n===== OFFICIAL TEST RESULTS =====")
    print(f"Test loss:     {test_loss:.4f}")
    print(f"Test Top-1:    {test_accuracy:.4f}")
    print(f"Test Top-5:    {test_top5:.4f}")
    print(f"Test Macro F1: {test_macro_f1:.4f}")

    results = {
        "architecture": "swin_t",
        "pretrained": True,
        "weighted_cross_entropy_training": True,
        "num_classes": num_classes,
        "best_epoch": checkpoint["epoch"],
        "best_val_macro_f1": checkpoint["val_macro_f1"],
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_top5_accuracy": test_top5,
        "test_macro_f1": test_macro_f1,
        "checkpoint": str(CHECKPOINT_PATH),
    }

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = RESULT_DIR / "swin_t_final_metrics.json"

    with open(output, "w") as f:
        json.dump(results, f, indent=4)

    print("\nSaved:", output)
    print("FINAL COLOUR TEST EVALUATION SUCCESSFUL")


if __name__ == "__main__":
    main()
