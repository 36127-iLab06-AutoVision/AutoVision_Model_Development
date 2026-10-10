import gc
import json
import math
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "reports" / "final_training" / "calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_ACCEPTED_ACCURACY = 0.95
BATCH_SIZE = 128
WORKERS = 4


TASKS = {
    "make_model": {
        "csv": ROOT / "reports/final_dataset_preparation/final_make_model_split.csv",
        "root": ROOT / "data/compcars_kaggle/extracted",
        "checkpoint": ROOT / "models/final_training/make_model/resnet50_final_best.pt",
        "architecture": "resnet50",
    },
    "body_type": {
        "csv": ROOT / "reports/final_dataset_preparation/final_body_type_split.csv",
        "root": ROOT / "data/compcars_kaggle/extracted",
        "checkpoint": ROOT / "models/final_training/body_type/convnext_tiny_final_best.pt",
        "architecture": "convnext_tiny",
    },
    "colour": {
        "csv": ROOT / "reports/final_dataset_preparation/final_colour_split.csv",
        "root": ROOT / "data/compcars_sv/extracted/sv_data",
        "checkpoint": ROOT / "models/final_training/colour/swin_t_final_best.pt",
        "architecture": "swin_t",
    },
}


EVAL_TRANSFORM = transforms.Compose(
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


class ClassificationDataset(Dataset):
    def __init__(self, df, image_root):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.image_root / row["relative_path"]

        with Image.open(path) as image:
            image = image.convert("RGB")
            image = EVAL_TRANSFORM(image)

        return image, int(row["class_id"])


def build_model(architecture, num_classes):
    if architecture == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )

    elif architecture == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(
            model.classifier[2].in_features,
            num_classes,
        )

    elif architecture == "swin_t":
        model = models.swin_t(weights=None)
        model.head = nn.Linear(
            model.head.in_features,
            num_classes,
        )

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model


def collect_logits(model, df, image_root, device):
    dataset = ClassificationDataset(df, image_root)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
        persistent_workers=False,
    )

    all_logits = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)

            with torch.amp.autocast(
                "cuda",
                enabled=device.type == "cuda",
            ):
                logits = model(images)

            all_logits.append(logits.float().cpu())
            all_labels.append(labels.long().cpu())

            if batch_idx % 50 == 0:
                print(
                    f"  batch {batch_idx}/{len(loader)}"
                )

    return (
        torch.cat(all_logits),
        torch.cat(all_labels),
    )


def fit_temperature(logits, labels):
    logits = logits.double()
    labels = labels.long()

    log_temperature = nn.Parameter(
        torch.zeros(1, dtype=torch.double)
    )

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()

        temperature = torch.exp(log_temperature)

        loss = criterion(
            logits / temperature,
            labels,
        )

        loss.backward()
        return loss

    optimizer.step(closure)

    temperature = float(
        torch.exp(log_temperature).item()
    )

    return max(0.05, min(temperature, 10.0))


def calibration_metrics(logits, labels, temperature=1.0):
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=1)

    confidence, predictions = probs.max(dim=1)

    accuracy = (
        predictions == labels
    ).float().mean().item()

    nll = nn.functional.cross_entropy(
        scaled_logits,
        labels,
    ).item()

    # Multiclass Brier score:
    # sum(p^2) - 2*p(correct class) + 1
    true_probs = probs[
        torch.arange(len(labels)),
        labels,
    ]

    brier = (
        probs.pow(2).sum(dim=1)
        - 2 * true_probs
        + 1
    ).mean().item()

    # Expected Calibration Error
    ece = 0.0

    bin_boundaries = torch.linspace(
        0,
        1,
        16,
    )

    correct = predictions.eq(labels)

    for lower, upper in zip(
        bin_boundaries[:-1],
        bin_boundaries[1:],
    ):
        mask = (
            confidence.gt(lower)
            & confidence.le(upper)
        )

        if mask.any():
            bin_accuracy = (
                correct[mask]
                .float()
                .mean()
                .item()
            )

            bin_confidence = (
                confidence[mask]
                .mean()
                .item()
            )

            weight = mask.float().mean().item()

            ece += weight * abs(
                bin_accuracy - bin_confidence
            )

    return {
        "accuracy": accuracy,
        "nll": nll,
        "brier": brier,
        "ece": ece,
    }


def threshold_table(logits, labels, temperature):
    probs = torch.softmax(
        logits / temperature,
        dim=1,
    )

    confidence, predictions = probs.max(dim=1)
    correct = predictions.eq(labels)

    rows = []

    for i in range(50, 100):
        threshold = i / 100

        accepted = confidence >= threshold
        accepted_count = int(accepted.sum())

        if accepted_count == 0:
            coverage = 0.0
            accepted_accuracy = None
        else:
            coverage = accepted.float().mean().item()

            accepted_accuracy = (
                correct[accepted]
                .float()
                .mean()
                .item()
            )

        rows.append(
            {
                "threshold": threshold,
                "accepted": accepted_count,
                "total": len(labels),
                "coverage": coverage,
                "accepted_accuracy": accepted_accuracy,
            }
        )

    return pd.DataFrame(rows)


def choose_threshold(table):
    valid = table[
        table["accepted_accuracy"].notna()
        & (
            table["accepted_accuracy"]
            >= TARGET_ACCEPTED_ACCURACY
        )
    ].copy()

    if len(valid):
        # Highest coverage while satisfying target accuracy
        chosen = valid.sort_values(
            ["coverage", "threshold"],
            ascending=[False, True],
        ).iloc[0]

        reason = (
            f"validation accepted accuracy >= "
            f"{TARGET_ACCEPTED_ACCURACY:.0%}"
        )

    else:
        available = table[
            table["accepted_accuracy"].notna()
        ].copy()

        chosen = available.sort_values(
            ["accepted_accuracy", "coverage"],
            ascending=[False, False],
        ).iloc[0]

        reason = (
            "95% target unavailable; selected "
            "highest validation accepted accuracy"
        )

    return chosen, reason


def threshold_result(
    logits,
    labels,
    temperature,
    threshold,
):
    probs = torch.softmax(
        logits / temperature,
        dim=1,
    )

    confidence, predictions = probs.max(dim=1)

    accepted = confidence >= threshold
    accepted_count = int(accepted.sum())

    if accepted_count == 0:
        accepted_accuracy = None
    else:
        accepted_accuracy = (
            predictions[accepted]
            .eq(labels[accepted])
            .float()
            .mean()
            .item()
        )

    return {
        "threshold": threshold,
        "accepted": accepted_count,
        "total": len(labels),
        "coverage": accepted.float().mean().item(),
        "accepted_accuracy": accepted_accuracy,
    }


def run_task(task_name, config, device):
    print("\n" + "=" * 70)
    print("TASK:", task_name.upper())
    print("=" * 70)

    df = pd.read_csv(config["csv"])

    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    num_classes = int(df["class_id"].nunique())

    print("Architecture:", config["architecture"])
    print("Classes:", num_classes)
    print("Validation:", len(val_df))
    print("Test:", len(test_df))
    print("Checkpoint:", config["checkpoint"])

    model = build_model(
        config["architecture"],
        num_classes,
    )

    checkpoint = torch.load(
        config["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    model.load_state_dict(state_dict)
    model = model.to(device)

    print("\nCollecting validation logits...")
    val_logits, val_labels = collect_logits(
        model,
        val_df,
        config["root"],
        device,
    )

    print("\nCollecting test logits...")
    test_logits, test_labels = collect_logits(
        model,
        test_df,
        config["root"],
        device,
    )

    print("\nFitting temperature on validation split...")

    fitted_temperature = fit_temperature(
        val_logits,
        val_labels,
    )

    print(
        "Fitted temperature:",
        fitted_temperature,
    )

    val_raw = calibration_metrics(
        val_logits,
        val_labels,
        1.0,
    )

    val_cal = calibration_metrics(
        val_logits,
        val_labels,
        fitted_temperature,
    )

    # Deployment calibration decision is made from validation only.
    if val_cal["nll"] < val_raw["nll"] - 1e-6:
        deployment_temperature = fitted_temperature
        calibration_used = True
    else:
        deployment_temperature = 1.0
        calibration_used = False

    print(
        "Deployment temperature:",
        deployment_temperature,
    )

    print(
        "Calibration retained:",
        calibration_used,
    )

    test_raw = calibration_metrics(
        test_logits,
        test_labels,
        1.0,
    )

    test_cal = calibration_metrics(
        test_logits,
        test_labels,
        deployment_temperature,
    )

    thresholds = threshold_table(
        val_logits,
        val_labels,
        deployment_temperature,
    )

    threshold_file = (
        OUTPUT_DIR
        / f"{task_name}_threshold_analysis.csv"
    )

    thresholds.to_csv(
        threshold_file,
        index=False,
    )

    chosen, threshold_reason = choose_threshold(
        thresholds
    )

    chosen_threshold = float(
        chosen["threshold"]
    )

    val_threshold = threshold_result(
        val_logits,
        val_labels,
        deployment_temperature,
        chosen_threshold,
    )

    test_threshold = threshold_result(
        test_logits,
        test_labels,
        deployment_temperature,
        chosen_threshold,
    )

    result = {
        "architecture": config["architecture"],
        "num_classes": num_classes,
        "validation_images": len(val_df),
        "test_images": len(test_df),

        "fitted_temperature": fitted_temperature,
        "deployment_temperature": deployment_temperature,
        "calibration_used": calibration_used,

        "validation_raw": val_raw,
        "validation_calibrated": val_cal,

        "test_raw": test_raw,
        "test_calibrated": test_cal,

        "confidence_threshold": chosen_threshold,
        "threshold_selection_reason": threshold_reason,

        "validation_threshold_result": val_threshold,
        "test_threshold_result": test_threshold,

        "checkpoint": str(config["checkpoint"]),
    }

    print("\n----- CALIBRATION -----")
    print(
        f"Temperature: {deployment_temperature:.4f}"
    )

    print(
        f"Test ECE: "
        f"{test_raw['ece']:.4f} -> "
        f"{test_cal['ece']:.4f}"
    )

    print(
        f"Test NLL: "
        f"{test_raw['nll']:.4f} -> "
        f"{test_cal['nll']:.4f}"
    )

    print(
        f"Test Brier: "
        f"{test_raw['brier']:.4f} -> "
        f"{test_cal['brier']:.4f}"
    )

    print("\n----- CONFIDENCE THRESHOLD -----")
    print(
        f"Threshold: {chosen_threshold:.2f}"
    )

    print(
        f"Validation coverage: "
        f"{val_threshold['coverage']:.2%}"
    )

    print(
        f"Validation accepted accuracy: "
        f"{val_threshold['accepted_accuracy']:.2%}"
    )

    print(
        f"Test coverage: "
        f"{test_threshold['coverage']:.2%}"
    )

    if test_threshold["accepted_accuracy"] is not None:
        print(
            f"Test accepted accuracy: "
            f"{test_threshold['accepted_accuracy']:.2%}"
        )

    del model
    del checkpoint
    del val_logits
    del test_logits

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    all_results = {}

    for task_name, config in TASKS.items():
        all_results[task_name] = run_task(
            task_name,
            config,
            device,
        )

    deployment = {
        task: {
            "temperature": result[
                "deployment_temperature"
            ],
            "confidence_threshold": result[
                "confidence_threshold"
            ],
        }
        for task, result in all_results.items()
    }

    with open(
        OUTPUT_DIR / "final_calibration_results.json",
        "w",
    ) as f:
        json.dump(
            all_results,
            f,
            indent=4,
        )

    with open(
        OUTPUT_DIR / "deployment_confidence_settings.json",
        "w",
    ) as f:
        json.dump(
            deployment,
            f,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("FINAL DEPLOYMENT SETTINGS")
    print("=" * 70)

    print(
        json.dumps(
            deployment,
            indent=4,
        )
    )

    print(
        "\nFINAL CLASSIFIER CALIBRATION "
        "FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
