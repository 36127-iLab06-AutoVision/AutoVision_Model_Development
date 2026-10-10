import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "models"
    / "final_training"
    / "damage"
    / "yolo11s_negative_aware_final.pt"
)

DATA_YAML = (
    ROOT
    / "reports"
    / "final_training"
    / "damage"
    / "cardd_test.yaml"
)

CLEAN_DIR = (
    ROOT
    / "data"
    / "processed"
    / "cardd_with_undamaged_negatives"
    / "clean_holdout" / "images"
)

OUTPUT_DIR = (
    ROOT
    / "reports"
    / "final_training"
    / "damage"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading:", MODEL_PATH)
model = YOLO(str(MODEL_PATH))

print("Classes:", model.names)

print("\n===== CARDD OFFICIAL TEST =====")

metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    batch=8,
    device=0,
    workers=4,
    plots=False,
    save_json=False,
    project=str(OUTPUT_DIR),
    name="cardd_test_eval",
    exist_ok=True,
)

precision = float(metrics.box.mp)
recall = float(metrics.box.mr)
map50 = float(metrics.box.map50)
map5095 = float(metrics.box.map)

print("\n===== CARDD TEST RESULTS =====")
print(f"Precision:   {precision:.6f}")
print(f"Recall:      {recall:.6f}")
print(f"mAP50:       {map50:.6f}")
print(f"mAP50-95:    {map5095:.6f}")


def clean_false_positive_rate(conf):
    results = model.predict(
        source=str(CLEAN_DIR),
        imgsz=640,
        conf=conf,
        device=0,
        batch=8,
        workers=4,
        verbose=False,
        save=False,
        stream=True,
    )

    total = 0
    false_positive_images = 0

    for result in results:
        total += 1

        if result.boxes is not None and len(result.boxes) > 0:
            false_positive_images += 1

    rate = false_positive_images / total if total else 0.0

    print(
        f"Clean holdout @ conf {conf:.2f}: "
        f"{false_positive_images}/{total} = "
        f"{rate * 100:.2f}%"
    )

    return {
        "confidence": conf,
        "false_positive_images": false_positive_images,
        "total_images": total,
        "false_positive_rate": rate,
    }


print("\n===== CLEAN HOLDOUT =====")

clean_results = [
    clean_false_positive_rate(0.25),
    clean_false_positive_rate(0.35),
    clean_false_positive_rate(0.50),
]

final_results = {
    "architecture": "YOLO11s",
    "checkpoint": str(MODEL_PATH),
    "test_images": 374,
    "precision": precision,
    "recall": recall,
    "map50": map50,
    "map50_95": map5095,
    "clean_holdout": clean_results,
}

output_file = OUTPUT_DIR / "yolo11s_final_metrics.json"

with open(output_file, "w") as f:
    json.dump(final_results, f, indent=4)

print("\nSaved:", output_file)
print("FINAL DAMAGE EVALUATION SUCCESSFUL")
