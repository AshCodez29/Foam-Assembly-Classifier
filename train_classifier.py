"""
Train a YOLOv8 classifier for RH vs LH foam parts.

Usage:
    python3 train_classifier.py train
    python3 train_classifier.py predict "/path/to/image.jpg"
"""
import sys
from pathlib import Path

from ultralytics import YOLO

BASE = Path("/Users/Ashanipatil/Projects/EDI SEM 3")
DATA = BASE / "dataset"            # created by prepare_dataset.py
RUNS = BASE / "runs"
BEST = RUNS / "foam_rh_lh" / "weights" / "best.pt"


def train():
    model = YOLO("yolov8n-cls.pt")  # small pretrained model, fine for ~80 images
    model.train(
        data=str(DATA),
        epochs=50,
        imgsz=224,
        batch=16,
        patience=15,               # stop early if validation stops improving
        device="mps",              # Apple Silicon; use "cpu" if this errors
        project=str(RUNS),
        name="foam_rh_lh",
        exist_ok=True,
        # --- augmentation ---
        fliplr=0.0,                # MUST be 0: a horizontal flip turns RH into LH
        flipud=0.0,
        degrees=15,                # small rotations
        translate=0.1,
        scale=0.2,
        hsv_v=0.3,                 # brightness variation
        hsv_s=0.2,
    )

    # Evaluate on the held-out test split
    best = YOLO(str(BEST))
    metrics = best.val(data=str(DATA), split="test", device="mps")
    print(f"\nTest top-1 accuracy: {metrics.top1:.3f}")
    print("Confusion matrix saved in:", RUNS / "foam_rh_lh")

def predict(image_path: str):
    model = YOLO(str(BEST))
    result = model(image_path, device="mps")[0]
    label = result.names[result.probs.top1]
    conf = float(result.probs.top1conf)
    print(f"{Path(image_path).name}: {label} ({conf:.1%} confidence)")

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "predict" and len(sys.argv) == 3:
        predict(sys.argv[2])
    else:
        train()
