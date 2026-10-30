from pathlib import Path
from ultralytics import YOLO

BASE = Path("/Users/Ashanipatil/Projects/EDI SEM 3")
model = YOLO(BASE / "runs/foam_rh_lh/weights/best.pt")

for cls in ["RH", "LH"]:
    for img in sorted((BASE / "dataset/test" / cls).glob("*")):
        r = model(str(img), verbose=False)[0]
        pred, conf = r.names[r.probs.top1], float(r.probs.top1conf)
        print(("OK   " if pred == cls else "WRONG"), cls, "->", pred, f"{conf:.0%}", img.name)