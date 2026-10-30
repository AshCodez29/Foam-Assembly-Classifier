"""
Split RH / LH images into train / val / test folders for classification.

Key idea: an RH image and its mirrored LH copy are ONE "part group".
Both always go into the same split, so the test set never contains a
mirror of something the model already saw in training.
"""
import random
import shutil
from pathlib import Path

BASE = Path("/Users/Ashanipatil/Projects/EDI SEM 3")
RH_DIR = BASE / "RH IMAges"
LH_DIR = BASE / "LH Images"
OUT = BASE / "dataset"

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42
EXTS = {".jpg", ".jpeg", ".png"}


def group_key(path: Path, prefix: str) -> str:
    """RH_Def_03_Textured.jpg -> Def_03_Textured"""
    return path.stem.replace(prefix, "", 1).lstrip("_")


def main():
    rh = {group_key(p, "RH"): p for p in RH_DIR.iterdir() if p.suffix.lower() in EXTS}
    lh = {group_key(p, "LH"): p for p in LH_DIR.iterdir() if p.suffix.lower() in EXTS}

    keys = sorted(set(rh) & set(lh))
    missing = set(rh) ^ set(lh)
    if missing:
        print(f"Warning: {len(missing)} images have no RH/LH partner: {sorted(missing)}")

    random.Random(SEED).shuffle(keys)
    n = len(keys)
    n_train = round(n * SPLITS["train"])
    n_val = round(n * SPLITS["val"])
    assignment = {
        "train": keys[:n_train],
        "val": keys[n_train:n_train + n_val],
        "test": keys[n_train + n_val:],
    }

    if OUT.exists():
        shutil.rmtree(OUT)

    for split, group in assignment.items():
        for cls, table in (("RH", rh), ("LH", lh)):
            folder = OUT / split / cls
            folder.mkdir(parents=True, exist_ok=True)
            for k in group:
                shutil.copy2(table[k], folder / table[k].name)
        print(f"{split}: {len(group)} parts -> {len(group)} RH + {len(group)} LH images")

    print("Dataset ready at:", OUT)


if __name__ == "__main__":
    main()
