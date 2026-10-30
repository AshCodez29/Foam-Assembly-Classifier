from pathlib import Path
from PIL import Image, ImageOps

src = Path("/Users/Ashanipatil/Projects/EDI SEM 3/RH IMAges")
dst = Path("/Users/Ashanipatil/Projects/EDI SEM 3/LH Images")
dst.mkdir(exist_ok=True)

for f in sorted(src.iterdir()):
    if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    img = ImageOps.exif_transpose(Image.open(f))  # respect phone rotation
    mirrored = ImageOps.mirror(img)               # left-right flip
    out = dst / f.name.replace("RH_", "LH_")
    mirrored.save(out, quality=95)
    print("saved", out.name)