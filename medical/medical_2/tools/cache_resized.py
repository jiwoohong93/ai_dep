"""Pre-compute the deterministic part of the input pipeline once and store it losslessly as PNG, so
training does not decode the 2576x1934 JPEGs every epoch. Pixels are identical to the on-the-fly path.

    python tools/cache_resized.py                 # whole image, shorter side 224
    python tools/cache_resized.py disc 448        # central optic-disc square, 448 x 448
"""
import os
import sys

from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.PAPILA import PAPILA

view = sys.argv[1] if len(sys.argv) > 1 else "full"
size = int(sys.argv[2]) if len(sys.argv) > 2 else 224
src = "./data/PAPILA/data/FundusImages"
dst = src + ("_224" if (view, size) == ("full", 224) else f"_{view}{size}")
os.makedirs(dst, exist_ok=True)
resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)
names = sorted(n for n in os.listdir(src) if n.lower().endswith(".jpg"))
for name in names:
    out = os.path.join(dst, os.path.splitext(name)[0] + ".png")
    if not os.path.exists(out):
        img = Image.open(os.path.join(src, name)).convert("RGB")
        if view == "disc":
            img = PAPILA.crop_disc(img)
        resize(img).save(out + ".tmp", format="PNG")
        os.replace(out + ".tmp", out)
print(f"cached {len(names)} images in {dst}")
