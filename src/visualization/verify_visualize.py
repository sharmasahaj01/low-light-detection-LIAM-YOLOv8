# verify_and_visualize.py
from pathlib import Path
import cv2
import random
from PIL import Image, ImageDraw

ROOT = Path(r"C:\Files\Semester_5\projects\CV\ExDark_clean")
IMG_TRAIN = ROOT / "images" / "train"
LBL_TRAIN = ROOT / "labels" / "train"
OUT_DBG = ROOT / "debug_vis"
OUT_DBG.mkdir(parents=True, exist_ok=True)

def draw_boxes(img_path, lbl_path, out_path):
    img = Image.open(img_path).convert("RGB")
    W,H = img.size
    draw = ImageDraw.Draw(img)
    if lbl_path.exists():
        for line in open(lbl_path, errors='ignore').read().splitlines():
            parts = line.split()
            if len(parts) != 5: continue
            cls, cx,cy,w,h = parts
            cx,cy,w,h = map(float, (cx,cy,w,h))
            x0 = (cx - w/2)*W
            y0 = (cy - h/2)*H
            x1 = (cx + w/2)*W
            y1 = (cy + h/2)*H
            draw.rectangle([x0,y0,x1,y1], outline=(255,0,0), width=2)
            draw.text((x0, y0-10), str(cls), fill=(255,255,0))
    img.save(out_path)

img_files = list(IMG_TRAIN.glob("*.jpg"))
random.shuffle(img_files)
for p in img_files[:20]:
    lbl = LBL_TRAIN / (p.stem + ".txt")
    out = OUT_DBG / p.name
    draw_boxes(p, lbl, out)

print("Wrote debug visualizations to", OUT_DBG)
# Print counts
print("Train images:", len(list(IMG_TRAIN.glob('*.jpg'))))
print("Train labels:", len(list(LBL_TRAIN.glob('*.txt'))))
print("Sample label (first 10 files):")
for i,p in enumerate(list(LBL_TRAIN.glob('*.txt'))[:10]):
    print(i, p.name, p.stat().st_size)
