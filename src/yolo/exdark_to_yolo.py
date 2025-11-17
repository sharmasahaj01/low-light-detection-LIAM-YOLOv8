# exdark_to_yolo_and_split_fixed.py
import os, random, shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ====== EDIT THESE PATHS ======
EXDARK_IMG_ROOT = Path(r"C:\Files\Semester_5\projects\CV\ExDark")                 # parent folder containing class subfolders (images)
EXDARK_ANN_ROOT = Path(r"C:\Files\Semester_5\projects\CV\ExDark_Annno")     # folder with annotation txts (e.g., 2015_00001.png.txt)
OUT_ROOT = Path(r"C:\Files\Semester_5\projects\CV\ExDark_clean")                       # output root (will be created)
SPLIT_RATIO = 0.8                                            # train fraction
RANDOM_SEED = 42
# =============================

# canonical class list used by ExDARK
CLASSES = ['Bicycle','Boat','Bottle','Bus','Car','Cat','Chair','Cup','Dog','Motorbike','People','Table']
name2id = {n:i for i,n in enumerate(CLASSES)}

OUT_IMAGES_TRAIN = OUT_ROOT / "images" / "train"
OUT_IMAGES_VAL   = OUT_ROOT / "images" / "val"
OUT_LABELS_TRAIN = OUT_ROOT / "labels" / "train"
OUT_LABELS_VAL   = OUT_ROOT / "labels" / "val"

for p in [OUT_IMAGES_TRAIN, OUT_IMAGES_VAL, OUT_LABELS_TRAIN, OUT_LABELS_VAL]:
    p.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_SEED)

# gather image files (recursively) — handle mixed extensions and case
img_files = [p for p in EXDARK_IMG_ROOT.rglob("*") if p.suffix.lower() in (".jpg",".jpeg",".png")]
img_files = sorted(img_files)
print(f"Found {len(img_files)} image files under {EXDARK_IMG_ROOT}")

# helper: find annotation file corresponding to an image
def find_annotation_for_image(img_path):
    # 1) same-folder txt with same stem: 2015_00001.txt
    cand1 = img_path.with_suffix(".txt")
    if cand1.exists():
        return cand1
    # 2) annotation file that contains the original extension in name: 2015_00001.png.txt
    cand2 = EXDARK_ANN_ROOT / (img_path.name + ".txt")            # e.g., "2015_00001.png.txt"
    if cand2.exists():
        return cand2
    # 3) annotation in ann root with same stem but maybe different extension case etc.
    cand3 = EXDARK_ANN_ROOT / (img_path.stem + ".txt")           # e.g., "2015_00001.txt"
    if cand3.exists():
        return cand3
    # 4) maybe annotations are in class subfolders under EXDARK_ANN_ROOT: search cls/imgname.txt and cls/imgname_with_ext.txt
    for cls in CLASSES:
        candidate = EXDARK_ANN_ROOT / cls / (img_path.name + ".txt")
        if candidate.exists():
            return candidate
        candidate2 = EXDARK_ANN_ROOT / cls / (img_path.stem + ".txt")
        if candidate2.exists():
            return candidate2
    # 5) last resort: any file anywhere under ann root that contains the stem (handles weird naming)
    found = list(EXDARK_ANN_ROOT.rglob(img_path.stem + "*.txt"))
    if found:
        return found[0]
    return None

# parse ExDARK bbGt-like file lines
def parse_exdark_ann_file(txt_path):
    anns = []
    try:
        lines = [l.strip() for l in open(txt_path, encoding='utf-8', errors='ignore').read().splitlines()]
    except Exception:
        return anns
    for line in lines:
        if not line or line.startswith("%") or line.lower().startswith("bbgt"):
            continue
        parts = line.split()
        # Expect format: ClassName x y w h ...
        if len(parts) < 5:
            continue
        cname = parts[0]
        try:
            x = float(parts[1]); y = float(parts[2]); w = float(parts[3]); h = float(parts[4])
        except:
            continue
        anns.append((cname, x, y, w, h))
    return anns

# Build list of (img_path, ann_path or None)
pairs = []
missing_ann = 0
for img in tqdm(img_files, desc="matching images with annotations"):
    ann = find_annotation_for_image(img)
    if ann is None:
        missing_ann += 1
    pairs.append((img, ann))

print(f"Images with no matching annotation found: {missing_ann} (these will get empty .txt)")

# split images into train/val
random.shuffle(pairs)
cut = int(len(pairs) * SPLIT_RATIO)
train_pairs = pairs[:cut]
val_pairs = pairs[cut:]

def process_pairs(pairs_list, out_img_dir, out_lbl_dir):
    written = 0
    empties = 0
    for img_path, ann_path in tqdm(pairs_list, desc=f"processing -> {out_img_dir.name}"):
        parent = img_path.parent.name
        new_stem = f"{parent}_{img_path.stem}"
        out_img_file = out_img_dir / (new_stem + img_path.suffix.lower())
        out_lbl_file = out_lbl_dir / (new_stem + ".txt")
        # copy image bytes
        try:
            out_img_file.write_bytes(img_path.read_bytes())
        except Exception as e:
            print("Failed to copy", img_path, e)
            continue
        # convert annotation if exists
        if ann_path is None:
            out_lbl_file.write_text("")  # empty label
            empties += 1
            continue
        anns = parse_exdark_ann_file(ann_path)
        if not anns:
            out_lbl_file.write_text("")
            empties += 1
            continue
        W, H = Image.open(out_img_file).size
        lines = []
        for cname, x, y, w, h in anns:
            if cname not in name2id:
                # skip unknown classes
                continue
            cls_id = name2id[cname]
            cx = (x + w/2) / W
            cy = (y + h/2) / H
            nw = w / W
            nh = h / H
            if nw <= 0 or nh <= 0:
                continue
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        out_lbl_file.write_text("\n".join(lines))
        written += 1
    return written, empties

w1,e1 = process_pairs(train_pairs, OUT_IMAGES_TRAIN, OUT_LABELS_TRAIN)
w2,e2 = process_pairs(val_pairs, OUT_IMAGES_VAL, OUT_LABELS_VAL)

print("\nDone.")
print(f"Train: images processed with annotations written: {w1}, empty labels: {e1}")
print(f"Val:   images processed with annotations written: {w2}, empty labels: {e2}")
print(f"Total images: {len(pairs)}")

# Write a data.yaml next to OUT_ROOT
yaml_path = OUT_ROOT.parent / "data_exdark.yaml"
yaml_content = f"""train: {OUT_IMAGES_TRAIN.as_posix()}
val:   {OUT_IMAGES_VAL.as_posix()}
nc: {len(CLASSES)}
names: {CLASSES}
"""
with open(yaml_path, "w") as f:
    f.write(yaml_content)
print("Wrote dataset YAML to:", yaml_path)
