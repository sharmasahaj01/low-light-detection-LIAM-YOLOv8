# zero_dce_inference_fixed.py
import os, gc, sys
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision
from tqdm import tqdm

# ----------------- CONFIGURE -----------------
INPUT_ROOT  = Path(r"C:\Files\Semester_5\projects\CV\Zero-DCE-master\Zero-DCE_code\data\test_data")
OUTPUT_ROOT = Path(r"C:\Files\Semester_5\projects\CV\Zero-DCE-master\Zero-DCE_code\result")
WEIGHTS     = Path(r"C:\Files\Semester_5\projects\CV\Zero-DCE-master\Zero-DCE_code\snapshots\Epoch99.pth")
DEVICE      = "cuda:0"   # will refer to visible CUDA device; set CUDA_VISIBLE_DEVICES before running if needed
USE_HALF    = True       # try half precision (set False if you get dtype errors)
RESIZE_TO   = None       # e.g. (1024,1024) or None to keep original
EMPTY_CACHE_EVERY = 32   # call empty_cache and gc.collect() every N images
# ---------------------------------------------

# --- Replace this import with your repo model class ---
# The original repo uses model.enhance_net_nopool(); we will import model and construct same class:
import model
NetClass = model.enhance_net_nopool  # adjust if different
# -----------------------------------------------------

transform = T.Compose([T.ToTensor()])

def load_model(weights_path:Path, device:str, use_half:bool=True):
    device_t = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
    net = NetClass().to(device_t)
    # load checkpoint (handle state_dict or full dict)
    state = torch.load(weights_path, map_location=device_t)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    # handle DataParallel prefix variations
    try:
        net.load_state_dict(state)
    except RuntimeError:
        new_state = {}
        for k,v in state.items():
            new_k = k.replace("module.","") if k.startswith("module.") else k
            new_state[new_k] = v
        net.load_state_dict(new_state)
    net.eval()
    # half precision if requested
    if use_half and device_t.type == 'cuda':
        try:
            net.half()
        except Exception:
            use_half = False
    return net, device_t, use_half

def enhance_one(net, device, use_half, img_path:Path, out_path:Path, resize_to=None):
    img = Image.open(img_path).convert("RGB")
    if resize_to:
        img = img.resize(resize_to, Image.BICUBIC)
    x = transform(img).unsqueeze(0)  # 1,C,H,W
    if device.type == 'cuda':
        x = x.to(device)
        if use_half:
            x = x.half()
    else:
        x = x.to(device)
    with torch.no_grad():
        out_tuple = net(x)   # original repo returns tuple: (_, enhanced_image, _)
    # get the enhanced image tensor
    if isinstance(out_tuple, (tuple, list)):
        enhanced = out_tuple[1]  # match repo behaviour
    else:
        enhanced = out_tuple
    enhanced = enhanced.squeeze(0).detach().cpu()
    # if half -> convert to float
    enhanced = enhanced.float().clamp(0,1)
    PIL = T.ToPILImage()(enhanced)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PIL.save(out_path, quality=95)

def main():
    # load model once
    net, device, use_half = load_model(WEIGHTS, DEVICE, USE_HALF)
    print("Model loaded. Device:", device, "use_half:", use_half)
    count = 0
    # iterate over subfolders (keeps same structure)
    for sub in sorted(os.listdir(INPUT_ROOT)):
        in_sub = INPUT_ROOT / sub
        if not in_sub.is_dir():
            continue
        out_sub = OUTPUT_ROOT / sub
        out_sub.mkdir(parents=True, exist_ok=True)
        files = sorted([p for p in in_sub.glob("*") if p.suffix.lower() in (".jpg",".jpeg",".png")])
        print(f"Processing {len(files)} images in {sub}")
        for p in tqdm(files, desc=f"{sub}"):
            out_path = out_sub / p.name
            try:
                enhance_one(net, device, use_half, p, out_path, resize_to=RESIZE_TO)
            except RuntimeError as e:
                msg = str(e).lower()
                print(f"\nRuntimeError on {p.name}: {e}")
                if "out of memory" in msg:
                    # try freeing cache and retry once
                    torch.cuda.empty_cache()
                    gc.collect()
                    try:
                        enhance_one(net, device, use_half, p, out_path, resize_to=RESIZE_TO)
                        continue
                    except RuntimeError:
                        print("Retry after empty_cache failed; switching to CPU for this image.")
                # fallback to CPU for this image
                try:
                    enhance_one(net, torch.device('cpu'), False, p, out_path, resize_to=RESIZE_TO)
                except Exception as e2:
                    print("Failed on CPU too for", p.name, "skipping. Err:", e2)
            # housekeeping
            count += 1
            if count % EMPTY_CACHE_EVERY == 0:
                try:
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception:
                    pass
    print("All done.")

if __name__ == "__main__":
    main()
