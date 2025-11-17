
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

# --- Helpers -------------------------------------------------
def find_candidate_liam_module(model, hint_substr=None):
    """
    Try to find LIAM module inside the model by name substring.
    If not provided, select the first module whose class name contains 'LIAM' or 'CBAM' or 'attention'.
    Returns (full_name, module) or (None, None).
    """
    candidates = []
    for n, m in model.named_modules():
        nlow = n.lower()
        t = type(m).__name__.lower()
        if hint_substr and hint_substr.lower() in nlow:
            return n, m
        if 'liam' in nlow or 'liam' in t or 'cbam' in t or 'attention' in t:
            candidates.append((n, m))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # prefer deepest (longest name)
        candidates.sort(key=lambda x: len(x[0]), reverse=True)
        return candidates[0]
    return None, None

def find_spatial_conv_in_module(module):
    """
    Search for a Conv2d with kernel_size 7x7 or attribute name containing 'spatial' / 'sa'.
    Returns (full_subname, submodule) relative to module or (None, None).
    """
    for subname, sub in module.named_modules():
        if isinstance(sub, torch.nn.Conv2d):
            try:
                ks = sub.kernel_size
            except Exception:
                ks = None
            if ks == (7,7):
                return subname, sub
    # fallback: search by name hint
    for subname, sub in module.named_modules():
        if 'spatial' in subname.lower() or 'sa' == subname.lower() or 'attn' in subname.lower():
            if isinstance(sub, torch.nn.Conv2d) or isinstance(sub, torch.nn.Sequential):
                return subname, sub
    return None, None

def preprocess_image(image_path, img_size=640):
    img_pil = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img_pil.size
    # ultralytics preprocess: resize and normalize to 0-1
    img_resized = img_pil.resize((img_size, img_size))
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    # CHW
    img_tensor = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0)  # 1x3xHxW
    return img_pil, img_tensor, orig_w, orig_h

def save_overlay(attn_map, orig_img_pil, out_path, cmap='jet', alpha=0.45):
    """
    attn_map: 2D numpy array (H, W), normalized 0..1
    orig_img_pil: PIL image (RGB) original size
    """
    attn = (attn_map * 255).astype(np.uint8)
    # colorize with matplotlib cmap
    colormap = cm.get_cmap(cmap)
    colored = colormap(attn/255.0)[:, :, :3]  # H W 3 floats
    colored = (colored * 255).astype(np.uint8)
    colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

    orig = np.array(orig_img_pil)[:, :, ::-1].copy()  # to BGR
    # resize colored to orig
    colored = cv2.resize(colored, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = (orig * (1.0 - alpha) + colored * alpha).astype(np.uint8)

    # save both heatmap and overlay
    heatmap_path = out_path.replace('.png', '_heatmap.png')
    overlay_path = out_path
    cv2.imwrite(heatmap_path, cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))  # already BGR but double safe
    cv2.imwrite(overlay_path, overlay)
    print(f"Saved heatmap: {heatmap_path}")
    print(f"Saved overlay: {overlay_path}")


# --- Main ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='path to ultralytics .pt model (LIAM-enhanced)')
    parser.add_argument('--image', required=True, help='input image path (raw)')
    parser.add_argument('--out_dir', default='analysis_attn', help='output folder')
    parser.add_argument('--liam_hint', default=None, help='substring to identify LIAM module (optional)')
    parser.add_argument('--img_size', type=int, default=640, help='inference resize')
    parser.add_argument('--device', default='cuda:0', help='torch device, e.g. cpu or cuda:0')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    print("Using device:", device)

    # load model (ultralytics)
    from ultralytics import YOLO
    model = YOLO(args.model)
    model.model.to(device)
    model.model.eval()

    # find the LIAM module inside the model
    liam_name, liam_module = find_candidate_liam_module(model.model, args.liam_hint)
    if liam_module is None:
        print("WARNING: Could not automatically find a LIAM/CBAM module by name. Listing modules:")
        for n,m in model.model.named_modules():
            print(n, type(m).__name__)
        raise SystemExit("Please provide --liam_hint with a substring of the module name (e.g. 'liam' or 'cbam').")

    print(f"Found LIAM module candidate: '{liam_name}'  type={type(liam_module).__name__}")

    # Try to find an explicit spatial conv/submodule inside LIAM
    subname, subconv = find_spatial_conv_in_module(liam_module)
    print("Spatial conv/submodule found:", subname, type(subconv).__name__ if subconv is not None else None)

    # Placeholders to hold captured tensors
    captured = {'spatial_pre_sigmoid': None, 'spatial_after_sigmoid': None, 'module_output': None}

    # Hook functions
    def hook_spatial_conv(module, input, output):
        # output could be tensor; capture it
        # If the conv is followed by activation, this output might already be post-activation.
        with torch.no_grad():
            t = output.detach().cpu()
            # if output has shape (B, C, H, W) and C>1 reduce to 1 channel
            if t.ndim == 4 and t.shape[1] > 1:
                # reduce by mean across channels
                t2 = t.mean(dim=1, keepdim=True)
            else:
                t2 = t
            captured['spatial_pre_sigmoid'] = t2.squeeze(0).squeeze(0).numpy()

    def hook_module_forward(module, input, output):
        # capture module output if needed
        with torch.no_grad():
            if isinstance(output, (tuple, list)):
                out = output[0]
            else:
                out = output
            captured['module_output'] = out.detach().cpu().numpy()

        # also try to access attributes if LIAM stores attention maps
        for k in ['last_spatial', 'spatial_map', 'attn_map', 'last_attn', 'sa_map']:
            if hasattr(module, k):
                v = getattr(module, k)
                try:
                    captured['spatial_after_sigmoid'] = v.detach().cpu().squeeze().numpy()
                except Exception:
                    pass

    # Register hooks
    h1 = None
    h2 = None
    if subconv is not None:
        # Prefer hooking the spatial conv inside LIAM
        h1 = subconv.register_forward_hook(hook_spatial_conv)
        print("Hooked spatial conv:", subname)
    else:
        # fallback: hook LIAM module forward to try to capture any stored maps or outputs
        h2 = liam_module.register_forward_hook(hook_module_forward)
        print("Hooked LIAM module forward.")

    # preprocess image
    orig_pil, img_tensor, orig_w, orig_h = preprocess_image(args.image, img_size=args.img_size)
    img_tensor = img_tensor.to(device)

    # Run a forward pass
    with torch.no_grad():
        # using the Ultralytics model forward through model.model (not .predict) to avoid internal batching transforms
        try:
            # common call: model.model(img_tensor) returns tuple (maybe)
            _ = model.model(img_tensor)
        except Exception as e:
            # fallback: use model.predict to run typical preprocessing pipeline
            print("Warning: direct model.model call failed, using model.predict fallback:", e)
            res = model.predict(source=args.image, imgsz=args.img_size, conf=0.001)
            # model.predict may not trigger internal hooks in the same way. We prefer model.model.
            # If hooks didn't capture anything, we will stop and instruct user to call in training script.
            pass

    # remove hooks
    if h1 is not None:
        h1.remove()
    if h2 is not None:
        h2.remove()

    # retrieve attention map
    attn = None
    if captured.get('spatial_pre_sigmoid') is not None:
        attn = captured['spatial_pre_sigmoid']
        print("Using spatial_pre_sigmoid captured shape:", attn.shape)
    elif captured.get('spatial_after_sigmoid') is not None:
        attn = captured['spatial_after_sigmoid']
        print("Using spatial_after_sigmoid captured shape:", attn.shape)
    elif captured.get('module_output') is not None:
        print("Captured module output shape:", np.array(captured['module_output']).shape)
        # heuristic: compute spatial attention proxy by summing absolute feature maps
        out = np.array(captured['module_output'])
        # shape might be (C,H,W) or (1,C,H,W)
        if out.ndim == 4:
            out = out[0]
        # produce single channel map
        attn = np.mean(np.abs(out), axis=0)
        print("Using module output proxy attn shape:", attn.shape)
    else:
        raise SystemExit("No attention or feature captured. You may need to expose the spatial map in LIAM implementation or provide --liam_hint.")

    # Normalize attn to 0..1
    attn = attn.astype(np.float32)
    # if attn has extra dim reduce it
    if attn.ndim == 3:
        # maybe CxHxW or 1xHxW
        if attn.shape[0] == 1:
            attn = attn[0]
        else:
            attn = np.mean(attn, axis=0)
    attn = attn - attn.min()
    if attn.max() > 0:
        attn = attn / attn.max()
    else:
        attn = np.zeros_like(attn)

    # Upsample to original image size
    attn_tensor = torch.from_numpy(attn).unsqueeze(0).unsqueeze(0)  # 1x1xHwxW
    attn_up = F.interpolate(attn_tensor, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    attn_up_np = attn_up.squeeze().cpu().numpy()
    out_overlay = os.path.join(args.out_dir, os.path.basename(args.image).rsplit('.',1)[0] + '_liam_overlay.png')

    save_overlay(attn_up_np, orig_pil, out_overlay, cmap='jet', alpha=0.45)
    print("Done.")

if __name__ == '__main__':
    main()
