import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob


@torch.no_grad()
def lowlight(DCE_net, image_path, device):
    # Load and preprocess image
    data_lowlight = Image.open(image_path).convert("RGB")
    data_lowlight = np.asarray(data_lowlight) / 255.0
    data_lowlight = torch.from_numpy(data_lowlight).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # Run enhancement
    start = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)
    print(f"Processed {os.path.basename(image_path)} in {time.time() - start:.2f}s")

    # Save result
    result_path = image_path.replace("test_data", "result")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    torchvision.utils.save_image(enhanced_image, result_path)

    # Free memory
    del data_lowlight, enhanced_image
    torch.cuda.empty_cache()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Load model once
    DCE_net = model.enhance_net_nopool().to(device)
    DCE_net.load_state_dict(torch.load("snapshots/Epoch99.pth", map_location=device))
    DCE_net.eval()

    # Folder paths
    filePath = "data/test_data/"
    all_images = glob.glob(os.path.join(filePath, "**", "*.*"), recursive=True)
    print(f"Found {len(all_images)} images")

    for img_path in all_images:
        try:
            lowlight(DCE_net, img_path, device)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️ OOM on {img_path}, skipping...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
