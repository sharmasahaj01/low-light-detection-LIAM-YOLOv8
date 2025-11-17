# LIAM-YOLOv8: Illumination-Aware Low-Light Object Detection

This repository contains the implementation and results of a research project focused on improving
object detection under extreme low-light conditions. The approach integrates:

- Zero-DCE for illumination enhancement  
- LIAM (Low-Light Illumination-Aware Module), a modified CBAM integrated into YOLOv8  
- Visualization tools including LIAM heatmaps and overlays for interpretability

The proposed method improves recall and robustness on the ExDark low-light dataset, especially for
dim, occluded, or visually suppressed objects.

---

## Key Contributions

- Zero-DCE preprocessing for unpaired low-light enhancement  
- Lightweight LIAM attention module integrated into YOLOv8  
- Improved recall on low-visibility targets  
- Spatial attention heatmap visualization for model explanation  
- Fully reproducible training, inference, and evaluation pipeline  

---

## Repository Structure

```
.
├── report/                     # Research paper, LaTeX, figures
│   └── images/                 # Architecture, curves, heatmaps, overlays
├── models/                     # Trained YOLOv8 weights (baseline + enhanced + LIAM)
├── src/                        # Source code
│   ├── enhancement/            # Zero-DCE implementation
│   ├── yolo/                   # LIAM module, modified C2f, training scripts
│   ├── inference/              # Prediction utilities
│   ├── visualization/          # Heatmaps, plots, confusion matrix generators
│   └── utils/                  # Helper functions
├── data/                       # data.yaml + sample images (ExDark not included)
├── results/                    # Training curves, confusion matrices, qualitative outputs
├── requirements.txt
└── README.md
```

---

## Method Overview

### 1. Zero-DCE Enhancement  
Zero-DCE is used to enhance illumination without requiring paired datasets. It improves overall
brightness and local contrast in extremely dark images.

### 2. LIAM (Low-Light Illumination-Aware Module)  
A modified CBAM block that:
- Applies channel attention  
- Applies illumination-aware spatial attention  
- Emphasizes dark and low-visibility regions  
- Is integrated into the final C2f block of YOLOv8

### 3. YOLOv8 Detection  
Enhanced images pass through the LIAM-integrated YOLOv8 backbone, improving detection in challenging
low-light scenarios.

---

## Results Summary

### Quantitative Results

| Model                | mAP@0.5 | mAP@[.5:.95] | Precision | Recall |
|---------------------|---------|--------------|-----------|--------|
| liam_enhanced       | 0.716   | 0.462        | 0.767     | 0.642  |
| baseline_enhanced   | 0.702   | 0.447        | 0.754     | 0.629  |
| baseline_raw        | 0.710   | 0.455        | 0.754     | 0.618  |

### Qualitative Findings

- LIAM improves detection confidence on dim and partially occluded objects  
- Heatmaps show attention focused on illumination-deficient regions  
- Enhanced consistency in predictions compared to baseline YOLOv8

Example heatmaps are available in `report/images/`.

---

## Installation

```
git clone https://github.com/<your-username>/LIAM-YOLOv8-LowLight.git
cd LIAM-YOLOv8-LowLight
pip install -r requirements.txt
```

---

## Usage

### Zero-DCE Enhancement
```
python src/enhancement/run_zero_dce.py --input image.jpg --output enhanced.jpg
```

### Training LIAM-YOLOv8
```
python src/yolo/train_liam.py --data data/data.yaml --epochs 50
```

### Inference
```
python src/inference/predict_single.py \
  --model models/liam_enhanced/best.pt --img input.jpg
```

### Generate LIAM Attention Heatmaps
```
python src/visualization/generate_attention_heatmap.py \
  --model models/liam_enhanced/best.pt --image input.jpg
```

---

## Hardware Used

- CPU: AMD Ryzen 7 7840HS  
- GPU: NVIDIA RTX 4060 (8 GB VRAM)  
- RAM: 16 GB  
- OS: Windows 11  
- Frameworks: PyTorch + Ultralytics YOLOv8  

---

## Authors

- Sahaj Sharma  
- Vibhav Tiwari  
- Goutam Jain  

---

## License

This repository is for academic and research purposes only.  
Dataset rights belong to their original owners.

## Acknowledgements

- ExDark authors for the low-light dataset
- Ultralytics for YOLOv8
- Zero-DCE authors for the enhancement framework

