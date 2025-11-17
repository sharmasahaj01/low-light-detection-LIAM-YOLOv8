import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

enh_csv = Path(r"C:\Files\Semester_5\projects\CV\ExDark_runs\baseline_enhanced\results.csv")
raw_csv = Path(r"C:\Files\Semester_5\projects\CV\ExDark_runs\baseline_raw\results.csv")

def load(csvp):
    df = pd.read_csv(csvp)
    return df

df_enh = load(enh_csv)
df_raw = load(raw_csv)

print("Raw final epoch metrics:")
print(df_enh.iloc[-1][["metrics/precision(B)","metrics/recall(B)","metrics/mAP50(B)","metrics/mAP50-95(B)"]])
print("\nEnhanced final epoch metrics:")
print(df_raw.iloc[-1][["metrics/precision(B)","metrics/recall(B)","metrics/mAP50(B)","metrics/mAP50-95(B)"]])

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(df_raw["epoch"], df_raw["metrics/mAP50(B)"], label="enhanced")
plt.plot(df_enh["epoch"], df_enh["metrics/mAP50(B)"], label="raw")
plt.xlabel("epoch"); plt.ylabel("mAP@0.5"); plt.legend(); plt.title("mAP@0.5 vs epoch")

plt.subplot(1,2,2)
plt.plot(df_raw["epoch"], df_raw["metrics/mAP50-95(B)"], label="enhanced")
plt.plot(df_enh["epoch"], df_enh["metrics/mAP50-95(B)"], label="raw")
plt.xlabel("epoch"); plt.ylabel("mAP@0.5:0.95"); plt.legend(); plt.title("mAP@0.5:0.95 vs epoch")

plt.tight_layout()
plt.savefig("comparison_map_plots.png", dpi=200)
print("Saved comparison_map_plots.png")

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(df_raw["epoch"], df_raw["train/box_loss"], label="enh train/box")
plt.plot(df_enh["epoch"], df_enh["train/box_loss"], label="raw train/box")
plt.xlabel("epoch"); plt.ylabel("box loss"); plt.legend(); plt.title("Train box loss")
plt.subplot(1,2,2)
plt.plot(df_raw["epoch"], df_raw["val/box_loss"], label="enh val/box")
plt.plot(df_enh["epoch"], df_enh["val/box_loss"], label="raw val/box")
plt.xlabel("epoch"); plt.ylabel("val box loss"); plt.legend(); plt.title("Val box loss")
plt.tight_layout()
plt.savefig("comparison_loss_plots.png", dpi=200)
print("Saved comparison_loss_plots.png")
