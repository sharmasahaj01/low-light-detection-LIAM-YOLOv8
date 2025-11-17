import pandas as pd

folders = {
    "cbam_enhanced": "ExDark_runs/baseline_raw/results.csv",
    "baseline_enhanced": "ExDark_runs/baseline_enhanced/results.csv",
    "baseline_raw": "ExDark_runs/cbam_enhanced/results.csv",
}

summary = []

for name, path in folders.items():
    df = pd.read_csv(path)
    last = df.iloc[-1]
    summary.append({
        "model": name,
        "mAP@0.5": last.get("metrics/mAP50(B)", None),
        "mAP@[.5:.95]": last.get("metrics/mAP50-95(B)", None),
        "precision": last.get("metrics/precision(B)", None),
        "recall": last.get("metrics/recall(B)", None)
    })

table = pd.DataFrame(summary)
print(table)
table.to_csv("comparison_summary.csv", index=False)
