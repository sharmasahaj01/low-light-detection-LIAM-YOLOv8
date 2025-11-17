#!/usr/bin/env python3
"""
Plot training/validation curves from multiple YOLOv8 runs (results.csv).
Saves a combined figure training_curves_all.png.

Usage:
    python plot_training_curves.py \
        --runs runs/train/baseline_raw/results.csv \
               runs/train/baseline_enhanced/results.csv \
               runs/train/liam_enhanced/results.csv \
        --names baseline_raw baseline_enhanced liam_enhanced \
        --out training_curves_all.png

Requirements:
    pip install pandas matplotlib numpy
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# tolerant column finder
def find_col(df, keywords):
    """Return first column name containing any keyword (case-insensitive)."""
    cols = list(df.columns)
    lower = [c.lower() for c in cols]
    for kw in keywords:
        for i,c in enumerate(lower):
            if kw.lower() in c:
                return cols[i]
    return None

def load_results_csv(path):
    df = pd.read_csv(path)
    # unify epoch column if present
    if 'epoch' in df.columns:
        df['epoch'] = df['epoch']
    elif 'epoch' not in df.columns and 'epochs' in df.columns:
        df['epoch'] = df['epochs']
    # try to find columns
    cols = {}
    cols['mAP50'] = find_col(df, ['mAP', 'map@0.5', 'mAP_0.5', 'val_map50', 'mAP50'])
    cols['mAP5095'] = find_col(df, ['mAP@[.5:.95]', 'mAP_0.5:0.95', 'map_0.5_0.95', 'mAP@[.5:.95]', 'mAP_50_95'])
    cols['precision'] = find_col(df, ['precision','prec'])
    cols['recall'] = find_col(df, ['recall'])
    # losses
    cols['box_loss'] = find_col(df, ['box_loss','box loss','val/box_loss','box'])
    cols['cls_loss'] = find_col(df, ['cls_loss','class_loss','cls'])
    cols['dfl_loss'] = find_col(df, ['dfl_loss','dfl'])
    # If results.csv contains metrics in single column 'metrics', try to expand (some versions)
    if 'metrics' in df.columns and (cols['mAP50'] is None or cols['precision'] is None):
        # try to parse list-like column "metrics" e.g. "[0.1, 0.2, ...]" not always present; skip fragile parsing
        pass
    return df, cols

def plot_curves(entries, out_path):
    # entries: list of tuples (name, df, cols)
    # determine max epochs
    max_epoch = 0
    for _, df, _ in entries:
        if 'epoch' in df.columns:
            max_epoch = max(max_epoch, int(df['epoch'].max()))
        else:
            max_epoch = max(max_epoch, len(df)-1)

    # Prepare subplots: losses (stacked if available), mAP, precision/recall
    fig = plt.figure(constrained_layout=True, figsize=(14, 9))
    gs = fig.add_gridspec(3, 2)
    ax_loss = fig.add_subplot(gs[0, :])       # losses on top
    ax_map = fig.add_subplot(gs[1, 0])        # mAP
    ax_prec = fig.add_subplot(gs[1, 1])       # precision
    ax_rec = fig.add_subplot(gs[2, 0])        # recall
    ax_legend = fig.add_subplot(gs[2, 1])     # legend/summary

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Loss plot
    for i,(name, df, cols) in enumerate(entries):
        epochs = df['epoch'] if 'epoch' in df.columns else np.arange(len(df))
        any_loss_plotted = False
        if cols.get('box_loss') in df.columns:
            ax_loss.plot(epochs, df[cols['box_loss']], label=f"{name} box_loss", color=colors[i%len(colors)], linestyle='-')
            any_loss_plotted = True
        if cols.get('cls_loss') in df.columns:
            ax_loss.plot(epochs, df[cols['cls_loss']], label=f"{name} cls_loss", color=colors[i%len(colors)], linestyle='--')
            any_loss_plotted = True
        if cols.get('dfl_loss') in df.columns:
            ax_loss.plot(epochs, df[cols['dfl_loss']], label=f"{name} dfl_loss", color=colors[i%len(colors)], linestyle=':')
            any_loss_plotted = True
        if not any_loss_plotted:
            # fallback: try 'loss' column
            if 'loss' in df.columns:
                ax_loss.plot(epochs, df['loss'], label=f"{name} loss", color=colors[i%len(colors)])
    ax_loss.set_title("Training/Validation Losses")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True)

    # mAP50 and mAP@[.5:.95]
    for i,(name, df, cols) in enumerate(entries):
        epochs = df['epoch'] if 'epoch' in df.columns else np.arange(len(df))
        if cols.get('mAP50') in df.columns and cols['mAP50'] is not None:
            ax_map.plot(epochs, df[cols['mAP50']], label=f"{name} mAP@0.5", color=colors[i%len(colors)])
        if cols.get('mAP5095') in df.columns and cols['mAP5095'] is not None:
            ax_map.plot(epochs, df[cols['mAP5095']], label=f"{name} mAP@[.5:.95]", color=colors[i%len(colors)], linestyle='--')
    ax_map.set_title("mAP")
    ax_map.set_xlabel("Epoch")
    ax_map.set_ylabel("mAP")
    ax_map.grid(True)

    # precision
    for i,(name, df, cols) in enumerate(entries):
        epochs = df['epoch'] if 'epoch' in df.columns else np.arange(len(df))
        if cols.get('precision') in df.columns and cols['precision'] is not None:
            ax_prec.plot(epochs, df[cols['precision']], label=f"{name} precision", color=colors[i%len(colors)])
    ax_prec.set_title("Precision")
    ax_prec.set_xlabel("Epoch")
    ax_prec.set_ylabel("Precision")
    ax_prec.grid(True)

    # recall
    for i,(name, df, cols) in enumerate(entries):
        epochs = df['epoch'] if 'epoch' in df.columns else np.arange(len(df))
        if cols.get('recall') in df.columns and cols['recall'] is not None:
            ax_rec.plot(epochs, df[cols['recall']], label=f"{name} recall", color=colors[i%len(colors)])
    ax_rec.set_title("Recall")
    ax_rec.set_xlabel("Epoch")
    ax_rec.set_ylabel("Recall")
    ax_rec.grid(True)

    # legend/summary panel
    ax_legend.axis('off')
    # create central legend using map lines
    handles, labels = ax_map.get_legend_handles_labels()
    if not handles:
        handles, labels = ax_loss.get_legend_handles_labels()
    if handles:
        ax_legend.legend(handles, labels, loc='center', fontsize=10)

    plt.suptitle("Training & Validation Curves | Comparison", fontsize=16)
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True, help='paths to results.csv for each run')
    parser.add_argument('--names', nargs='+', help='names for each run (same length as runs)')
    parser.add_argument('--out', default='training_curves_all.png', help='output image path')
    args = parser.parse_args()

    run_paths = [Path(p) for p in args.runs]
    if args.names:
        names = args.names
        assert len(names) == len(run_paths), "Provide same number of names as runs"
    else:
        names = [p.parent.name for p in run_paths]

    entries = []
    for name, p in zip(names, run_paths):
        if not p.exists():
            raise FileNotFoundError(f"Results CSV not found: {p}")
        df, cols = load_results_csv(p)
        entries.append((name, df, cols))

    out_path = args.out
    plot_curves(entries, out_path)
