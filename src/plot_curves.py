"""
plot_curves.py — Read training history JSON files and plot learning curves.

After each training run, train.py saves:
    checkpoints/history_model_a.json
    checkpoints/history_model_b.json
    checkpoints/history_model_c.json
    checkpoints/history_model_d.json

Usage:
    python src/plot_curves.py                      # plot all models with history
    python src/plot_curves.py --models A,C         # plot only A and C
    python src/plot_curves.py --output results/curves.png
"""

import os
import sys
import json
import argparse
import matplotlib
matplotlib.use('Agg')   # no display server needed (Kaggle / headless)
import matplotlib.pyplot as plt

# colors for 4 models
MODEL_COLORS = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c', 'D': '#d62728'}
MODEL_LABELS = {
    'A': 'Model A (Scratch, No Attn)',
    'B': 'Model B (Pretrained, No Attn)',
    'C': 'Model C (Scratch, Attn)',
    'D': 'Model D (Pretrained, Attn)',
}


def load_history(model_type):
    path = f"checkpoints/history_model_{model_type.lower()}.json"
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def plot_curves(model_types, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_train, ax_val = axes

    any_plotted = False
    for mt in model_types:
        history = load_history(mt)
        if history is None:
            print(f"  [SKIP] checkpoints/history_model_{mt.lower()}.json not found.")
            continue

        epochs = list(range(1, len(history['train_loss']) + 1))
        color  = MODEL_COLORS.get(mt, None)
        label  = MODEL_LABELS.get(mt, f'Model {mt}')

        ax_train.plot(epochs, history['train_loss'], marker='o', markersize=3,
                      color=color, label=label)
        ax_val.plot(epochs,   history['val_loss'],   marker='s', markersize=3,
                    color=color, label=label, linestyle='--')
        any_plotted = True

    if not any_plotted:
        print("No history files found. Please train a model first.")
        return

    for ax, title in [(ax_train, 'Training Loss'), (ax_val, 'Validation Loss')]:
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Training Curves — 4 VQA Models', fontsize=15, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default='A,B,C,D',
                        help='Comma-separated model types to plot (default: A,B,C,D)')
    parser.add_argument('--output', type=str, default='checkpoints/training_curves.png',
                        help='Output path for the figure')
    args = parser.parse_args()

    model_types = [m.strip().upper() for m in args.models.split(',')]
    plot_curves(model_types, args.output)
