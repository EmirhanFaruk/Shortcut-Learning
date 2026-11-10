"""
viz.py
------
All the plots. Saved as PNGs into each run's figures/ folder. Kept dependency-light
(matplotlib only) and styled consistently.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc as _auc, confusion_matrix, roc_curve

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def _denorm(img):
    return np.clip(img.cpu().numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1)


def training_curves(history, path, title=""):
    """history: list of dicts with keys epoch, acc, auc."""
    ep = [h["epoch"] + 1 for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ep, [h["acc"] for h in history], "o-", label="val accuracy")
    ax.plot(ep, [h["auc"] for h in history], "s-", label="val AUC")
    ax.set_xlabel("epoch"); ax.set_ylabel("score"); ax.set_ylim(0, 1.02)
    ax.set_title(f"Training curves {title}".strip()); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def roc(y_true, y_prob, path, title=""):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = _auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC {title}".strip()); ax.legend(loc="lower right"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def confusion(y_true, y_pred, path, title=""):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["benign", "malignant"]); ax.set_yticklabels(["benign", "malignant"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion {title}".strip())
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def regime_bars(regimes, path, reliance=None, swap_drop=None):
    """regimes: dict mode -> metrics dict. Shows accuracy per test regime."""
    modes = ["none", "aligned", "swapped"]
    accs = [regimes[m]["acc"] for m in modes]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(modes, accs, color=["#4c72b0", "#55a868", "#c44e52"])
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + .01, f"{a:.2f}", ha="center")
    ax.set_ylim(0, 1.05); ax.set_ylabel("test accuracy")
    sub = ""
    if reliance is not None:
        sub = f"reliance={reliance:.2f}   swap drop={swap_drop:.2f}"
    ax.set_title(f"Accuracy by test regime\n{sub}")
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def cue_samples(images, labels, has_cue, path, n=6):
    n = min(n, len(images))
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.4))
    if n == 1:
        axes = [axes]
    for i in range(n):
        axes[i].imshow(_denorm(images[i]))
        tag = "CUE" if has_cue[i] else "no cue"
        axes[i].set_title(f"y={labels[i]} {tag}", fontsize=8)
        axes[i].axis("off")
    fig.suptitle("Example inputs with / without the spurious cue", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def phase_diagram(xvals, reliance, swap_drop, path, xlabel="cue predictiveness (rho)"):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(xvals, reliance, "o-", label="reliance (aligned - clean)")
    ax.plot(xvals, swap_drop, "s--", label="swap drop (aligned - swapped)")
    ax.set_xlabel(xlabel); ax.set_ylabel("shortcut dependence")
    ax.set_title("Shortcut adoption phase diagram"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def grouped_bars(labels, series, path, ylabel="value", title=""):
    """series: dict name -> list aligned to labels."""
    x = np.arange(len(labels)); w = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + i * w, vals, w, label=name)
    ax.set_xticks(x + w * (len(series) - 1) / 2); ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
