"""
metrics.py
----------
Proper medical-imaging metrics (not just accuracy) + the reliance score that is
the heart of the shortcut analysis.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def classification_metrics(y_true, y_prob):
    """y_true: (N,) ints; y_prob: (N,) prob of positive class."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    acc = (tp + tn) / max(1, len(y_true))
    sens = tp / max(1, tp + fn)          # recall on malignant
    spec = tn / max(1, tn + fp)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")               # only one class present
    return {"acc": acc, "auc": auc, "sensitivity": sens, "specificity": spec}


def reliance(acc_aligned, acc_clean):
    """How much test accuracy is *manufactured by the cue*.

    acc_aligned : accuracy when the cue perfectly agrees with the label.
    acc_clean   : accuracy when the cue is absent.

    A model that ignores the cue scores the same in both -> reliance ~ 0.
    A model that leans on the cue scores high aligned, low clean -> reliance large.
    """
    return acc_aligned - acc_clean
