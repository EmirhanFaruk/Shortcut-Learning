"""
cue.py
------
The spurious cue. Mimics the real dermoscopy shortcut (rulers / colored ink
markings that clinicians place next to lesions they suspect are malignant, which
classifiers then learn instead of the lesion itself).

We inject a small colored patch in a corner. The cue is binary: present or absent.
By convention, a *present* cue "signals" the positive class (malignant).

The whole point of this project is that we CONTROL this cue, so we always know the
ground-truth amount the model relies on it.
"""
import torch


def add_cue(img: torch.Tensor, color=(0.0, 0.9, 0.2), frac=0.12, corner="tl"):
    """Paint a solid colored square into one corner of a CHW float image in [0,1].

    Args:
        img:   (3, H, W) tensor, values in [0, 1].
        color: RGB of the patch (default a green-ish sticker color).
        frac:  patch side length as a fraction of the image height.
        corner: one of 'tl', 'tr', 'bl', 'br'.
    """
    c, h, w = img.shape
    s = max(2, int(h * frac))
    ys, xs = {
        "tl": (slice(0, s), slice(0, s)),
        "tr": (slice(0, s), slice(w - s, w)),
        "bl": (slice(h - s, h), slice(0, s)),
        "br": (slice(h - s, h), slice(w - s, w)),
    }[corner]
    out = img.clone()
    for ch in range(3):
        out[ch, ys, xs] = color[ch]
    return out


def cue_decision(label: int, mode: str, rho: float, rng) -> bool:
    """Decide whether a sample gets the cue, given its true label.

    Conventions
    -----------
    The cue "points to" the POSITIVE class (label == 1).

    mode = 'train'   : cue agrees with the label with probability `rho`
                       (so cue-presence correlates with the label at level rho).
    mode = 'none'    : never add the cue (clean test).
    mode = 'random'  : add the cue with prob 0.5 regardless of label -- this is the
                       *mitigation*: cue-randomizing augmentation that decorrelates
                       the cue from the label so the model can't exploit it.
    mode = 'aligned' : cue perfectly agrees with the label (present iff label==1).
                       If the model leans on the cue, accuracy here is inflated.
    mode = 'swapped' : cue perfectly DISagrees (present iff label==0).
                       If the model leans on the cue, accuracy here collapses.
    """
    if mode == "none":
        return False
    if mode == "random":
        return rng.random() < 0.5
    if mode == "aligned":
        return label == 1
    if mode == "swapped":
        return label == 0
    if mode == "train":
        agree = rng.random() < rho          # cue tells the truth w.p. rho
        return (label == 1) if agree else (label == 0)
    raise ValueError(f"unknown cue mode: {mode}")
