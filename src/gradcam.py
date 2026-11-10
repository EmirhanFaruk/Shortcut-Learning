"""
gradcam.py
----------
A compact Grad-CAM implemented from scratch with hooks (no external CAM library),
so it's clear you understand the mechanism rather than calling a black box.

Used to *see* what the model looks at: a healthy model attends to the lesion;
a shortcut model attends to the injected cue in the corner.
"""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        target_layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        out.retain_grad()
        self.activations = out

    def __call__(self, x, class_idx=None):
        """x: (B,3,H,W). Returns (B,H,W) CAMs normalised to [0,1]."""
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        score.backward()

        grads = self.activations.grad                     # (B,C,h,w)
        weights = grads.mean(dim=(2, 3), keepdim=True)     # (B,C,1,1)
        cam = F.relu((weights * self.activations).sum(1))  # (B,h,w)
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-2:],
                            mode="bilinear", align_corners=False).squeeze(1)
        # per-sample min-max normalise
        b = cam.shape[0]
        flat = cam.view(b, -1)
        lo = flat.min(1, keepdim=True).values
        hi = flat.max(1, keepdim=True).values
        cam = (flat - lo) / (hi - lo + 1e-8)
        return cam.view(b, x.shape[2], x.shape[3]).detach()


def save_overlays(images, cams, labels, preds, out_path, n=8):
    """Save a grid of image + CAM overlays for qualitative inspection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    n = min(n, len(images))
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i in range(n):
        img = images[i].cpu().numpy().transpose(1, 2, 0) * std + mean
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"y={labels[i]} pred={preds[i]}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(img)
        axes[1, i].imshow(cams[i].cpu().numpy(), cmap="jet", alpha=0.5)
        axes[1, i].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
