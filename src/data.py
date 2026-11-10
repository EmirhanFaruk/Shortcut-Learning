"""
data.py
-------
Two data sources behind one interface:

1. REAL  -- a binary-organised image folder, e.g. HAM10000 collapsed to
            malignant vs benign:
                data_dir/
                    benign/    *.jpg
                    malignant/ *.jpg
            (benign -> label 0, malignant -> label 1)

2. SYNTH -- generated on the fly. Each image has a *real* class signal (a bright
            central blob whose intensity depends on the label) PLUS the optional
            injected cue. This lets the whole pipeline run on a laptop with no
            download, and -- crucially -- gives the model a genuine feature it
            COULD use, so when it prefers the cue instead, that means something.

Both apply the cue according to (mode, rho) from cue.py.
"""
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from cue import add_cue, cue_decision


class LesionDataset(Dataset):
    def __init__(self, data_dir, split, img_size, cue_mode, rho, seed=0, cue_frac=0.12):
        self.img_size = img_size
        self.cue_mode = cue_mode
        self.rho = rho
        self.cue_frac = cue_frac
        self.rng = random.Random(seed + {"train": 1, "val": 2, "test": 3}[split])
        self.to_tensor = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # -> [0,1]
        ])
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.normalize = norm

        self.samples = []  # list of (path, label)
        base = Path(data_dir) / split if data_dir else None
        if base and base.exists():
            for label, name in enumerate(["benign", "malignant"]):
                for p in (base / name).glob("*"):
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        self.samples.append((p, label))
            self.synthetic = False
        else:
            # synthetic fallback
            n = {"train": 1500, "val": 300, "test": 600}[split]
            self.synthetic = True
            self.samples = [(None, i % 2) for i in range(n)]

    def __len__(self):
        return len(self.samples)

    def _synth_image(self, label, idx):
        g = torch.Generator().manual_seed(1000 * idx + label)
        img = 0.25 + 0.05 * torch.rand(3, self.img_size, self.img_size, generator=g)
        # real signal: central blob, brighter for malignant (label 1)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, self.img_size),
            torch.linspace(-1, 1, self.img_size),
            indexing="ij",
        )
        blob = torch.exp(-((yy ** 2 + xx ** 2) / 0.15))
        intensity = 0.55 if label == 1 else 0.30
        for ch in range(3):
            img[ch] = (img[ch] + intensity * blob).clamp(0, 1)
        return img

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        if self.synthetic:
            img = self._synth_image(label, idx)
        else:
            img = self.to_tensor(Image.open(path).convert("RGB"))

        add = cue_decision(label, self.cue_mode, self.rho, self.rng)
        if add:
            img = add_cue(img, frac=self.cue_frac)

        return self.normalize(img), label
