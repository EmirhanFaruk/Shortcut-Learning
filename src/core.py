"""
core.py
-------
The reusable engine. `train_eval(cfg, logger)` runs one stage end-to-end:
build -> (resume if interrupted) -> train with per-epoch checkpoints ->
promote best model to final_model.pth -> delete checkpoints -> evaluate ->
Grad-CAM -> write results.json. Everything is logged with timestamps.

Checkpointing granularity is per-EPOCH, on purpose: epochs here are short, so
per-batch checkpointing would cost throughput for no real safety benefit. The
checkpoint lets a preempted / time-limited cluster job resume where it left off;
once the stage finishes cleanly only final_model.pth survives.
"""
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import LesionDataset
from gradcam import GradCAM, save_overlays
from logutil import Timer, fmt_secs
from metrics import classification_metrics, reliance
from models import build_model


@dataclass
class Config:
    phase: str = "clean"           # 'clean' | 'shortcut'
    model: str = "resnet50"        # 'resnet50' | 'efficientnet'
    data_dir: str = "data/ham_binary"
    synthetic: bool = False
    rho: float = 0.95
    img_size: int = 224
    epochs: int = 8
    bs: int = 32
    lr: float = 3e-4
    workers: int = 4
    out: str = "outputs/run"       # results + checkpoints
    fig_dir: str = "outputs/visualizations/run"   # figures (inside outputs)
    seed: int = 0
    limit: int = 0
    pretrained: bool = True
    cue_frac: float = 0.12         # cue size (fraction of image height) -- saliency knob
    mitigate: bool = False         # cue-randomizing augmentation at train time


def make_loader(cfg, split, cue_mode, rho):
    data_dir = None if cfg.synthetic else cfg.data_dir
    ds = LesionDataset(data_dir, split, cfg.img_size, cue_mode, rho, cue_frac=cfg.cue_frac)
    if cfg.limit and cfg.limit < len(ds):
        ds.samples = ds.samples[:cfg.limit]
    return DataLoader(ds, batch_size=cfg.bs, shuffle=(split == "train"),
                      num_workers=cfg.workers, pin_memory=True)


@torch.no_grad()
def predict(model, dl, device):
    model.eval()
    ys, ps = [], []
    for x, y in dl:
        prob = torch.softmax(model(x.to(device)), dim=1)[:, 1]
        ys.append(y.numpy())
        ps.append(prob.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


@torch.no_grad()
def evaluate(model, dl, device):
    y_true, y_prob = predict(model, dl, device)
    return classification_metrics(y_true, y_prob)


def train_eval(cfg, logger):
    t_stage = time.time()
    logger.info("=" * 70)
    logger.info(f"STAGE phase={cfg.phase} model={cfg.model} rho={cfg.rho} out={cfg.out}")
    logger.info(f"config: {json.dumps(asdict(cfg))}")

    os.makedirs(cfg.out, exist_ok=True)
    ckpt_dir = os.path.join(cfg.out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    latest = os.path.join(ckpt_dir, "latest.pth")
    # final model goes in a dedicated outputs/models/<stage> tree so the
    # results.json files are easy to find without big .pth files alongside them
    if cfg.out.startswith("outputs/"):
        model_dir = os.path.join("outputs/models", cfg.out[len("outputs/"):])
    else:
        model_dir = os.path.join(cfg.out, "model")
    os.makedirs(model_dir, exist_ok=True)
    final_model = os.path.join(model_dir, "final_model.pth")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}")

    model, target_layer = build_model(cfg.model, pretrained=cfg.pretrained)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    crit = nn.CrossEntropyLoss()

    # ---- resume from an interrupted run, if a checkpoint exists ----
    start_epoch, best_acc, best_state = 0, -1.0, None
    if os.path.exists(latest):
        ck = torch.load(latest, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        best_acc = ck["best_acc"]
        best_state = ck.get("best_state")
        logger.info(f"RESUME from epoch {start_epoch} (best_acc={best_acc:.3f})")

    if cfg.phase == "clean":
        train_cue, train_rho = "none", 0.0
    elif cfg.mitigate:
        train_cue, train_rho = "random", 0.0     # decorrelate the cue from the label
        logger.info("MITIGATION on: cue-randomizing augmentation at train time")
    else:
        train_cue, train_rho = "train", cfg.rho
    dl_train = make_loader(cfg, "train", train_cue, train_rho)
    dl_val = make_loader(cfg, "val", train_cue, train_rho)

    # ---- training loop with per-epoch checkpointing ----
    history = []
    for ep in range(start_epoch, cfg.epochs):
        t0 = time.time()
        model.train()
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        m = evaluate(model, dl_val, device)
        history.append({"epoch": ep, "acc": m["acc"], "auc": m["auc"]})
        improved = m["acc"] > best_acc
        if improved:
            best_acc = m["acc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep, "best_acc": best_acc, "best_state": best_state}, latest)
        logger.info(f"epoch {ep+1:2d}/{cfg.epochs}  val_acc={m['acc']:.3f}  "
                    f"auc={m['auc']:.3f}  ({time.time()-t0:.1f}s)"
                    f"{'  <-- best, checkpoint saved' if improved else '  checkpoint saved'}")

    # ---- promote best to final model, then delete checkpoints ----
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), final_model)
    logger.info(f"saved final model (best val_acc={best_acc:.3f}) -> {final_model}")
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    logger.info(f"deleted checkpoints in {ckpt_dir} (training complete)")

    # ---- evaluation ----
    import viz
    fig_dir = cfg.fig_dir
    os.makedirs(fig_dir, exist_ok=True)
    results = {"config": asdict(cfg), "best_val_acc": best_acc, "history": history}

    if history:
        viz.training_curves(history, os.path.join(fig_dir, "training_curves.png"),
                            title=f"({cfg.model})")

    if cfg.phase == "clean":
        dl_test = make_loader(cfg, "test", "none", 0.0)
        y_true, y_prob = predict(model, dl_test, device)
        m = classification_metrics(y_true, y_prob)
        results["test_clean"] = m
        results["preds"] = {"none": {"y_true": y_true.astype(int).tolist(),
                                     "y_pred": (y_prob >= .5).astype(int).tolist()}}
        logger.info(f"TEST clean: {json.dumps({k: round(v,3) for k,v in m.items()})}")
        viz.roc(y_true, y_prob, os.path.join(fig_dir, "roc.png"), title=f"({cfg.model})")
        viz.confusion(y_true, (y_prob >= .5).astype(int),
                      os.path.join(fig_dir, "confusion.png"), title=f"({cfg.model})")
    else:
        regimes = {}
        preds = {}
        for mode in ["none", "aligned", "swapped"]:
            y_true, y_prob = predict(model, make_loader(cfg, "test", mode, 0.0), device)
            regimes[mode] = classification_metrics(y_true, y_prob)
            preds[mode] = {"y_true": y_true.astype(int).tolist(),
                           "y_pred": (y_prob >= .5).astype(int).tolist()}
            logger.info(f"TEST [{mode:7s}]: "
                        f"{json.dumps({k: round(v,3) for k,v in regimes[mode].items()})}")
        r = reliance(regimes["aligned"]["acc"], regimes["none"]["acc"])
        swap_drop = regimes["aligned"]["acc"] - regimes["swapped"]["acc"]
        results["test_regimes"] = regimes
        results["preds"] = preds
        results["reliance"] = r
        results["swap_drop"] = swap_drop
        logger.info(f">>> RELIANCE={r:.3f}  SWAP_DROP={swap_drop:.3f}")
        viz.regime_bars(regimes, os.path.join(fig_dir, "regime_bars.png"),
                        reliance=r, swap_drop=swap_drop)
        # show a few example inputs with the cue
        ds = make_loader(cfg, "test", "aligned", 0.0).dataset
        imgs = [ds[i][0] for i in range(min(6, len(ds)))]
        labs = [ds[i][1] for i in range(min(6, len(ds)))]
        viz.cue_samples(imgs, labs, [l == 1 for l in labs],
                        os.path.join(fig_dir, "cue_samples.png"))

    # ---- Grad-CAM snapshot ----
    with Timer(logger, "grad-cam overlays"):
        cam = GradCAM(model, target_layer)
        vis_mode = "none" if cfg.phase == "clean" else "aligned"
        x, y = next(iter(make_loader(cfg, "test", vis_mode, 0.0)))
        x = x[:8].to(device)
        cams = cam(x)
        preds = model(x).argmax(1).cpu().numpy()
        out_png = os.path.join(fig_dir, f"gradcam_{cfg.phase}_{cfg.model}.png")
        save_overlays(x, cams, y[:8].numpy(), preds, out_png)
    results["gradcam"] = out_png

    with open(os.path.join(cfg.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"STAGE complete in {fmt_secs(time.time()-t_stage)} -> {cfg.out}/results.json")
    return results
