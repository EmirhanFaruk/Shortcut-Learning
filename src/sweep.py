"""
sweep.py  --  STEP 3: the adoption phase diagram
------------------------------------------------
Trains a shortcut model at several cue settings and plots how much the model
relies on the cue as that setting changes. Two axes you can sweep:

    --vary rho       : cue predictiveness (how often the cue agrees with the label)
    --vary saliency  : cue size (how easy the cue is to see) at fixed rho

The rho sweep answers "how predictive must a shortcut be before the model grabs
it?"; the saliency sweep probes whether *simplicity* (not just predictiveness)
drives adoption. This is the step that turns the demo into a small study.

    python src/sweep.py --vary rho --data-dir data/ham_binary
    python src/sweep.py --vary saliency --data-dir data/ham_binary
    python src/sweep.py --vary rho --synthetic --no-pretrained --img-size 64 --epochs 2 --limit 256
"""
import argparse
import csv
import os

import viz
from core import Config, train_eval
from logutil import setup_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vary", choices=["rho", "saliency"], default="rho")
    ap.add_argument("--data-dir", default="data/ham_binary")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/sweep")
    ap.add_argument("--viz", default="outputs/visualizations/sweep")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.viz, exist_ok=True)
    logger = setup_logger(args.logs, name=f"sweep_{args.vary}")

    if args.vary == "rho":
        values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.97]
        xlabel = "cue predictiveness (rho)"
    else:
        values = [0.04, 0.08, 0.12, 0.18, 0.25]      # cue size as fraction of image
        xlabel = "cue size (fraction of image)"

    rel, swap = [], []
    for v in values:
        cfg = Config(
            phase="shortcut", model="resnet50",
            data_dir=args.data_dir, synthetic=args.synthetic,
            pretrained=not args.no_pretrained, img_size=args.img_size,
            epochs=args.epochs, bs=args.bs, workers=args.workers, limit=args.limit,
            rho=v if args.vary == "rho" else 0.97,
            cue_frac=v if args.vary == "saliency" else 0.12,
            out=os.path.join(args.out, f"{args.vary}_{v}"),
            fig_dir=os.path.join(args.viz, f"{args.vary}_{v}"),
        )
        import json
        rp = os.path.join(cfg.out, "results.json")
        if os.path.exists(rp) and not args.force:
            logger.info(f"SKIP {cfg.out} (done)")
            res = json.load(open(rp))
        else:
            res = train_eval(cfg, logger)
        rel.append(res["reliance"]); swap.append(res["swap_drop"])

    # save table (outputs) + plot (visualizations)
    with open(os.path.join(args.out, f"sweep_{args.vary}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow([args.vary, "reliance", "swap_drop"])
        for v, r, s in zip(values, rel, swap):
            w.writerow([v, r, s])
    viz.phase_diagram(values, rel, swap,
                      os.path.join(args.viz, f"phase_diagram_{args.vary}.png"),
                      xlabel=xlabel)
    logger.info(f"wrote CSV to {args.out}, phase diagram to {args.viz}")


if __name__ == "__main__":
    main()
