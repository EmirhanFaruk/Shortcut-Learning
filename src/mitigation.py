"""
mitigation.py  --  STEP 4: can we break the shortcut?
-----------------------------------------------------
Trains the SAME biased setup twice -- once normally, once with cue-randomizing
augmentation (the cue is added at random during training so it no longer predicts
the label) -- and compares reliance. A successful mitigation shows reliance drop
toward zero and the clean-test accuracy recover.

Honest framing for your write-up: this works because we *know* the cue and can
randomize it. The genuinely open problem is mitigating a shortcut you have NOT
identified -- which is exactly what the larger research project tackles.

    python src/mitigation.py --data-dir data/ham_binary
    python src/mitigation.py --synthetic --no-pretrained --img-size 64 --epochs 2 --limit 256
"""
import argparse
import json
import os

import viz
from core import Config, train_eval
from logutil import setup_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/ham_binary")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--rho", type=float, default=0.95)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/mitigation")
    ap.add_argument("--viz", default="outputs/visualizations/mitigation")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.viz, exist_ok=True)
    logger = setup_logger(args.logs, name="mitigation")

    common = dict(
        phase="shortcut", model="resnet50", rho=args.rho,
        data_dir=args.data_dir, synthetic=args.synthetic,
        pretrained=not args.no_pretrained, img_size=args.img_size,
        epochs=args.epochs, bs=args.bs, workers=args.workers, limit=args.limit,
    )
    arms = {
        "baseline": Config(mitigate=False, out=f"{args.out}/baseline",
                           fig_dir=f"{args.viz}/baseline", **common),
        "mitigated": Config(mitigate=True, out=f"{args.out}/mitigated",
                            fig_dir=f"{args.viz}/mitigated", **common),
    }

    res = {}
    for name, cfg in arms.items():
        rp = os.path.join(cfg.out, "results.json")
        if os.path.exists(rp) and not args.force:
            logger.info(f"SKIP {name} (done)")
            res[name] = json.load(open(rp))
        else:
            res[name] = train_eval(cfg, logger)

    viz.grouped_bars(
        labels=["reliance", "swap drop", "clean acc"],
        series={
            "baseline": [res["baseline"]["reliance"], res["baseline"]["swap_drop"],
                         res["baseline"]["test_regimes"]["none"]["acc"]],
            "mitigated": [res["mitigated"]["reliance"], res["mitigated"]["swap_drop"],
                          res["mitigated"]["test_regimes"]["none"]["acc"]],
        },
        path=os.path.join(args.viz, "mitigation.png"),
        ylabel="value", title="Cue-randomizing augmentation vs baseline",
    )
    logger.info(f"baseline reliance={res['baseline']['reliance']:.3f}  "
                f"mitigated reliance={res['mitigated']['reliance']:.3f}")
    logger.info(f"wrote mitigation figure to {args.viz}/mitigation.png")


if __name__ == "__main__":
    main()
