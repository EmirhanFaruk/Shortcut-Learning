"""
train.py
--------
CLI for a SINGLE stage. (To run the whole study in one go, use run_experiment.py.)

    python src/train.py --phase clean    --model resnet50 --data-dir data/ham_binary
    python src/train.py --phase shortcut --model resnet50 --rho 0.95 --data-dir data/ham_binary
    python src/train.py --phase shortcut --synthetic --no-pretrained --limit 256   # smoke test
"""
import argparse

from core import Config, train_eval
from logutil import setup_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["clean", "shortcut"], default="clean")
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--data-dir", default="data/ham_binary")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rho", type=float, default=0.95)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="outputs/single")
    ap.add_argument("--viz", default="outputs/visualizations/single")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    cfg = Config(
        phase=args.phase, model=args.model, data_dir=args.data_dir,
        synthetic=args.synthetic, rho=args.rho, img_size=args.img_size,
        epochs=args.epochs, bs=args.bs, lr=args.lr, workers=args.workers,
        out=args.out, fig_dir=args.viz, seed=args.seed, limit=args.limit,
        pretrained=not args.no_pretrained,
    )
    logger = setup_logger(args.logs, name="train")
    train_eval(cfg, logger)


if __name__ == "__main__":
    main()
