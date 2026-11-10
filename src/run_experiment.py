"""
run_experiment.py
-----------------
Run the ENTIRE study with one command:

    python src/run_experiment.py --data-dir data/ham_binary

    # laptop smoke test, no data / no download:
    python src/run_experiment.py --synthetic --no-pretrained --img-size 64 --epochs 2 --limit 256

Stages:
    P1a  clean   ResNet-50          (architecture comparison)
    P1b  clean   EfficientNet-B0
    P2   shortcut ResNet-50  rho    (injection + reliance)

Resumability:
  * Per-EPOCH checkpoints inside each stage (handled in core.py) -> a preempted
    stage resumes mid-training.
  * Per-STAGE: a finished stage writes results.json; on re-run, completed stages
    are SKIPPED (use --force to redo). So a 5-run job that dies on stage 3 picks
    up at stage 3, not stage 1.
  * Checkpoints are deleted once a stage finishes cleanly; only final_model.pth
    and results.json remain.

Everything is timestamped in runs/logs/experiment_*.log and echoed to console.
A human-readable runs/SUMMARY.md is written at the end.
"""
import argparse
import json
import os
import time

from core import Config, train_eval
from logutil import fmt_secs, setup_logger


def build_stages(args):
    common = dict(
        data_dir=args.data_dir, synthetic=args.synthetic,
        pretrained=not args.no_pretrained, img_size=args.img_size,
        epochs=args.epochs, bs=args.bs, workers=args.workers, limit=args.limit,
    )
    return [
        ("p1_resnet50", Config(phase="clean", model="resnet50",
                               out=f"{args.out}/p1_resnet50",
                               fig_dir=f"{args.viz}/p1_resnet50", **common)),
        ("p1_effnet", Config(phase="clean", model="efficientnet",
                             out=f"{args.out}/p1_effnet",
                             fig_dir=f"{args.viz}/p1_effnet", **common)),
        ("p2_resnet50", Config(phase="shortcut", model="resnet50", rho=args.rho,
                               out=f"{args.out}/p2_resnet50",
                               fig_dir=f"{args.viz}/p2_resnet50", **common)),
    ]


def write_summary(path, summary):
    lines = ["# Experiment summary\n"]
    lines.append("## Phase 1 — architecture comparison (clean data)\n")
    lines.append("| Stage | Model | AUC | Sensitivity | Specificity |")
    lines.append("|-------|-------|-----|-------------|-------------|")
    for name, res in summary.items():
        if "test_clean" in res:
            m = res["test_clean"]
            lines.append(f"| {name} | {res['config']['model']} | {m['auc']:.3f} | "
                         f"{m['sensitivity']:.3f} | {m['specificity']:.3f} |")
    lines.append("\n## Phase 2 — shortcut injection\n")
    for name, res in summary.items():
        if "test_regimes" in res:
            lines.append(f"### {name} (rho={res['config']['rho']})\n")
            lines.append("| Regime | Accuracy | AUC |")
            lines.append("|--------|----------|-----|")
            for mode, m in res["test_regimes"].items():
                lines.append(f"| {mode} | {m['acc']:.3f} | {m['auc']:.3f} |")
            lines.append(f"\n**Reliance (aligned − clean) = {res['reliance']:.3f}**  ")
            lines.append(f"**Swap drop (aligned − swapped) = {res['swap_drop']:.3f}**\n")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


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
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--viz", default="outputs/visualizations")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--force", action="store_true", help="re-run stages even if done")
    args = ap.parse_args()

    # keep synthetic smoke-test artifacts out of the real output folders, so a
    # later real run is never skipped by leftover smoke results.json files
    if args.synthetic:
        if args.out == "outputs":
            args.out = "outputs/smoke"
        if args.viz == "outputs/visualizations":
            args.viz = "outputs/visualizations/smoke"

    os.makedirs(args.out, exist_ok=True)
    logger = setup_logger(args.logs, name="experiment")
    t0 = time.time()
    logger.info("######## FULL EXPERIMENT START ########")

    stages = build_stages(args)
    summary = {}
    for name, cfg in stages:
        results_path = os.path.join(cfg.out, "results.json")
        if os.path.exists(results_path) and not args.force:
            logger.info(f"SKIP stage '{name}' (already complete: {results_path})")
            with open(results_path) as f:
                summary[name] = json.load(f)
            continue
        summary[name] = train_eval(cfg, logger)

    summary_path = os.path.join(args.out, "SUMMARY.md")
    write_summary(summary_path, summary)
    logger.info(f"wrote {summary_path}")
    logger.info(f"######## EXPERIMENT DONE in {fmt_secs(time.time()-t0)} ########")


if __name__ == "__main__":
    main()
