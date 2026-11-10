#!/usr/bin/env bash
#
# run_all.sh -- set up the environment and run the whole pipeline in one command.
#
# By default it uses the prepared dataset at data/ham_binary. If that folder
# doesn't exist yet but a raw HAM10000/ folder does, it prepares it for you first.
#
# Usage:
#   ./run_all.sh                       # full pipeline on data/ham_binary (auto-prepares if needed)
#   ./run_all.sh --smoke               # quick synthetic test, no data required
#   ./run_all.sh --epochs 5 --skip-sweep
#   ./run_all.sh --data-dir some/other/path
#
# Flags:
#   --smoke            run a tiny synthetic test instead of the real pipeline
#   --data-dir PATH    prepared binary dataset (default data/ham_binary)
#   --raw-dir PATH     raw HAM10000 folder to auto-prepare from (default HAM10000)
#   --epochs N         epochs per training (default 8)
#   --skip-sweep       skip Step 3 (the rho sweep is the slowest part)
#   --skip-mitigation  skip Step 4
#
set -euo pipefail
cd "$(dirname "$0")"

PY=python3
SMOKE=0
DATA_DIR="data/ham_binary"
RAW_DIR="HAM10000"
EPOCHS=8
RUN_SWEEP=1
RUN_MITIGATION=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)           SMOKE=1; shift;;
    --data-dir)        DATA_DIR="$2"; shift 2;;
    --raw-dir)         RAW_DIR="$2"; shift 2;;
    --epochs)          EPOCHS="$2"; shift 2;;
    --skip-sweep)      RUN_SWEEP=0; shift;;
    --skip-mitigation) RUN_MITIGATION=0; shift;;
    -h|--help)         sed -n '2,21p' "$0"; exit 0;;
    *) echo "unknown argument: $1 (use --help)"; exit 1;;
  esac
done

# ---------------------------------------------------------------------------
# 1. environment
# ---------------------------------------------------------------------------
if [[ ! -d .venv ]]; then
  echo "[setup] creating virtual environment (.venv) with $PY ..."
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
echo "[setup] installing requirements ..."
python -m pip install -r requirements.txt

# ---------------------------------------------------------------------------
# 2. smoke test (no data)
# ---------------------------------------------------------------------------
if [[ "$SMOKE" -eq 1 ]]; then
  echo "[run] synthetic smoke test"
  python src/run_experiment.py --synthetic --no-pretrained --img-size 64 --epochs 2 --limit 256
  echo "[done] smoke test complete."
  exit 0
fi

# ---------------------------------------------------------------------------
# 3. ensure the dataset exists (auto-prepare from raw HAM10000 if needed)
# ---------------------------------------------------------------------------
if [[ ! -d "$DATA_DIR" ]]; then
  if [[ -d "$RAW_DIR" ]]; then
    echo "[data] $DATA_DIR not found -> preparing it from $RAW_DIR ..."
    python scripts/prepare_ham10000.py --images-dir "$RAW_DIR" \
        --metadata "$RAW_DIR/HAM10000_metadata.csv" --out "$DATA_DIR" --link
  else
    echo "[error] neither $DATA_DIR nor raw $RAW_DIR found."
    echo "        Download HAM10000 into ./$RAW_DIR, or pass --raw-dir / --data-dir, or use --smoke."
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# 4. run the pipeline
# ---------------------------------------------------------------------------
echo "[run] Step 1+2: classifier comparison + shortcut demo"
python src/run_experiment.py --data-dir "$DATA_DIR" --epochs "$EPOCHS"

if [[ "$RUN_SWEEP" -eq 1 ]]; then
  echo "[run] Step 3: adoption phase diagram (rho sweep)"
  python src/sweep.py --vary rho --data-dir "$DATA_DIR" --epochs "$EPOCHS"
fi

if [[ "$RUN_MITIGATION" -eq 1 ]]; then
  echo "[run] Step 4: mitigation"
  python src/mitigation.py --data-dir "$DATA_DIR" --epochs "$EPOCHS"
fi

echo "[run] generating report ..."
python src/report.py

echo "[done] pipeline complete."
echo "       read this first -> outputs/REPORT.md"
echo "       models -> outputs/models/   figures -> outputs/visualizations/   logs -> logs/"
