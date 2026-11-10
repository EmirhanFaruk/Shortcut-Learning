#!/bin/bash
#SBATCH --job-name=lesion-shortcut
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm_%j.log

# --- adjust for your cluster -----------------------------------------------
# A single RTX 2080 / 3080 node is plenty; a V100 (ppti-gpu-4) is overkill.
# Inspect with `sinfo`, then set e.g.:
#   #SBATCH --partition=<your_partition>
#   #SBATCH --nodelist=ppti-14-302-01      # pin a 3080 if you like
# ---------------------------------------------------------------------------

module load cuda 2>/dev/null || true
source ~/venvs/lesion/bin/activate 2>/dev/null || true

# The whole study in one command. If the job is preempted / hits the time
# limit, just `sbatch` it again: finished stages are skipped and an
# interrupted stage resumes from its last epoch checkpoint.
python3 src/run_experiment.py --epochs 8   # uses data/ham_binary by default
