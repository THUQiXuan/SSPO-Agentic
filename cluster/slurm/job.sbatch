#!/bin/bash
# Generic single-node Slurm wrapper used for every stage (adapt account / partition / constraint to your cluster):
#   SSPO_ROOT=/path/to/SSPO-Agentic SSPO_ENV="source /path/to/venv/bin/activate" RBD_MODEL=qwen3_4b \
#     sbatch --gpus-per-node=1 --constraint=h200 cluster/slurm/job.sbatch "bash scripts/train.sh ..."
# Every command in this repository is resumable; the job requeues itself before the time limit or on preemption.
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@900
set -uo pipefail
trap 'echo "[job] USR1: requeue $(date)"; scontrol requeue ${SLURM_JOB_ID}; exit 0' USR1
cd "${SSPO_ROOT:?set SSPO_ROOT to the repository root}"
eval "${SSPO_ENV:-true}"                       # activates the Python environment with vLLM, torch, peft
export PYTHON="${PYTHON:-python}"
export RBD_MODEL_PATH="${RBD_MODEL_PATH:-}"     # optional local snapshot; otherwise the HF id from configs/models
export RBD_PORT_BASE=$((20000 + ${SLURM_JOB_ID:-0} % 15000)) RBD_WORKDIR=/tmp/sspo-${SLURM_JOB_ID:-local}
export no_proxy="127.0.0.1,localhost,${no_proxy:-}" NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
echo "job=${SLURM_JOB_ID:-} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-none} $(date)"
bash -c "$1" &
wait $!
rc=$?
echo "[job] exit $rc $(date)"
exit $rc
