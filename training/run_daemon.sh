#!/usr/bin/env bash
# pm2 entrypoint for indefinite self-play training. From the project root
# (~/Workspace/quoridor-ai-engine):
#   pm2 start repo/training/run_daemon.sh --name quoridor-train --kill-timeout 20000 \
#     --output logs/train.out.log --error logs/train.err.log
#
# The orchestrator runs the trainer in this process and spawns the standalone
# Rust self-play worker (selfplay_rs serve: MCTS workers + AOTInductor bf16
# forward). Every snapshot it launches evaluate.py detached. Resumes
# $OUT_DIR/latest.pt if present.
#
# Build both native pieces first (after every pull that touches them):
#   repo/training/quoridor_rs/build.sh   (pyo3 extension: Board, MatchEngine)
#   repo/training/selfplay_rs/build.sh   (self-play binary)
#
# Every file the run writes — checkpoints, compile caches, temp files — stays
# under $ROOT.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${QUORIDOR_ROOT:-$(dirname "$REPO_DIR")}"
VENV="$ROOT/venv"
OUT_DIR="${OUT_DIR:-$ROOT/runs/run1}"

export TMPDIR="$ROOT/cache/tmp"
export XDG_CACHE_HOME="$ROOT/cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor"
export TRITON_CACHE_DIR="$ROOT/cache/triton"
export CUDA_CACHE_PATH="$ROOT/cache/nv"
export PIP_CACHE_DIR="$ROOT/cache/pip"
export MPLCONFIGDIR="$ROOT/cache/mpl"
mkdir -p "$TMPDIR" "$OUT_DIR"

# One-time warm start: set INIT_FROM to seed a fresh run. Ignored once latest.pt exists.
INIT_ARGS=()
if [[ -n "${INIT_FROM:-}" ]]; then
  INIT_ARGS=(--init-from "$INIT_FROM")
fi

cd "$REPO_DIR/training"
exec "$VENV/bin/python" orchestrator.py \
  --out-dir "$OUT_DIR" \
  --snapshot-every "${SNAPSHOT_EVERY:-1h}" \
  --save-latest-every 300s \
  --sims "${SIMS:-200}" \
  --n-threads "${N_THREADS:-16}" \
  --eval-args "${EVAL_ARGS:-}" \
  "${INIT_ARGS[@]}"
