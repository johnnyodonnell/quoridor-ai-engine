#!/usr/bin/env bash
# Build the selfplay_rs binaries (release: selfplay_rs + evaluate_rs) against the
# project venv's PyTorch libtorch. Run on the aarch64 training box; output lands
# in selfplay_rs/target/release/, where orchestrator.py looks for it.
#   ./build.sh
# Everything (cargo registry, temp files) stays under the project root.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${QUORIDOR_ROOT:-$HOME/Workspace/quoridor-ai-engine}"
VENV="${VENV:-$ROOT/venv}"
export PATH="$VENV/bin:$HOME/.cargo/bin:$PATH"
export CARGO_HOME="$ROOT/cache/cargo"
export TMPDIR="$ROOT/cache/tmp"
# build.rs and torch-sys (LIBTORCH_USE_PYTORCH=1, see .cargo/config.toml) both
# ask this interpreter where torch and the CUDA runtime live.
export SELFPLAY_PYTHON="$VENV/bin/python"
export PYTHON_SYS_EXECUTABLE="$VENV/bin/python"
mkdir -p "$TMPDIR"

cd "$HERE"
cargo build --release
echo "built $HERE/target/release/{selfplay_rs,evaluate_rs}"
