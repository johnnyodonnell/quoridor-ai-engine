#!/usr/bin/env bash
# Build the quoridor_rs wheel (release) and install it into the training venv.
# Run on the aarch64 box; the wheel is arch-specific so it must be built there.
#   ./build.sh
# Everything (cargo registry, build output) stays under the project root.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${QUORIDOR_ROOT:-$HOME/Workspace/quoridor-ai-engine}"
VENV="${VENV:-$ROOT/venv}"
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_HOME="$ROOT/cache/cargo"
export CARGO_TARGET_DIR="$ROOT/cache/target"
export TMPDIR="$ROOT/cache/tmp"
export PIP_CACHE_DIR="$ROOT/cache/pip"
mkdir -p "$TMPDIR"

cd "$HERE"
rm -f "$ROOT"/cache/wheels/quoridor_rs-*.whl
"$VENV/bin/maturin" build --release -i "$VENV/bin/python" --out "$ROOT/cache/wheels"
"$VENV/bin/pip" install --force-reinstall --no-deps "$ROOT"/cache/wheels/quoridor_rs-*.whl
echo "installed quoridor_rs into $VENV"
