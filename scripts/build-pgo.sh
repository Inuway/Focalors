#!/usr/bin/env bash
#
# Two-pass profile-guided optimization (PGO) build for focalors.
#
# Pass 1 builds an instrumented binary, runs it through a representative
# search workload, and captures profile data. Pass 2 rebuilds with the
# captured profile so the optimizer can specialize hot paths.
#
# Output: a PGO-optimized binary at target/release/focalors (overwrites the
# regular release build). PGO is purely a compiler optimization; no source
# changes, no semantic changes — search results are identical to a regular
# release build, just typically 10–20% faster.
#
# Requires: `rustup component add llvm-tools-preview`
#
# Usage:
#   ./scripts/build-pgo.sh

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PROFDATA_DIR="$(pwd)/target/pgo-data"
PROFDATA_FILE="$PROFDATA_DIR/merged.profdata"

# Locate llvm-profdata from the active rustup toolchain.
HOST="$(rustc -vV | sed -n 's|host: ||p')"
LLVM_PROFDATA="$(rustc --print sysroot)/lib/rustlib/${HOST}/bin/llvm-profdata"

if [[ ! -x "$LLVM_PROFDATA" ]]; then
    echo "Error: llvm-profdata not found at $LLVM_PROFDATA"
    echo
    echo "Install the llvm-tools-preview component:"
    echo "    rustup component add llvm-tools-preview"
    exit 1
fi

# Clean prior profile data so stale .profraw files don't bias the merge.
rm -rf "$PROFDATA_DIR"
mkdir -p "$PROFDATA_DIR"

echo "=== Pass 1: instrumented build ==="
RUSTFLAGS="-Cprofile-generate=$PROFDATA_DIR" \
    cargo build --release --bin focalors

echo "=== Pass 1: running profiling workload ==="
# Representative workload — three searches at depth 12 covering opening,
# tactical, and middlegame patterns. Enough variety to give the optimizer
# good signal without taking forever.
./target/release/focalors uci > /dev/null 2>&1 << 'EOF'
uci
isready
position startpos
go depth 12
position fen r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4
go depth 12
position fen r2q1rk1/ppp1bppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP1QPPP/R1B2RK1 w - - 0 8
go depth 12
quit
EOF

echo "=== Merging profile data ==="
"$LLVM_PROFDATA" merge -o "$PROFDATA_FILE" "$PROFDATA_DIR"

echo "=== Pass 2: PGO-optimized build ==="
RUSTFLAGS="-Cprofile-use=$PROFDATA_FILE" \
    cargo build --release --bin focalors

echo
echo "Done. PGO binary at target/release/focalors"
echo "Verify with: cargo test --release --locked"
