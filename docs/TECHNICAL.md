# Technical Overview

This document covers the architecture, engine internals, and development-oriented commands behind Focalors.

## System Overview

Focalors is an offline-first chess engine and learning application written in Rust. The same core engine is used across three user-facing entry points:

- the desktop GUI for local play, review, puzzles, and stats
- standard UCI mode for engine integration
- a headless Lichess bot mode

Beyond those runtime modes, the repository also includes tooling for self-play generation, NNUE training, network promotion, and hand-crafted evaluation tuning.

## Evaluation Design

Focalors uses a dual-evaluation design.

- NNUE is used during search for playing strength.
- HCE is kept alongside it for interpretation, so analysis can explain why a move or position was good or bad.

That split is the core design choice of the project: strong play and readable feedback in the same codebase.

## Source Layout

At a high level the project is organized like this:

```text
src/
|- gui.rs        desktop interface
|- analysis.rs   post-game review and coaching logic
|- db.rs         SQLite persistence
|- search.rs     alpha-beta search
|- eval.rs       hand-crafted evaluation for explanations
|- nnue/         neural network inference
|- lichess.rs    Lichess integration
|- uci.rs        UCI protocol support
|- trainer.rs    NNUE training
`- selfplay.rs   self-play data generation
```

Additional core modules in the repository include:

- move generation, attacks, and board state handling
- opening, puzzle, and statistics flows
- transposition tables and zobrist hashing
- tuning and strength-management helpers

## Runtime Modes

The most common runtime commands are:

```bash
cargo run --release -- gui
cargo run --release -- uci
cargo run --release
cargo run --release -- lichess
```

- `gui` launches the egui desktop application.
- `uci` runs the engine with the standard UCI protocol.
- the default command without a subcommand is equivalent to `uci`.
- `lichess` runs the headless bot integration.

If you want to use Lichess mode, create a `.env` file with your bot token:

```bash
printf "LICHESS_TOKEN=your_token_here\n" > .env
cargo run --release -- lichess
```

## Default Network

The default NNUE net is embedded directly into the binary, so normal use does not require downloading an extra model file.

## Engine Development Commands

The most common engine-development commands are:

```bash
# Generate self-play data
cargo run --release -- selfplay 100000 nets/gen2v2-data.bin --nnue nets/gen1v2.nnue

# Train or continue an NNUE network
cargo run --release -- train nets/gen1v2-data.bin \
  --data nets/gen2v2-data.bin \
  --mix 0.3,0.7 \
  --resume nets/gen1v2.nnue \
  --epochs 30 \
  --output nets/gen2v2.nnue

# Promote a trained network as the shipped default
cargo run --release -- promote nets/gen2v2.nnue

# Tune the hand-crafted evaluation
cargo run --release -- tune dataset.txt
```

For contributor-oriented build, test, and training notes, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Current Technical Direction

The engine side already includes:

- bitboard move generation with castling, promotion, and en passant
- alpha-beta search with iterative deepening, TT, null move pruning, LMR, SEE, singular extensions, and time management
- pure-Rust NNUE inference with incremental accumulators and AVX2 support
- a hand-crafted evaluation with multiple explained components for analysis
- self-play and training tooling for new NNUE generations

The application side already includes:

- a compact egui-based desktop interface
- player profile storage and saved local game history
- analysis review, accuracy tracking, and statistics views
- puzzle generation and solving flows
- opening tracking and coaching summaries

## Related Docs

- [README.md](../README.md) is the public showcase and quick-start entry point.
- [CONTRIBUTING.md](../CONTRIBUTING.md) covers contribution workflow and development expectations.