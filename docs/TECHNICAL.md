# Technical Overview

This document is the engine-side reference for Focalors: how the code is laid out, why the evaluation is split in two, and how the NNUE training loop works. The user-facing intro lives in the [README](../README.md); contributor workflow lives in [CONTRIBUTING.md](../CONTRIBUTING.md).

## Why two evaluations

Focalors carries both an NNUE network and a hand-crafted evaluation (HCE), and both are live during normal use. NNUE is what the search calls during play — it's stronger and faster than HCE on its own. HCE stays in the codebase because its outputs are *interpretable*: each component (material, pawn structure, king safety, mobility, …) is a real number you can read and reason about. When the app explains a mistake, it diffs HCE components before and after the move and surfaces the largest signed swings. The result is play that uses the strong engine and feedback that uses the readable one — without bolting on a separate analysis backend.

## Source layout

```text
src/
├── main.rs        entry point; routes to subcommands
├── gui.rs         egui desktop app (play, review, puzzles, stats)
├── analysis.rs    post-game review and explanation generation
├── search.rs      alpha-beta search with iterative deepening
├── eval.rs        hand-crafted evaluation
├── nnue/          NNUE inference (incremental accumulators, AVX2)
├── trainer.rs     NNUE training loop
├── selfplay.rs    self-play data generation
├── tuning.rs      HCE weight tuning
├── db.rs          SQLite persistence
├── pgn.rs         PGN import/export
├── puzzles.rs     puzzle extraction and trainer
├── openings.rs    opening name lookup
├── strength.rs    skill levels and adaptive difficulty
├── uci.rs         UCI protocol
├── board.rs       bitboard position state
├── movegen.rs     legal move generation
├── attacks.rs     attack tables
├── moves.rs       packed move encoding
├── tt.rs          transposition table
├── zobrist.rs     position hashing
└── types.rs       shared types
```

## Runtime modes

The compiled binary takes a subcommand:

```bash
./target/release/focalors gui        # desktop app
./target/release/focalors uci        # standard UCI engine
./target/release/focalors            # same as uci
```

`gui` is the everyday mode. `uci` lets external front-ends (Arena, Cute Chess, etc.) drive the engine.

## Default network

The shipped NNUE net is embedded into the binary at build time, so a normal install doesn't need to fetch anything separately. Promoting a newly trained net to be the default rebuilds the binary against that file (see below).

## NNUE training workflow

The training pipeline is three commands. Self-play generates positions:

```bash
cargo run --release -- selfplay 100000 nets/gen2v2-data.bin --nnue nets/gen1v2.nnue
```

Then training turns positions into a network. `--mix` blends two datasets, `--resume` warm-starts from an existing net so you don't relearn from scratch:

```bash
cargo run --release -- train nets/gen1v2-data.bin \
  --data nets/gen2v2-data.bin \
  --mix 0.3,0.7 \
  --resume nets/gen1v2.nnue \
  --epochs 30 \
  --output nets/gen2v2.nnue
```

Promoting installs a trained net as the new default and rebuilds:

```bash
cargo run --release -- promote nets/gen2v2.nnue
cargo build --release
```

The HCE side has its own tuner, used independently:

```bash
cargo run --release -- tune dataset.txt
```
