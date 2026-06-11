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
├── trainer.rs     NNUE training loop (CPU)
├── trainer_gpu.rs GPU training via Burn (optional `gpu-training` feature)
├── selfplay.rs    self-play data generation
├── selfmatch.rs   engine-vs-engine match runner for strength benchmarking
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
./target/release/focalors            # same as gui (double-click behavior)
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

There is also a GPU trainer (`train-gpu`, built on Burn) behind the optional `gpu-training` Cargo feature. It accepts the same flags, reads the same data files, and exports the same `.nnue` format — a drop-in alternative when training on a GPU. Build it with `cargo build --release --features gpu-training`; details in [GPU_TRAINING.md](GPU_TRAINING.md).

Promoting installs a trained net as the new default and rebuilds:

```bash
cargo run --release -- promote nets/gen2v2.nnue
cargo build --release
```

The HCE side has its own tuner, used independently:

```bash
cargo run --release -- tune dataset.txt
```

## Measuring engine strength

After training a new net the obvious question is: did it actually get stronger? A low training loss is a necessary but not sufficient condition — wrong-sign targets, overfitting, or mode collapse can produce a net that fits the dataset well but plays worse. The `selfmatch` subcommand answers the question directly: play N games between two engine configurations and report W/L/D, elo delta vs the standard logistic, a 95% confidence interval, and LOS (likelihood of superiority).

```bash
./target/release/focalors selfmatch 100 --challenger-net nets/gen2v2.nnue
```

This loads the embedded default net into Engine A and `nets/gen2v2.nnue` into Engine B (via an alternate-net slot in `nnue/network.rs`), then plays 100 games at fixed depth. All elo numbers are reported from Engine A's perspective — **positive elo means Engine A (the embedded default) is stronger; negative means the challenger is stronger**.

Options:

| Flag | Default | Notes |
|---|---|---|
| `--depth N` | `8` | Fixed search depth per move. Higher depth ⇒ stronger play ⇒ more meaningful signal, at the cost of time. ~1 min/100 games at depth 6, a few minutes at depth 8, ~10 min at depth 10. |
| `--challenger-net PATH` | (none) | NNUE file to load into Engine B. If omitted, both sides use the embedded default — useful only as an infrastructure smoke test. |
| `--seed N` | wall-clock | Reproducible openings. Pass the same seed to replay the exact same set of games. |
| `--random-plies N` | `8` | Random plies played at the start of each game from startpos to diversify openings. |
| `--max-moves N` | `200` | Cap on game length. Reached games are scored as draws. |
| `--threads N` | all cores | Games run in parallel across threads; opening pairs stay matched. |

The match uses **matched-pair openings**: each random opening is played twice with the engines swapping colors. This is the standard variance-reduction trick used by cutechess-cli / fastchess — any positional imbalance from the random walk cancels across the pair, so the resulting WLD reflects engine strength rather than which side got the better opening.

Interpreting the output:

- **Elo delta** — magnitude tells you the strength gap; sign tells you who's ahead.
- **95% CI** — the noise floor. A 100-game match at depth 8 typically gives a half-width around ±30–50 elo. If your delta is inside the CI, the match didn't prove anything either way.
- **LOS** — probability that Engine A is *truly* stronger given the observed WLD. >95% is solid evidence, 80–95% is suggestive, below 80% means you need more games (or the engines are genuinely close).

Game-termination handling mirrors the GUI's: checkmate, stalemate, 50-move rule, insufficient material, threefold repetition, and the max-moves cap.

## Full retraining workflow

The end-to-end loop with validation:

```bash
# 1. Generate self-play training data with the current net
cargo run --release -- selfplay 100000 nets/genN-data.bin --nnue nets/current.nnue

# 2. Train, warm-starting from the current net
cargo run --release -- train nets/genN-data.bin \
  --resume nets/current.nnue \
  --epochs 30 \
  --output nets/genN.nnue

# 3. Validate: does the candidate beat the current embedded net?
#    Engine A = embedded default (current);  Engine B = candidate (challenger).
#    NEGATIVE elo + high LOS for B ⇒ candidate is stronger.
./target/release/focalors selfmatch 100 --challenger-net nets/genN.nnue

# 4. Promote only if the validation match says the candidate is stronger.
cargo run --release -- promote nets/genN.nnue
cargo build --release
```

Step 3 is the safety net: it catches regressions that low training loss alone would miss. Without it, a bad net can sit as the default until you notice the engine playing weaker in actual games. With it, you only promote nets that demonstrably beat the previous default at the depth you tested.
