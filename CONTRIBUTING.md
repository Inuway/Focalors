# Contributing

Focalors is a Rust codebase that mixes engine work, GUI work, persistence, and training tooling. This document keeps the contributor-facing workflow out of the public README while preserving the practical details needed to work on the project.

## Local Setup

Requirements:

- Rust stable toolchain
- A working desktop OpenGL environment if you want to run the GUI locally

Common commands:

```bash
cargo test
cargo build --release
cargo run --release -- gui
cargo run --release -- uci
```

The repository uses a local Cargo config with `target-cpu=native` for development builds. That is convenient for local iteration, but prebuilt public binaries should use a more portable CPU target. The GitHub Actions release workflow handles that automatically.

## Run Modes

| Command | Purpose |
| --- | --- |
| `cargo run --release -- gui` | Desktop GUI with local play, analysis, puzzles, and stats |
| `cargo run --release -- uci` | Standard UCI engine mode |
| `cargo run --release` | Same as `uci` |
| `cargo run --release -- lichess` | Headless Lichess bot mode |
| `cargo test` | Full test suite |
| `cargo build --release` | Optimized local build |

For Lichess mode, create a `.env` file with:

```text
LICHESS_TOKEN=your_token_here
```

## Engine Development

The NNUE workflow is built into the repository.

Generate self-play data:

```bash
cargo run --release -- selfplay 100000 nets/gen2v2-data.bin --nnue nets/gen1v2.nnue
```

Train a new network with warm-start plus data mixing:

```bash
cargo run --release -- train nets/gen1v2-data.bin \
  --data nets/gen2v2-data.bin \
  --mix 0.3,0.7 \
  --resume nets/gen1v2.nnue \
  --epochs 30 \
  --output nets/gen2v2.nnue
```

Promote a trained net as the shipped default:

```bash
cargo run --release -- promote nets/gen2v2.nnue
cargo build --release
```

Tune the hand-crafted evaluation separately:

```bash
cargo run --release -- tune dataset.txt
```

## Repo Guidelines

- Keep patches focused. Separate engine changes, GUI changes, and documentation changes when practical.
- Add or update tests when changing search, move generation, evaluation, persistence, or analysis logic.
- Preserve the existing compact GUI style rather than turning the app into a dashboard-heavy interface.
- Keep public branding on `Focalors`. Internal helper names are fine if they are not user-facing.
- Do not commit local secrets, `.env`, or generated `target/` output.

## Release Notes

Public release automation lives in `.github/workflows/ci.yml`.

- Pull requests and pushes build and test on Linux, macOS, and Windows.
- Tag pushes matching `v*` also build portable release binaries and attach packaged artifacts to the GitHub release.

## Where To Start

If you want a good first contribution, pick one area and stay focused:

- engine strength improvements in `src/search.rs`, `src/eval.rs`, and `src/nnue/`
- GUI and player experience in `src/gui.rs`
- persistence and stats in `src/db.rs` and `src/analysis.rs`
- documentation and release polish in the root docs and `.github/workflows/`