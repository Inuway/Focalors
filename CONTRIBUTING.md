# Contributing

If you want to send a patch, here's the practical workflow.

## Setup

You need a Rust stable toolchain. The GUI also needs a working desktop OpenGL stack on your machine (any modern Linux/macOS/Windows install with up-to-date graphics drivers should be fine). Then:

```bash
cargo test
cargo build --release
./target/release/focalors gui
```

The repo's `.cargo/config.toml` sets `target-cpu=native` for local builds — handy when iterating but not portable. The GitHub Actions release workflow builds with `target-cpu=x86-64` for the binaries it attaches to releases, so distributed builds run on any 64-bit machine.

## Engine development

The NNUE training pipeline and the HCE tuner are documented in [docs/TECHNICAL.md](docs/TECHNICAL.md) — the same commands work for contributors. There's nothing special to set up beyond what's already in the repo.

## Working on patches

Keep changes focused. If a patch touches the engine, the GUI, and the docs all at once, it's usually easier to review (and easier to revert) when split. Add or update tests when changing search, move generation, evaluation, persistence, or analysis logic — the test suite is the cheap insurance against regressions in those areas.

The GUI is intentionally compact rather than dashboard-heavy. New features land best when they fit the existing layout instead of adding new top-level surfaces.

Don't commit `.env`, secrets, or anything under `target/`.

## CI and releases

CI lives in `.github/workflows/ci.yml`. Pull requests and pushes run the test suite on Linux, macOS, and Windows. Pushing a tag of the form `v*` also builds release binaries for all three and attaches them to the GitHub release.

## Where to start

Engine work happens in `src/search.rs`, `src/eval.rs`, and `src/nnue/`. GUI and player experience live in `src/gui.rs`. Persistence and stats are in `src/db.rs` and `src/analysis.rs`. Documentation and release polish are in the root docs and `.github/workflows/`. Pick one and stay there.
