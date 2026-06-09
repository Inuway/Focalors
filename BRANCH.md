# `gpu-training` — experimental GPU NNUE training branch

This branch is a long-lived experimental fork of `main` that adds a
GPU-accelerated NNUE training path on top of the existing CPU pipeline.
The shipping `focalors` binary (engine + GUI) is unchanged; the GPU
trainer is gated behind a Cargo feature and is dev-only.

## Status

Scaffolding only. The `gpu-training` Cargo feature wires in [Burn][]
(Apache-2.0/MIT) and exposes a `focalors train-gpu` subcommand stub.
The autodiff modules, training loop, and `.nnue` export path are not
yet implemented.

[Burn]: https://burn.dev/

## Building

The default build is **identical to `main`** — no extra deps pulled in,
no extra binaries, same shipping behaviour:

```bash
cargo build --release          # CPU build, Burn not pulled
./target/release/focalors gui  # same as main
```

The GPU trainer is opt-in:

```bash
cargo build --release --features gpu-training
./target/release/focalors train-gpu <data> [opts]
```

Enabling the feature pulls in Burn + `wgpu` + `naga` and noticeably
extends compile times. None of those crates touch the shipping
`focalors` binary built without the feature.

## Goals

- Train larger NNUE nets than the CPU pipeline can practically reach
- Keep the existing data generation (`focalors selfplay`), FEN parsing,
  feature indexing, and quantized `.nnue` export — only the autodiff
  forward/backward pass moves to GPU
- Output `.nnue` files **bit-identical in format** to those produced by
  the CPU trainer, so they drop straight into the shipping binary via
  `focalors promote`

## Non-goals

- Replacing the CPU trainer — it stays the source of shipping nets
  until this branch produces clearly stronger nets validated via
  `focalors selfmatch`
- GPU inference at runtime — NNUE is small enough that CPU SIMD wins,
  and the host↔device round-trip would dominate the actual compute
- Merging this branch back to `main` (the Burn dep is not shipped to
  users; that's the whole point of the split)

## Maintenance

`main` periodically merges **into** this branch to absorb engine
improvements:

```bash
git checkout gpu-training
git merge main
```

This branch never merges back to `main`. Tag working snapshots
(`gpu-training-v0.1`, etc.) to mark stable parking points.

## License & credits

Burn is dual-licensed Apache-2.0 OR MIT, both GPL-compatible with the
rest of Focalors (GPL-3.0-or-later). See [CREDITS.md](CREDITS.md) for
the full attribution.
