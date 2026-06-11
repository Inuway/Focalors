# GPU NNUE training (optional `gpu-training` feature)

Focalors has two NNUE trainers that share one pipeline: the pure-Rust
CPU trainer (`focalors train`, always built) and a GPU trainer
(`focalors train-gpu`) built on the [Burn][] ML framework, gated behind
the `gpu-training` Cargo feature. Both consume the same self-play data
format, use the same loss and hyperparameter defaults, and export the
same byte-identical `.nnue` layout — a net from either trainer drops
straight into the shipping binary via `focalors promote`.

[Burn]: https://burn.dev/

## Building

The default build never compiles (or downloads) Burn — the shipping
binary is unaffected by the GPU code's existence:

```bash
cargo build --release          # CPU-only, what users get
```

The GPU trainer is opt-in:

```bash
cargo build --release --features gpu-training
./target/release/focalors train-gpu <data> [opts]
```

Enabling the feature pulls in Burn + `wgpu` + `naga` and noticeably
extends compile times. Training runs on any wgpu-capable GPU (Vulkan,
Metal, DX12) — NVIDIA, AMD, Intel, and Apple Silicon all work; no CUDA
required.

`train-gpu` accepts the same flags as `train`: `--data`, `--mix`,
`--resume`, `--warmup-lr-factor`, `--warmup-epochs`, `--epochs`,
`--batch-size`, `--lr`, `--wdl`, `--output`, `--save-rate`
(`--threads` is accepted-and-ignored; the GPU replaces the thread pool).

## Design

- The Burn forward pass mirrors the CPU trainer's f32 forward exactly
  (SCReLU clamp 0..QA=255 squared, CReLU clamp 0..QB=64, the same
  division points). The regression gate
  (`trainer_gpu::tests::burn_forward_matches_cpu_forward`) asserts
  numerical agreement across seeds, guards against activation collapse,
  and includes a swapped-perspective negative control.
- Memory scales with `--batch-size`: the feature-transformer gather
  holds roughly `2 × batch × 32 × 256 × 4` bytes live for backward
  (~1 GB at the default 16384). On GPUs with less than 8 GB, use
  `--batch-size 8192` or `4096`.
- GPU *inference* is deliberately not supported: at this network size,
  CPU SIMD evaluates in nanoseconds and a host↔device round-trip would
  cost more than the compute. The GPU earns its keep in training only.

## Validating a trained net

Same gate as any net change — a fixed-depth self-match against the
current default:

```bash
./target/release/focalors selfmatch 200 --depth 8 --challenger-net path/to/new.nnue
```

Ship only on positive elo with high LOS.

## License & credits

Burn is dual-licensed Apache-2.0 OR MIT, both compatible with
Focalors' GPL-3.0-or-later. See [CREDITS.md](../CREDITS.md) for the
attribution.
