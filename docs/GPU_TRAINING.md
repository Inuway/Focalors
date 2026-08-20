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
(`--threads` sets the number of host threads used to marshal batches;
it defaults to all cores).

## Design

- The Burn forward pass mirrors the CPU trainer's f32 forward exactly
  (SCReLU clamp 0..QA=255 squared, CReLU clamp 0..QB=64, the same
  division points). The regression gate
  (`trainer_gpu::tests::burn_forward_matches_cpu_forward`) asserts
  numerical agreement across seeds, guards against activation collapse,
  and includes a swapped-perspective negative control.
- The feature transformer is evaluated as a multi-hot matmul: active
  feature indices are scattered into a `[batch, NUM_FEATURES]` matrix
  which is multiplied by the FT weights. The obvious alternative — gather
  the active rows and sum them — materializes `[batch × K, 256]` (536 MB
  per perspective at the default batch size, retained by autodiff for the
  backward pass) and makes training bandwidth-bound; the matmul form
  measured ~15x faster end to end on an RTX 5060 Laptop, with all weight
  differences at ±1 quantization level.
- Memory scales with `--batch-size`: ~900 MB peak at the default 16384.
  On GPUs with less than 4 GB, use `--batch-size 8192` or `4096`.
- Batches are packed on worker threads and prefetched one ahead of the
  device, so host marshalling overlaps device work instead of alternating
  with it. On this workload the GPU is the limit and the packing is fully
  hidden; the overlap is insurance for hosts with weak CPUs.
- GPU *inference* is deliberately not supported: at this network size,
  CPU SIMD evaluates in nanoseconds and a host↔device round-trip would
  cost more than the compute. The GPU earns its keep in training only.

## Validating a trained net

Same gate as any net change — a fixed-depth self-match against the
current default:

```bash
./target/release/focalors selfmatch 1000 --depth 8 --challenger-net path/to/new.nnue
```

Ship only on positive elo with high LOS. 1000 games is the promotion
standard: typical per-generation gains (+10-30 elo) sit below the noise
floor of a short match.

## License & credits

Burn is dual-licensed Apache-2.0 OR MIT, both compatible with
Focalors' GPL-3.0-or-later. See [CREDITS.md](../CREDITS.md) for the
attribution.
