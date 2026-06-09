//! GPU NNUE trainer scaffold (experimental, `gpu-training` feature only).
//!
//! Entry point for a Burn-based GPU training pipeline that will eventually
//! mirror the CPU trainer in [`crate::trainer`] but run forward/backward
//! passes on a GPU via Burn's wgpu backend.
//!
//! Status: scaffold only. No autodiff modules, no training loop, no export
//! yet. See `BRANCH.md` on the `gpu-training` branch for design and roadmap.
//!
//! Output contract (when implemented): emit `.nnue` files bit-identical in
//! layout to those produced by [`crate::trainer`], so they drop straight
//! into the shipping binary via `focalors promote`.

pub fn run_training_gpu(_data_path: &str, _args: &[String]) {
    eprintln!("focalors: GPU NNUE training is not yet implemented.");
    eprintln!();
    eprintln!("This is the experimental `gpu-training` branch. The Burn-based");
    eprintln!("trainer is scaffolded but the training loop is not yet wired up.");
    eprintln!("See BRANCH.md for the design and current status.");
    eprintln!();
    eprintln!("For working CPU NNUE training, use the `train` subcommand on `main`.");
    std::process::exit(2);
}
