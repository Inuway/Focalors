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

#[cfg(test)]
mod tests {
    //! Smoke tests that the trainer's public surface is reachable from
    //! a sibling module. These guard against accidental visibility
    //! regressions before the Burn integration starts using these APIs
    //! for real.

    use crate::nnue::network::{FT_SIZE, L1_INPUT, L1_SIZE, L2_SIZE};
    use crate::nnue::features::NUM_FEATURES;
    use crate::trainer::{
        SIGMOID_SCALE, Sample, TrainNet, load_data, loss_and_gradient,
        parse_record, quantize, sigmoid,
    };

    #[test]
    fn shared_trainer_api_is_reachable() {
        // Compile-time check that every item the GPU trainer will need
        // is actually `pub` and importable from this module. The body
        // just exercises the easy ones; the heavy items are exercised
        // by the CPU trainer's own tests.
        assert_eq!(SIGMOID_SCALE, 400.0);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        let (loss, _) = loss_and_gradient(0.0, 0.0, 0.5, 0.5);
        assert!(loss.is_finite());

        // Reference the constructable types so `unused_imports` doesn't
        // fire if a later refactor removes a real usage above.
        let _: Option<fn(&[u8]) -> Option<Sample>> = Some(parse_record);
        let _: Option<fn(&str) -> Vec<Sample>> = Some(load_data);
    }

    #[test]
    fn train_net_constructor_validates_sizes_and_quantizes() {
        // Build a zero-weight TrainNet via the public constructor,
        // quantize it, and verify the output is the expected on-disk
        // size and loads back through Network::from_bytes. Acts as the
        // contract test for the CPU↔GPU boundary.
        let net = TrainNet::from_f32_weights(
            vec![0.0; NUM_FEATURES * FT_SIZE],
            vec![0.0; FT_SIZE],
            vec![0.0; L1_INPUT * L1_SIZE],
            vec![0.0; L1_SIZE],
            vec![0.0; L1_SIZE * L2_SIZE],
            vec![0.0; L2_SIZE],
            vec![0.0; L2_SIZE],
            0.0,
        );
        let bytes = quantize(&net);
        assert_eq!(bytes.len(), 411_428, "shipping .nnue layout must stay 411428 bytes");
        crate::nnue::network::Network::from_bytes(&bytes)
            .expect("quantized zero-net must load through the production loader");
    }

    #[test]
    #[should_panic(expected = "ft_weights size mismatch")]
    fn train_net_constructor_rejects_wrong_size() {
        let _ = TrainNet::from_f32_weights(
            vec![0.0; 0],
            vec![0.0; FT_SIZE],
            vec![0.0; L1_INPUT * L1_SIZE],
            vec![0.0; L1_SIZE],
            vec![0.0; L1_SIZE * L2_SIZE],
            vec![0.0; L2_SIZE],
            vec![0.0; L2_SIZE],
            0.0,
        );
    }
}
