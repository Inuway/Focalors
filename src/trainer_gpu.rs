//! GPU NNUE trainer (experimental, `gpu-training` feature only).
//!
//! Burn 0.21 NNUE model + weight bridge + numerical-match test. The
//! Burn forward mirrors [`crate::trainer::forward`] (src/trainer.rs:432)
//! bit-for-bit in f32 so any net produced here is layout- and
//! semantically-identical to one produced by the CPU trainer (modulo
//! f32 reordering, well within the 1e-4 test tolerance).
//!
//! Status: model + numerical-match gate. No training loop yet. See
//! [BRANCH.md](../../BRANCH.md) for the design and roadmap.

// The model + weight bridge are forward-looking surface: today only
// the test module consumes them, the Phase 3 training loop will. The
// non-test build sees them as dead; the test build (cargo test
// --features gpu-training) catches actually-dead code as usual.
#![cfg_attr(not(test), allow(dead_code))]

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, NdArray, Wgpu};
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Shape, Tensor, TensorData};

use crate::nnue::features::NUM_FEATURES;
use crate::nnue::network::{FT_SIZE, L1_INPUT, L1_SIZE, L2_SIZE, QA, QB};

// ════════════════════════════════════════════════════════════════════════════
// Backend type aliases
// ════════════════════════════════════════════════════════════════════════════

/// CPU backend for the numerical-match test. Plain `NdArray` — the test
/// is forward-only, so Autodiff would just bloat compile times.
pub type TestBackend = NdArray;

/// GPU backend for the (eventual) training loop. Autodiff wraps Wgpu
/// because training needs gradients. `Wgpu<f32, i32>` is the canonical
/// Burn 0.21 alias; fusion is intentionally off in Cargo.toml so kernels
/// stay un-fused (tighter numerical agreement with NdArray).
#[allow(dead_code)]
pub type TrainBackend = Autodiff<Wgpu<f32, i32>>;

pub fn test_device() -> NdArrayDevice {
    NdArrayDevice::Cpu
}

#[allow(dead_code)]
pub fn train_device() -> WgpuDevice {
    WgpuDevice::default()
}

// ════════════════════════════════════════════════════════════════════════════
// Model
// ════════════════════════════════════════════════════════════════════════════

/// Burn-side NNUE forward model. Layout-equivalent to the CPU
/// `TrainNet` in [`crate::trainer`] (private struct there).
///
/// FT is a raw `Param<Tensor<B, 2>>` rather than `nn::Embedding` because
/// `Embedding::forward` returns one row per index — we need the
/// SCReLU-on-sum semantic from src/trainer.rs:437-451 (sum all active
/// rows into the FT bias, apply clamp² to the perspective accumulator).
#[derive(Module, Debug)]
pub struct NnueModel<B: Backend> {
    /// FT weights: `[NUM_FEATURES, FT_SIZE]` row-major.
    pub ft_weight: Param<Tensor<B, 2>>,
    /// FT bias: `[FT_SIZE]`. Added once per perspective
    /// (src/trainer.rs:437, 445).
    pub ft_bias: Param<Tensor<B, 1>>,
    /// L1: `[L1_INPUT, L1_SIZE]` weight + `[L1_SIZE]` bias. Burn's
    /// `Linear.weight` is `[d_in, d_out]` — matches the trainer's
    /// offset `i * L1_SIZE + j` at src/trainer.rs:469.
    pub l1: Linear<B>,
    /// L2: `[L1_SIZE, L2_SIZE]` (src/trainer.rs:483).
    pub l2: Linear<B>,
    /// L3: `[L2_SIZE, 1]` — the trainer's 1-D dot product at
    /// src/trainer.rs:493-496 expressed as a single-output Linear.
    pub l3: Linear<B>,
}

impl<B: Backend> NnueModel<B> {
    /// Construct with zero weights. Every real use path overwrites
    /// these via [`load_raw_weights`] before running the forward.
    pub fn new(device: &B::Device) -> Self {
        let ft_weight = Param::from_tensor(Tensor::<B, 2>::zeros(
            [NUM_FEATURES, FT_SIZE],
            device,
        ));
        let ft_bias = Param::from_tensor(Tensor::<B, 1>::zeros([FT_SIZE], device));

        NnueModel {
            ft_weight,
            ft_bias,
            l1: LinearConfig::new(L1_INPUT, L1_SIZE).with_bias(true).init(device),
            l2: LinearConfig::new(L1_SIZE, L2_SIZE).with_bias(true).init(device),
            l3: LinearConfig::new(L2_SIZE, 1).with_bias(true).init(device),
        }
    }

    /// Forward pass mirroring [`crate::trainer::forward`]
    /// (src/trainer.rs:432-499). Batched: per-perspective active feature
    /// indices padded to `[B, K]` plus a `[B, K]` float mask (1.0 live /
    /// 0.0 pad). Returns `[B]` — the scaled centipawn output, same
    /// scalar the CPU forward writes into `ForwardState::output`.
    pub fn forward(
        &self,
        stm_idx: Tensor<B, 2, Int>,
        stm_mask: Tensor<B, 2>,
        opp_idx: Tensor<B, 2, Int>,
        opp_mask: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let qa = QA as f32; // 255.0 (src/trainer.rs:433)
        let qb = QB as f32; //  64.0 (src/trainer.rs:434)

        let stm_acc = self.accumulate(stm_idx, stm_mask);
        let opp_acc = self.accumulate(opp_idx, opp_mask);

        // SCReLU: clamp(x, 0, QA)²  (src/trainer.rs:453-462).
        // QA=255, NOT 1.0 — the trainer operates in quantized-value space.
        let stm_clamped = stm_acc.clamp(0.0_f32, qa);
        let stm_sq = stm_clamped.clone().mul(stm_clamped);
        let opp_clamped = opp_acc.clamp(0.0_f32, qa);
        let opp_sq = opp_clamped.clone().mul(opp_clamped);

        // Concat STM-first, OPP-second along dim 1 — must match the
        // `l1_input` layout at src/trainer.rs:455-462. Swapping this
        // order silently breaks eval by ~25%, with no compile error.
        let l1_input = Tensor::cat(vec![stm_sq, opp_sq], 1);

        // L1: (bias + matmul) / QA, then CReLU clamp(0, QB)
        //     (src/trainer.rs:464-476)
        let l1_pre = self.l1.forward(l1_input).div_scalar(qa);
        let l1_out = l1_pre.clamp(0.0_f32, qb);

        // L2: bias + matmul, then CReLU. No division.
        //     (src/trainer.rs:478-490)
        let l2_pre = self.l2.forward(l1_out);
        let l2_out = l2_pre.clamp(0.0_f32, qb);

        // L3: (bias + dot) / QB  (src/trainer.rs:492-497)
        let l3_pre = self.l3.forward(l2_out); // [B, 1]
        let [batch, _] = l3_pre.dims();
        l3_pre.div_scalar(qb).reshape([batch])
    }

    /// Per-perspective accumulator: `ft_bias + sum_k(ft_weight[idx_k])`.
    /// Mirrors src/trainer.rs:437-443 / 445-451. Padding rows are
    /// zeroed out by `mask` so they contribute nothing to the sum.
    fn accumulate(
        &self,
        idx: Tensor<B, 2, Int>, // [B, K]
        mask: Tensor<B, 2>,     // [B, K], 1.0 live / 0.0 pad
    ) -> Tensor<B, 2> {
        let [batch, k] = idx.dims();
        let w = self.ft_weight.val(); // [NUM_FEATURES, FT_SIZE]

        // Flatten [B, K] indices, gather, reshape back to [B, K, FT].
        let flat_idx: Tensor<B, 1, Int> = idx.reshape([batch * k]);
        let gathered: Tensor<B, 2> = w.select(0, flat_idx); // [B*K, FT_SIZE]
        let looked: Tensor<B, 3> = gathered.reshape([batch, k, FT_SIZE]);

        // Mask [B, K] -> [B, K, 1] for broadcast against [B, K, FT_SIZE].
        let mask3: Tensor<B, 3> = mask.reshape([batch, k, 1]);
        let masked = looked.mul(mask3);

        // Sum over K -> [B, FT_SIZE], add bias broadcast across batch.
        let summed: Tensor<B, 2> = masked.sum_dim(1).reshape([batch, FT_SIZE]);
        let bias: Tensor<B, 2> = self.ft_bias.val().reshape([1, FT_SIZE]);
        summed.add(bias)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Weight bridge: raw f32 vecs -> NnueModel params
// ════════════════════════════════════════════════════════════════════════════

/// Raw f32 weight bundle — same shape/order as the args to
/// [`crate::trainer::TrainNet::from_f32_weights`] (src/trainer.rs:218).
/// The numerical-match test builds one of these, hands it to BOTH the
/// CPU `TrainNet` and the Burn `NnueModel`, so the two forwards see
/// identical inputs.
pub struct RawWeights {
    pub ft_weights: Vec<f32>, // [NUM_FEATURES * FT_SIZE]
    pub ft_biases: Vec<f32>,  // [FT_SIZE]
    pub l1_weights: Vec<f32>, // [L1_INPUT * L1_SIZE]
    pub l1_biases: Vec<f32>,  // [L1_SIZE]
    pub l2_weights: Vec<f32>, // [L1_SIZE * L2_SIZE]
    pub l2_biases: Vec<f32>,  // [L2_SIZE]
    pub l3_weights: Vec<f32>, // [L2_SIZE]
    pub l3_bias: f32,
}

fn p2<B: Backend>(v: Vec<f32>, shape: [usize; 2], device: &B::Device) -> Param<Tensor<B, 2>> {
    let data = TensorData::new(v, Shape::new(shape));
    Param::from_tensor(Tensor::<B, 2>::from_data(data, device))
}

fn p1<B: Backend>(v: Vec<f32>, shape: [usize; 1], device: &B::Device) -> Param<Tensor<B, 1>> {
    let data = TensorData::new(v, Shape::new(shape));
    Param::from_tensor(Tensor::<B, 1>::from_data(data, device))
}

/// Overwrite every parameter of `model` with the supplied f32 vectors.
/// Param IDs are regenerated via `Param::from_tensor` — fine for the
/// numerical-match test and the initial GPU-side weight handoff. If we
/// ever round-trip through Burn's record format we'll switch to
/// `Param::initialized(old.id, ...)`.
pub fn load_raw_weights<B: Backend>(
    model: &mut NnueModel<B>,
    raw: RawWeights,
    device: &B::Device,
) {
    assert_eq!(raw.ft_weights.len(), NUM_FEATURES * FT_SIZE);
    assert_eq!(raw.ft_biases.len(), FT_SIZE);
    assert_eq!(raw.l1_weights.len(), L1_INPUT * L1_SIZE);
    assert_eq!(raw.l1_biases.len(), L1_SIZE);
    assert_eq!(raw.l2_weights.len(), L1_SIZE * L2_SIZE);
    assert_eq!(raw.l2_biases.len(), L2_SIZE);
    assert_eq!(raw.l3_weights.len(), L2_SIZE);

    model.ft_weight = p2(raw.ft_weights, [NUM_FEATURES, FT_SIZE], device);
    model.ft_bias = p1(raw.ft_biases, [FT_SIZE], device);

    // Linear.weight layout `[d_in, d_out]` matches the trainer's
    // row-major `[L1_INPUT][L1_SIZE]` — no transpose needed.
    model.l1.weight = p2(raw.l1_weights, [L1_INPUT, L1_SIZE], device);
    model.l1.bias = Some(p1(raw.l1_biases, [L1_SIZE], device));

    model.l2.weight = p2(raw.l2_weights, [L1_SIZE, L2_SIZE], device);
    model.l2.bias = Some(p1(raw.l2_biases, [L2_SIZE], device));

    // L3: trainer's 1-D dot product becomes a [L2_SIZE, 1] Linear.
    model.l3.weight = p2(raw.l3_weights, [L2_SIZE, 1], device);
    model.l3.bias = Some(p1(vec![raw.l3_bias], [1], device));
}

// ════════════════════════════════════════════════════════════════════════════
// CLI entry point (training loop comes in a later phase)
// ════════════════════════════════════════════════════════════════════════════

pub fn run_training_gpu(_data_path: &str, _args: &[String]) {
    eprintln!("focalors: GPU NNUE training loop is not yet implemented.");
    eprintln!();
    eprintln!("The Burn model and weight bridge are in place (Phase 2);");
    eprintln!("the training loop is the next phase. See BRANCH.md.");
    eprintln!();
    eprintln!("For working CPU NNUE training, use the `train` subcommand on `main`.");
    std::process::exit(2);
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    //! Phase 1 smoke tests guard the trainer's pub surface from
    //! visibility regressions; the Phase 2 numerical-match test
    //! (`burn_forward_matches_cpu_forward`) is the real Phase 2 gate.

    use super::*;

    use crate::trainer::{
        SIGMOID_SCALE, Sample, TrainNet, forward_output, load_data,
        loss_and_gradient, parse_record, quantize, sigmoid,
    };

    // ─── Phase 1: trainer pub-surface visibility ─────────────────────────

    #[test]
    fn shared_trainer_api_is_reachable() {
        assert_eq!(SIGMOID_SCALE, 400.0);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        let (loss, _) = loss_and_gradient(0.0, 0.0, 0.5, 0.5);
        assert!(loss.is_finite());

        let _: Option<fn(&[u8]) -> Option<Sample>> = Some(parse_record);
        let _: Option<fn(&str) -> Vec<Sample>> = Some(load_data);
    }

    #[test]
    fn train_net_constructor_validates_sizes_and_quantizes() {
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

    // ─── Phase 2: Burn vs CPU forward numerical-match gate ───────────────

    /// Deterministic LCG with output scaled down by 16× to keep
    /// neurons safely inside the SCReLU and CReLU clamp ranges.
    /// Without the down-scaling, the L1 accumulator routinely lands
    /// near the QB=64 clamp boundary, and f32 reordering between the
    /// trainer's scalar loop and Burn's batched matmul could put the
    /// two forwards on opposite sides of the clamp — i.e. flaky.
    fn lcg_scaled(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (*state >> 33) as f32 / u32::MAX as f32;
        // -1..=1 then scaled to keep matmul magnitudes mild.
        (u * 2.0 - 1.0) / 16.0
    }

    fn random_raw_weights(seed: u64) -> RawWeights {
        let mut s = seed;
        let mut mk = |n: usize| -> Vec<f32> {
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                out.push(lcg_scaled(&mut s));
            }
            out
        };
        let ft_weights = mk(NUM_FEATURES * FT_SIZE);
        let ft_biases = mk(FT_SIZE);
        let l1_weights = mk(L1_INPUT * L1_SIZE);
        let l1_biases = mk(L1_SIZE);
        let l2_weights = mk(L1_SIZE * L2_SIZE);
        let l2_biases = mk(L2_SIZE);
        let l3_weights = mk(L2_SIZE);
        let l3_bias = mk(1)[0];
        RawWeights {
            ft_weights, ft_biases,
            l1_weights, l1_biases,
            l2_weights, l2_biases,
            l3_weights, l3_bias,
        }
    }

    /// Distinct, asymmetric STM/OPP feature lists so a perspective
    /// swap or concat-order bug shows up as a clear numerical
    /// difference instead of silent equivalence.
    fn synthetic_sample() -> Sample {
        Sample {
            stm_features: vec![0, 12, 64, 100, 250, 400, 500, 700],
            opp_features: vec![1, 13, 65, 101, 251, 401, 501, 701],
            score: 0.0,
            result: 0.5,
        }
    }

    fn pack_sample(
        sample: &Sample,
        device: &NdArrayDevice,
    ) -> (
        Tensor<TestBackend, 2, Int>,
        Tensor<TestBackend, 2>,
        Tensor<TestBackend, 2, Int>,
        Tensor<TestBackend, 2>,
    ) {
        let k = sample.stm_features.len().max(sample.opp_features.len());
        let pad = |v: &[usize]| -> (Vec<i64>, Vec<f32>) {
            let mut idx = Vec::with_capacity(k);
            let mut mask = Vec::with_capacity(k);
            for &i in v {
                idx.push(i as i64);
                mask.push(1.0);
            }
            while idx.len() < k {
                idx.push(0);
                mask.push(0.0);
            }
            (idx, mask)
        };
        let (sidx, smask) = pad(&sample.stm_features);
        let (oidx, omask) = pad(&sample.opp_features);

        let mk_i = |v: Vec<i64>| -> Tensor<TestBackend, 2, Int> {
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(v, Shape::new([1, k])),
                device,
            )
        };
        let mk_f = |v: Vec<f32>| -> Tensor<TestBackend, 2> {
            Tensor::<TestBackend, 2>::from_data(
                TensorData::new(v, Shape::new([1, k])),
                device,
            )
        };
        (mk_i(sidx), mk_f(smask), mk_i(oidx), mk_f(omask))
    }

    #[test]
    fn burn_forward_matches_cpu_forward() {
        let device = test_device();
        let raw = random_raw_weights(0xC0FFEE);

        // CPU side: build a TrainNet from the same raw vectors, run the
        // trainer's f32 forward via the pub(crate) shim.
        let cpu_net = TrainNet::from_f32_weights(
            raw.ft_weights.clone(),
            raw.ft_biases.clone(),
            raw.l1_weights.clone(),
            raw.l1_biases.clone(),
            raw.l2_weights.clone(),
            raw.l2_biases.clone(),
            raw.l3_weights.clone(),
            raw.l3_bias,
        );
        let sample = synthetic_sample();
        let cpu_out = forward_output(&cpu_net, &sample);

        // Burn side: zero-init model, inject the same weights, run forward.
        let mut model: NnueModel<TestBackend> = NnueModel::new(&device);
        load_raw_weights(&mut model, raw, &device);
        let (sidx, smask, oidx, omask) = pack_sample(&sample, &device);
        let burn_out: f32 = model
            .forward(sidx, smask, oidx, omask)
            .into_scalar();

        let diff = (burn_out - cpu_out).abs();
        assert!(
            diff < 1e-4,
            "Burn vs CPU forward mismatch: burn={burn_out}, cpu={cpu_out}, diff={diff}"
        );
    }
}
