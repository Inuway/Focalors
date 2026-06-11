//! GPU NNUE trainer (`gpu-training` feature only).
//!
//! Complete Burn 0.21 NNUE training pipeline: model + data batcher +
//! manual training loop (Adam, LR schedule, WDL-blended MSE loss) +
//! export to the same byte-identical `.nnue` layout the CPU trainer
//! produces (via [`crate::trainer::quantize`]).
//!
//! All forward semantics mirror [`crate::trainer::forward`] bit-for-bit
//! in f32 — the numerical-match test (`burn_forward_matches_cpu_forward`)
//! is the regression gate. Outer training shape (epochs, LR drops at
//! `epochs*3/4` and `epochs*9/10`, warmup ramp on `--resume`, save_rate
//! checkpoints) mirrors [`crate::trainer::train`] one-for-one.
//!
//! See `docs/GPU_TRAINING.md` for build instructions, design notes, and
//! the validation workflow.

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::data::dataloader::batcher::Batcher;
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Shape, Tensor, TensorData};

use crate::nnue::features::NUM_FEATURES;
use crate::nnue::network::{FT_SIZE, L1_INPUT, L1_SIZE, L2_SIZE, QA, QB};
use crate::trainer::{
    Rng, SIGMOID_SCALE, Sample, TrainNet, load_data, load_data_multi, lr_for_epoch, quantize,
    sigmoid,
};

// ════════════════════════════════════════════════════════════════════════════
// Backend type aliases
// ════════════════════════════════════════════════════════════════════════════

/// GPU backend for the training loop. Autodiff is required for gradient
/// computation; `Wgpu<f32, i32>` is the canonical Burn 0.21 alias.
/// Fusion is intentionally off in Cargo.toml so kernels stay un-fused
/// (tighter numerical agreement across backends).
pub type TrainBackend = Autodiff<Wgpu<f32, i32>>;

pub fn train_device() -> WgpuDevice {
    WgpuDevice::default()
}

/// CPU backend for the numerical-match test. Forward-only, so no
/// Autodiff wrapper — keeps the test hermetic and the compile light.
#[cfg(test)]
type TestBackend = burn::backend::NdArray;

#[cfg(test)]
fn test_device() -> burn::backend::ndarray::NdArrayDevice {
    burn::backend::ndarray::NdArrayDevice::Cpu
}

// ════════════════════════════════════════════════════════════════════════════
// Model
// ════════════════════════════════════════════════════════════════════════════

/// Burn-side NNUE forward model. Layout-equivalent to the CPU
/// `TrainNet` in [`crate::trainer`] (private struct there).
///
/// FT is a raw `Param<Tensor<B, 2>>` rather than `nn::Embedding` because
/// `Embedding::forward` returns one row per index — we need the
/// SCReLU-on-sum semantic from the CPU forward (sum all active rows
/// into the FT bias, apply clamp² to the perspective accumulator).
#[derive(Module, Debug)]
pub struct NnueModel<B: Backend> {
    pub ft_weight: Param<Tensor<B, 2>>,
    pub ft_bias: Param<Tensor<B, 1>>,
    pub l1: Linear<B>,
    pub l2: Linear<B>,
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

    /// Forward pass mirroring [`crate::trainer::forward`]. Batched:
    /// per-perspective active feature indices padded to `[B, K]` plus
    /// a `[B, K]` float mask (1.0 live / 0.0 pad). Returns `[B]` —
    /// the scaled centipawn output, same scalar the CPU forward writes
    /// into `ForwardState::output`.
    pub fn forward(
        &self,
        stm_idx: Tensor<B, 2, Int>,
        stm_mask: Tensor<B, 2>,
        opp_idx: Tensor<B, 2, Int>,
        opp_mask: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let qa = QA as f32; // 255.0
        let qb = QB as f32; //  64.0

        let stm_acc = self.accumulate(stm_idx, stm_mask);
        let opp_acc = self.accumulate(opp_idx, opp_mask);

        // SCReLU: clamp(x, 0, QA)² — QA=255, NOT 1.0, since the trainer
        // operates in quantized-value space (matches CPU forward).
        let stm_clamped = stm_acc.clamp(0.0_f32, qa);
        let stm_sq = stm_clamped.clone().mul(stm_clamped);
        let opp_clamped = opp_acc.clamp(0.0_f32, qa);
        let opp_sq = opp_clamped.clone().mul(opp_clamped);

        // Concat STM-first, OPP-second — must match CPU's l1_input
        // layout. Swapping silently breaks eval by ~25%.
        let l1_input = Tensor::cat(vec![stm_sq, opp_sq], 1);

        // L1: (bias + matmul) / QA, then CReLU clamp(0, QB).
        let l1_pre = self.l1.forward(l1_input).div_scalar(qa);
        let l1_out = l1_pre.clamp(0.0_f32, qb);

        // L2: bias + matmul, then CReLU. No division.
        let l2_pre = self.l2.forward(l1_out);
        let l2_out = l2_pre.clamp(0.0_f32, qb);

        // L3: (bias + dot) / QB.
        let l3_pre = self.l3.forward(l2_out); // [B, 1]
        let [batch, _] = l3_pre.dims();
        l3_pre.div_scalar(qb).reshape([batch])
    }

    /// Per-perspective accumulator: `ft_bias + sum_k(ft_weight[idx_k])`.
    /// Padding rows are zeroed via `mask` so they contribute nothing.
    fn accumulate(
        &self,
        idx: Tensor<B, 2, Int>, // [B, K]
        mask: Tensor<B, 2>,     // [B, K], 1.0 live / 0.0 pad
    ) -> Tensor<B, 2> {
        let [batch, k] = idx.dims();
        let w = self.ft_weight.val();

        let flat_idx: Tensor<B, 1, Int> = idx.reshape([batch * k]);
        let gathered: Tensor<B, 2> = w.select(0, flat_idx); // [B*K, FT_SIZE]
        let looked: Tensor<B, 3> = gathered.reshape([batch, k, FT_SIZE]);

        let mask3: Tensor<B, 3> = mask.reshape([batch, k, 1]);
        let masked = looked.mul(mask3);

        let summed: Tensor<B, 2> = masked.sum_dim(1).reshape([batch, FT_SIZE]);
        let bias: Tensor<B, 2> = self.ft_bias.val().reshape([1, FT_SIZE]);
        summed.add(bias)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Weight bridge: raw f32 vecs <-> NnueModel params
// ════════════════════════════════════════════════════════════════════════════

/// Raw f32 weight bundle — same shape/order as
/// [`crate::trainer::TrainNet::from_f32_weights`] args. The training
/// loop builds one of these (random init or warm-start from `.nnue`),
/// injects it into the Burn model, and re-emits one at export time.
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
/// numerical-match test and the initial-weight handoff at training
/// start. If we ever round-trip through Burn's record format we'll
/// switch to `Param::initialized(old.id, ...)`.
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

    model.l1.weight = p2(raw.l1_weights, [L1_INPUT, L1_SIZE], device);
    model.l1.bias = Some(p1(raw.l1_biases, [L1_SIZE], device));

    model.l2.weight = p2(raw.l2_weights, [L1_SIZE, L2_SIZE], device);
    model.l2.bias = Some(p1(raw.l2_biases, [L2_SIZE], device));

    model.l3.weight = p2(raw.l3_weights, [L2_SIZE, 1], device);
    model.l3.bias = Some(p1(vec![raw.l3_bias], [1], device));
}

// ════════════════════════════════════════════════════════════════════════════
// Data pipeline: pad-to-batch-K_max Batcher producing per-batch tensors
// ════════════════════════════════════════════════════════════════════════════

/// Per-batch packer. We invoke `Batcher::batch` directly on each
/// `Vec<Sample>` slice (no `DataLoader` machinery) because the CPU
/// trainer mirrors the same shape — shuffle the slice, step_by
/// batch_size, pack and upload one batch at a time.
#[derive(Clone, Debug)]
pub(crate) struct NnueBatcher {
    /// WDL blend weight, captured at construction. We precompute the
    /// blended target host-side so the GPU only sees a single f32
    /// target per sample.
    pub wdl_weight: f32,
}

/// One minibatch on the training device. Padded to per-batch K_max for
/// both perspectives; padding rows are zeroed via the float mask before
/// the embedding-sum in [`NnueModel::accumulate`].
#[derive(Clone, Debug)]
pub(crate) struct NnueBatch<B: Backend> {
    pub stm_idx: Tensor<B, 2, Int>,
    pub stm_mask: Tensor<B, 2>,
    pub opp_idx: Tensor<B, 2, Int>,
    pub opp_mask: Tensor<B, 2>,
    /// WDL-blended sigmoid target in [0, 1], computed host-side.
    pub target: Tensor<B, 1>,
    /// Number of samples in this batch (may be < `batch_size` on the
    /// tail of an epoch).
    pub len: usize,
}

impl<B: Backend> Batcher<B, Sample, NnueBatch<B>> for NnueBatcher {
    fn batch(&self, items: Vec<Sample>, device: &B::Device) -> NnueBatch<B> {
        let batch = items.len();
        assert!(batch > 0, "NnueBatcher::batch called with empty items");

        // K_max across BOTH perspectives — keeps the two index tensors
        // the same width so we can reuse one shape constant.
        let k_max = items
            .iter()
            .map(|s| s.stm_features.len().max(s.opp_features.len()))
            .max()
            .unwrap_or(0)
            .max(1); // avoid a zero-width tensor on a degenerate batch

        let mut stm_idx_buf: Vec<i64> = Vec::with_capacity(batch * k_max);
        let mut stm_mask_buf: Vec<f32> = Vec::with_capacity(batch * k_max);
        let mut opp_idx_buf: Vec<i64> = Vec::with_capacity(batch * k_max);
        let mut opp_mask_buf: Vec<f32> = Vec::with_capacity(batch * k_max);
        let mut target_buf: Vec<f32> = Vec::with_capacity(batch);

        for s in &items {
            for &f in &s.stm_features {
                stm_idx_buf.push(f as i64);
                stm_mask_buf.push(1.0);
            }
            for _ in s.stm_features.len()..k_max {
                stm_idx_buf.push(0); // value irrelevant, masked out
                stm_mask_buf.push(0.0);
            }
            for &f in &s.opp_features {
                opp_idx_buf.push(f as i64);
                opp_mask_buf.push(1.0);
            }
            for _ in s.opp_features.len()..k_max {
                opp_idx_buf.push(0);
                opp_mask_buf.push(0.0);
            }
            // WDL-blended target — same formula as the CPU trainer's loss.
            let target_sig = sigmoid(s.score);
            let target = self.wdl_weight * s.result + (1.0 - self.wdl_weight) * target_sig;
            target_buf.push(target);
        }

        let shape_2d = Shape::new([batch, k_max]);
        let stm_idx = Tensor::<B, 2, Int>::from_data(
            TensorData::new(stm_idx_buf, shape_2d.clone()),
            device,
        );
        let stm_mask = Tensor::<B, 2>::from_data(
            TensorData::new(stm_mask_buf, shape_2d.clone()),
            device,
        );
        let opp_idx = Tensor::<B, 2, Int>::from_data(
            TensorData::new(opp_idx_buf, shape_2d.clone()),
            device,
        );
        let opp_mask = Tensor::<B, 2>::from_data(
            TensorData::new(opp_mask_buf, shape_2d),
            device,
        );
        let target = Tensor::<B, 1>::from_data(
            TensorData::new(target_buf, Shape::new([batch])),
            device,
        );

        NnueBatch { stm_idx, stm_mask, opp_idx, opp_mask, target, len: batch }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Export: NnueModel -> f32 vecs -> TrainNet -> quantize -> .nnue bytes
// ════════════════════════════════════════════════════════════════════════════
//
// Generic over `B: AutodiffBackend` so the Phase 5 round-trip test can
// exercise the same path on `Autodiff<NdArray>` without spinning up a
// wgpu adapter.

fn extract_f32_1d<B: AutodiffBackend>(p: &Param<Tensor<B, 1>>) -> Vec<f32> {
    p.val()
        .inner()
        .into_data()
        .into_vec::<f32>()
        .expect("Param tensor must be f32")
}

fn extract_f32_2d<B: AutodiffBackend>(p: &Param<Tensor<B, 2>>) -> Vec<f32> {
    p.val()
        .inner()
        .into_data()
        .into_vec::<f32>()
        .expect("Param tensor must be f32")
}

/// Pull every Param off `model`, route through
/// [`TrainNet::from_f32_weights`] + [`quantize`], and emit the on-disk
/// `.nnue` byte layout. Identical output to what the CPU trainer
/// produces — same 411 428-byte format the shipping binary loads.
fn export_to_nnue_bytes<B: AutodiffBackend>(model: &NnueModel<B>) -> Vec<u8> {
    let ft_weights = extract_f32_2d(&model.ft_weight);
    let ft_biases = extract_f32_1d(&model.ft_bias);

    let l1_weights = extract_f32_2d(&model.l1.weight);
    let l1_biases = extract_f32_1d(
        model.l1.bias.as_ref().expect("l1 bias was Some at construction"),
    );
    let l2_weights = extract_f32_2d(&model.l2.weight);
    let l2_biases = extract_f32_1d(
        model.l2.bias.as_ref().expect("l2 bias was Some at construction"),
    );

    // L3: weight is [L2_SIZE, 1] row-major — flat layout = [L2_SIZE].
    let l3_weights_flat = extract_f32_2d(&model.l3.weight);
    debug_assert_eq!(l3_weights_flat.len(), L2_SIZE);
    let l3_bias_vec = extract_f32_1d(
        model.l3.bias.as_ref().expect("l3 bias was Some at construction"),
    );
    let l3_bias = l3_bias_vec[0];

    let net = TrainNet::from_f32_weights(
        ft_weights,
        ft_biases,
        l1_weights,
        l1_biases,
        l2_weights,
        l2_biases,
        l3_weights_flat,
        l3_bias,
    );
    quantize(&net)
}

// ════════════════════════════════════════════════════════════════════════════
// Initial weights: random He init OR warm-start from existing .nnue
// ════════════════════════════════════════════════════════════════════════════

fn build_initial_raw_weights(resume_path: Option<&str>, rng: &mut Rng) -> RawWeights {
    if let Some(path) = resume_path {
        eprintln!("Resuming training from existing net: {path}");
        let bytes = std::fs::read(path)
            .unwrap_or_else(|e| panic!("Failed to read resume net '{path}': {e}"));
        let loaded = crate::nnue::network::Network::from_bytes(&bytes)
            .unwrap_or_else(|e| panic!("Failed to parse resume net: {e}"));
        RawWeights {
            ft_weights: loaded.ft_weights.iter().map(|&v| v as f32).collect(),
            ft_biases: loaded.ft_biases.iter().map(|&v| v as f32).collect(),
            l1_weights: loaded.l1_weights.iter().map(|&v| v as f32).collect(),
            l1_biases: loaded.l1_biases.iter().map(|&v| v as f32).collect(),
            l2_weights: loaded.l2_weights.iter().map(|&v| v as f32).collect(),
            l2_biases: loaded.l2_biases.iter().map(|&v| v as f32).collect(),
            l3_weights: loaded.l3_weights.iter().map(|&v| v as f32).collect(),
            l3_bias: loaded.l3_bias as f32,
        }
    } else {
        eprintln!("Initializing fresh random network.");
        // Match CPU trainer std values exactly. Biases stay zero.
        let ft_std = (2.0_f32 / 30.0).sqrt() * 10.0;
        let l1_std = (2.0_f32 / L1_INPUT as f32).sqrt() * 16.0;
        let l2_std = (2.0_f32 / L1_SIZE as f32).sqrt() * 16.0;
        let l3_std = (2.0_f32 / L2_SIZE as f32).sqrt() * 16.0;

        let mut ft_weights = vec![0.0_f32; NUM_FEATURES * FT_SIZE];
        for w in &mut ft_weights {
            *w = rng.next_gaussian() * ft_std;
        }
        let mut l1_weights = vec![0.0_f32; L1_INPUT * L1_SIZE];
        for w in &mut l1_weights {
            *w = rng.next_gaussian() * l1_std;
        }
        let mut l2_weights = vec![0.0_f32; L1_SIZE * L2_SIZE];
        for w in &mut l2_weights {
            *w = rng.next_gaussian() * l2_std;
        }
        let mut l3_weights = vec![0.0_f32; L2_SIZE];
        for w in &mut l3_weights {
            *w = rng.next_gaussian() * l3_std;
        }

        RawWeights {
            ft_weights,
            ft_biases: vec![0.0; FT_SIZE],
            l1_weights,
            l1_biases: vec![0.0; L1_SIZE],
            l2_weights,
            l2_biases: vec![0.0; L2_SIZE],
            l3_weights,
            l3_bias: 0.0,
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Training loop
// ════════════════════════════════════════════════════════════════════════════

pub(crate) struct GpuTrainConfig {
    pub data_paths: Vec<String>,
    pub data_weights: Vec<f32>,
    pub output_path: String,
    pub resume_path: Option<String>,
    pub warmup_lr_factor: f32,
    pub warmup_epochs: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f32,
    pub wdl_weight: f32,
    pub save_rate: usize,
}

fn train_gpu(config: &GpuTrainConfig) {
    let device = train_device();
    let mut rng = Rng::new(0xDEAD_CAFE_1234);

    eprintln!("Loading data...");
    let mut samples: Vec<Sample> = if config.data_paths.len() == 1 {
        load_data(&config.data_paths[0])
    } else {
        load_data_multi(&config.data_paths, &config.data_weights, &mut rng)
    };
    eprintln!("Loaded {} total positions.", samples.len());
    if samples.is_empty() {
        eprintln!("No training data. Aborting.");
        return;
    }

    // Build model on wgpu device and inject initial weights.
    let mut model: NnueModel<TrainBackend> = NnueModel::new(&device);
    let raw = build_initial_raw_weights(config.resume_path.as_deref(), &mut rng);
    load_raw_weights(&mut model, raw, &device);

    // Adam with CPU-equivalent constants.
    let adam_config = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8);
    let mut optim = adam_config.init::<TrainBackend, NnueModel<TrainBackend>>();

    // LR schedule: the same lr_for_epoch the CPU trainer uses — warmup
    // and decay composed in one function, so a warmup restore can never
    // overwrite an already-applied drop on short resumed runs.
    let lr_drop_1 = config.epochs * 3 / 4;
    let lr_drop_2 = config.epochs * 9 / 10;
    let mut current_lr: f64 = lr_for_epoch(
        config.lr,
        config.resume_path.is_some(),
        config.warmup_epochs,
        config.warmup_lr_factor,
        1,
        lr_drop_1,
        lr_drop_2,
    ) as f64;

    eprintln!(
        "Training (GPU): {} epochs, batch_size={}, lr={}, wdl={:.2}, save every {} epochs",
        config.epochs, config.batch_size, config.lr, config.wdl_weight, config.save_rate
    );
    eprintln!("LR drops at epochs {lr_drop_1} and {lr_drop_2}");
    if config.resume_path.is_some() && config.warmup_epochs > 0 {
        eprintln!(
            "Warm-start: LR x{:.2} for first {} epochs, then ramp to {}",
            config.warmup_lr_factor, config.warmup_epochs, config.lr
        );
    }

    let batcher = NnueBatcher { wdl_weight: config.wdl_weight };

    for epoch in 1..=config.epochs {
        let epoch_lr = lr_for_epoch(
            config.lr,
            config.resume_path.is_some(),
            config.warmup_epochs,
            config.warmup_lr_factor,
            epoch,
            lr_drop_1,
            lr_drop_2,
        ) as f64;
        if epoch_lr != current_lr {
            current_lr = epoch_lr;
            eprintln!("  LR now {current_lr:.6}");
        }

        rng.shuffle(&mut samples);

        let mut epoch_loss = 0.0_f64;
        let mut epoch_count: u64 = 0;

        for batch_start in (0..samples.len()).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(samples.len());
            let batch_items: Vec<Sample> = samples[batch_start..batch_end].to_vec();

            let batch = batcher.batch(batch_items, &device);
            let batch_len = batch.len;

            let predicted_cp = model.forward(
                batch.stm_idx,
                batch.stm_mask,
                batch.opp_idx,
                batch.opp_mask,
            );

            // Loss: MSE between sigmoid(pred / SIGMOID_SCALE) and the
            // WDL-blended target (precomputed in the batcher).
            let scaled = predicted_cp.div_scalar(SIGMOID_SCALE);
            let pred_sig = burn::tensor::activation::sigmoid(scaled);
            let diff = batch.target - pred_sig;
            let loss = diff.powf_scalar(2.0).mean();

            // `backward(&self)` borrows, so reading the scalar after
            // it returns is fine.
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            let loss_scalar: f32 = loss.into_scalar();
            model = optim.step(current_lr, model, grads);

            // CPU trainer reports per-sample mean loss; we accumulate
            // `mean_loss * batch_len` and divide at end of epoch.
            epoch_loss += (loss_scalar as f64) * (batch_len as f64);
            epoch_count += batch_len as u64;
        }

        let avg_loss = if epoch_count == 0 { 0.0 } else { epoch_loss / epoch_count as f64 };
        eprintln!("Epoch {epoch}/{}: loss = {avg_loss:.8}", config.epochs);

        if config.save_rate > 0 && epoch % config.save_rate == 0 && epoch < config.epochs {
            let checkpoint_path = format!(
                "{}-epoch{}.nnue",
                config.output_path.trim_end_matches(".nnue"),
                epoch
            );
            let bytes = export_to_nnue_bytes(&model);
            match std::fs::write(&checkpoint_path, &bytes) {
                Ok(()) => eprintln!("  Checkpoint saved: {checkpoint_path}"),
                Err(e) => eprintln!("  WARNING: failed to save checkpoint: {e}"),
            }
        }

    }

    eprintln!("Quantizing and saving to '{}'...", config.output_path);
    let bytes = export_to_nnue_bytes(&model);
    std::fs::write(&config.output_path, &bytes)
        .unwrap_or_else(|e| panic!("Failed to write '{}': {e}", config.output_path));
    eprintln!("Done! Saved {} bytes ({} KB).", bytes.len(), bytes.len() / 1024);
    match crate::nnue::network::Network::from_bytes(&bytes) {
        Ok(_) => eprintln!("Verification: net file loads successfully."),
        Err(e) => eprintln!("WARNING: net file failed to load: {e}"),
    }
}

// ════════════════════════════════════════════════════════════════════════════
// CLI entry point
// ════════════════════════════════════════════════════════════════════════════

pub fn run_training_gpu(first_data_path: &str, args: &[String]) {
    let mut config = GpuTrainConfig {
        data_paths: vec![first_data_path.to_string()],
        data_weights: Vec::new(),
        output_path: "focalors.nnue".to_string(),
        resume_path: None,
        warmup_lr_factor: 0.3,
        warmup_epochs: 3,
        epochs: 20,
        batch_size: 16384,
        lr: 0.001,
        wdl_weight: 0.5,
        save_rate: 5,
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => {
                if let Some(p) = args.get(i + 1) {
                    config.data_paths.push(p.clone());
                }
                i += 2;
            }
            "--mix" => {
                if let Some(s) = args.get(i + 1) {
                    config.data_weights = s
                        .split(',')
                        .filter_map(|p| p.parse::<f32>().ok())
                        .collect();
                }
                i += 2;
            }
            "--resume" => {
                config.resume_path = args.get(i + 1).cloned();
                i += 2;
            }
            "--warmup-lr-factor" => {
                config.warmup_lr_factor =
                    args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0.3);
                i += 2;
            }
            "--warmup-epochs" => {
                config.warmup_epochs =
                    args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(3);
                i += 2;
            }
            "--epochs" => {
                config.epochs = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(20);
                i += 2;
            }
            "--batch-size" => {
                config.batch_size = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(16384);
                i += 2;
            }
            "--lr" => {
                config.lr = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0.001);
                i += 2;
            }
            "--wdl" => {
                config.wdl_weight = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0.5);
                i += 2;
            }
            "--output" => {
                config.output_path = args
                    .get(i + 1)
                    .cloned()
                    .unwrap_or_else(|| "focalors.nnue".into());
                i += 2;
            }
            "--save-rate" => {
                config.save_rate = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(5);
                i += 2;
            }
            "--threads" => {
                // Accepted-and-ignored for CLI parity with the CPU
                // trainer. GPU training uses one device — Burn handles
                // intra-op parallelism via the backend kernels.
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
    if !config.data_weights.is_empty() && config.data_weights.len() != config.data_paths.len() {
        eprintln!(
            "WARNING: --mix has {} weights but {} --data files; using equal weights",
            config.data_weights.len(),
            config.data_paths.len()
        );
        config.data_weights = vec![1.0; config.data_paths.len()];
    } else if config.data_weights.is_empty() && config.data_paths.len() > 1 {
        config.data_weights = vec![1.0; config.data_paths.len()];
    }
    train_gpu(&config);
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;

    use crate::trainer::{
        SIGMOID_SCALE, Sample, TrainNet, forward_output, load_data, loss_and_gradient,
        parse_record, quantize, sigmoid,
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

    // ─── Phase 2: Burn vs CPU forward numerical match ────────────────────

    /// Deterministic LCG mapped to uniform [-1, 1). The original version
    /// of this helper had a bit-math bug (`>> 33` keeps 31 bits but
    /// divided by `u32::MAX`) that made every output NEGATIVE — all
    /// weights and biases negative meant SCReLU zeroed every activation
    /// and both forwards collapsed to the constant `l3_bias / QB`,
    /// silently making the match gate vacuous (it passed even with the
    /// perspectives' concat order deliberately swapped).
    fn lcg_uniform(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Top 24 bits / 2^24 → [0, 1), same construction as
        // trainer::Rng::next_f32.
        let u = (*state >> 40) as f32 / (1u64 << 24) as f32;
        u * 2.0 - 1.0
    }

    /// Random weights at REALISTIC quantized-space magnitudes, scaled
    /// per layer so the FT accumulators genuinely sweep the SCReLU
    /// 0..QA=255 range, hidden layers sweep 0..QB=64, and the output
    /// lands at centipawn scale. At these scales a structural bug
    /// (swapped concat, transposed weight load, wrong clamp bound or
    /// division point) shifts the output by whole centipawns instead of
    /// hiding in the noise floor that near-zero weights produce.
    fn random_raw_weights(seed: u64) -> RawWeights {
        let mut s = seed;
        let mut mk = |n: usize, scale: f32| -> Vec<f32> {
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                out.push(lcg_uniform(&mut s) * scale);
            }
            out
        };
        // Scales chosen so pre-activations land mostly INSIDE the clamp
        // windows (some clipping, not wholesale saturation): a heavily
        // saturated net washes out small structural differences and
        // would blunt the negative control below.
        let ft_weights = mk(NUM_FEATURES * FT_SIZE, 40.0);
        let ft_biases = mk(FT_SIZE, 50.0);
        let l1_weights = mk(L1_INPUT * L1_SIZE, 0.08);
        let l1_biases = mk(L1_SIZE, 500.0);
        let l2_weights = mk(L1_SIZE * L2_SIZE, 0.3);
        let l2_biases = mk(L2_SIZE, 20.0);
        let l3_weights = mk(L2_SIZE, 100.0);
        let l3_bias = mk(1, 500.0)[0];
        RawWeights {
            ft_weights,
            ft_biases,
            l1_weights,
            l1_biases,
            l2_weights,
            l2_biases,
            l3_weights,
            l3_bias,
        }
    }

    /// Asymmetric STM/OPP feature lists (offset shifts both) so a
    /// perspective swap or concat-order bug produces a materially
    /// different output instead of silent equivalence.
    fn synthetic_sample(offset: usize) -> Sample {
        let shift = |base: [usize; 8], by: usize| -> Vec<usize> {
            base.iter().map(|f| (f + by) % NUM_FEATURES).collect()
        };
        Sample {
            stm_features: shift([0, 12, 64, 100, 250, 400, 500, 700], offset),
            opp_features: shift([1, 13, 65, 101, 251, 401, 501, 701], offset * 2),
            score: 0.0,
            result: 0.5,
        }
    }

    #[test]
    fn burn_forward_matches_cpu_forward() {
        let device = test_device();
        let batcher = NnueBatcher { wdl_weight: 0.5 };

        // Several independent seeds + samples so a single accidental
        // cancellation can't mask a structural bug.
        for (i, seed) in [0xC0FFEE_u64, 0xB00C, 0x5EED_5EED].into_iter().enumerate() {
            let raw = random_raw_weights(seed);
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
            let sample = synthetic_sample(i * 37);
            let cpu_out = forward_output(&cpu_net, &sample);

            // Anti-collapse guard: if the activations all die, the
            // forward degenerates to the constant l3_bias/QB and the
            // comparison below stops testing anything. Fail loudly
            // instead of passing vacuously.
            let collapsed = raw.l3_bias / QB as f32;
            assert!(
                (cpu_out - collapsed).abs() > 1.0,
                "seed {seed:#x}: test weights collapsed to l3_bias/QB; the gate is vacuous (cpu_out={cpu_out})"
            );

            let mut model: NnueModel<TestBackend> = NnueModel::new(&device);
            load_raw_weights(&mut model, raw, &device);

            // Pack through the production batcher so it is exercised by
            // the gate too.
            let batch: NnueBatch<TestBackend> = batcher.batch(vec![sample.clone()], &device);
            let burn_out: f32 = model
                .forward(batch.stm_idx, batch.stm_mask, batch.opp_idx, batch.opp_mask)
                .into_scalar();

            // Relative tolerance: at centipawn scale, f32 reordering
            // between the CPU scalar loops and Burn's batched matmuls
            // costs ~1e-5 relative; a structural bug costs whole
            // centipawns. Floor keeps near-zero outputs meaningful.
            let tol = (cpu_out.abs() * 1e-3).max(0.01);
            let diff = (burn_out - cpu_out).abs();
            assert!(
                diff < tol,
                "seed {seed:#x}: Burn vs CPU forward mismatch: burn={burn_out}, cpu={cpu_out}, diff={diff}, tol={tol}"
            );

            // Negative control: feeding the perspectives swapped must
            // change the output materially. Pins the STM-first concat
            // order and perspective wiring — the exact class of bug the
            // collapsed gate provably failed to catch.
            let batch2: NnueBatch<TestBackend> = batcher.batch(vec![sample], &device);
            let swapped: f32 = model
                .forward(batch2.opp_idx, batch2.opp_mask, batch2.stm_idx, batch2.stm_mask)
                .into_scalar();
            assert!(
                (swapped - cpu_out).abs() > 1.0,
                "seed {seed:#x}: swapped perspectives gave ~the same output ({swapped} vs {cpu_out}); gate is insensitive to perspective wiring"
            );
        }
    }

    // ─── Phase 3: batcher tensor shapes + WDL blend ──────────────────────

    #[test]
    fn batcher_pads_to_k_max_and_masks_padding() {
        let device = test_device();
        let s1 = Sample {
            stm_features: vec![3, 5],
            opp_features: vec![7],
            score: 0.0,
            result: 0.5,
        };
        let s2 = Sample {
            stm_features: vec![1, 2, 4, 6],
            opp_features: vec![8, 9],
            score: 0.0,
            result: 1.0,
        };
        let batcher = NnueBatcher { wdl_weight: 0.5 };
        let batch: NnueBatch<TestBackend> = batcher.batch(vec![s1, s2], &device);

        assert_eq!(batch.len, 2);
        let [b, k] = batch.stm_idx.dims();
        assert_eq!(b, 2);
        assert_eq!(k, 4, "K_max across both perspectives = 4 (s2.stm_features)");
        assert_eq!(batch.opp_idx.dims(), [2, 4]);
        assert_eq!(batch.stm_mask.dims(), [2, 4]);

        // Row 0 mask: [1, 1, 0, 0] (s1 had 2 stm features).
        // Row 1 mask: [1, 1, 1, 1] (s2 had 4 stm features).
        let stm_mask: Vec<f32> = batch.stm_mask.into_data().into_vec::<f32>().unwrap();
        assert_eq!(stm_mask, vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        // Target is WDL-blended: 0.5*result + 0.5*sigmoid(0)=0.5*0.5.
        let targets: Vec<f32> = batch.target.into_data().into_vec::<f32>().unwrap();
        assert!((targets[0] - (0.5 * 0.5 + 0.5 * 0.5)).abs() < 1e-6);
        assert!((targets[1] - (0.5 * 1.0 + 0.5 * 0.5)).abs() < 1e-6);
    }

    // ─── Phase 4: one Adam step reduces a synthetic loss ─────────────────
    //
    // Runs on Autodiff<NdArray> so it's hermetic — no wgpu adapter
    // needed in CI. The training loop's inner step shape (forward +
    // sigmoid loss + backward + step) is exercised verbatim.

    type AutodiffNd = Autodiff<NdArray>;

    #[test]
    fn single_step_reduces_loss() {
        let device = NdArrayDevice::Cpu;
        let raw = random_raw_weights(0xDEADBEEF);
        let mut model: NnueModel<AutodiffNd> = NnueModel::new(&device);
        load_raw_weights(&mut model, raw, &device);

        // Non-trivial target: score=600, result=1.0 -> blended target
        // ≈ 0.5*1.0 + 0.5*sigmoid(600/400) ≈ 0.93, well above the
        // initial prediction (~sigmoid(0)=0.5). The MSE then has a real
        // gradient for Adam to chase.
        let sample = Sample {
            stm_features: vec![0, 12, 64, 100, 250, 400, 500, 700],
            opp_features: vec![1, 13, 65, 101, 251, 401, 501, 701],
            score: 600.0,
            result: 1.0,
        };
        let batcher = NnueBatcher { wdl_weight: 0.5 };
        let batch: NnueBatch<AutodiffNd> = batcher.batch(vec![sample.clone()], &device);

        // Snapshot inputs so we can re-run forward after the step.
        let (idx_s, mask_s, idx_o, mask_o, tgt) = (
            batch.stm_idx.clone(),
            batch.stm_mask.clone(),
            batch.opp_idx.clone(),
            batch.opp_mask.clone(),
            batch.target.clone(),
        );

        let loss_before: f32 = {
            let pred = model.forward(
                idx_s.clone(),
                mask_s.clone(),
                idx_o.clone(),
                mask_o.clone(),
            );
            let pred_sig = burn::tensor::activation::sigmoid(pred.div_scalar(SIGMOID_SCALE));
            (tgt.clone() - pred_sig)
                .powf_scalar(2.0)
                .mean()
                .into_scalar()
        };

        let mut optim = AdamConfig::new()
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_epsilon(1e-8)
            .init::<AutodiffNd, NnueModel<AutodiffNd>>();

        // Snapshot the FT row of an active feature (index 0 is in
        // stm_features) so we can prove gradients flow all the way
        // through the select/mask/sum embedding path — not just into
        // the output bias, which is all the old all-negative-weights
        // version of this test actually validated.
        let ft_row_before: Vec<f32> = model
            .ft_weight
            .val()
            .inner()
            .into_data()
            .into_vec::<f32>()
            .unwrap()[0..FT_SIZE]
            .to_vec();

        let pred = model.forward(batch.stm_idx, batch.stm_mask, batch.opp_idx, batch.opp_mask);
        let pred_sig = burn::tensor::activation::sigmoid(pred.div_scalar(SIGMOID_SCALE));
        let loss = (batch.target - pred_sig).powf_scalar(2.0).mean();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        // Aggressive LR so a single step makes a measurable dent.
        model = optim.step(1e-2_f64, model, grads);

        let ft_row_after: Vec<f32> = model
            .ft_weight
            .val()
            .inner()
            .into_data()
            .into_vec::<f32>()
            .unwrap()[0..FT_SIZE]
            .to_vec();
        assert!(
            ft_row_before
                .iter()
                .zip(&ft_row_after)
                .any(|(b, a)| b != a),
            "feature-transformer row of an active feature did not move — gradients are not flowing through the embedding path"
        );

        let loss_after: f32 = {
            let pred = model.forward(idx_s, mask_s, idx_o, mask_o);
            let pred_sig = burn::tensor::activation::sigmoid(pred.div_scalar(SIGMOID_SCALE));
            (tgt - pred_sig).powf_scalar(2.0).mean().into_scalar()
        };

        assert!(
            loss_after < loss_before,
            "loss should decrease after one Adam step: before={loss_before}, after={loss_after}"
        );
    }

    // ─── Phase 5: end-to-end export produces a loadable .nnue ────────────

    #[test]
    fn export_round_trips_through_network_loader() {
        let device = NdArrayDevice::Cpu;
        let raw = random_raw_weights(0xCAFEF00D);
        let mut model: NnueModel<AutodiffNd> = NnueModel::new(&device);
        load_raw_weights(&mut model, raw, &device);

        // Call the real production export path — the same function
        // train_gpu calls for checkpoint + final save.
        let bytes = export_to_nnue_bytes(&model);
        assert_eq!(bytes.len(), 411_428);
        crate::nnue::network::Network::from_bytes(&bytes)
            .expect("quantized GPU-export must load through the production loader");
    }
}
