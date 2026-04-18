//! AVX2-accelerated NNUE inference primitives.
//!
//! Pure stdlib (no external crates). Uses runtime CPU detection
//! via `is_x86_feature_detected!` and dispatches to scalar fallback
//! on non-AVX2 CPUs.

use super::network::{L1_INPUT, L1_SIZE, QA, QB, Network};

/// Cached result of CPU feature detection.
use std::sync::OnceLock;
static AVX2_AVAILABLE: OnceLock<bool> = OnceLock::new();

#[inline]
pub fn has_avx2() -> bool {
    *AVX2_AVAILABLE.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// Scalar implementation of the L1 layer:
///   l1_out[j] = clamp((bias[j] + sum_i(input[i] * w[i][j])) / QA, 0, QB)
///
/// Used as the reference implementation and as the fallback on non-AVX2 CPUs.
#[inline]
pub fn l1_forward_scalar(
    l1_input: &[i32; L1_INPUT],
    net: &Network,
    l1_out: &mut [i32; L1_SIZE],
) {
    for j in 0..L1_SIZE {
        let mut sum = net.l1_biases[j];
        for i in 0..L1_INPUT {
            sum += l1_input[i] * net.l1_weights[i * L1_SIZE + j] as i32;
        }
        l1_out[j] = (sum / QA).clamp(0, QB);
    }
}

/// AVX2 implementation of the L1 layer.
///
/// Bit-exact equivalent to `l1_forward_scalar`. Processes all 32 output
/// neurons in parallel using 4 × `__m256i` (8 i32 lanes each).
///
/// # Safety
/// Caller must ensure the host CPU supports AVX2. This is enforced by
/// the `#[target_feature]` attribute on this function — calling it on
/// a CPU without AVX2 is undefined behavior.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn l1_forward_avx2(
    l1_input: &[i32; L1_INPUT],
    net: &Network,
    l1_out: &mut [i32; L1_SIZE],
) { unsafe {
    use std::arch::x86_64::*;

    // Initialize 4 accumulators with the L1 biases (32 i32 total)
    let bias_ptr = net.l1_biases.as_ptr() as *const __m256i;
    let mut acc0 = _mm256_loadu_si256(bias_ptr.add(0));
    let mut acc1 = _mm256_loadu_si256(bias_ptr.add(1));
    let mut acc2 = _mm256_loadu_si256(bias_ptr.add(2));
    let mut acc3 = _mm256_loadu_si256(bias_ptr.add(3));

    let weights_ptr = net.l1_weights.as_ptr();

    // For each input, broadcast it and accumulate into all 32 output neurons.
    // Layout of l1_weights is row-major [L1_INPUT][L1_SIZE] = [512][32].
    // Row i contains the 32 weights for input i (one per output neuron).
    for i in 0..L1_INPUT {
        let x = _mm256_set1_epi32(l1_input[i]);

        // Load 32 i8 weights for input i (one full row)
        let w_i8 = _mm256_loadu_si256(weights_ptr.add(i * L1_SIZE) as *const __m256i);

        // Sign-extend 32 × i8 → 32 × i32, splitting into 4 × __m256i (8 lanes each).
        // Steps:
        //   1. Split the 32 i8s into two halves (16 i8s each)
        //   2. Sign-extend each half to 16 i16s
        //   3. Split each 16-i16 vector into halves and sign-extend to 8 i32s
        let lo_i8 = _mm256_extracti128_si256::<0>(w_i8); // first 16 i8s
        let hi_i8 = _mm256_extracti128_si256::<1>(w_i8); // last 16 i8s

        let lo_i16 = _mm256_cvtepi8_epi16(lo_i8); // 16 i16s
        let hi_i16 = _mm256_cvtepi8_epi16(hi_i8); // 16 i16s

        let w0 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<0>(lo_i16)); // outputs 0-7
        let w1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(lo_i16)); // outputs 8-15
        let w2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<0>(hi_i16)); // outputs 16-23
        let w3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(hi_i16)); // outputs 24-31

        acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(x, w0));
        acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(x, w1));
        acc2 = _mm256_add_epi32(acc2, _mm256_mullo_epi32(x, w2));
        acc3 = _mm256_add_epi32(acc3, _mm256_mullo_epi32(x, w3));
    }

    // Store the raw sums to a scratch array, then divide-by-QA and clamp scalar.
    // Integer division by a non-power-of-2 in SIMD requires multiply-high tricks
    // that aren't worth the complexity for 32 elements.
    let mut tmp = [0i32; L1_SIZE];
    let tmp_ptr = tmp.as_mut_ptr() as *mut __m256i;
    _mm256_storeu_si256(tmp_ptr.add(0), acc0);
    _mm256_storeu_si256(tmp_ptr.add(1), acc1);
    _mm256_storeu_si256(tmp_ptr.add(2), acc2);
    _mm256_storeu_si256(tmp_ptr.add(3), acc3);

    for j in 0..L1_SIZE {
        l1_out[j] = (tmp[j] / QA).clamp(0, QB);
    }
}}

/// Dispatch to the best available L1 implementation.
#[inline]
pub fn l1_forward(
    l1_input: &[i32; L1_INPUT],
    net: &Network,
    l1_out: &mut [i32; L1_SIZE],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe { l1_forward_avx2(l1_input, net, l1_out); }
            return;
        }
    }
    l1_forward_scalar(l1_input, net, l1_out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::network::Network;

    /// Ensures the AVX2 implementation is bit-exactly equivalent to the scalar code.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn l1_avx2_matches_scalar() {
        if !has_avx2() {
            eprintln!("Skipping: AVX2 not available on this CPU");
            return;
        }

        // Use a real (random) network — gives realistic weight magnitudes
        let net = Network::random_for_test();

        // Generate 100 random L1 input arrays in the realistic SCReLU output range [0, 65025]
        let mut state = 0xCAFEBABE_u64;
        for trial in 0..100 {
            let mut l1_input = [0i32; L1_INPUT];
            for v in &mut l1_input {
                // xorshift PRNG
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *v = (state % 65026) as i32;  // 0..=65025
            }

            let mut out_scalar = [0i32; L1_SIZE];
            let mut out_simd = [0i32; L1_SIZE];

            l1_forward_scalar(&l1_input, &net, &mut out_scalar);
            unsafe { l1_forward_avx2(&l1_input, &net, &mut out_simd); }

            assert_eq!(
                out_scalar, out_simd,
                "Trial {trial}: AVX2 output diverges from scalar"
            );
        }
    }

    #[test]
    fn dispatch_works() {
        let net = Network::random_for_test();
        let l1_input = [1000i32; L1_INPUT];
        let mut out = [0i32; L1_SIZE];
        l1_forward(&l1_input, &net, &mut out);
        // Should not panic; outputs are clamped to [0, QB]
        for &v in &out {
            assert!(v >= 0 && v <= QB);
        }
    }
}
