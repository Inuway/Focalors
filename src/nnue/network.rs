//! NNUE network weights: storage, loading, and binary format parsing.
//!
//! Architecture: 768 -> 256x2 -> 32 -> 32 -> 1
//! Format: `bullet` trainer compatible (little-endian, sequential).

use std::sync::OnceLock;

use super::features::NUM_FEATURES;

/// Feature transformer hidden size (per perspective).
pub const FT_SIZE: usize = 256;

/// Hidden layer 1 input size (both perspectives concatenated).
pub const L1_INPUT: usize = FT_SIZE * 2; // 512

/// Hidden layer sizes.
pub const L1_SIZE: usize = 32;
pub const L2_SIZE: usize = 32;

/// Quantization constants.
pub const QA: i32 = 255; // Feature transformer output clamp range [0, QA]
pub const QB: i32 = 64;  // Hidden layer output clamp range [0, QB]

/// The NNUE network weights (loaded once, shared read-only).
pub struct Network {
    /// Feature transformer weights: [NUM_FEATURES][FT_SIZE] stored row-major.
    pub ft_weights: Vec<i16>,
    /// Feature transformer biases: [FT_SIZE].
    pub ft_biases: Vec<i16>,

    /// Layer 1 weights: [L1_INPUT][L1_SIZE] stored row-major.
    pub l1_weights: Vec<i8>,
    /// Layer 1 biases: [L1_SIZE].
    pub l1_biases: Vec<i32>,

    /// Layer 2 weights: [L1_SIZE][L2_SIZE] stored row-major.
    pub l2_weights: Vec<i8>,
    /// Layer 2 biases: [L2_SIZE].
    pub l2_biases: Vec<i32>,

    /// Output layer weights: [L2_SIZE].
    pub l3_weights: Vec<i8>,
    /// Output layer bias.
    pub l3_bias: i32,
}

/// Global network instance.
static NETWORK: OnceLock<Network> = OnceLock::new();

/// Get a reference to the loaded network.
pub fn get_network() -> Option<&'static Network> {
    NETWORK.get()
}

/// Initialize the network from raw bytes. Returns error if already initialized
/// or if the data is malformed.
pub fn init_from_bytes(data: &[u8]) -> Result<(), String> {
    let net = Network::from_bytes(data)?;
    NETWORK.set(net).map_err(|_| "Network already initialized".into())
}

/// Generate a randomly-initialized network for testing.
/// Uses a simple deterministic PRNG so results are reproducible.
#[cfg(test)]
pub fn init_random() -> Result<(), String> {
    let net = Network::random();
    NETWORK.set(net).map_err(|_| "Network already initialized".into())
}

impl Network {
    /// Parse network weights from a binary blob (bullet trainer format).
    ///
    /// Layout (all little-endian):
    ///   ft_biases:  [FT_SIZE] i16
    ///   ft_weights: [NUM_FEATURES * FT_SIZE] i16
    ///   l1_biases:  [L1_SIZE] i32
    ///   l1_weights: [L1_INPUT * L1_SIZE] i8
    ///   l2_biases:  [L2_SIZE] i32
    ///   l2_weights: [L1_SIZE * L2_SIZE] i8
    ///   l3_bias:    1 i32
    ///   l3_weights: [L2_SIZE] i8
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let mut cursor = 0usize;

        let ft_biases = read_i16s(data, &mut cursor, FT_SIZE)?;
        let ft_weights = read_i16s(data, &mut cursor, NUM_FEATURES * FT_SIZE)?;
        let l1_biases = read_i32s(data, &mut cursor, L1_SIZE)?;
        let l1_weights = read_i8s(data, &mut cursor, L1_INPUT * L1_SIZE)?;
        let l2_biases = read_i32s(data, &mut cursor, L2_SIZE)?;
        let l2_weights = read_i8s(data, &mut cursor, L1_SIZE * L2_SIZE)?;
        let l3_bias = {
            let vals = read_i32s(data, &mut cursor, 1)?;
            vals[0]
        };
        let l3_weights = read_i8s(data, &mut cursor, L2_SIZE)?;

        Ok(Network {
            ft_weights,
            ft_biases,
            l1_weights,
            l1_biases,
            l2_weights,
            l2_biases,
            l3_weights,
            l3_bias,
        })
    }

    /// Create a randomly-initialized network for testing.
    #[cfg(test)]
    fn random() -> Self {
        let mut rng = SimpleRng::new(0xDEAD_BEEF);

        let ft_biases: Vec<i16> = (0..FT_SIZE).map(|_| rng.next_i16_small()).collect();
        let ft_weights: Vec<i16> = (0..NUM_FEATURES * FT_SIZE)
            .map(|_| rng.next_i16_small())
            .collect();
        let l1_biases: Vec<i32> = (0..L1_SIZE).map(|_| rng.next_i32_small()).collect();
        let l1_weights: Vec<i8> = (0..L1_INPUT * L1_SIZE)
            .map(|_| rng.next_i8())
            .collect();
        let l2_biases: Vec<i32> = (0..L2_SIZE).map(|_| rng.next_i32_small()).collect();
        let l2_weights: Vec<i8> = (0..L1_SIZE * L2_SIZE)
            .map(|_| rng.next_i8())
            .collect();
        let l3_bias = rng.next_i32_small();
        let l3_weights: Vec<i8> = (0..L2_SIZE).map(|_| rng.next_i8()).collect();

        Network {
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

    /// Create a random network for use in tests (not stored in the global).
    #[cfg(test)]
    pub fn random_for_test() -> Self {
        Self::random()
    }

    /// Layer 2 weight for input `i`, output `o`.
    #[inline]
    pub fn l2_weight(&self, input: usize, output: usize) -> i8 {
        self.l2_weights[input * L2_SIZE + output]
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Binary reading helpers
// ════════════════════════════════════════════════════════════════════════════

fn read_i8s(data: &[u8], cursor: &mut usize, count: usize) -> Result<Vec<i8>, String> {
    let end = *cursor + count;
    if end > data.len() {
        return Err(format!(
            "Unexpected EOF reading i8s: need {end} bytes, have {}",
            data.len()
        ));
    }
    let vals: Vec<i8> = data[*cursor..end].iter().map(|&b| b as i8).collect();
    *cursor = end;
    Ok(vals)
}

fn read_i16s(data: &[u8], cursor: &mut usize, count: usize) -> Result<Vec<i16>, String> {
    let bytes_needed = count * 2;
    let end = *cursor + bytes_needed;
    if end > data.len() {
        return Err(format!(
            "Unexpected EOF reading i16s: need {end} bytes, have {}",
            data.len()
        ));
    }
    let mut vals = Vec::with_capacity(count);
    for i in 0..count {
        let offset = *cursor + i * 2;
        let v = i16::from_le_bytes([data[offset], data[offset + 1]]);
        vals.push(v);
    }
    *cursor = end;
    Ok(vals)
}

fn read_i32s(data: &[u8], cursor: &mut usize, count: usize) -> Result<Vec<i32>, String> {
    let bytes_needed = count * 4;
    let end = *cursor + bytes_needed;
    if end > data.len() {
        return Err(format!(
            "Unexpected EOF reading i32s: need {end} bytes, have {}",
            data.len()
        ));
    }
    let mut vals = Vec::with_capacity(count);
    for i in 0..count {
        let offset = *cursor + i * 4;
        let v = i32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        vals.push(v);
    }
    *cursor = end;
    Ok(vals)
}

// ════════════════════════════════════════════════════════════════════════════
// Simple deterministic PRNG for test net generation
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
struct SimpleRng {
    state: u64,
}

#[cfg(test)]
impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_i8(&mut self) -> i8 {
        (self.next_u64() & 0xFF) as i8
    }

    /// Small i16 values suitable for feature transformer weights.
    fn next_i16_small(&mut self) -> i16 {
        ((self.next_u64() % 201) as i16) - 100 // range [-100, 100]
    }

    /// Small i32 values suitable for biases.
    fn next_i32_small(&mut self) -> i32 {
        ((self.next_u64() % 2001) as i32) - 1000 // range [-1000, 1000]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_network_has_correct_sizes() {
        let net = Network::random();
        assert_eq!(net.ft_biases.len(), FT_SIZE);
        assert_eq!(net.ft_weights.len(), NUM_FEATURES * FT_SIZE);
        assert_eq!(net.l1_biases.len(), L1_SIZE);
        assert_eq!(net.l1_weights.len(), L1_INPUT * L1_SIZE);
        assert_eq!(net.l2_biases.len(), L2_SIZE);
        assert_eq!(net.l2_weights.len(), L1_SIZE * L2_SIZE);
        assert_eq!(net.l3_weights.len(), L2_SIZE);
    }

    #[test]
    fn roundtrip_serialization() {
        let net = Network::random();

        // Serialize to bytes
        let mut bytes = Vec::new();
        for &v in &net.ft_biases {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &net.ft_weights {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &net.l1_biases {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &net.l1_weights {
            bytes.push(v as u8);
        }
        for &v in &net.l2_biases {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &net.l2_weights {
            bytes.push(v as u8);
        }
        bytes.extend_from_slice(&net.l3_bias.to_le_bytes());
        for &v in &net.l3_weights {
            bytes.push(v as u8);
        }

        // Deserialize and compare
        let net2 = Network::from_bytes(&bytes).expect("Should parse successfully");
        assert_eq!(net.ft_biases, net2.ft_biases);
        assert_eq!(net.ft_weights, net2.ft_weights);
        assert_eq!(net.l1_biases, net2.l1_biases);
        assert_eq!(net.l1_weights, net2.l1_weights);
        assert_eq!(net.l2_biases, net2.l2_biases);
        assert_eq!(net.l2_weights, net2.l2_weights);
        assert_eq!(net.l3_bias, net2.l3_bias);
        assert_eq!(net.l3_weights, net2.l3_weights);
    }

    #[test]
    fn rejects_truncated_data() {
        let result = Network::from_bytes(&[0u8; 100]);
        assert!(result.is_err());
    }
}
