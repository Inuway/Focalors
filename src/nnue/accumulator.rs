//! NNUE accumulator: caches the feature transformer output for each perspective.
//!
//! The accumulator stores the result of: bias + sum(active_feature_weights)
//! for both white and black perspectives. Incremental updates add/subtract
//! weight columns when pieces move, avoiding full recomputation.

use super::features::{self, NUM_FEATURES};
use super::network::{Network, FT_SIZE};
use crate::board::Board;
use crate::types::*;

/// Feature transformer accumulator for one position.
#[derive(Clone)]
pub struct Accumulator {
    /// White perspective: FT output values.
    pub white: [i16; FT_SIZE],
    /// Black perspective: FT output values.
    pub black: [i16; FT_SIZE],
}

impl Default for Accumulator {
    fn default() -> Self {
        Accumulator {
            white: [0i16; FT_SIZE],
            black: [0i16; FT_SIZE],
        }
    }
}

impl Accumulator {
    /// Full refresh: recompute accumulator from scratch for the given board.
    pub fn refresh(&mut self, board: &Board, net: &Network) {
        // Start with biases
        self.white.copy_from_slice(&net.ft_biases);
        self.black.copy_from_slice(&net.ft_biases);

        // Add weight columns for every piece on the board
        for color in [Color::White, Color::Black] {
            for piece in Piece::ALL {
                let mut bb = board.piece_bb(color, piece);
                while bb.is_not_empty() {
                    let sq = bb.pop_lsb();
                    let (w_idx, b_idx) = features::feature_indices(color, piece, sq);
                    self.add_feature(net, w_idx, b_idx);
                }
            }
        }
    }

    /// Add a feature (piece placed on the board).
    #[inline]
    pub fn add_feature(&mut self, net: &Network, white_idx: usize, black_idx: usize) {
        debug_assert!(white_idx < NUM_FEATURES);
        debug_assert!(black_idx < NUM_FEATURES);
        let w_offset = white_idx * FT_SIZE;
        let b_offset = black_idx * FT_SIZE;
        for i in 0..FT_SIZE {
            self.white[i] += net.ft_weights[w_offset + i];
            self.black[i] += net.ft_weights[b_offset + i];
        }
    }

    /// Remove a feature (piece removed from the board).
    #[inline]
    pub fn remove_feature(&mut self, net: &Network, white_idx: usize, black_idx: usize) {
        debug_assert!(white_idx < NUM_FEATURES);
        debug_assert!(black_idx < NUM_FEATURES);
        let w_offset = white_idx * FT_SIZE;
        let b_offset = black_idx * FT_SIZE;
        for i in 0..FT_SIZE {
            self.white[i] -= net.ft_weights[w_offset + i];
            self.black[i] -= net.ft_weights[b_offset + i];
        }
    }

    /// Move a feature (piece moves from one square to another).
    /// More efficient than separate remove + add.
    #[inline]
    pub fn move_feature(
        &mut self,
        net: &Network,
        old_white: usize,
        old_black: usize,
        new_white: usize,
        new_black: usize,
    ) {
        let ow = old_white * FT_SIZE;
        let ob = old_black * FT_SIZE;
        let nw = new_white * FT_SIZE;
        let nb = new_black * FT_SIZE;
        for i in 0..FT_SIZE {
            self.white[i] += net.ft_weights[nw + i] - net.ft_weights[ow + i];
            self.black[i] += net.ft_weights[nb + i] - net.ft_weights[ob + i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refresh_startpos_produces_nonzero() {
        crate::attacks::init();
        let board = Board::startpos();
        let net = Network::random_for_test();
        let mut acc = Accumulator::default();
        acc.refresh(&board, &net);

        // With 32 pieces on the board and random weights, the accumulator
        // should be non-zero (overwhelmingly likely with random weights).
        let sum_w: i64 = acc.white.iter().map(|&v| v as i64).sum();
        let sum_b: i64 = acc.black.iter().map(|&v| v as i64).sum();
        assert!(sum_w != 0 || sum_b != 0, "Accumulator should be non-zero for startpos");
    }

    #[test]
    fn add_then_remove_restores_state() {
        let net = Network::random_for_test();
        let mut acc = Accumulator::default();
        acc.white.copy_from_slice(&net.ft_biases);
        acc.black.copy_from_slice(&net.ft_biases);

        let original_white = acc.white;
        let original_black = acc.black;

        // Add a feature
        let (w_idx, b_idx) = features::feature_indices(Color::White, Piece::Knight, Square(27));
        acc.add_feature(&net, w_idx, b_idx);

        // Should have changed
        assert_ne!(acc.white, original_white);

        // Remove it
        acc.remove_feature(&net, w_idx, b_idx);

        // Should be back to original
        assert_eq!(acc.white, original_white);
        assert_eq!(acc.black, original_black);
    }
}
