//! NNUE feature index computation.
//!
//! Architecture: HalfKA 768 inputs per perspective.
//! Feature index = piece_color * 384 + piece_type * 64 + square
//!
//! White perspective: features as-is.
//! Black perspective: flip square vertically (sq XOR 56), swap colors.

use crate::types::*;

/// Total number of input features per perspective.
pub const NUM_FEATURES: usize = 768; // 2 colors * 6 piece types * 64 squares

/// Compute the feature index for a piece from white's perspective.
#[inline]
pub fn feature_index_white(piece_color: Color, piece: Piece, sq: Square) -> usize {
    piece_color as usize * 384 + piece as usize * 64 + sq.0 as usize
}

/// Compute the feature index for a piece from black's perspective.
/// Mirrors the board vertically and swaps colors.
#[inline]
pub fn feature_index_black(piece_color: Color, piece: Piece, sq: Square) -> usize {
    let flipped_color = piece_color.flip();
    let flipped_sq = sq.0 ^ 56; // vertical flip: rank 0↔7, 1↔6, etc.
    flipped_color as usize * 384 + piece as usize * 64 + flipped_sq as usize
}

/// Compute feature indices for both perspectives at once.
#[inline]
pub fn feature_indices(piece_color: Color, piece: Piece, sq: Square) -> (usize, usize) {
    (
        feature_index_white(piece_color, piece, sq),
        feature_index_black(piece_color, piece, sq),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_pawn_e2() {
        // White pawn on e2 (square 12) from white's perspective
        let idx = feature_index_white(Color::White, Piece::Pawn, Square(12));
        // color=0 * 384 + piece=0 * 64 + sq=12 = 12
        assert_eq!(idx, 12);
    }

    #[test]
    fn black_pawn_e7_from_white() {
        // Black pawn on e7 (square 52) from white's perspective
        let idx = feature_index_white(Color::Black, Piece::Pawn, Square(52));
        // color=1 * 384 + piece=0 * 64 + sq=52 = 384 + 52 = 436
        assert_eq!(idx, 436);
    }

    #[test]
    fn perspective_symmetry() {
        // A white knight on d4 (square 27) from white's perspective
        // should equal a black knight on d5 (square 27^56=35) from black's perspective
        // but with flipped color.
        //
        // White perspective: color=0, Knight=1, sq=27 → 0*384 + 1*64 + 27 = 91
        // Black perspective: flipped_color=1(Black→White=0 wait no...
        //
        // Actually: from black's perspective, we flip the color and the square.
        // So a white knight on d4:
        //   white_idx = 0*384 + 1*64 + 27 = 91
        //   black_idx: flipped_color = Black(1), flipped_sq = 27^56 = 35
        //              = 1*384 + 1*64 + 35 = 384 + 64 + 35 = 483
        let (w_idx, b_idx) = feature_indices(Color::White, Piece::Knight, Square(27));
        assert_eq!(w_idx, 91);
        assert_eq!(b_idx, 483);
    }

    #[test]
    fn mirror_symmetry() {
        // White king on e1 from white's perspective should mirror to
        // black king on e8 from black's perspective with same index.
        // White king e1 (sq=4): white_idx = 0*384 + 5*64 + 4 = 324
        // Black king e8 (sq=60): black_idx = flip_color(Black)=White(0), flip_sq=60^56=4
        //   = 0*384 + 5*64 + 4 = 324
        let w_idx = feature_index_white(Color::White, Piece::King, Square(4));
        let b_idx = feature_index_black(Color::Black, Piece::King, Square(60));
        assert_eq!(w_idx, b_idx, "Symmetric positions should have same feature index");
    }

    #[test]
    fn feature_index_bounds() {
        // All valid combinations should produce indices in [0, 768)
        for color in [Color::White, Color::Black] {
            for piece in Piece::ALL {
                for sq in 0..64u8 {
                    let w = feature_index_white(color, piece, Square(sq));
                    let b = feature_index_black(color, piece, Square(sq));
                    assert!(w < NUM_FEATURES, "White index {w} out of bounds");
                    assert!(b < NUM_FEATURES, "Black index {b} out of bounds");
                }
            }
        }
    }
}
