//! NNUE (Efficiently Updatable Neural Network) evaluation.
//!
//! Architecture: 768 -> 256x2 -> 32 -> 32 -> 1 (HalfKA, SCReLU)
//!
//! Used for playing strength in search. The existing HCE in eval.rs is kept
//! for analysis/teaching explanations — that's Focalors' differentiator.

pub mod accumulator;
pub mod features;
pub mod network;
pub mod simd;

use accumulator::Accumulator;
use features::feature_indices;
use network::*;

use crate::board::Board;
use crate::eval::Score;
use crate::moves::{Move, MoveFlag};
use crate::types::*;

const MAX_PLY: usize = 128;

/// NNUE state for a search: pre-allocated accumulator stack.
pub struct NnueState {
    accumulators: Vec<Accumulator>,
}

impl NnueState {
    pub fn new() -> Self {
        let mut accumulators = Vec::with_capacity(MAX_PLY);
        for _ in 0..MAX_PLY {
            accumulators.push(Accumulator::default());
        }
        NnueState { accumulators }
    }

    /// Full refresh at the given ply from a board position.
    pub fn refresh(&mut self, board: &Board, ply: usize) {
        let net = get_network().expect("NNUE network not initialized");
        self.accumulators[ply].refresh(board, net);
    }

    /// Copy the accumulator at `ply` to `ply + 1` and incrementally update
    /// it for the given move. Call BEFORE recursing into the next ply.
    ///
    /// `board` is the position BEFORE the move is made.
    pub fn push_and_update(&mut self, board: &Board, mv: Move, ply: usize) {
        let net = get_network().expect("NNUE network not initialized");
        let us = board.side_to_move;
        let them = us.flip();
        let from = mv.from_sq();
        let to = mv.to_sq();

        // Copy current accumulator to next ply
        self.accumulators[ply + 1] = self.accumulators[ply].clone();
        let acc = &mut self.accumulators[ply + 1];

        match mv.flag() {
            MoveFlag::Normal => {
                let piece = board.piece_type_on(from).expect("No piece on from square");

                // Handle capture
                if let Some((cap_color, cap_piece)) = board.piece_on(to) {
                    if cap_color == them {
                        let (w_idx, b_idx) = feature_indices(cap_color, cap_piece, to);
                        acc.remove_feature(net, w_idx, b_idx);
                    }
                }

                // Move the piece
                let (old_w, old_b) = feature_indices(us, piece, from);
                let (new_w, new_b) = feature_indices(us, piece, to);
                acc.move_feature(net, old_w, old_b, new_w, new_b);
            }

            MoveFlag::Promotion => {
                let promo_piece = mv.promotion_piece();

                // Handle capture
                if let Some((cap_color, cap_piece)) = board.piece_on(to) {
                    if cap_color == them {
                        let (w_idx, b_idx) = feature_indices(cap_color, cap_piece, to);
                        acc.remove_feature(net, w_idx, b_idx);
                    }
                }

                // Remove pawn from source
                let (old_w, old_b) = feature_indices(us, Piece::Pawn, from);
                acc.remove_feature(net, old_w, old_b);

                // Add promoted piece at destination
                let (new_w, new_b) = feature_indices(us, promo_piece, to);
                acc.add_feature(net, new_w, new_b);
            }

            MoveFlag::EnPassant => {
                let captured_sq = Square::new(to.file(), from.rank());

                // Remove captured pawn
                let (cap_w, cap_b) = feature_indices(them, Piece::Pawn, captured_sq);
                acc.remove_feature(net, cap_w, cap_b);

                // Move our pawn
                let (old_w, old_b) = feature_indices(us, Piece::Pawn, from);
                let (new_w, new_b) = feature_indices(us, Piece::Pawn, to);
                acc.move_feature(net, old_w, old_b, new_w, new_b);
            }

            MoveFlag::Castling => {
                // Move king
                let (k_old_w, k_old_b) = feature_indices(us, Piece::King, from);
                let (k_new_w, k_new_b) = feature_indices(us, Piece::King, to);
                acc.move_feature(net, k_old_w, k_old_b, k_new_w, k_new_b);

                // Determine rook squares
                let (rook_from, rook_to) = if to.file() > from.file() {
                    // Kingside
                    (Square::new(7, from.rank()), Square::new(5, from.rank()))
                } else {
                    // Queenside
                    (Square::new(0, from.rank()), Square::new(3, from.rank()))
                };

                // Move rook
                let (r_old_w, r_old_b) = feature_indices(us, Piece::Rook, rook_from);
                let (r_new_w, r_new_b) = feature_indices(us, Piece::Rook, rook_to);
                acc.move_feature(net, r_old_w, r_old_b, r_new_w, r_new_b);
            }
        }
    }

    /// Copy the accumulator for a null move (no features change).
    pub fn push_null(&mut self, ply: usize) {
        self.accumulators[ply + 1] = self.accumulators[ply].clone();
    }

    /// Evaluate at a specific ply.
    pub fn evaluate_at_ply(&self, board: &Board, ply: usize) -> Score {
        let net = get_network().expect("NNUE network not initialized");
        forward(&self.accumulators[ply], board.side_to_move, net)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Forward pass
// ════════════════════════════════════════════════════════════════════════════

/// NNUE forward pass: accumulator -> hidden layers -> output score.
/// Returns evaluation in centipawns from the side-to-move's perspective.
fn forward(acc: &Accumulator, side_to_move: Color, net: &Network) -> Score {
    // Step 1: Select perspectives — side-to-move first
    let (stm_acc, opp_acc) = match side_to_move {
        Color::White => (&acc.white, &acc.black),
        Color::Black => (&acc.black, &acc.white),
    };

    // Step 2: SCReLU activation on feature transformer output
    // activated[i] = clamp(x, 0, QA)^2
    // We process both perspectives into a single [L1_INPUT] array.
    let mut l1_input = [0i32; L1_INPUT];
    for i in 0..FT_SIZE {
        let v = (stm_acc[i] as i32).clamp(0, QA);
        l1_input[i] = v * v;
    }
    for i in 0..FT_SIZE {
        let v = (opp_acc[i] as i32).clamp(0, QA);
        l1_input[FT_SIZE + i] = v * v;
    }

    // Step 3: Layer 1 — l1_input[512] * l1_weights[512][32] + l1_biases[32]
    // Dispatches to AVX2 implementation when available; bit-exact equivalent.
    let mut l1_out = [0i32; L1_SIZE];
    simd::l1_forward(&l1_input, net, &mut l1_out);

    // Step 4: Layer 2 — l1_out[32] * l2_weights[32][32] + l2_biases[32]
    let mut l2_out = [0i32; L2_SIZE];
    for j in 0..L2_SIZE {
        let mut sum = net.l2_biases[j];
        for i in 0..L1_SIZE {
            sum += l1_out[i] * net.l2_weight(i, j) as i32;
        }
        l2_out[j] = sum.clamp(0, QB);
    }

    // Step 5: Output layer — dot product + bias
    let mut output = net.l3_bias;
    for j in 0..L2_SIZE {
        output += l2_out[j] * net.l3_weights[j] as i32;
    }

    // Scale to centipawns.
    // The network output is in internal quantized units.
    // Divide by QB to account for hidden layer quantization.
    // The result is roughly in centipawn-ish units depending on training.
    output / QB
}

/// The default net shipped with the binary. Embedded at compile time.
/// Update by running: `cargo run --release -- promote <path-to-new.nnue>`
/// then rebuild.
const DEFAULT_NET: &[u8] = include_bytes!("../../nets/current.nnue");

/// Initialize the NNUE subsystem. Loads from `path` if provided,
/// otherwise uses the embedded default net.
pub fn init(path: Option<&str>) -> Result<(), String> {
    if get_network().is_some() {
        return Ok(()); // already initialized
    }

    if let Some(p) = path {
        let data = std::fs::read(p)
            .map_err(|e| format!("Failed to read NNUE net file '{}': {}", p, e))?;
        init_from_bytes(&data)
    } else {
        // Load the embedded default net
        init_from_bytes(DEFAULT_NET)
    }
}

/// Evaluate a board position from scratch (no incremental updates).
/// Convenience function for non-performance-critical use (tests, analysis).
#[cfg(test)]
pub fn evaluate_from_board(board: &Board) -> Score {
    let net = get_network().expect("NNUE network not initialized");
    let mut acc = Accumulator::default();
    acc.refresh(board, net);
    forward(&acc, board.side_to_move, net)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen;

    fn init_test_net() {
        let _ = init_random();
    }

    #[test]
    fn evaluate_startpos_does_not_crash() {
        crate::attacks::init();
        init_test_net();
        let board = Board::startpos();
        let score = evaluate_from_board(&board);
        // With random weights, we can't predict the exact value,
        // but it should be finite and not absurdly large.
        assert!(
            score.abs() < 100_000,
            "Score should be reasonable, got {score}"
        );
    }

    #[test]
    fn evaluate_is_symmetric() {
        crate::attacks::init();
        init_test_net();

        // Startpos should give ~same eval (it's symmetric, both sides equal)
        // But since side_to_move differs, we check consistency.
        let board = Board::startpos();
        let score = evaluate_from_board(&board);

        // The starting position is symmetric — score should be near 0
        // with a perfectly trained net. With random weights, just check it's stable.
        let score2 = evaluate_from_board(&board);
        assert_eq!(score, score2, "Same position should give same score");
    }

    #[test]
    fn nnue_state_refresh_matches_from_board() {
        crate::attacks::init();
        init_test_net();

        let board = Board::startpos();
        let score_direct = evaluate_from_board(&board);

        let mut state = NnueState::new();
        state.refresh(&board, 0);
        let score_state = state.evaluate_at_ply(&board, 0);

        assert_eq!(score_direct, score_state);
    }

    #[test]
    fn incremental_matches_full_refresh_after_e2e4() {
        crate::attacks::init();
        init_test_net();

        let board = Board::startpos();
        let mv = Move::new(Square(12), Square(28)); // e2e4

        // Incremental path
        let mut state = NnueState::new();
        state.refresh(&board, 0);
        state.push_and_update(&board, mv, 0);

        let mut board_after = board.clone();
        movegen::make_move(&mut board_after, mv);
        let score_incremental = state.evaluate_at_ply(&board_after, 1);

        // Full refresh path
        let mut state2 = NnueState::new();
        state2.refresh(&board_after, 0);
        let score_refresh = state2.evaluate_at_ply(&board_after, 0);

        assert_eq!(
            score_incremental, score_refresh,
            "Incremental update must match full refresh"
        );
    }

    #[test]
    fn incremental_matches_after_capture() {
        crate::attacks::init();
        init_test_net();

        // Set up a position where a capture is possible
        // Italian Game after 1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. Ng5
        // Then Bxf7+ is a capture
        let board = Board::from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 4 4")
            .unwrap();

        // Bxf7+ (c4 to f7 = square 26 to square 53)
        let mv = Move::new(Square(26), Square(53));

        let mut state = NnueState::new();
        state.refresh(&board, 0);
        state.push_and_update(&board, mv, 0);

        let mut board_after = board.clone();
        movegen::make_move(&mut board_after, mv);
        let score_inc = state.evaluate_at_ply(&board_after, 1);

        let mut state2 = NnueState::new();
        state2.refresh(&board_after, 0);
        let score_ref = state2.evaluate_at_ply(&board_after, 0);

        assert_eq!(score_inc, score_ref, "Capture: incremental must match refresh");
    }

    #[test]
    fn incremental_matches_after_castling() {
        crate::attacks::init();
        init_test_net();

        // Position where white can castle kingside
        let board = Board::from_fen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
            .unwrap();

        // O-O: king e1->g1 (square 4 to 6)
        let mv = Move::new_castling(Square(4), Square(6));

        let mut state = NnueState::new();
        state.refresh(&board, 0);
        state.push_and_update(&board, mv, 0);

        let mut board_after = board.clone();
        movegen::make_move(&mut board_after, mv);
        let score_inc = state.evaluate_at_ply(&board_after, 1);

        let mut state2 = NnueState::new();
        state2.refresh(&board_after, 0);
        let score_ref = state2.evaluate_at_ply(&board_after, 0);

        assert_eq!(score_inc, score_ref, "Castling: incremental must match refresh");
    }

    #[test]
    fn incremental_matches_after_en_passant() {
        crate::attacks::init();
        init_test_net();

        // Position with en passant available
        let board = Board::from_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
            .unwrap();

        // exd6 en passant: e5(36) to d6(43)
        let mv = Move::new_en_passant(Square(36), Square(43));

        let mut state = NnueState::new();
        state.refresh(&board, 0);
        state.push_and_update(&board, mv, 0);

        let mut board_after = board.clone();
        movegen::make_move(&mut board_after, mv);
        let score_inc = state.evaluate_at_ply(&board_after, 1);

        let mut state2 = NnueState::new();
        state2.refresh(&board_after, 0);
        let score_ref = state2.evaluate_at_ply(&board_after, 0);

        assert_eq!(score_inc, score_ref, "En passant: incremental must match refresh");
    }

    #[test]
    fn incremental_matches_after_promotion() {
        crate::attacks::init();
        init_test_net();

        // Position with a pawn about to promote
        let board = Board::from_fen("8/4P3/8/8/8/8/4k3/4K3 w - - 0 1").unwrap();

        // e7e8=Q: e7(52) to e8(60), promote to Queen
        let mv = Move::new_promotion(Square(52), Square(60), Piece::Queen);

        let mut state = NnueState::new();
        state.refresh(&board, 0);
        state.push_and_update(&board, mv, 0);

        let mut board_after = board.clone();
        movegen::make_move(&mut board_after, mv);
        let score_inc = state.evaluate_at_ply(&board_after, 1);

        let mut state2 = NnueState::new();
        state2.refresh(&board_after, 0);
        let score_ref = state2.evaluate_at_ply(&board_after, 0);

        assert_eq!(score_inc, score_ref, "Promotion: incremental must match refresh");
    }

    #[test]
    fn null_move_preserves_accumulator() {
        crate::attacks::init();
        init_test_net();

        let board = Board::startpos();
        let mut state = NnueState::new();
        state.refresh(&board, 0);

        state.push_null(0);

        // Accumulators at ply 0 and 1 should be identical
        assert_eq!(state.accumulators[0].white, state.accumulators[1].white);
        assert_eq!(state.accumulators[0].black, state.accumulators[1].black);
    }
}
