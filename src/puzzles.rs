//! Puzzle extraction from post-game analysis.
//!
//! When the user blunders (CPL > 150, eval swing > 200cp), the position before
//! the blunder becomes a puzzle: "find the best move." Themes are detected via
//! attack-map heuristics.

use crate::analysis::{GameAnalysis, MoveClass};
use crate::attacks;
use crate::board::Board;
use crate::eval::{Score, MATE_SCORE};
use crate::movegen::{generate_legal_moves, make_move};
use crate::types::*;
use crate::uci::parse_move;

// ════════════════════════════════════════════════════════════════════════════
// Types
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PuzzleTheme {
    MateIn1,
    MateIn2,
    MateIn3,
    Fork,
    HangingPiece,
    BackRankMate,
    Tactical,
}

impl PuzzleTheme {
    pub fn label(self) -> &'static str {
        match self {
            PuzzleTheme::MateIn1 => "Mate in 1",
            PuzzleTheme::MateIn2 => "Mate in 2",
            PuzzleTheme::MateIn3 => "Mate in 3",
            PuzzleTheme::Fork => "Fork",
            PuzzleTheme::HangingPiece => "Hanging Piece",
            PuzzleTheme::BackRankMate => "Back Rank Mate",
            PuzzleTheme::Tactical => "Tactical",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            PuzzleTheme::MateIn1 => "There's a forced checkmate in 1 move.",
            PuzzleTheme::MateIn2 => "There's a forced checkmate in 2 moves.",
            PuzzleTheme::MateIn3 => "There's a forced checkmate in 3 moves.",
            PuzzleTheme::Fork => "Look for a piece that can attack two targets at once.",
            PuzzleTheme::HangingPiece => "One of the opponent's pieces is undefended.",
            PuzzleTheme::BackRankMate => "The opponent's back rank is weak.",
            PuzzleTheme::Tactical => "Find the strongest move in this position.",
        }
    }

    pub fn to_db_str(self) -> &'static str {
        match self {
            PuzzleTheme::MateIn1 => "mate_in_1",
            PuzzleTheme::MateIn2 => "mate_in_2",
            PuzzleTheme::MateIn3 => "mate_in_3",
            PuzzleTheme::Fork => "fork",
            PuzzleTheme::HangingPiece => "hanging_piece",
            PuzzleTheme::BackRankMate => "back_rank_mate",
            PuzzleTheme::Tactical => "tactical",
        }
    }

    pub fn from_db_str(s: &str) -> Self {
        match s {
            "mate_in_1" => PuzzleTheme::MateIn1,
            "mate_in_2" => PuzzleTheme::MateIn2,
            "mate_in_3" => PuzzleTheme::MateIn3,
            "fork" => PuzzleTheme::Fork,
            "hanging_piece" => PuzzleTheme::HangingPiece,
            "back_rank_mate" => PuzzleTheme::BackRankMate,
            _ => PuzzleTheme::Tactical,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PuzzleCandidate {
    pub fen: String,
    pub solution_uci: String,
    pub theme: PuzzleTheme,
    pub rating: i32,
    pub game_id: Option<i64>,
}

// ════════════════════════════════════════════════════════════════════════════
// Puzzle extraction
// ════════════════════════════════════════════════════════════════════════════

/// Extract puzzles from a completed game analysis.
/// Replays the move list to reconstruct board positions at blunder points.
pub fn extract_puzzles(
    uci_moves: &[String],
    analysis: &GameAnalysis,
    user_color: Color,
    user_rating: i32,
) -> Vec<PuzzleCandidate> {
    let mut puzzles = Vec::new();
    let mut board = Board::startpos();

    for (i, ma) in analysis.moves.iter().enumerate() {
        // Only extract from user's blunders
        if ma.side != user_color || !matches!(ma.classification, MoveClass::Blunder) {
            // Still need to advance the board
            if let Some(mv) = parse_move(&board, &uci_moves[i]) {
                make_move(&mut board, mv);
            }
            continue;
        }

        // Check eval swing is significant (> 200cp from user's perspective)
        let swing = if user_color == Color::White {
            ma.best_eval - ma.eval_after
        } else {
            ma.eval_after - ma.best_eval
        };
        if swing < 200 {
            if let Some(mv) = parse_move(&board, &uci_moves[i]) {
                make_move(&mut board, mv);
            }
            continue;
        }

        // This is a puzzle candidate: position before the blunder, solution = best move
        let best_move_uci = &ma.best_move_uci;

        // Detect theme
        let theme = if let Some(best_mv) = parse_move(&board, best_move_uci) {
            detect_theme(&board, best_mv, ma.best_eval)
        } else {
            PuzzleTheme::Tactical
        };

        // Puzzle rating: user rating + scaled CPL, clamped
        let rating = (user_rating + ma.cpl / 5).clamp(400, 2800);

        puzzles.push(PuzzleCandidate {
            fen: board.to_fen(),
            solution_uci: best_move_uci.clone(),
            theme,
            rating,
            game_id: None, // set by caller
        });

        // Advance board with the played move
        if let Some(mv) = parse_move(&board, &uci_moves[i]) {
            make_move(&mut board, mv);
        }
    }

    puzzles
}

// ════════════════════════════════════════════════════════════════════════════
// Theme detection
// ════════════════════════════════════════════════════════════════════════════

/// Detect the tactical theme of a puzzle position.
fn detect_theme(board: &Board, best_move: crate::moves::Move, best_eval: Score) -> PuzzleTheme {
    // Check for mate
    if best_eval.abs() > MATE_SCORE - 100 {
        let plies_to_mate = (MATE_SCORE - best_eval.abs()) as u32;
        let moves_to_mate = (plies_to_mate + 1) / 2;

        // Check for back rank mate
        if moves_to_mate <= 1 {
            if is_back_rank_mate(board, best_move) {
                return PuzzleTheme::BackRankMate;
            }
            return PuzzleTheme::MateIn1;
        }
        return match moves_to_mate {
            2 => PuzzleTheme::MateIn2,
            3 => PuzzleTheme::MateIn3,
            _ => PuzzleTheme::Tactical,
        };
    }

    // Check for fork: moved piece attacks 2+ valuable enemy pieces
    if is_fork(board, best_move) {
        return PuzzleTheme::Fork;
    }

    // Check for hanging piece: capturing an undefended piece
    if is_hanging_piece_capture(board, best_move) {
        return PuzzleTheme::HangingPiece;
    }

    PuzzleTheme::Tactical
}

/// Check if a mate is a back-rank mate (king on rank 1 or 8, blocked by own pieces).
fn is_back_rank_mate(board: &Board, best_move: crate::moves::Move) -> bool {
    let mut after = board.clone();
    make_move(&mut after, best_move);

    // Check if this is actually checkmate
    let legal = generate_legal_moves(&after);
    if !legal.is_empty() {
        return false;
    }

    // Find the mated king
    let mated_color = after.side_to_move;
    let king_bb = after.piece_bb(mated_color, Piece::King);
    if king_bb.0 == 0 {
        return false;
    }
    let king_sq_raw = king_bb.0.trailing_zeros() as u8;
    let king_rank = king_sq_raw / 8;

    // Back rank = rank 0 (black's back rank) or rank 7 (white's back rank)
    king_rank == 0 || king_rank == 7
}

/// Check if the best move creates a fork (piece attacks 2+ valuable enemy pieces).
fn is_fork(board: &Board, best_move: crate::moves::Move) -> bool {
    let mut after = board.clone();
    make_move(&mut after, best_move);

    let mover_color = board.side_to_move;
    let to_sq = best_move.to_sq();
    let occupancy = after.all_occupied();

    // What piece landed on the target square?
    let moved_piece = match after.piece_on(to_sq) {
        Some((_, p)) => p,
        None => return false,
    };

    // Get attack bitboard of the moved piece from its new square
    let attacks = match moved_piece {
        Piece::Knight => attacks::knight_attacks(to_sq),
        Piece::Bishop => attacks::bishop_attacks(to_sq, occupancy),
        Piece::Rook => attacks::rook_attacks(to_sq, occupancy),
        Piece::Queen => attacks::queen_attacks(to_sq, occupancy),
        Piece::Pawn => attacks::pawn_attacks(mover_color, to_sq),
        Piece::King => return false, // kings don't fork
    };

    // Count attacked enemy pieces worth >= knight (320cp)
    let enemy = mover_color.flip();
    let valuable_enemies = after.piece_bb(enemy, Piece::Knight)
        | after.piece_bb(enemy, Piece::Bishop)
        | after.piece_bb(enemy, Piece::Rook)
        | after.piece_bb(enemy, Piece::Queen)
        | after.piece_bb(enemy, Piece::King);

    let attacked = attacks & valuable_enemies;
    attacked.0.count_ones() >= 2
}

/// Check if the best move captures an undefended piece.
fn is_hanging_piece_capture(board: &Board, best_move: crate::moves::Move) -> bool {
    let to_sq = best_move.to_sq();
    let enemy = board.side_to_move.flip();

    // Is there an enemy piece on the target square?
    let captured = match board.piece_on(to_sq) {
        Some((color, piece)) if color == enemy && piece != Piece::Pawn => piece,
        _ => return false,
    };

    // Is the captured piece defended?
    let _ = captured; // we just need it to be a non-pawn
    !attacks::is_square_attacked(board, to_sq, enemy)
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn theme_labels_roundtrip() {
        for theme in [
            PuzzleTheme::MateIn1,
            PuzzleTheme::MateIn2,
            PuzzleTheme::MateIn3,
            PuzzleTheme::Fork,
            PuzzleTheme::HangingPiece,
            PuzzleTheme::BackRankMate,
            PuzzleTheme::Tactical,
        ] {
            let s = theme.to_db_str();
            assert_eq!(PuzzleTheme::from_db_str(s), theme);
        }
    }

    #[test]
    fn fork_detection_knight() {
        attacks::init();
        // Knight on d3 can jump to e5 forking king g6 and rook c6
        let board = Board::from_fen("8/8/2r3k1/8/8/3N4/8/4K3 w - - 0 1").unwrap();
        let mv = parse_move(&board, "d3e5").unwrap();
        assert!(is_fork(&board, mv));
    }

    #[test]
    fn hanging_piece_detection() {
        attacks::init();
        // Black rook on a5 is undefended, white queen can capture
        let board = Board::from_fen("8/8/8/r7/8/8/8/Q3K2k w - - 0 1").unwrap();
        let mv = parse_move(&board, "a1a5").unwrap();
        assert!(is_hanging_piece_capture(&board, mv));
    }

    #[test]
    fn defended_piece_not_hanging() {
        attacks::init();
        // Black rook on a5 defended by pawn on b6
        let board = Board::from_fen("8/8/1p6/r7/8/8/8/Q3K2k w - - 0 1").unwrap();
        let mv = parse_move(&board, "a1a5").unwrap();
        assert!(!is_hanging_piece_capture(&board, mv));
    }

    #[test]
    fn back_rank_mate_detection() {
        attacks::init();
        // Classic back rank: Rook delivers mate on e8
        let board = Board::from_fen("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        let mv = parse_move(&board, "a1a8").unwrap();
        assert!(is_back_rank_mate(&board, mv));
    }

    #[test]
    fn mate_theme_detected() {
        attacks::init();
        let board = Board::from_fen("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        let mv = parse_move(&board, "a1a8").unwrap();
        let theme = detect_theme(&board, mv, MATE_SCORE - 1);
        assert_eq!(theme, PuzzleTheme::BackRankMate);
    }

    #[test]
    fn non_mate_tactical_fallback() {
        attacks::init();
        // Position where best move is just positionally strong, no specific theme
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap();
        let mv = parse_move(&board, "e7e5").unwrap();
        let theme = detect_theme(&board, mv, 50);
        assert_eq!(theme, PuzzleTheme::Tactical);
    }
}
