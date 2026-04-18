use crate::attacks::*;
use crate::board::Board;
use crate::moves::*;
use crate::types::*;

/// Generate all pseudo-legal moves, then filter to only legal ones.
pub fn generate_legal_moves(board: &Board) -> MoveList {
    let moves = generate_pseudo_legal_moves(board);
    let mut legal = MoveList::new();

    for i in 0..moves.len() {
        let mv = moves[i];
        if is_legal(board, mv) {
            legal.push(mv);
        }
    }

    legal
}

/// Check if a pseudo-legal move is actually legal (doesn't leave own king in check).
fn is_legal(board: &Board, mv: Move) -> bool {
    let mut clone = board.clone();
    make_move(&mut clone, mv);
    // After making the move, side_to_move has flipped — check if the *previous*
    // side's king is in check (which would make the move illegal).
    let king_color = board.side_to_move;
    let king_bb = clone.piece_bb(king_color, Piece::King);
    if king_bb.is_empty() {
        return false;
    }
    let king_sq = king_bb.lsb();
    !is_square_attacked(&clone, king_sq, king_color.flip())
}

/// Apply a move to the board (mutates in place). This is a minimal implementation
/// that handles all move types. It doesn't validate legality.
/// Incrementally updates the Zobrist hash.
pub fn make_move(board: &mut Board, mv: Move) {
    use crate::zobrist;

    let from = mv.from_sq();
    let to = mv.to_sq();
    let us = board.side_to_move;
    let them = us.flip();

    // Remove old en passant and castling from hash (we'll add new ones back)
    if let Some(ep_sq) = board.en_passant {
        board.hash ^= zobrist::en_passant_key(ep_sq.file());
    }
    let old_castling = board.castling;

    match mv.flag() {
        MoveFlag::Normal => {
            let piece = board.piece_type_on(from).expect("No piece on from square");
            let was_capture = matches!(board.piece_on(to), Some((cap_color, _)) if cap_color == them);

            // Capture
            if let Some((cap_color, cap_piece)) = board.piece_on(to) {
                if cap_color == them {
                    board.remove_piece(them, cap_piece, to);
                    board.hash ^= zobrist::piece_key(them, cap_piece, to);
                    if cap_piece == Piece::Pawn {
                        board.pawn_hash ^= zobrist::piece_key(them, Piece::Pawn, to);
                    }
                }
            }

            // Move piece
            board.remove_piece(us, piece, from);
            board.put_piece(us, piece, to);
            board.hash ^= zobrist::piece_key(us, piece, from);
            board.hash ^= zobrist::piece_key(us, piece, to);
            if piece == Piece::Pawn {
                board.pawn_hash ^= zobrist::piece_key(us, Piece::Pawn, from);
                board.pawn_hash ^= zobrist::piece_key(us, Piece::Pawn, to);
            }

            update_castling_rights(board, from, to);

            // En passant
            board.en_passant = None;
            if piece == Piece::Pawn {
                let diff = (to.0 as i8 - from.0 as i8).unsigned_abs();
                if diff == 16 {
                    let ep_sq = Square((from.0 as i16 + (to.0 as i16 - from.0 as i16) / 2) as u8);
                    board.en_passant = Some(ep_sq);
                    board.hash ^= zobrist::en_passant_key(ep_sq.file());
                }
            }

            if piece == Piece::Pawn || was_capture {
                board.halfmove_clock = 0;
            } else {
                board.halfmove_clock += 1;
            }
        }
        MoveFlag::Promotion => {
            if let Some((cap_color, cap_piece)) = board.piece_on(to) {
                if cap_color == them {
                    board.remove_piece(them, cap_piece, to);
                    board.hash ^= zobrist::piece_key(them, cap_piece, to);
                    if cap_piece == Piece::Pawn {
                        board.pawn_hash ^= zobrist::piece_key(them, Piece::Pawn, to);
                    }
                }
            }

            board.remove_piece(us, Piece::Pawn, from);
            board.put_piece(us, mv.promotion_piece(), to);
            board.hash ^= zobrist::piece_key(us, Piece::Pawn, from);
            board.hash ^= zobrist::piece_key(us, mv.promotion_piece(), to);
            // Pawn disappears (promoted) — remove from pawn hash
            board.pawn_hash ^= zobrist::piece_key(us, Piece::Pawn, from);

            update_castling_rights(board, from, to);
            board.en_passant = None;
            board.halfmove_clock = 0;
        }
        MoveFlag::EnPassant => {
            let captured_sq = Square::new(to.file(), from.rank());
            board.remove_piece(them, Piece::Pawn, captured_sq);
            board.remove_piece(us, Piece::Pawn, from);
            board.put_piece(us, Piece::Pawn, to);
            board.hash ^= zobrist::piece_key(them, Piece::Pawn, captured_sq);
            board.hash ^= zobrist::piece_key(us, Piece::Pawn, from);
            board.hash ^= zobrist::piece_key(us, Piece::Pawn, to);
            board.pawn_hash ^= zobrist::piece_key(them, Piece::Pawn, captured_sq);
            board.pawn_hash ^= zobrist::piece_key(us, Piece::Pawn, from);
            board.pawn_hash ^= zobrist::piece_key(us, Piece::Pawn, to);

            board.en_passant = None;
            board.halfmove_clock = 0;
        }
        MoveFlag::Castling => {
            board.remove_piece(us, Piece::King, from);
            board.put_piece(us, Piece::King, to);
            board.hash ^= zobrist::piece_key(us, Piece::King, from);
            board.hash ^= zobrist::piece_key(us, Piece::King, to);

            let (rook_from, rook_to) = if to.file() > from.file() {
                (Square::new(7, from.rank()), Square::new(5, from.rank()))
            } else {
                (Square::new(0, from.rank()), Square::new(3, from.rank()))
            };

            board.remove_piece(us, Piece::Rook, rook_from);
            board.put_piece(us, Piece::Rook, rook_to);
            board.hash ^= zobrist::piece_key(us, Piece::Rook, rook_from);
            board.hash ^= zobrist::piece_key(us, Piece::Rook, rook_to);

            match us {
                Color::White => board.castling.remove(CastlingRights::WHITE),
                Color::Black => board.castling.remove(CastlingRights::BLACK),
            }

            board.en_passant = None;
            board.halfmove_clock += 1;
        }
    }

    // Update castling hash (XOR out old, XOR in new)
    if board.castling != old_castling {
        board.hash ^= zobrist::castling_key(old_castling);
        board.hash ^= zobrist::castling_key(board.castling);
    }

    if us == Color::Black {
        board.fullmove_number += 1;
    }

    board.side_to_move = them;
    board.hash ^= zobrist::side_key();
}

fn update_castling_rights(board: &mut Board, from: Square, to: Square) {
    // If king moves, remove all castling for that side
    // If rook moves from its home square, remove that side's castling
    // If a rook is captured on its home square, remove that castling right

    // King home squares
    if from.0 == 4 {
        board.castling.remove(CastlingRights::WHITE);
    }
    if from.0 == 60 {
        board.castling.remove(CastlingRights::BLACK);
    }

    // Rook home squares (from)
    if from.0 == 0 {
        board.castling.remove(CastlingRights::WHITE_QUEEN);
    }
    if from.0 == 7 {
        board.castling.remove(CastlingRights::WHITE_KING);
    }
    if from.0 == 56 {
        board.castling.remove(CastlingRights::BLACK_QUEEN);
    }
    if from.0 == 63 {
        board.castling.remove(CastlingRights::BLACK_KING);
    }

    // Rook home squares (to — captures)
    if to.0 == 0 {
        board.castling.remove(CastlingRights::WHITE_QUEEN);
    }
    if to.0 == 7 {
        board.castling.remove(CastlingRights::WHITE_KING);
    }
    if to.0 == 56 {
        board.castling.remove(CastlingRights::BLACK_QUEEN);
    }
    if to.0 == 63 {
        board.castling.remove(CastlingRights::BLACK_KING);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Pseudo-legal move generation
// ════════════════════════════════════════════════════════════════════════════

fn generate_pseudo_legal_moves(board: &Board) -> MoveList {
    let mut moves = MoveList::new();
    let us = board.side_to_move;
    let them = us.flip();
    let our_pieces = board.color_bb(us);
    let their_pieces = board.color_bb(them);
    let occupied = board.all_occupied();
    let empty = board.all_empty();

    // ── Pawns ──────────────────────────────────────────────────────────

    generate_pawn_moves(board, us, our_pieces, their_pieces, empty, &mut moves);

    // ── Knights ────────────────────────────────────────────────────────

    let mut knights = board.piece_bb(us, Piece::Knight);
    while knights.is_not_empty() {
        let from = knights.pop_lsb();
        let targets = knight_attacks(from) & !our_pieces;
        add_moves_from_targets(from, targets, &mut moves);
    }

    // ── Bishops ────────────────────────────────────────────────────────

    let mut bishops = board.piece_bb(us, Piece::Bishop);
    while bishops.is_not_empty() {
        let from = bishops.pop_lsb();
        let targets = bishop_attacks(from, occupied) & !our_pieces;
        add_moves_from_targets(from, targets, &mut moves);
    }

    // ── Rooks ──────────────────────────────────────────────────────────

    let mut rooks = board.piece_bb(us, Piece::Rook);
    while rooks.is_not_empty() {
        let from = rooks.pop_lsb();
        let targets = rook_attacks(from, occupied) & !our_pieces;
        add_moves_from_targets(from, targets, &mut moves);
    }

    // ── Queens ─────────────────────────────────────────────────────────

    let mut queens = board.piece_bb(us, Piece::Queen);
    while queens.is_not_empty() {
        let from = queens.pop_lsb();
        let targets = queen_attacks(from, occupied) & !our_pieces;
        add_moves_from_targets(from, targets, &mut moves);
    }

    // ── King ───────────────────────────────────────────────────────────

    let king_sq = board.piece_bb(us, Piece::King).lsb();
    let king_targets = king_attacks(king_sq) & !our_pieces;
    add_moves_from_targets(king_sq, king_targets, &mut moves);

    // ── Castling ───────────────────────────────────────────────────────

    generate_castling_moves(board, us, occupied, &mut moves);

    moves
}

fn generate_pawn_moves(
    board: &Board,
    us: Color,
    _our_pieces: Bitboard,
    their_pieces: Bitboard,
    empty: Bitboard,
    moves: &mut MoveList,
) {
    let pawns = board.piece_bb(us, Piece::Pawn);
    let promo_rank = if us == Color::White { 7u8 } else { 0u8 };

    match us {
        Color::White => {
            // Single push
            let mut single = Bitboard((pawns.0 << 8) & empty.0);
            // Double push (only from rank 2)
            let rank3 = Bitboard(0x0000_0000_00FF_0000); // rank 3
            let mut double = Bitboard(((single.0 & rank3.0) << 8) & empty.0);

            while single.is_not_empty() {
                let to = single.pop_lsb();
                let from = Square(to.0 - 8);
                if to.rank() == promo_rank {
                    add_promotions(from, to, moves);
                } else {
                    moves.push(Move::new(from, to));
                }
            }

            while double.is_not_empty() {
                let to = double.pop_lsb();
                let from = Square(to.0 - 16);
                moves.push(Move::new(from, to));
            }

            // Captures
            let mut left_cap = Bitboard((pawns.0 << 7) & !FILE_H.0 & their_pieces.0);
            let mut right_cap = Bitboard((pawns.0 << 9) & !FILE_A.0 & their_pieces.0);

            while left_cap.is_not_empty() {
                let to = left_cap.pop_lsb();
                let from = Square(to.0 - 7);
                if to.rank() == promo_rank {
                    add_promotions(from, to, moves);
                } else {
                    moves.push(Move::new(from, to));
                }
            }

            while right_cap.is_not_empty() {
                let to = right_cap.pop_lsb();
                let from = Square(to.0 - 9);
                if to.rank() == promo_rank {
                    add_promotions(from, to, moves);
                } else {
                    moves.push(Move::new(from, to));
                }
            }
        }
        Color::Black => {
            // Single push
            let mut single = Bitboard((pawns.0 >> 8) & empty.0);
            // Double push (only from rank 7)
            let rank6 = Bitboard(0x0000_FF00_0000_0000); // rank 6
            let mut double = Bitboard(((single.0 & rank6.0) >> 8) & empty.0);

            while single.is_not_empty() {
                let to = single.pop_lsb();
                let from = Square(to.0 + 8);
                if to.rank() == promo_rank {
                    add_promotions(from, to, moves);
                } else {
                    moves.push(Move::new(from, to));
                }
            }

            while double.is_not_empty() {
                let to = double.pop_lsb();
                let from = Square(to.0 + 16);
                moves.push(Move::new(from, to));
            }

            // Captures
            let mut left_cap = Bitboard((pawns.0 >> 9) & !FILE_H.0 & their_pieces.0);
            let mut right_cap = Bitboard((pawns.0 >> 7) & !FILE_A.0 & their_pieces.0);

            while left_cap.is_not_empty() {
                let to = left_cap.pop_lsb();
                let from = Square(to.0 + 9);
                if to.rank() == promo_rank {
                    add_promotions(from, to, moves);
                } else {
                    moves.push(Move::new(from, to));
                }
            }

            while right_cap.is_not_empty() {
                let to = right_cap.pop_lsb();
                let from = Square(to.0 + 7);
                if to.rank() == promo_rank {
                    add_promotions(from, to, moves);
                } else {
                    moves.push(Move::new(from, to));
                }
            }
        }
    }

    // En passant
    if let Some(ep_sq) = board.en_passant {
        let attackers = pawn_attacks(us.flip(), ep_sq) & pawns;
        let mut att = attackers;
        while att.is_not_empty() {
            let from = att.pop_lsb();
            moves.push(Move::new_en_passant(from, ep_sq));
        }
    }
}

fn add_promotions(from: Square, to: Square, moves: &mut MoveList) {
    moves.push(Move::new_promotion(from, to, Piece::Queen));
    moves.push(Move::new_promotion(from, to, Piece::Rook));
    moves.push(Move::new_promotion(from, to, Piece::Bishop));
    moves.push(Move::new_promotion(from, to, Piece::Knight));
}

fn add_moves_from_targets(from: Square, mut targets: Bitboard, moves: &mut MoveList) {
    while targets.is_not_empty() {
        let to = targets.pop_lsb();
        moves.push(Move::new(from, to));
    }
}

fn generate_castling_moves(board: &Board, us: Color, occupied: Bitboard, moves: &mut MoveList) {
    let (king_sq, kingside_right, queenside_right, rank) = match us {
        Color::White => (
            Square::new(4, 0),
            CastlingRights::WHITE_KING,
            CastlingRights::WHITE_QUEEN,
            0u8,
        ),
        Color::Black => (
            Square::new(4, 7),
            CastlingRights::BLACK_KING,
            CastlingRights::BLACK_QUEEN,
            7u8,
        ),
    };

    let them = us.flip();

    // Kingside: squares f1/f8 and g1/g8 must be empty and not attacked, king not in check
    if board.castling.contains(kingside_right) {
        let f = Square::new(5, rank);
        let g = Square::new(6, rank);

        if !occupied.contains(f)
            && !occupied.contains(g)
            && !is_square_attacked(board, king_sq, them)
            && !is_square_attacked(board, f, them)
            && !is_square_attacked(board, g, them)
        {
            moves.push(Move::new_castling(king_sq, g));
        }
    }

    // Queenside: squares b1/b8, c1/c8, d1/d8 must be empty; c1/c8, d1/d8 not attacked
    if board.castling.contains(queenside_right) {
        let b = Square::new(1, rank);
        let c = Square::new(2, rank);
        let d = Square::new(3, rank);

        if !occupied.contains(b)
            && !occupied.contains(c)
            && !occupied.contains(d)
            && !is_square_attacked(board, king_sq, them)
            && !is_square_attacked(board, c, them)
            && !is_square_attacked(board, d, them)
        {
            moves.push(Move::new_castling(king_sq, c));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startpos_has_20_moves() {
        init();
        let board = Board::startpos();
        let moves = generate_legal_moves(&board);
        assert_eq!(moves.len(), 20, "Starting position should have exactly 20 legal moves");
    }

    #[test]
    fn moves_after_e4() {
        init();
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1").unwrap();
        let moves = generate_legal_moves(&board);
        assert_eq!(moves.len(), 20, "After 1.e4, black should have 20 legal moves");
    }

    #[test]
    fn promotion_moves_generated() {
        init();
        // White pawn on e7, kings far apart — e8 is free for promotion
        let board = Board::from_fen("8/4P3/8/8/8/8/8/K6k w - - 0 1").unwrap();
        let moves = generate_legal_moves(&board);
        // Should include 4 promotion moves for e7e8
        let promos: Vec<_> = moves.iter().filter(|m| matches!(m.flag(), MoveFlag::Promotion)).collect();
        assert_eq!(promos.len(), 4, "Should have 4 promotion moves (Q/R/B/N)");
    }

    #[test]
    fn en_passant_generated() {
        init();
        // White pawn on e5, black just played d7d5
        let board = Board::from_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 1").unwrap();
        let moves = generate_legal_moves(&board);
        let ep: Vec<_> = moves.iter().filter(|m| m.flag() as u8 == MoveFlag::EnPassant as u8).collect();
        assert_eq!(ep.len(), 1, "Should have 1 en passant capture");
        assert_eq!(ep[0].to_sq().to_algebraic(), "d6");
    }

    #[test]
    fn castling_both_sides() {
        init();
        // Position where white can castle both sides
        let board = Board::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1").unwrap();
        let moves = generate_legal_moves(&board);
        let castles: Vec<_> = moves.iter().filter(|m| m.flag() as u8 == MoveFlag::Castling as u8).collect();
        assert_eq!(castles.len(), 2, "White should be able to castle both sides");
    }

    #[test]
    fn no_moves_in_checkmate() {
        init();
        // Scholar's mate position — black is checkmated
        let board = Board::from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1").unwrap();
        let moves = generate_legal_moves(&board);
        assert_eq!(moves.len(), 0, "Checkmated position should have 0 legal moves");
    }

    /// Perft: count leaf nodes at a given depth. This is THE standard correctness
    /// test for move generators — results must match known values exactly.
    fn perft(board: &Board, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }

        let moves = generate_legal_moves(board);

        if depth == 1 {
            return moves.len() as u64;
        }

        let mut nodes = 0u64;
        for i in 0..moves.len() {
            let mut clone = board.clone();
            make_move(&mut clone, moves[i]);
            nodes += perft(&clone, depth - 1);
        }
        nodes
    }

    #[test]
    fn perft_startpos_depth_1() {
        init();
        let board = Board::startpos();
        assert_eq!(perft(&board, 1), 20);
    }

    #[test]
    fn perft_startpos_depth_2() {
        init();
        let board = Board::startpos();
        assert_eq!(perft(&board, 2), 400);
    }

    #[test]
    fn perft_startpos_depth_3() {
        init();
        let board = Board::startpos();
        assert_eq!(perft(&board, 3), 8902);
    }

    #[test]
    fn perft_startpos_depth_4() {
        init();
        let board = Board::startpos();
        assert_eq!(perft(&board, 4), 197281);
    }
}
