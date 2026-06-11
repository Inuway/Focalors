use crate::types::*;

/// Full board state — everything needed to determine the legal moves in a position.
#[derive(Debug, Clone)]
pub struct Board {
    /// Bitboards indexed by [Color][Piece]
    /// e.g. `pieces[Color::White as usize][Piece::Pawn as usize]` = all white pawns
    pieces: [[Bitboard; Piece::COUNT]; Color::COUNT],

    /// Aggregate bitboard per color (union of all piece bitboards for that side)
    occupancy: [Bitboard; Color::COUNT],

    /// Side to move
    pub side_to_move: Color,

    /// Castling rights
    pub castling: CastlingRights,

    /// En passant target square (the square a pawn can capture *to*, not the pawn itself)
    pub en_passant: Option<Square>,

    /// Halfmove clock (for 50-move rule)
    pub halfmove_clock: u8,

    /// Fullmove number (starts at 1, incremented after Black moves)
    pub fullmove_number: u16,

    /// Incrementally updated Zobrist hash of the position
    pub hash: u64,

    /// Zobrist hash of pawn positions only (for pawn hash table)
    pub pawn_hash: u64,
}

impl Board {
    // ── Accessors ──────────────────────────────────────────────────────

    /// Bitboard of a specific piece type for a specific color
    pub fn piece_bb(&self, color: Color, piece: Piece) -> Bitboard {
        self.pieces[color as usize][piece as usize]
    }

    /// All pieces of one color
    pub fn color_bb(&self, color: Color) -> Bitboard {
        self.occupancy[color as usize]
    }

    /// All occupied squares
    pub fn all_occupied(&self) -> Bitboard {
        self.occupancy[0] | self.occupancy[1]
    }

    /// All empty squares
    pub fn all_empty(&self) -> Bitboard {
        !self.all_occupied()
    }

    /// What piece (if any) sits on a given square? Returns (Color, Piece).
    pub fn piece_on(&self, sq: Square) -> Option<(Color, Piece)> {
        let bb = Bitboard::from_square(sq);
        for color in [Color::White, Color::Black] {
            if (self.occupancy[color as usize] & bb).is_empty() {
                continue;
            }
            for piece in Piece::ALL {
                if (self.pieces[color as usize][piece as usize] & bb).is_not_empty() {
                    return Some((color, piece));
                }
            }
        }
        None
    }

    /// What piece type sits on a given square (ignores color)?
    pub fn piece_type_on(&self, sq: Square) -> Option<Piece> {
        self.piece_on(sq).map(|(_, p)| p)
    }

    // ── Mutation helpers ───────────────────────────────────────────────

    /// Place a piece on the board (does not check if square is occupied)
    pub fn put_piece(&mut self, color: Color, piece: Piece, sq: Square) {
        let bb = Bitboard::from_square(sq);
        self.pieces[color as usize][piece as usize] |= bb;
        self.occupancy[color as usize] |= bb;
    }

    /// Remove a piece from the board
    pub fn remove_piece(&mut self, color: Color, piece: Piece, sq: Square) {
        let bb = !Bitboard::from_square(sq);
        self.pieces[color as usize][piece as usize] &= bb;
        self.occupancy[color as usize] &= bb;
    }


    /// Make a "null move" — pass the turn without moving.
    /// Used for null move pruning in search.
    pub fn make_null_move(&mut self) {
        // Remove old en passant from hash
        if let Some(ep_sq) = self.en_passant {
            self.hash ^= crate::zobrist::en_passant_key(ep_sq.file());
        }
        self.side_to_move = self.side_to_move.flip();
        self.en_passant = None;
        self.hash ^= crate::zobrist::side_key();
    }

    /// Check if the position has enough material for either side to
    /// deliver checkmate. Used to avoid null move pruning in zugzwang-prone
    /// endgames (e.g. king + pawns only).
    pub fn has_non_pawn_material(&self, color: Color) -> bool {
        let c = color as usize;
        (self.pieces[c][Piece::Knight as usize]
            | self.pieces[c][Piece::Bishop as usize]
            | self.pieces[c][Piece::Rook as usize]
            | self.pieces[c][Piece::Queen as usize])
        .is_not_empty()
    }

    pub fn is_insufficient_material(&self) -> bool {
        let white_pawns = self.piece_bb(Color::White, Piece::Pawn).popcount();
        let black_pawns = self.piece_bb(Color::Black, Piece::Pawn).popcount();
        let white_rooks = self.piece_bb(Color::White, Piece::Rook).popcount();
        let black_rooks = self.piece_bb(Color::Black, Piece::Rook).popcount();
        let white_queens = self.piece_bb(Color::White, Piece::Queen).popcount();
        let black_queens = self.piece_bb(Color::Black, Piece::Queen).popcount();

        if white_pawns + black_pawns + white_rooks + black_rooks + white_queens + black_queens > 0 {
            return false;
        }

        let white_bishops = self.piece_bb(Color::White, Piece::Bishop).popcount();
        let black_bishops = self.piece_bb(Color::Black, Piece::Bishop).popcount();
        let white_knights = self.piece_bb(Color::White, Piece::Knight).popcount();
        let black_knights = self.piece_bb(Color::Black, Piece::Knight).popcount();

        let white_minors = white_bishops + white_knights;
        let black_minors = black_bishops + black_knights;

        if white_minors == 0 && black_minors == 0 {
            return true;
        }

        if (white_minors == 1 && black_minors == 0) || (white_minors == 0 && black_minors == 1) {
            return true;
        }

        if white_knights == 0 && black_knights == 0 && white_bishops == 1 && black_bishops == 1 {
            let white_sq = self.piece_bb(Color::White, Piece::Bishop).lsb();
            let black_sq = self.piece_bb(Color::Black, Piece::Bishop).lsb();
            if is_light_square(white_sq) == is_light_square(black_sq) {
                return true;
            }
        }

        false
    }

    // ── FEN parsing ────────────────────────────────────────────────────
    //
    // FEN (Forsyth-Edwards Notation) is the standard way to describe a
    // chess position as a single string. Example for the starting position:
    //   rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return Err("FEN must have at least 4 fields".into());
        }

        let mut board = Board {
            pieces: [[Bitboard::EMPTY; Piece::COUNT]; Color::COUNT],
            occupancy: [Bitboard::EMPTY; Color::COUNT],
            side_to_move: Color::White,
            castling: CastlingRights::NONE,
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            hash: 0,
            pawn_hash: 0,
        };

        // 1) Piece placement (rank 8 first, separated by '/')
        let ranks: Vec<&str> = parts[0].split('/').collect();
        if ranks.len() != 8 {
            return Err("FEN piece placement must have 8 ranks".into());
        }

        for (rank_idx, rank_str) in ranks.iter().enumerate() {
            let rank = 7 - rank_idx as u8; // FEN starts from rank 8
            let mut file: u8 = 0;

            for ch in rank_str.chars() {
                if let Some(skip) = ch.to_digit(10) {
                    // Bounds-check BEFORE accumulating: a crafted rank
                    // like "999..." would otherwise overflow the u8 (and
                    // in release silently wrap, placing pieces on wrong
                    // squares while still passing the file == 8 check).
                    file = file
                        .checked_add(skip as u8)
                        .filter(|f| *f <= 8)
                        .ok_or_else(|| format!("Rank {rank} runs past file h"))?;
                } else {
                    if file >= 8 {
                        return Err(format!("Rank {rank} runs past file h"));
                    }
                    let color = if ch.is_uppercase() {
                        Color::White
                    } else {
                        Color::Black
                    };
                    let piece = match ch.to_ascii_lowercase() {
                        'p' => Piece::Pawn,
                        'n' => Piece::Knight,
                        'b' => Piece::Bishop,
                        'r' => Piece::Rook,
                        'q' => Piece::Queen,
                        'k' => Piece::King,
                        _ => return Err(format!("Invalid piece character: {ch}")),
                    };
                    board.put_piece(color, piece, Square::new(file, rank));
                    file += 1;
                }
            }

            if file != 8 {
                return Err(format!("Rank {rank} has {file} files instead of 8"));
            }
        }

        // Semantic validation: exactly one king per side. Movegen
        // assumes a king exists (it takes lsb() of the king bitboard
        // unconditionally); a kingless board would index out of bounds
        // and abort the process.
        for color in [Color::White, Color::Black] {
            let kings = board.piece_bb(color, Piece::King).popcount();
            if kings != 1 {
                return Err(format!(
                    "{color:?} must have exactly one king, found {kings}"
                ));
            }
        }

        // 2) Side to move
        board.side_to_move = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            other => return Err(format!("Invalid side to move: {other}")),
        };

        // The side that just moved cannot have left its king in check.
        // Search would otherwise "capture" that king, and evaluating the
        // kingless position takes lsb() of an empty bitboard — indexing
        // out of bounds and aborting the process.
        let opp = board.side_to_move.flip();
        let opp_king = board.piece_bb(opp, Piece::King).lsb();
        if crate::attacks::is_square_attacked(&board, opp_king, board.side_to_move) {
            return Err("Side not to move is in check (illegal position)".into());
        }

        // 3) Castling rights
        if parts[2] != "-" {
            for ch in parts[2].chars() {
                board.castling |= match ch {
                    'K' => CastlingRights::WHITE_KING,
                    'Q' => CastlingRights::WHITE_QUEEN,
                    'k' => CastlingRights::BLACK_KING,
                    'q' => CastlingRights::BLACK_QUEEN,
                    _ => return Err(format!("Invalid castling character: {ch}")),
                };
            }

            // Mask rights against actual piece placement — castling
            // move generation trusts these bits and would otherwise
            // conjure a rook (or a second king) out of thin air when a
            // FEN claims rights without the pieces on their home squares.
            let has = |color: Color, piece: Piece, sq: u8| {
                (board.piece_bb(color, piece) & Bitboard::from_square(Square(sq))).is_not_empty()
            };
            let wk = has(Color::White, Piece::King, 4);
            let bk = has(Color::Black, Piece::King, 60);
            let keep_wk = wk && has(Color::White, Piece::Rook, 7);
            let keep_wq = wk && has(Color::White, Piece::Rook, 0);
            let keep_bk = bk && has(Color::Black, Piece::Rook, 63);
            let keep_bq = bk && has(Color::Black, Piece::Rook, 56);
            if !keep_wk {
                board.castling.remove(CastlingRights::WHITE_KING);
            }
            if !keep_wq {
                board.castling.remove(CastlingRights::WHITE_QUEEN);
            }
            if !keep_bk {
                board.castling.remove(CastlingRights::BLACK_KING);
            }
            if !keep_bq {
                board.castling.remove(CastlingRights::BLACK_QUEEN);
            }
        }

        // 4) En passant target square. Keep it only when a side-to-move
        // pawn can actually capture onto it — mirrors make_move's
        // condition so FEN-loaded and move-reached positions hash
        // identically (matters for repetition detection and the TT).
        if parts[3] != "-" {
            let ep_sq = Square::from_algebraic(parts[3])
                .ok_or_else(|| format!("Invalid en passant square: {}", parts[3]))?;
            let pusher = board.side_to_move.flip();
            if (crate::attacks::pawn_attacks(pusher, ep_sq)
                & board.piece_bb(board.side_to_move, Piece::Pawn))
            .is_not_empty()
            {
                board.en_passant = Some(ep_sq);
            }
        }

        // 5) Halfmove clock (optional)
        if let Some(hm) = parts.get(4) {
            board.halfmove_clock = hm.parse().map_err(|_| "Invalid halfmove clock")?;
        }

        // 6) Fullmove number (optional)
        if let Some(fm) = parts.get(5) {
            board.fullmove_number = fm.parse().map_err(|_| "Invalid fullmove number")?;
        }

        // Compute initial Zobrist hashes
        board.hash = crate::zobrist::hash_position(&board);
        board.pawn_hash = compute_pawn_hash(&board);

        Ok(board)
    }

    /// Standard starting position
    pub fn startpos() -> Self {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap()
    }

    /// Convert the board back to a FEN string
    pub fn to_fen(&self) -> String {
        let mut fen = String::new();

        // 1) Piece placement
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let sq = Square::new(file, rank);
                if let Some((color, piece)) = self.piece_on(sq) {
                    if empty > 0 {
                        fen.push_str(&empty.to_string());
                        empty = 0;
                    }
                    let ch = match piece {
                        Piece::Pawn => 'p',
                        Piece::Knight => 'n',
                        Piece::Bishop => 'b',
                        Piece::Rook => 'r',
                        Piece::Queen => 'q',
                        Piece::King => 'k',
                    };
                    if color == Color::White {
                        fen.push(ch.to_ascii_uppercase());
                    } else {
                        fen.push(ch);
                    }
                } else {
                    empty += 1;
                }
            }
            if empty > 0 {
                fen.push_str(&empty.to_string());
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        // 2) Side to move
        fen.push(' ');
        fen.push(match self.side_to_move {
            Color::White => 'w',
            Color::Black => 'b',
        });

        // 3) Castling
        fen.push(' ');
        if self.castling.is_empty() {
            fen.push('-');
        } else {
            if self.castling.contains(CastlingRights::WHITE_KING) {
                fen.push('K');
            }
            if self.castling.contains(CastlingRights::WHITE_QUEEN) {
                fen.push('Q');
            }
            if self.castling.contains(CastlingRights::BLACK_KING) {
                fen.push('k');
            }
            if self.castling.contains(CastlingRights::BLACK_QUEEN) {
                fen.push('q');
            }
        }

        // 4) En passant
        fen.push(' ');
        match self.en_passant {
            Some(sq) => fen.push_str(&sq.to_algebraic()),
            None => fen.push('-'),
        }

        // 5) Halfmove clock and fullmove number
        fen.push_str(&format!(" {} {}", self.halfmove_clock, self.fullmove_number));

        fen
    }

    /// Pretty-print the board (useful for debugging)
    pub fn display(&self) -> String {
        let mut s = String::new();
        s.push_str("\n  +---+---+---+---+---+---+---+---+\n");
        for rank in (0..8).rev() {
            s.push_str(&format!("{} ", rank + 1));
            for file in 0..8 {
                let sq = Square::new(file, rank);
                let ch = match self.piece_on(sq) {
                    Some((Color::White, Piece::Pawn)) => 'P',
                    Some((Color::White, Piece::Knight)) => 'N',
                    Some((Color::White, Piece::Bishop)) => 'B',
                    Some((Color::White, Piece::Rook)) => 'R',
                    Some((Color::White, Piece::Queen)) => 'Q',
                    Some((Color::White, Piece::King)) => 'K',
                    Some((Color::Black, Piece::Pawn)) => 'p',
                    Some((Color::Black, Piece::Knight)) => 'n',
                    Some((Color::Black, Piece::Bishop)) => 'b',
                    Some((Color::Black, Piece::Rook)) => 'r',
                    Some((Color::Black, Piece::Queen)) => 'q',
                    Some((Color::Black, Piece::King)) => 'k',
                    None => '.',
                };
                s.push_str(&format!("| {ch} "));
            }
            s.push_str("|\n  +---+---+---+---+---+---+---+---+\n");
        }
        s.push_str("    a   b   c   d   e   f   g   h\n");
        s
    }
}

impl std::fmt::Display for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display())
    }
}

fn is_light_square(sq: Square) -> bool {
    (sq.file() + sq.rank()) % 2 == 0
}

/// Compute pawn hash from scratch (XOR of all pawn Zobrist keys).
fn compute_pawn_hash(board: &Board) -> u64 {
    let mut hash = 0u64;
    for color in [Color::White, Color::Black] {
        let mut pawns = board.piece_bb(color, Piece::Pawn);
        while pawns.is_not_empty() {
            let sq = pawns.pop_lsb();
            hash ^= crate::zobrist::piece_key(color, Piece::Pawn, sq);
        }
    }
    hash
}

#[cfg(test)]
mod insufficient_material_tests {
    use super::Board;

    #[test]
    fn detects_king_vs_king() {
        let board = Board::from_fen("8/8/8/4k3/8/8/4K3/8 w - - 0 1").unwrap();
        assert!(board.is_insufficient_material());
    }

    #[test]
    fn detects_knight_vs_king() {
        let board = Board::from_fen("8/8/8/4k3/8/8/4K2N/8 w - - 0 1").unwrap();
        assert!(board.is_insufficient_material());
    }

    #[test]
    fn detects_bishop_vs_king() {
        let board = Board::from_fen("8/8/8/4k3/8/8/4K1B1/8 w - - 0 1").unwrap();
        assert!(board.is_insufficient_material());
    }

    #[test]
    fn detects_same_color_bishops_only() {
        let board = Board::from_fen("8/8/8/4k3/8/6b1/4K3/2B5 w - - 0 1").unwrap();
        assert!(board.is_insufficient_material());
    }

    #[test]
    fn keeps_pawn_endings_playable() {
        let board = Board::from_fen("8/8/8/4k3/8/8/4K1P1/8 w - - 0 1").unwrap();
        assert!(!board.is_insufficient_material());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startpos_roundtrip() {
        let board = Board::startpos();
        assert_eq!(
            board.to_fen(),
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        );
    }

    #[test]
    fn startpos_piece_counts() {
        let board = Board::startpos();
        // Each side has 16 pieces
        assert_eq!(board.color_bb(Color::White).popcount(), 16);
        assert_eq!(board.color_bb(Color::Black).popcount(), 16);
        // 8 pawns each
        assert_eq!(board.piece_bb(Color::White, Piece::Pawn).popcount(), 8);
        assert_eq!(board.piece_bb(Color::Black, Piece::Pawn).popcount(), 8);
        // 1 king each
        assert_eq!(board.piece_bb(Color::White, Piece::King).popcount(), 1);
        assert_eq!(board.piece_bb(Color::Black, Piece::King).popcount(), 1);
    }

    #[test]
    fn fen_with_en_passant() {
        // Black pawn on d4 can genuinely capture onto e3 → ep kept.
        let fen = "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3";
        let board = Board::from_fen(fen).unwrap();
        assert_eq!(board.en_passant, Some(Square::from_algebraic("e3").unwrap()));
        assert_eq!(board.side_to_move, Color::Black);
        assert_eq!(board.to_fen(), fen);
    }

    /// FIDE 9.2.2: an ep square only distinguishes positions when an ep
    /// capture is actually available. A phantom ep square (no enemy pawn
    /// adjacent) must be dropped so the position hashes identically to
    /// the same placement reached without the double push.
    #[test]
    fn phantom_en_passant_is_filtered() {
        // After 1.e4: no black pawn can capture onto e3.
        let with_phantom =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
                .unwrap();
        assert_eq!(with_phantom.en_passant, None, "uncapturable ep must be dropped");

        // Identical placement declared without the ep square: same hash.
        let without =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
                .unwrap();
        assert_eq!(
            with_phantom.hash, without.hash,
            "phantom ep must not change the zobrist hash"
        );
    }

    /// Same rule applied by make_move: a double push with no adjacent
    /// enemy pawn must not set the ep square, so the position hashes the
    /// same as one reached by slow maneuvering.
    #[test]
    fn double_push_without_capturer_sets_no_ep() {
        crate::attacks::init();
        let mut board = Board::startpos();
        // 1.e4 — no black pawn can take en passant.
        let e2e4 = crate::moves::Move::new(Square(12), Square(28));
        crate::movegen::make_move(&mut board, e2e4);
        assert_eq!(board.en_passant, None);
        // Hash must equal the from-scratch hash of the same position.
        assert_eq!(board.hash, crate::zobrist::hash_position(&board));
    }

    #[test]
    fn kingless_fen_is_rejected() {
        assert!(Board::from_fen("8/8/8/8/8/8/8/8 w - - 0 1").is_err());
        assert!(Board::from_fen("8/8/8/4k3/8/8/8/8 w - - 0 1").is_err());
        // Two kings of one color also rejected.
        assert!(Board::from_fen("4k3/8/8/8/8/8/8/2K1K3 w - - 0 1").is_err());
    }

    #[test]
    fn side_not_to_move_in_check_is_rejected() {
        // White to move while the black king is already attacked by the
        // f7 knight — illegal; search would "capture" the king and then
        // evaluate a kingless board.
        assert!(Board::from_fen("6rk/5Npp/8/8/8/8/8/6QK w - - 0 1").is_err());
        // The side TO move being in check is an ordinary legal position.
        assert!(Board::from_fen("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1").is_ok());
    }

    #[test]
    fn castling_rights_masked_against_placement() {
        // Rights claimed but no rooks: rights dropped, castling not generated.
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w KQ - 0 1").unwrap();
        assert_eq!(board.castling, CastlingRights::NONE);

        // King not on e1: all white rights dropped even with rooks present.
        let board = Board::from_fen("r3k2r/8/8/8/8/8/8/R2K3R w KQkq - 0 1").unwrap();
        assert!(!board.castling.contains(CastlingRights::WHITE_KING));
        assert!(!board.castling.contains(CastlingRights::WHITE_QUEEN));
        assert!(board.castling.contains(CastlingRights::BLACK_KING));
        assert!(board.castling.contains(CastlingRights::BLACK_QUEEN));
    }

    #[test]
    fn fen_digit_overflow_is_rejected() {
        // Digits summing past file h must error, not wrap the u8.
        let many_nines = "9".repeat(29);
        assert!(Board::from_fen(&format!("{many_nines}/8/8/8/8/8/8/4K2k w - - 0 1")).is_err());
        assert!(
            Board::from_fen("8r9999999999999999999999999993/8/8/8/8/8/8/4K2k w - - 0 1").is_err()
        );
    }

    #[test]
    fn dense_position_does_not_overflow_movelist() {
        crate::attacks::init();
        // 22 queens: >256 pseudo-legal moves for White. Must not abort.
        // With White to move the position is illegal (Black's king is en
        // prise), so from_fen now rejects it at the front door...
        let fen = "1QQkQQ2/Q5QK/Q5Q1/Q2Q2Q1/Q6Q/Q6Q/4QQ1Q/QQQ4Q";
        assert!(Board::from_fen(&format!("{fen} w - - 0 1")).is_err());
        // ...so parse it as the legal black-to-move position and flip the
        // side directly, exercising MoveList's guard as defense-in-depth
        // against absurd internal states.
        let mut board = Board::from_fen(&format!("{fen} b - - 0 1")).unwrap();
        board.side_to_move = Color::White;
        let moves = crate::movegen::generate_legal_moves(&board);
        assert!(moves.len() <= 256);
    }

    #[test]
    fn piece_on_startpos() {
        let board = Board::startpos();
        assert_eq!(
            board.piece_on(Square::from_algebraic("e1").unwrap()),
            Some((Color::White, Piece::King))
        );
        assert_eq!(
            board.piece_on(Square::from_algebraic("d8").unwrap()),
            Some((Color::Black, Piece::Queen))
        );
        assert_eq!(board.piece_on(Square::from_algebraic("e4").unwrap()), None);
    }
}
