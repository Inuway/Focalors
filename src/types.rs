/// Colors / sides
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    pub const COUNT: usize = 2;

    pub fn flip(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

/// Piece types (without color information)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Piece {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl Piece {
    pub const COUNT: usize = 6;

    pub const ALL: [Piece; 6] = [
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
        Piece::King,
    ];
}

/// Square index: 0 = a1, 1 = b1, ..., 63 = h8
/// Rank-major: squares 0..7 are rank 1 (a1-h1), 8..15 are rank 2, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Square(pub u8);

impl Square {
    pub const fn new(file: u8, rank: u8) -> Self {
        Square(rank * 8 + file)
    }

    pub const fn rank(self) -> u8 {
        self.0 / 8
    }

    pub const fn file(self) -> u8 {
        self.0 % 8
    }

    /// Parse a square from algebraic notation like "e4"
    pub fn from_algebraic(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.len() != 2 {
            return None;
        }
        let file = bytes[0].wrapping_sub(b'a');
        let rank = bytes[1].wrapping_sub(b'1');
        if file < 8 && rank < 8 {
            Some(Square::new(file, rank))
        } else {
            None
        }
    }

    pub fn to_algebraic(self) -> String {
        let file = (b'a' + self.file()) as char;
        let rank = (b'1' + self.rank()) as char;
        format!("{file}{rank}")
    }
}

impl std::fmt::Display for Square {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_algebraic())
    }
}

// ── Bitboard ───────────────────────────────────────────────────────────────
//
// A 64-bit integer where bit N corresponds to square N.
// This is the heart of the engine — nearly every operation on the board
// boils down to bitwise ops on these.

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Bitboard(pub u64);

impl Bitboard {
    pub const EMPTY: Self = Bitboard(0);

    /// Bitboard with a single bit set for the given square
    pub const fn from_square(sq: Square) -> Self {
        Bitboard(1u64 << sq.0)
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn is_not_empty(self) -> bool {
        self.0 != 0
    }

    /// Number of set bits (= number of pieces on this bitboard)
    pub const fn popcount(self) -> u32 {
        self.0.count_ones()
    }

    /// Index of the least significant set bit (= lowest square index)
    /// Undefined if the bitboard is empty.
    pub const fn lsb(self) -> Square {
        Square(self.0.trailing_zeros() as u8)
    }

    /// Remove and return the least significant set bit
    pub fn pop_lsb(&mut self) -> Square {
        let sq = self.lsb();
        self.0 &= self.0 - 1; // clears the lowest set bit
        sq
    }

    /// Set a square (used in tests)
    #[cfg(test)]
    pub fn set(&mut self, sq: Square) {
        self.0 |= 1u64 << sq.0;
    }

    /// Check whether a specific square is set
    pub const fn contains(self, sq: Square) -> bool {
        (self.0 & (1u64 << sq.0)) != 0
    }

}

// Bitwise operators so we can write `bb1 & bb2`, `bb1 | bb2`, `!bb`, etc.

impl std::ops::BitAnd for Bitboard {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        Bitboard(self.0 & rhs.0)
    }
}

impl std::ops::BitOr for Bitboard {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Bitboard(self.0 | rhs.0)
    }
}

impl std::ops::BitXor for Bitboard {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self {
        Bitboard(self.0 ^ rhs.0)
    }
}

impl std::ops::Not for Bitboard {
    type Output = Self;
    fn not(self) -> Self {
        Bitboard(!self.0)
    }
}

impl std::ops::BitAndAssign for Bitboard {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl std::ops::BitOrAssign for Bitboard {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::fmt::Debug for Bitboard {
    /// Pretty-print the bitboard as an 8x8 grid (rank 8 at top)
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        for rank in (0..8).rev() {
            write!(f, "  {} ", rank + 1)?;
            for file in 0..8 {
                let sq = Square::new(file, rank);
                if self.contains(sq) {
                    write!(f, " 1")?;
                } else {
                    write!(f, " .")?;
                }
            }
            writeln!(f)?;
        }
        writeln!(f, "    a b c d e f g h")
    }
}

// ── Castling rights ────────────────────────────────────────────────────────

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CastlingRights: u8 {
        const WHITE_KING  = 0b0001;
        const WHITE_QUEEN = 0b0010;
        const BLACK_KING  = 0b0100;
        const BLACK_QUEEN = 0b1000;

        const WHITE = Self::WHITE_KING.bits() | Self::WHITE_QUEEN.bits();
        const BLACK = Self::BLACK_KING.bits() | Self::BLACK_QUEEN.bits();
        const ALL   = Self::WHITE.bits() | Self::BLACK.bits();
        const NONE  = 0;
    }
}
