use crate::types::*;

/// Compact move representation packed into a u16.
///
/// Layout (matches Stockfish convention):
///   bits  0-5:  destination square (0-63)
///   bits  6-11: origin square (0-63)
///   bits 12-13: promotion piece (0=Knight, 1=Bishop, 2=Rook, 3=Queen)
///   bits 14-15: flags (0=Normal, 1=Promotion, 2=En passant, 3=Castling)
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Move(u16);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MoveFlag {
    Normal = 0,
    Promotion = 1,
    EnPassant = 2,
    Castling = 3,
}

impl Move {
    pub const NULL: Move = Move(0);

    pub fn new(from: Square, to: Square) -> Self {
        Move((from.0 as u16) << 6 | to.0 as u16)
    }

    pub fn new_promotion(from: Square, to: Square, promo: Piece) -> Self {
        let promo_bits = match promo {
            Piece::Knight => 0,
            Piece::Bishop => 1,
            Piece::Rook => 2,
            Piece::Queen => 3,
            _ => unreachable!(),
        };
        Move((MoveFlag::Promotion as u16) << 14 | promo_bits << 12 | (from.0 as u16) << 6 | to.0 as u16)
    }

    pub fn new_en_passant(from: Square, to: Square) -> Self {
        Move((MoveFlag::EnPassant as u16) << 14 | (from.0 as u16) << 6 | to.0 as u16)
    }

    pub fn new_castling(king_from: Square, king_to: Square) -> Self {
        Move((MoveFlag::Castling as u16) << 14 | (king_from.0 as u16) << 6 | king_to.0 as u16)
    }

    pub const fn from_sq(self) -> Square {
        Square(((self.0 >> 6) & 0x3F) as u8)
    }

    pub const fn to_sq(self) -> Square {
        Square((self.0 & 0x3F) as u8)
    }

    pub const fn flag(self) -> MoveFlag {
        match (self.0 >> 14) & 3 {
            0 => MoveFlag::Normal,
            1 => MoveFlag::Promotion,
            2 => MoveFlag::EnPassant,
            3 => MoveFlag::Castling,
            _ => unreachable!(),
        }
    }

    pub const fn promotion_piece(self) -> Piece {
        match (self.0 >> 12) & 3 {
            0 => Piece::Knight,
            1 => Piece::Bishop,
            2 => Piece::Rook,
            3 => Piece::Queen,
            _ => unreachable!(),
        }
    }

    pub const fn raw(self) -> u16 {
        self.0
    }

    pub const fn from_raw(raw: u16) -> Move {
        Move(raw)
    }

    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    /// UCI notation: e.g. "e2e4", "e7e8q" for promotions
    pub fn to_uci(self) -> String {
        let from = self.from_sq().to_algebraic();
        let to = self.to_sq().to_algebraic();
        if self.flag() as u8 == MoveFlag::Promotion as u8 {
            let promo = match self.promotion_piece() {
                Piece::Knight => 'n',
                Piece::Bishop => 'b',
                Piece::Rook => 'r',
                Piece::Queen => 'q',
                _ => unreachable!(),
            };
            format!("{from}{to}{promo}")
        } else {
            format!("{from}{to}")
        }
    }
}

impl std::fmt::Debug for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

/// A pre-allocated move list (avoids heap allocation in the hot path).
/// Maximum possible legal moves in any chess position is 218.
pub struct MoveList {
    moves: [Move; 256],
    len: usize,
}

impl MoveList {
    pub fn new() -> Self {
        MoveList {
            moves: [Move::NULL; 256],
            len: 0,
        }
    }

    pub fn push(&mut self, mv: Move) {
        self.moves[self.len] = mv;
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[cfg(test)]
    pub fn iter(&self) -> impl Iterator<Item = &Move> {
        self.moves[..self.len].iter()
    }
}

impl std::ops::Index<usize> for MoveList {
    type Output = Move;
    fn index(&self, idx: usize) -> &Move {
        &self.moves[idx]
    }
}
