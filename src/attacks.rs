use crate::types::*;

// ════════════════════════════════════════════════════════════════════════════
// Precomputed attack tables
//
// Non-sliding pieces (knight, king, pawn) have fixed attack patterns that
// only depend on the source square — we precompute and store them in arrays.
//
// Sliding pieces (bishop, rook, queen) depend on which squares are blocked,
// so we use "magic bitboards": a hash-based lookup that maps
//   (square, relevant_occupancy) -> attack_bitboard
// in O(1) with a single multiply + shift.
// ════════════════════════════════════════════════════════════════════════════

/// File masks (columns a through h)
pub const FILE_A: Bitboard = Bitboard(0x0101010101010101);
pub const FILE_B: Bitboard = Bitboard(0x0202020202020202);
pub const FILE_G: Bitboard = Bitboard(0x4040404040404040);
pub const FILE_H: Bitboard = Bitboard(0x8080808080808080);

pub const NOT_FILE_AB: Bitboard = Bitboard(!(FILE_A.0 | FILE_B.0));
pub const NOT_FILE_GH: Bitboard = Bitboard(!(FILE_G.0 | FILE_H.0));

// ── Knight attacks ─────────────────────────────────────────────────────────

const fn compute_knight_attacks(sq: u8) -> u64 {
    let bb = 1u64 << sq;
    let mut attacks = 0u64;

    // 8 possible L-shaped jumps, with file wrapping guards
    attacks |= (bb << 17) & !FILE_A.0; // up 2, right 1
    attacks |= (bb << 15) & !FILE_H.0; // up 2, left 1
    attacks |= (bb << 10) & NOT_FILE_AB.0; // up 1, right 2
    attacks |= (bb << 6) & NOT_FILE_GH.0; // up 1, left 2
    attacks |= (bb >> 6) & NOT_FILE_AB.0; // down 1, right 2
    attacks |= (bb >> 10) & NOT_FILE_GH.0; // down 1, left 2
    attacks |= (bb >> 15) & !FILE_A.0; // down 2, right 1
    attacks |= (bb >> 17) & !FILE_H.0; // down 2, left 1

    attacks
}

const fn init_knight_attacks() -> [Bitboard; 64] {
    let mut table = [Bitboard(0); 64];
    let mut sq = 0u8;
    while sq < 64 {
        table[sq as usize] = Bitboard(compute_knight_attacks(sq));
        sq += 1;
    }
    table
}

pub static KNIGHT_ATTACKS: [Bitboard; 64] = init_knight_attacks();

// ── King attacks ───────────────────────────────────────────────────────────

const fn compute_king_attacks(sq: u8) -> u64 {
    let bb = 1u64 << sq;
    let mut attacks = 0u64;

    attacks |= bb << 8; // up
    attacks |= bb >> 8; // down
    attacks |= (bb << 1) & !FILE_A.0; // right
    attacks |= (bb >> 1) & !FILE_H.0; // left
    attacks |= (bb << 9) & !FILE_A.0; // up-right
    attacks |= (bb << 7) & !FILE_H.0; // up-left
    attacks |= (bb >> 7) & !FILE_A.0; // down-right
    attacks |= (bb >> 9) & !FILE_H.0; // down-left

    attacks
}

const fn init_king_attacks() -> [Bitboard; 64] {
    let mut table = [Bitboard(0); 64];
    let mut sq = 0u8;
    while sq < 64 {
        table[sq as usize] = Bitboard(compute_king_attacks(sq));
        sq += 1;
    }
    table
}

pub static KING_ATTACKS: [Bitboard; 64] = init_king_attacks();

// ── Pawn attacks ───────────────────────────────────────────────────────────
// Note: these are *attacks* only (diagonal captures), not pushes.

const fn compute_white_pawn_attacks(sq: u8) -> u64 {
    let bb = 1u64 << sq;
    let mut attacks = 0u64;
    attacks |= (bb << 7) & !FILE_H.0; // capture left
    attacks |= (bb << 9) & !FILE_A.0; // capture right
    attacks
}

const fn compute_black_pawn_attacks(sq: u8) -> u64 {
    let bb = 1u64 << sq;
    let mut attacks = 0u64;
    attacks |= (bb >> 7) & !FILE_A.0; // capture right
    attacks |= (bb >> 9) & !FILE_H.0; // capture left
    attacks
}

const fn init_pawn_attacks() -> [[Bitboard; 64]; 2] {
    let mut table = [[Bitboard(0); 64]; 2];
    let mut sq = 0u8;
    while sq < 64 {
        table[Color::White as usize][sq as usize] = Bitboard(compute_white_pawn_attacks(sq));
        table[Color::Black as usize][sq as usize] = Bitboard(compute_black_pawn_attacks(sq));
        sq += 1;
    }
    table
}

pub static PAWN_ATTACKS: [[Bitboard; 64]; 2] = init_pawn_attacks();

// ════════════════════════════════════════════════════════════════════════════
// Magic bitboards for sliding pieces
//
// The idea: for a rook on e4, the squares it can attack depend on which
// squares along the rank/file are occupied (blockers). We take only the
// "relevant" occupancy bits (the squares that could block), multiply by a
// magic number, shift right, and use the result as an index into a table
// that gives the attack bitboard.
//
// attack = TABLE[square][(occupancy & MASK[square]) * MAGIC[square] >> SHIFT[square]]
// ════════════════════════════════════════════════════════════════════════════

/// Precomputed magic entry for one square
struct MagicEntry {
    mask: u64,  // relevant occupancy mask (excludes edges)
    magic: u64, // magic multiplier
    shift: u8,  // right-shift amount (64 - number of relevant bits)
    offset: u32, // starting index into the shared attack table
}

// The attack tables are initialized at runtime on first access.
// We use a flat array for all squares' attack tables.

// Rook: max 12 relevant bits → 4096 entries per square, 64 squares → 256K entries (worst case)
// Bishop: max 9 relevant bits → 512 entries per square
// In practice with our magics the total is much less, but we size conservatively.
static mut ROOK_TABLE: [Bitboard; 102400] = [Bitboard(0); 102400];
static mut BISHOP_TABLE: [Bitboard; 5248] = [Bitboard(0); 5248];

static ROOK_MAGICS: [MagicEntry; 64] = init_rook_magic_entries();
static BISHOP_MAGICS: [MagicEntry; 64] = init_bishop_magic_entries();

// ── Relevant occupancy masks ───────────────────────────────────────────────

const fn rook_relevant_mask(sq: u8) -> u64 {
    let rank = (sq / 8) as i8;
    let file = (sq % 8) as i8;
    let mut mask = 0u64;

    // Vertical (exclude first and last rank)
    let mut r = rank + 1;
    while r < 7 {
        mask |= 1u64 << (r * 8 + file);
        r += 1;
    }
    r = rank - 1;
    while r > 0 {
        mask |= 1u64 << (r * 8 + file);
        r -= 1;
    }

    // Horizontal (exclude first and last file)
    let mut f = file + 1;
    while f < 7 {
        mask |= 1u64 << (rank * 8 + f);
        f += 1;
    }
    f = file - 1;
    while f > 0 {
        mask |= 1u64 << (rank * 8 + f);
        f -= 1;
    }

    mask
}

const fn bishop_relevant_mask(sq: u8) -> u64 {
    let rank = (sq / 8) as i8;
    let file = (sq % 8) as i8;
    let mut mask = 0u64;

    // Four diagonals, excluding edge squares
    let mut r;
    let mut f;

    // Up-right
    r = rank + 1;
    f = file + 1;
    while r < 7 && f < 7 {
        mask |= 1u64 << (r * 8 + f);
        r += 1;
        f += 1;
    }

    // Up-left
    r = rank + 1;
    f = file - 1;
    while r < 7 && f > 0 {
        mask |= 1u64 << (r * 8 + f);
        r += 1;
        f -= 1;
    }

    // Down-right
    r = rank - 1;
    f = file + 1;
    while r > 0 && f < 7 {
        mask |= 1u64 << (r * 8 + f);
        r -= 1;
        f += 1;
    }

    // Down-left
    r = rank - 1;
    f = file - 1;
    while r > 0 && f > 0 {
        mask |= 1u64 << (r * 8 + f);
        r -= 1;
        f -= 1;
    }

    mask
}

// ── Sliding attack generation (used to fill the tables) ────────────────────

const fn rook_attacks_slow(sq: u8, occupancy: u64) -> u64 {
    let rank = (sq / 8) as i8;
    let file = (sq % 8) as i8;
    let mut attacks = 0u64;

    let mut r;
    let mut f;

    // Up
    r = rank + 1;
    while r <= 7 {
        attacks |= 1u64 << (r * 8 + file);
        if occupancy & (1u64 << (r * 8 + file)) != 0 {
            break;
        }
        r += 1;
    }

    // Down
    r = rank - 1;
    while r >= 0 {
        attacks |= 1u64 << (r * 8 + file);
        if occupancy & (1u64 << (r * 8 + file)) != 0 {
            break;
        }
        r -= 1;
    }

    // Right
    f = file + 1;
    while f <= 7 {
        attacks |= 1u64 << (rank * 8 + f);
        if occupancy & (1u64 << (rank * 8 + f)) != 0 {
            break;
        }
        f += 1;
    }

    // Left
    f = file - 1;
    while f >= 0 {
        attacks |= 1u64 << (rank * 8 + f);
        if occupancy & (1u64 << (rank * 8 + f)) != 0 {
            break;
        }
        f -= 1;
    }

    attacks
}

const fn bishop_attacks_slow(sq: u8, occupancy: u64) -> u64 {
    let rank = (sq / 8) as i8;
    let file = (sq % 8) as i8;
    let mut attacks = 0u64;

    let mut r;
    let mut f;

    // Up-right
    r = rank + 1;
    f = file + 1;
    while r <= 7 && f <= 7 {
        attacks |= 1u64 << (r * 8 + f);
        if occupancy & (1u64 << (r * 8 + f)) != 0 {
            break;
        }
        r += 1;
        f += 1;
    }

    // Up-left
    r = rank + 1;
    f = file - 1;
    while r <= 7 && f >= 0 {
        attacks |= 1u64 << (r * 8 + f);
        if occupancy & (1u64 << (r * 8 + f)) != 0 {
            break;
        }
        r += 1;
        f -= 1;
    }

    // Down-right
    r = rank - 1;
    f = file + 1;
    while r >= 0 && f <= 7 {
        attacks |= 1u64 << (r * 8 + f);
        if occupancy & (1u64 << (r * 8 + f)) != 0 {
            break;
        }
        r -= 1;
        f += 1;
    }

    // Down-left
    r = rank - 1;
    f = file - 1;
    while r >= 0 && f >= 0 {
        attacks |= 1u64 << (r * 8 + f);
        if occupancy & (1u64 << (r * 8 + f)) != 0 {
            break;
        }
        r -= 1;
        f -= 1;
    }

    attacks
}

// ── Known good magic numbers ───────────────────────────────────────────────
// These are well-tested magic numbers sourced from the chess programming wiki.

const ROOK_MAGIC_NUMBERS: [u64; 64] = [
    0x0080001020400080, 0x0040001000200040, 0x0080081000200080, 0x0080040800100080,
    0x0080020400080080, 0x0080010200040080, 0x0080008001000200, 0x0080002040800100,
    0x0000800020400080, 0x0000400020005000, 0x0000801000200080, 0x0000800800100080,
    0x0000800400080080, 0x0000800200040080, 0x0000800100020080, 0x0000800040800100,
    0x0000208000400080, 0x0000404000201000, 0x0000808010002000, 0x0000808008001000,
    0x0000808004000800, 0x0000808002000400, 0x0000010100020004, 0x0000020000408104,
    0x0000208080004000, 0x0000200040005000, 0x0000100080200080, 0x0000080080100080,
    0x0000040080080080, 0x0000020080040080, 0x0000010080800200, 0x0000800080004100,
    0x0000204000800080, 0x0000200040401000, 0x0000100080802000, 0x0000080080801000,
    0x0000040080800800, 0x0000020080800400, 0x0000020001010004, 0x0000800040800100,
    0x0000204000808000, 0x0000200040008080, 0x0000100020008080, 0x0000080010008080,
    0x0000040008008080, 0x0000020004008080, 0x0000010002008080, 0x0000004081020004,
    0x0000204000800080, 0x0000200040008080, 0x0000100020008080, 0x0000080010008080,
    0x0000040008008080, 0x0000020004008080, 0x0000800100020080, 0x0000800041000080,
    0x00FFFCDDFCED714A, 0x007FFCDDFCED714A, 0x003FFFCDFFD88096, 0x0000040810002101,
    0x0001000204080011, 0x0001000204000801, 0x0001000082000401, 0x0001FFFAABFAD1A2,
];

const BISHOP_MAGIC_NUMBERS: [u64; 64] = [
    0x0002020202020200, 0x0002020202020000, 0x0004010202000000, 0x0004040080000000,
    0x0001104000000000, 0x0000821040000000, 0x0000410410400000, 0x0000104104104000,
    0x0000040404040400, 0x0000020202020200, 0x0000040102020000, 0x0000040400800000,
    0x0000011040000000, 0x0000008210400000, 0x0000004104104000, 0x0000002082082000,
    0x0004000808080800, 0x0002000404040400, 0x0001000202020200, 0x0000800802004000,
    0x0000800400A00000, 0x0000200100884000, 0x0000400082082000, 0x0000200041041000,
    0x0002080010101000, 0x0001040008080800, 0x0000208004010400, 0x0000404004010200,
    0x0000840000802000, 0x0000404002011000, 0x0000808001041000, 0x0000404000820800,
    0x0001041000202000, 0x0000820800101000, 0x0000104400080800, 0x0000020080080080,
    0x0000404040040100, 0x0000808100020100, 0x0001010100020800, 0x0000808080010400,
    0x0000820820004000, 0x0000410410002000, 0x0000082088001000, 0x0000002011000800,
    0x0000080100400400, 0x0001010101000200, 0x0002020202000400, 0x0001010101000200,
    0x0000410410400000, 0x0000208208200000, 0x0000002084100000, 0x0000000020880000,
    0x0000001002020000, 0x0000040408020000, 0x0004040404040000, 0x0002020202020000,
    0x0000104104104000, 0x0000002082082000, 0x0000000020841000, 0x0000000000208800,
    0x0000000010020200, 0x0000000404080200, 0x0000040404040400, 0x0002020202020200,
];

// Number of relevant bits per square for rook/bishop
const ROOK_RELEVANT_BITS: [u8; 64] = [
    12, 11, 11, 11, 11, 11, 11, 12,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    12, 11, 11, 11, 11, 11, 11, 12,
];

const BISHOP_RELEVANT_BITS: [u8; 64] = [
    6, 5, 5, 5, 5, 5, 5, 6,
    5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 5, 5, 5, 5, 5, 5, 6,
];

// ── Magic entry initialization ─────────────────────────────────────────────

const fn init_rook_magic_entries() -> [MagicEntry; 64] {
    let mut entries: [MagicEntry; 64] = unsafe { std::mem::zeroed() };
    let mut offset = 0u32;
    let mut sq = 0usize;
    while sq < 64 {
        let bits = ROOK_RELEVANT_BITS[sq];
        entries[sq] = MagicEntry {
            mask: rook_relevant_mask(sq as u8),
            magic: ROOK_MAGIC_NUMBERS[sq],
            shift: 64 - bits,
            offset,
        };
        offset += 1u32 << bits;
        sq += 1;
    }
    entries
}

const fn init_bishop_magic_entries() -> [MagicEntry; 64] {
    let mut entries: [MagicEntry; 64] = unsafe { std::mem::zeroed() };
    let mut offset = 0u32;
    let mut sq = 0usize;
    while sq < 64 {
        let bits = BISHOP_RELEVANT_BITS[sq];
        entries[sq] = MagicEntry {
            mask: bishop_relevant_mask(sq as u8),
            magic: BISHOP_MAGIC_NUMBERS[sq],
            shift: 64 - bits,
            offset,
        };
        offset += 1u32 << bits;
        sq += 1;
    }
    entries
}

// ── Enumerate all subsets of a mask (Carry-Rippler trick) ──────────────────

fn enumerate_subsets(mask: u64) -> impl Iterator<Item = u64> {
    let mut subset = 0u64;
    let mut done = false;
    std::iter::from_fn(move || {
        if done {
            return None;
        }
        let current = subset;
        // Carry-Rippler: next subset
        subset = subset.wrapping_sub(mask) & mask;
        if subset == 0 {
            done = true;
        }
        Some(current)
    })
}

// ── Runtime initialization ─────────────────────────────────────────────────

/// Must be called once at startup before using sliding piece attacks.
pub fn init() {
    unsafe {
        // Fill rook table
        for sq in 0..64 {
            let entry = &ROOK_MAGICS[sq];
            for occ in enumerate_subsets(entry.mask) {
                let idx = ((occ.wrapping_mul(entry.magic)) >> entry.shift) as u32;
                ROOK_TABLE[(entry.offset + idx) as usize] =
                    Bitboard(rook_attacks_slow(sq as u8, occ));
            }
        }

        // Fill bishop table
        for sq in 0..64 {
            let entry = &BISHOP_MAGICS[sq];
            for occ in enumerate_subsets(entry.mask) {
                let idx = ((occ.wrapping_mul(entry.magic)) >> entry.shift) as u32;
                BISHOP_TABLE[(entry.offset + idx) as usize] =
                    Bitboard(bishop_attacks_slow(sq as u8, occ));
            }
        }
    }
}

// ── Public attack lookups ──────────────────────────────────────────────────

#[inline(always)]
pub fn knight_attacks(sq: Square) -> Bitboard {
    KNIGHT_ATTACKS[sq.0 as usize]
}

#[inline(always)]
pub fn king_attacks(sq: Square) -> Bitboard {
    KING_ATTACKS[sq.0 as usize]
}

#[inline(always)]
pub fn pawn_attacks(color: Color, sq: Square) -> Bitboard {
    PAWN_ATTACKS[color as usize][sq.0 as usize]
}

#[inline(always)]
pub fn rook_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
    let entry = &ROOK_MAGICS[sq.0 as usize];
    let idx = ((occupancy.0 & entry.mask).wrapping_mul(entry.magic) >> entry.shift) as u32;
    unsafe { ROOK_TABLE[(entry.offset + idx) as usize] }
}

#[inline(always)]
pub fn bishop_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
    let entry = &BISHOP_MAGICS[sq.0 as usize];
    let idx = ((occupancy.0 & entry.mask).wrapping_mul(entry.magic) >> entry.shift) as u32;
    unsafe { BISHOP_TABLE[(entry.offset + idx) as usize] }
}

#[inline(always)]
pub fn queen_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
    rook_attacks(sq, occupancy) | bishop_attacks(sq, occupancy)
}

/// Is a given square attacked by the specified side?
pub fn is_square_attacked(board: &crate::board::Board, sq: Square, by_color: Color) -> bool {
    // Knight attacks
    if (knight_attacks(sq) & board.piece_bb(by_color, Piece::Knight)).is_not_empty() {
        return true;
    }
    // King attacks
    if (king_attacks(sq) & board.piece_bb(by_color, Piece::King)).is_not_empty() {
        return true;
    }
    // Pawn attacks (check from the *other* side's perspective)
    let defend_color = by_color.flip();
    if (pawn_attacks(defend_color, sq) & board.piece_bb(by_color, Piece::Pawn)).is_not_empty() {
        return true;
    }
    // Sliding attacks
    let occ = board.all_occupied();
    if (rook_attacks(sq, occ)
        & (board.piece_bb(by_color, Piece::Rook) | board.piece_bb(by_color, Piece::Queen)))
    .is_not_empty()
    {
        return true;
    }
    if (bishop_attacks(sq, occ)
        & (board.piece_bb(by_color, Piece::Bishop) | board.piece_bb(by_color, Piece::Queen)))
    .is_not_empty()
    {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knight_attacks_center() {
        init();
        // Knight on e4 (square 28) should attack 8 squares
        let attacks = knight_attacks(Square::from_algebraic("e4").unwrap());
        assert_eq!(attacks.popcount(), 8);
    }

    #[test]
    fn knight_attacks_corner() {
        init();
        // Knight on a1 (square 0) should attack 2 squares
        let attacks = knight_attacks(Square::from_algebraic("a1").unwrap());
        assert_eq!(attacks.popcount(), 2);
    }

    #[test]
    fn king_attacks_center() {
        init();
        let attacks = king_attacks(Square::from_algebraic("e4").unwrap());
        assert_eq!(attacks.popcount(), 8);
    }

    #[test]
    fn rook_attacks_empty_board() {
        init();
        // Rook on e4 with no blockers should attack 14 squares (7 on rank + 7 on file)
        let attacks = rook_attacks(Square::from_algebraic("e4").unwrap(), Bitboard::EMPTY);
        assert_eq!(attacks.popcount(), 14);
    }

    #[test]
    fn bishop_attacks_empty_board() {
        init();
        // Bishop on e4 with no blockers should attack 13 squares
        let attacks = bishop_attacks(Square::from_algebraic("e4").unwrap(), Bitboard::EMPTY);
        assert_eq!(attacks.popcount(), 13);
    }

    #[test]
    fn rook_attacks_with_blockers() {
        init();
        // Rook on e4, blocker on e7 — should not see e8
        let mut occ = Bitboard::EMPTY;
        occ.set(Square::from_algebraic("e7").unwrap());
        let attacks = rook_attacks(Square::from_algebraic("e4").unwrap(), occ);
        assert!(attacks.contains(Square::from_algebraic("e7").unwrap())); // can capture blocker
        assert!(!attacks.contains(Square::from_algebraic("e8").unwrap())); // blocked
    }

    #[test]
    fn is_square_attacked_startpos() {
        init();
        let board = crate::board::Board::startpos();
        // e2 pawn attacks d3 and f3
        assert!(is_square_attacked(&board, Square::from_algebraic("d3").unwrap(), Color::White));
        assert!(is_square_attacked(&board, Square::from_algebraic("f3").unwrap(), Color::White));
        // e4 is not attacked by anyone in starting position
        assert!(!is_square_attacked(&board, Square::from_algebraic("e4").unwrap(), Color::White));
        assert!(!is_square_attacked(&board, Square::from_algebraic("e4").unwrap(), Color::Black));
    }
}
