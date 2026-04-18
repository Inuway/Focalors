use crate::types::*;

/// Zobrist hashing: represent a chess position as a single u64.
///
/// The idea: assign a random 64-bit number to every (piece, color, square)
/// combination, plus side-to-move, castling rights, and en passant file.
/// The hash of a position is the XOR of all applicable random numbers.
///
/// XOR is its own inverse, so updating the hash incrementally is trivial:
///   hash ^= PIECE_KEYS[color][piece][from]  // remove piece from old square
///   hash ^= PIECE_KEYS[color][piece][to]    // add piece to new square

/// Random numbers for [color][piece][square]
static PIECE_KEYS: [[[u64; 64]; Piece::COUNT]; Color::COUNT] = init_piece_keys();

/// Random number XOR'd when it's Black's turn
static SIDE_KEY: u64 = init_side_key();

/// Random numbers for castling rights (one per combination of 4 bits = 16)
static CASTLING_KEYS: [u64; 16] = init_castling_keys();

/// Random numbers for en passant file (0-7, or none)
static EN_PASSANT_KEYS: [u64; 8] = init_ep_keys();

// ── Deterministic PRNG for key generation ──────────────────────────────────
// We use a simple xorshift64 seeded with a fixed value so the keys are the
// same every run (important for reproducibility).

const fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

const fn init_piece_keys() -> [[[u64; 64]; Piece::COUNT]; Color::COUNT] {
    let mut keys = [[[0u64; 64]; Piece::COUNT]; Color::COUNT];
    let mut state = 0x12345678_9ABCDEF0u64;
    let mut color = 0;
    while color < Color::COUNT {
        let mut piece = 0;
        while piece < Piece::COUNT {
            let mut sq = 0;
            while sq < 64 {
                state = xorshift64(state);
                keys[color][piece][sq] = state;
                sq += 1;
            }
            piece += 1;
        }
        color += 1;
    }
    keys
}

const fn init_side_key() -> u64 {
    // Continue the sequence after piece keys
    let mut state = 0xFEDCBA98_76543210u64;
    state = xorshift64(state);
    state
}

const fn init_castling_keys() -> [u64; 16] {
    let mut keys = [0u64; 16];
    let mut state = 0xA5A5A5A5_5A5A5A5Au64;
    let mut i = 0;
    while i < 16 {
        state = xorshift64(state);
        keys[i] = state;
        i += 1;
    }
    keys
}

const fn init_ep_keys() -> [u64; 8] {
    let mut keys = [0u64; 8];
    let mut state = 0x1234ABCD_5678EF01u64;
    let mut i = 0;
    while i < 8 {
        state = xorshift64(state);
        keys[i] = state;
        i += 1;
    }
    keys
}

// ── Public API ─────────────────────────────────────────────────────────────

pub fn piece_key(color: Color, piece: Piece, sq: Square) -> u64 {
    PIECE_KEYS[color as usize][piece as usize][sq.0 as usize]
}

pub fn side_key() -> u64 {
    SIDE_KEY
}

pub fn castling_key(castling: CastlingRights) -> u64 {
    CASTLING_KEYS[castling.bits() as usize]
}

pub fn en_passant_key(file: u8) -> u64 {
    EN_PASSANT_KEYS[file as usize]
}

/// Compute the full Zobrist hash for a board position from scratch.
pub fn hash_position(board: &crate::board::Board) -> u64 {
    let mut hash = 0u64;

    for color in [Color::White, Color::Black] {
        for piece in Piece::ALL {
            let mut bb = board.piece_bb(color, piece);
            while bb.is_not_empty() {
                let sq = bb.pop_lsb();
                hash ^= piece_key(color, piece, sq);
            }
        }
    }

    if matches!(board.side_to_move, Color::Black) {
        hash ^= side_key();
    }

    hash ^= castling_key(board.castling);

    if let Some(ep_sq) = board.en_passant {
        hash ^= en_passant_key(ep_sq.file());
    }

    hash
}
