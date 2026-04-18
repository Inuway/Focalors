use crate::attacks;
use crate::board::Board;
use crate::types::*;

/// Evaluation score in centipawns (100 = 1 pawn).
/// Positive = good for the side to move.
pub type Score = i32;

pub const INFINITY: Score = 30000;
pub const MATE_SCORE: Score = 29000;

/// Material values in centipawns
const PIECE_VALUE: [Score; Piece::COUNT] = [
    100,  // Pawn
    320,  // Knight
    330,  // Bishop
    500,  // Rook
    900,  // Queen
    0,    // King
];

/// Phase contribution per piece type
const PHASE_WEIGHT: [i32; Piece::COUNT] = [0, 1, 1, 2, 4, 0];
const TOTAL_PHASE: i32 = 24;

// ════════════════════════════════════════════════════════════════════════════
// File and rank masks (precomputed)
// ════════════════════════════════════════════════════════════════════════════

const fn file_mask(file: u8) -> u64 {
    0x0101010101010101u64 << file
}

const FILE_MASKS: [Bitboard; 8] = {
    let mut masks = [Bitboard(0); 8];
    let mut f = 0u8;
    while f < 8 {
        masks[f as usize] = Bitboard(file_mask(f));
        f += 1;
    }
    masks
};

const ADJACENT_FILE_MASKS: [Bitboard; 8] = {
    let mut masks = [Bitboard(0); 8];
    let mut f = 0u8;
    while f < 8 {
        let mut m = 0u64;
        if f > 0 { m |= file_mask(f - 1); }
        if f < 7 { m |= file_mask(f + 1); }
        masks[f as usize] = Bitboard(m);
        f += 1;
    }
    masks
};

// ── Passed pawn masks ──────────────────────────────────────────────────────
// For a white pawn on square `sq`, the passed pawn mask covers all squares
// on the same file and adjacent files at higher ranks.

const PASSED_PAWN_MASKS_WHITE: [Bitboard; 64] = {
    let mut masks = [Bitboard(0); 64];
    let mut sq = 0u8;
    while sq < 64 {
        let rank = sq / 8;
        let file = sq % 8;
        let mut m = 0u64;
        let mut r = rank + 1;
        while r < 8 {
            m |= 1u64 << (r * 8 + file);
            if file > 0 { m |= 1u64 << (r * 8 + file - 1); }
            if file < 7 { m |= 1u64 << (r * 8 + file + 1); }
            r += 1;
        }
        masks[sq as usize] = Bitboard(m);
        sq += 1;
    }
    masks
};

const PASSED_PAWN_MASKS_BLACK: [Bitboard; 64] = {
    let mut masks = [Bitboard(0); 64];
    let mut sq = 0u8;
    while sq < 64 {
        let rank = sq / 8;
        let file = sq % 8;
        let mut m = 0u64;
        if rank > 0 {
            let mut r: i8 = rank as i8 - 1;
            while r >= 0 {
                m |= 1u64 << (r as u8 * 8 + file);
                if file > 0 { m |= 1u64 << (r as u8 * 8 + file - 1); }
                if file < 7 { m |= 1u64 << (r as u8 * 8 + file + 1); }
                r -= 1;
            }
        }
        masks[sq as usize] = Bitboard(m);
        sq += 1;
    }
    masks
};

// Passed pawn bonus by rank (from the pawn's perspective)
const PASSED_PAWN_BONUS_MG: [Score; 8] = [0, 5, 10, 20, 35, 55, 80, 0];
const PASSED_PAWN_BONUS_EG: [Score; 8] = [0, 10, 20, 40, 70, 110, 160, 0];

// ── New eval term constants ────────────────────────────────────────────────

const ROOK_OPEN_FILE_MG: Score = 20;
const ROOK_OPEN_FILE_EG: Score = 10;
const ROOK_SEMI_OPEN_FILE_MG: Score = 10;
const ROOK_SEMI_OPEN_FILE_EG: Score = 5;
const ROOK_SEVENTH_RANK_MG: Score = 20;
const ROOK_SEVENTH_RANK_EG: Score = 40;
const ROOK_BEHIND_PASSER_MG: Score = 15;
const ROOK_BEHIND_PASSER_EG: Score = 25;

const KNIGHT_OUTPOST_MG: Score = 20;
const KNIGHT_OUTPOST_EG: Score = 15;

const CONNECTED_PASSER_MG: Score = 10;
const CONNECTED_PASSER_EG: Score = 20;

const KING_PAWN_DIST_EG: Score = 5; // per-square bonus/penalty

const TEMPO_MG: Score = 10;

// ════════════════════════════════════════════════════════════════════════════
// Piece-square tables — Middlegame
// ════════════════════════════════════════════════════════════════════════════

#[rustfmt::skip]
const PAWN_MG: [Score; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0,
];

#[rustfmt::skip]
const KNIGHT_MG: [Score; 64] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
];

#[rustfmt::skip]
const BISHOP_MG: [Score; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
];

#[rustfmt::skip]
const ROOK_MG: [Score; 64] = [
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
];

#[rustfmt::skip]
const QUEEN_MG: [Score; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -10,  5,  5,  5,  5,  5,  0,-10,
      0,  0,  5,  5,  5,  5,  0, -5,
     -5,  0,  5,  5,  5,  5,  0, -5,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
];

#[rustfmt::skip]
const KING_MG: [Score; 64] = [
     20, 30, 10,  0,  0, 10, 30, 20,
     20, 20,  0,  0,  0,  0, 20, 20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
];

const PST_MG: [[Score; 64]; Piece::COUNT] = [
    PAWN_MG, KNIGHT_MG, BISHOP_MG, ROOK_MG, QUEEN_MG, KING_MG,
];

// ════════════════════════════════════════════════════════════════════════════
// Piece-square tables — Endgame
// ════════════════════════════════════════════════════════════════════════════

#[rustfmt::skip]
const PAWN_EG: [Score; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
    10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10,
    20, 20, 20, 20, 20, 20, 20, 20,
    30, 30, 30, 30, 30, 30, 30, 30,
    50, 50, 50, 50, 50, 50, 50, 50,
    80, 80, 80, 80, 80, 80, 80, 80,
     0,  0,  0,  0,  0,  0,  0,  0,
];

#[rustfmt::skip]
const KNIGHT_EG: [Score; 64] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
];

#[rustfmt::skip]
const BISHOP_EG: [Score; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  0, 10, 15, 15, 10,  0,-10,
    -10,  0, 10, 15, 15, 10,  0,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
];

#[rustfmt::skip]
const ROOK_EG: [Score; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
     5,  5,  5,  5,  5,  5,  5,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
];

#[rustfmt::skip]
const QUEEN_EG: [Score; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
     -5,  0,  5,  5,  5,  5,  0, -5,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
];

#[rustfmt::skip]
const KING_EG: [Score; 64] = [
    -50,-30,-30,-30,-30,-30,-30,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
];

const PST_EG: [[Score; 64]; Piece::COUNT] = [
    PAWN_EG, KNIGHT_EG, BISHOP_EG, ROOK_EG, QUEEN_EG, KING_EG,
];

// ════════════════════════════════════════════════════════════════════════════
// Mobility baselines (squares a piece "should" have; below = penalty)
// ════════════════════════════════════════════════════════════════════════════

const MOBILITY_WEIGHT_MG: [Score; Piece::COUNT] = [0, 4, 5, 2, 1, 0];
const MOBILITY_WEIGHT_EG: [Score; Piece::COUNT] = [0, 4, 5, 4, 2, 0];
const MOBILITY_BASELINE: [Score; Piece::COUNT] = [0, 4, 6, 7, 14, 0];

// ════════════════════════════════════════════════════════════════════════════
// Evaluation
// ════════════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════════════
// Pawn hash table — caches pawn structure + passed pawn evaluation
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
struct PawnEntry {
    key: u64,
    pawn_mg: Score,
    pawn_eg: Score,
    pass_mg: Score,
    pass_eg: Score,
    white_passed: Bitboard,
    black_passed: Bitboard,
}

impl Default for PawnEntry {
    fn default() -> Self {
        PawnEntry {
            key: 0, pawn_mg: 0, pawn_eg: 0, pass_mg: 0, pass_eg: 0,
            white_passed: Bitboard::EMPTY, black_passed: Bitboard::EMPTY,
        }
    }
}

pub struct PawnHashTable {
    entries: Vec<PawnEntry>,
    mask: usize,
}

impl PawnHashTable {
    pub fn new(size_kb: usize) -> Self {
        let entry_size = std::mem::size_of::<PawnEntry>();
        let num = ((size_kb * 1024) / entry_size).next_power_of_two().max(1);
        PawnHashTable {
            entries: vec![PawnEntry::default(); num],
            mask: num - 1,
        }
    }

    fn probe(&self, key: u64) -> Option<&PawnEntry> {
        let entry = &self.entries[key as usize & self.mask];
        if entry.key == key { Some(entry) } else { None }
    }

    fn store(&mut self, entry: PawnEntry) {
        let idx = entry.key as usize & self.mask;
        self.entries[idx] = entry;
    }
}

const fn flip_sq(sq: usize) -> usize {
    sq ^ 56
}

fn game_phase(board: &Board) -> i32 {
    let mut phase = 0;
    for piece in Piece::ALL {
        let count = board.piece_bb(Color::White, piece).popcount()
            + board.piece_bb(Color::Black, piece).popcount();
        phase += PHASE_WEIGHT[piece as usize] * count as i32;
    }
    phase.min(TOTAL_PHASE)
}

/// Tapered score helper
fn taper(mg: Score, eg: Score, phase: i32) -> Score {
    (mg * phase + eg * (TOTAL_PHASE - phase)) / TOTAL_PHASE
}

/// Decomposed evaluation showing individual component contributions.
/// All values are from White's perspective (positive = White is better).
#[derive(Debug, Clone, Default)]
pub struct EvalBreakdown {
    pub material: Score,
    pub pst: Score,
    pub mobility: Score,
    pub pawn_structure: Score,
    pub passed_pawns: Score,
    pub king_safety: Score,
    pub bishop_pair: Score,
    pub rook_placement: Score,
    pub knight_outpost: Score,
    pub connected_passers: Score,
    pub king_pawn_proximity: Score,
    pub tempo: Score,
    pub total: Score,
}

pub fn evaluate(board: &Board) -> Score {
    eval_full(board, None).total
}

pub fn evaluate_with_pht(board: &Board, pht: &mut PawnHashTable) -> Score {
    eval_full(board, Some(pht)).total
}

pub fn eval_components(board: &Board) -> EvalBreakdown {
    eval_full(board, None)
}

/// Full evaluation with component breakdown.
fn eval_full(board: &Board, mut pht: Option<&mut PawnHashTable>) -> EvalBreakdown {
    let phase = game_phase(board);

    let white_occ = board.color_bb(Color::White);
    let black_occ = board.color_bb(Color::Black);
    let all_occ = board.all_occupied();

    // ── Material + PST (tracked separately) ────────────────────────────
    let mut mat_mg: Score = 0;
    let mut mat_eg: Score = 0;
    let mut pst_mg: Score = 0;
    let mut pst_eg: Score = 0;

    for piece in Piece::ALL {
        let mut wbb = board.piece_bb(Color::White, piece);
        while wbb.is_not_empty() {
            let sq = wbb.pop_lsb();
            let idx = sq.0 as usize;
            mat_mg += PIECE_VALUE[piece as usize];
            mat_eg += PIECE_VALUE[piece as usize];
            pst_mg += PST_MG[piece as usize][idx];
            pst_eg += PST_EG[piece as usize][idx];
        }
        let mut bbb = board.piece_bb(Color::Black, piece);
        while bbb.is_not_empty() {
            let sq = bbb.pop_lsb();
            let idx = flip_sq(sq.0 as usize);
            mat_mg -= PIECE_VALUE[piece as usize];
            mat_eg -= PIECE_VALUE[piece as usize];
            pst_mg -= PST_MG[piece as usize][idx];
            pst_eg -= PST_EG[piece as usize][idx];
        }
    }

    // ── Mobility ───────────────────────────────────────────────────────
    let mut mob_mg: Score = 0;
    let mut mob_eg: Score = 0;
    for piece in [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
        let pi = piece as usize;
        let mut wbb = board.piece_bb(Color::White, piece);
        while wbb.is_not_empty() {
            let sq = wbb.pop_lsb();
            let atk = piece_attacks(piece, sq, all_occ);
            let mob = (atk & !white_occ).popcount() as Score - MOBILITY_BASELINE[pi];
            mob_mg += mob * MOBILITY_WEIGHT_MG[pi];
            mob_eg += mob * MOBILITY_WEIGHT_EG[pi];
        }
        let mut bbb = board.piece_bb(Color::Black, piece);
        while bbb.is_not_empty() {
            let sq = bbb.pop_lsb();
            let atk = piece_attacks(piece, sq, all_occ);
            let mob = (atk & !black_occ).popcount() as Score - MOBILITY_BASELINE[pi];
            mob_mg -= mob * MOBILITY_WEIGHT_MG[pi];
            mob_eg -= mob * MOBILITY_WEIGHT_EG[pi];
        }
    }

    // ── Pawn structure + passed pawns (with hash table caching) ───────
    let white_pawns = board.piece_bb(Color::White, Piece::Pawn);
    let black_pawns = board.piece_bb(Color::Black, Piece::Pawn);

    let (pawn_mg, pawn_eg, pass_mg, pass_eg, white_passed, black_passed) =
        if let Some(ref mut pht) = pht {
            if let Some(entry) = pht.probe(board.pawn_hash) {
                (entry.pawn_mg, entry.pawn_eg, entry.pass_mg, entry.pass_eg,
                 entry.white_passed, entry.black_passed)
            } else {
                let (wpawn_mg, wpawn_eg) = eval_pawn_structure(white_pawns, black_pawns, Color::White);
                let (bpawn_mg, bpawn_eg) = eval_pawn_structure(black_pawns, white_pawns, Color::Black);
                let pm = wpawn_mg - bpawn_mg;
                let pe = wpawn_eg - bpawn_eg;
                let (wpass_mg, wpass_eg, wp) = eval_passed_pawns(white_pawns, black_pawns, Color::White);
                let (bpass_mg, bpass_eg, bp) = eval_passed_pawns(black_pawns, white_pawns, Color::Black);
                let xm = wpass_mg - bpass_mg;
                let xe = wpass_eg - bpass_eg;
                pht.store(PawnEntry {
                    key: board.pawn_hash, pawn_mg: pm, pawn_eg: pe,
                    pass_mg: xm, pass_eg: xe, white_passed: wp, black_passed: bp,
                });
                (pm, pe, xm, xe, wp, bp)
            }
        } else {
            let (wpawn_mg, wpawn_eg) = eval_pawn_structure(white_pawns, black_pawns, Color::White);
            let (bpawn_mg, bpawn_eg) = eval_pawn_structure(black_pawns, white_pawns, Color::Black);
            let (wpass_mg, wpass_eg, wp) = eval_passed_pawns(white_pawns, black_pawns, Color::White);
            let (bpass_mg, bpass_eg, bp) = eval_passed_pawns(black_pawns, white_pawns, Color::Black);
            (wpawn_mg - bpawn_mg, wpawn_eg - bpawn_eg,
             wpass_mg - bpass_mg, wpass_eg - bpass_eg, wp, bp)
        };

    // ── King safety (middlegame only, when opponent has queen) ──────────
    let wks = eval_king_safety(board, Color::White, white_pawns, all_occ);
    let bks = eval_king_safety(board, Color::Black, black_pawns, all_occ);
    let ks_mg = wks - bks;

    // ── Bishop pair ────────────────────────────────────────────────────
    let mut bp_mg: Score = 0;
    let mut bp_eg: Score = 0;
    if board.piece_bb(Color::White, Piece::Bishop).popcount() >= 2 {
        bp_mg += 30;
        bp_eg += 50;
    }
    if board.piece_bb(Color::Black, Piece::Bishop).popcount() >= 2 {
        bp_mg -= 30;
        bp_eg -= 50;
    }

    // ── Rook placement ─────────────────────────────────────────────────
    let (wrp_mg, wrp_eg) = eval_rook_placement(board, Color::White, white_pawns, black_pawns, white_passed);
    let (brp_mg, brp_eg) = eval_rook_placement(board, Color::Black, black_pawns, white_pawns, black_passed);
    let rp_mg = wrp_mg - brp_mg;
    let rp_eg = wrp_eg - brp_eg;

    // ── Knight outposts ────────────────────────────────────────────────
    let (wno_mg, wno_eg) = eval_knight_outposts(board, Color::White, white_pawns, black_pawns);
    let (bno_mg, bno_eg) = eval_knight_outposts(board, Color::Black, black_pawns, white_pawns);
    let no_mg = wno_mg - bno_mg;
    let no_eg = wno_eg - bno_eg;

    // ── Connected passed pawns ─────────────────────────────────────────
    let (wcp_mg, wcp_eg) = eval_connected_passers(white_passed);
    let (bcp_mg, bcp_eg) = eval_connected_passers(black_passed);
    let cp_mg = wcp_mg - bcp_mg;
    let cp_eg = wcp_eg - bcp_eg;

    // ── King proximity to passed pawns (endgame only) ──────────────────
    let wkp = eval_king_pawn_proximity(board, Color::White, white_passed, black_passed);
    let bkp = eval_king_pawn_proximity(board, Color::Black, black_passed, white_passed);
    let kp_eg = wkp - bkp;

    // ── Tempo ──────────────────────────────────────────────────────────
    let tempo_mg: Score = match board.side_to_move {
        Color::White => TEMPO_MG,
        Color::Black => -TEMPO_MG,
    };

    // ── Taper each component ───────────────────────────────────────────
    let material = taper(mat_mg, mat_eg, phase);
    let pst = taper(pst_mg, pst_eg, phase);
    let mobility = taper(mob_mg, mob_eg, phase);
    let pawn_structure = taper(pawn_mg, pawn_eg, phase);
    let passed_pawns = taper(pass_mg, pass_eg, phase);
    let king_safety = taper(ks_mg, 0, phase);
    let bishop_pair = taper(bp_mg, bp_eg, phase);
    let rook_placement = taper(rp_mg, rp_eg, phase);
    let knight_outpost = taper(no_mg, no_eg, phase);
    let connected_passers = taper(cp_mg, cp_eg, phase);
    let king_pawn_proximity = taper(0, kp_eg, phase);
    let tempo = taper(tempo_mg, 0, phase);

    let raw = material + pst + mobility + pawn_structure + passed_pawns
        + king_safety + bishop_pair + rook_placement + knight_outpost
        + connected_passers + king_pawn_proximity + tempo;

    let total = match board.side_to_move {
        Color::White => raw,
        Color::Black => -raw,
    };

    EvalBreakdown {
        material,
        pst,
        mobility,
        pawn_structure,
        passed_pawns,
        king_safety,
        bishop_pair,
        rook_placement,
        knight_outpost,
        connected_passers,
        king_pawn_proximity,
        tempo,
        total,
    }
}

fn piece_attacks(piece: Piece, sq: Square, occ: Bitboard) -> Bitboard {
    match piece {
        Piece::Knight => attacks::knight_attacks(sq),
        Piece::Bishop => attacks::bishop_attacks(sq, occ),
        Piece::Rook => attacks::rook_attacks(sq, occ),
        Piece::Queen => attacks::queen_attacks(sq, occ),
        _ => Bitboard::EMPTY,
    }
}

// ── Pawn structure evaluation ──────────────────────────────────────────────

fn eval_pawn_structure(our_pawns: Bitboard, _enemy_pawns: Bitboard, _color: Color) -> (Score, Score) {
    let mut mg = 0;
    let mut eg = 0;

    for f in 0..8 {
        let file_pawns = our_pawns & FILE_MASKS[f];
        let count = file_pawns.popcount();
        if count == 0 { continue; }

        // Doubled pawns: penalty for each extra pawn on the same file
        if count > 1 {
            let extra = (count - 1) as Score;
            mg -= 10 * extra;
            eg -= 20 * extra;
        }

        // Isolated pawns: no friendly pawns on adjacent files
        if (our_pawns & ADJACENT_FILE_MASKS[f]).is_empty() {
            mg -= 15 * count as Score;
            eg -= 20 * count as Score;
        }
    }

    // Backward pawns: a pawn is backward if no friendly pawn on adjacent files
    // can support its advance, and the advance square is controlled by an enemy pawn.
    let mut pawns = our_pawns;
    while pawns.is_not_empty() {
        let sq = pawns.pop_lsb();
        let file = sq.file();
        let rank = sq.rank();

        // Check if any friendly pawn on adjacent files is at same rank or behind
        let supporters = our_pawns & ADJACENT_FILE_MASKS[file as usize];
        if supporters.is_not_empty() {
            let mut has_support = false;
            let mut s = supporters;
            while s.is_not_empty() {
                let ssq = s.pop_lsb();
                let sr = match _color {
                    Color::White => ssq.rank() <= rank,
                    Color::Black => ssq.rank() >= rank,
                };
                if sr { has_support = true; break; }
            }
            if has_support { continue; }
        }

        // Check if the advance square is attacked by enemy pawn
        let advance_sq = match _color {
            Color::White => if rank < 7 { Square::new(file, rank + 1) } else { continue },
            Color::Black => if rank > 0 { Square::new(file, rank - 1) } else { continue },
        };
        let enemy_pawn_attacks = attacks::pawn_attacks(_color, advance_sq);
        if (_enemy_pawns & enemy_pawn_attacks).is_not_empty() {
            mg -= 10;
            eg -= 15;
        }
    }

    (mg, eg)
}

// ── Passed pawn evaluation ─────────────────────────────────────────────────

/// Returns (mg, eg, passed_pawn_bitboard)
fn eval_passed_pawns(our_pawns: Bitboard, enemy_pawns: Bitboard, color: Color) -> (Score, Score, Bitboard) {
    let mut mg = 0;
    let mut eg = 0;
    let mut passed = Bitboard::EMPTY;

    let mut pawns = our_pawns;
    while pawns.is_not_empty() {
        let sq = pawns.pop_lsb();
        let mask = match color {
            Color::White => PASSED_PAWN_MASKS_WHITE[sq.0 as usize],
            Color::Black => PASSED_PAWN_MASKS_BLACK[sq.0 as usize],
        };

        if (enemy_pawns & mask).is_empty() {
            passed.0 |= 1u64 << sq.0;
            let rank = match color {
                Color::White => sq.rank() as usize,
                Color::Black => (7 - sq.rank()) as usize,
            };
            mg += PASSED_PAWN_BONUS_MG[rank];
            eg += PASSED_PAWN_BONUS_EG[rank];
        }
    }

    (mg, eg, passed)
}

// ── Rook placement evaluation ──────────────────────────────────────────────

fn eval_rook_placement(
    board: &Board,
    color: Color,
    friendly_pawns: Bitboard,
    enemy_pawns: Bitboard,
    our_passed: Bitboard,
) -> (Score, Score) {
    let mut mg = 0;
    let mut eg = 0;

    let seventh_rank = match color {
        Color::White => 6u8,
        Color::Black => 1u8,
    };

    let mut rooks = board.piece_bb(color, Piece::Rook);
    while rooks.is_not_empty() {
        let sq = rooks.pop_lsb();
        let file = sq.file() as usize;

        // Open / semi-open file
        if (friendly_pawns & FILE_MASKS[file]).is_empty() {
            if (enemy_pawns & FILE_MASKS[file]).is_empty() {
                mg += ROOK_OPEN_FILE_MG;
                eg += ROOK_OPEN_FILE_EG;
            } else {
                mg += ROOK_SEMI_OPEN_FILE_MG;
                eg += ROOK_SEMI_OPEN_FILE_EG;
            }
        }

        // 7th rank
        if sq.rank() == seventh_rank {
            mg += ROOK_SEVENTH_RANK_MG;
            eg += ROOK_SEVENTH_RANK_EG;
        }

        // Behind passed pawn
        if (our_passed & FILE_MASKS[file]).is_not_empty() {
            let behind = match color {
                Color::White => {
                    let mut pp = our_passed & FILE_MASKS[file];
                    let passer_sq = pp.pop_lsb();
                    sq.rank() < passer_sq.rank()
                }
                Color::Black => {
                    let mut pp = our_passed & FILE_MASKS[file];
                    let passer_sq = pp.pop_lsb();
                    sq.rank() > passer_sq.rank()
                }
            };
            if behind {
                mg += ROOK_BEHIND_PASSER_MG;
                eg += ROOK_BEHIND_PASSER_EG;
            }
        }
    }

    (mg, eg)
}

// ── Knight outpost evaluation ──────────────────────────────────────────────

fn eval_knight_outposts(
    board: &Board,
    color: Color,
    friendly_pawns: Bitboard,
    enemy_pawns: Bitboard,
) -> (Score, Score) {
    let mut mg = 0;
    let mut eg = 0;

    let mut knights = board.piece_bb(color, Piece::Knight);
    while knights.is_not_empty() {
        let sq = knights.pop_lsb();
        let rel_rank = match color {
            Color::White => sq.rank(),
            Color::Black => 7 - sq.rank(),
        };

        // Must be on rank 4-6 (relative)
        if rel_rank < 4 || rel_rank > 6 { continue; }

        // Must be supported by own pawn
        let supporters = attacks::pawn_attacks(color.flip(), sq) & friendly_pawns;
        if supporters.is_empty() { continue; }

        // No enemy pawn can attack it (no enemy pawns on adjacent files ahead)
        let file = sq.file() as usize;
        let enemy_on_adj = enemy_pawns & ADJACENT_FILE_MASKS[file];
        let mut can_be_attacked = false;
        let mut e = enemy_on_adj;
        while e.is_not_empty() {
            let esq = e.pop_lsb();
            let enemy_rel_rank = match color {
                Color::White => esq.rank(),
                Color::Black => 7 - esq.rank(),
            };
            // Enemy pawn behind the knight can advance to attack it
            if enemy_rel_rank > rel_rank {
                // This enemy pawn is further advanced (from our perspective), can't reach back
            } else {
                can_be_attacked = true;
                break;
            }
        }
        if can_be_attacked { continue; }

        mg += KNIGHT_OUTPOST_MG;
        eg += KNIGHT_OUTPOST_EG;
    }

    (mg, eg)
}

// ── Connected passed pawns ─────────────────────────────────────────────────

fn eval_connected_passers(passed: Bitboard) -> (Score, Score) {
    let mut mg = 0;
    let mut eg = 0;

    let mut pp = passed;
    while pp.is_not_empty() {
        let sq = pp.pop_lsb();
        let file = sq.file() as usize;

        // Check if another passed pawn exists on an adjacent file
        if (passed & ADJACENT_FILE_MASKS[file]).is_not_empty() {
            mg += CONNECTED_PASSER_MG;
            eg += CONNECTED_PASSER_EG;
        }
    }

    (mg, eg)
}

// ── King proximity to passed pawns (endgame) ───────────────────────────────

fn eval_king_pawn_proximity(board: &Board, color: Color, our_passed: Bitboard, their_passed: Bitboard) -> Score {
    let king_sq = board.piece_bb(color, Piece::King).lsb();
    let mut bonus: Score = 0;

    // Our king close to our passed pawns = good (escort them)
    let mut pp = our_passed;
    while pp.is_not_empty() {
        let sq = pp.pop_lsb();
        let dist = chebyshev_distance(king_sq, sq);
        bonus += KING_PAWN_DIST_EG * (7 - dist as Score);
    }

    // Our king close to their passed pawns = good (block them)
    let mut ep = their_passed;
    while ep.is_not_empty() {
        let sq = ep.pop_lsb();
        let dist = chebyshev_distance(king_sq, sq);
        bonus += KING_PAWN_DIST_EG * (7 - dist as Score);
    }

    bonus
}

fn chebyshev_distance(a: Square, b: Square) -> u8 {
    let dr = (a.rank() as i8 - b.rank() as i8).unsigned_abs();
    let df = (a.file() as i8 - b.file() as i8).unsigned_abs();
    dr.max(df)
}

// ── King safety evaluation ─────────────────────────────────────────────────

fn eval_king_safety(board: &Board, color: Color, friendly_pawns: Bitboard, occ: Bitboard) -> Score {
    let them = color.flip();

    // Only evaluate king safety if opponent has a queen
    if board.piece_bb(them, Piece::Queen).is_empty() {
        return 0;
    }

    let king_sq = board.piece_bb(color, Piece::King).lsb();
    let king_file = king_sq.file();
    let mut safety = 0;

    // ── Pawn shield ────────────────────────────────────────────────────
    // Check pawns directly in front of the king (and adjacent files)
    let shield_rank = match color {
        Color::White => king_sq.rank() + 1,
        Color::Black => king_sq.rank().wrapping_sub(1),
    };

    if shield_rank < 8 {
        for f in king_file.saturating_sub(1)..=(king_file + 1).min(7) {
            let shield_sq = Square::new(f, shield_rank);
            if !friendly_pawns.contains(shield_sq) {
                safety -= 15;

                // Extra penalty if the second rank is also missing
                let second_rank = match color {
                    Color::White => shield_rank + 1,
                    Color::Black => shield_rank.wrapping_sub(1),
                };
                if second_rank < 8 {
                    let second_sq = Square::new(f, second_rank);
                    if !friendly_pawns.contains(second_sq) {
                        safety -= 10;
                    }
                }
            }
        }
    }

    // ── Open/semi-open files near king ─────────────────────────────────
    for f in king_file.saturating_sub(1)..=(king_file + 1).min(7) {
        let file_bb = FILE_MASKS[f as usize];
        if (friendly_pawns & file_bb).is_empty() {
            if (board.piece_bb(them, Piece::Pawn) & file_bb).is_empty() {
                safety -= 20; // open file
            } else {
                safety -= 10; // semi-open file
            }
        }
    }

    // ── Attacker count in king zone ────────────────────────────────────
    let king_zone = attacks::king_attacks(king_sq) | Bitboard::from_square(king_sq);
    let mut attacker_weight = 0;

    // Knights
    let mut knights = board.piece_bb(them, Piece::Knight);
    while knights.is_not_empty() {
        let sq = knights.pop_lsb();
        if (attacks::knight_attacks(sq) & king_zone).is_not_empty() {
            attacker_weight += 2;
        }
    }

    // Bishops
    let mut bishops = board.piece_bb(them, Piece::Bishop);
    while bishops.is_not_empty() {
        let sq = bishops.pop_lsb();
        if (attacks::bishop_attacks(sq, occ) & king_zone).is_not_empty() {
            attacker_weight += 2;
        }
    }

    // Rooks
    let mut rooks = board.piece_bb(them, Piece::Rook);
    while rooks.is_not_empty() {
        let sq = rooks.pop_lsb();
        if (attacks::rook_attacks(sq, occ) & king_zone).is_not_empty() {
            attacker_weight += 3;
        }
    }

    // Queens
    let mut queens = board.piece_bb(them, Piece::Queen);
    while queens.is_not_empty() {
        let sq = queens.pop_lsb();
        if (attacks::queen_attacks(sq, occ) & king_zone).is_not_empty() {
            attacker_weight += 5;
        }
    }

    // Quadratic penalty based on attacker pressure
    if attacker_weight >= 2 {
        safety -= attacker_weight * attacker_weight * 3;
    }

    safety
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startpos_is_roughly_equal() {
        crate::attacks::init();
        let board = Board::startpos();
        let score = evaluate(&board);
        assert!(score.abs() < 50, "Starting position eval near 0, got {score}");
    }

    #[test]
    fn up_a_queen_is_winning() {
        crate::attacks::init();
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let base = evaluate(&board);
        let board2 = Board::from_fen("rnbqkbnr/pppppppp/8/3Q4/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let extra = evaluate(&board2);
        assert!(extra > base + 800, "Extra queen should be worth ~900cp more");
    }

    #[test]
    fn king_active_in_endgame() {
        crate::attacks::init();
        let active = Board::from_fen("8/8/8/8/4K3/8/4P3/7k w - - 0 1").unwrap();
        let passive = Board::from_fen("K7/8/8/8/8/8/4P3/7k w - - 0 1").unwrap();
        assert!(evaluate(&active) > evaluate(&passive), "Active king better in endgame");
    }

    #[test]
    fn passed_pawn_bonus() {
        crate::attacks::init();
        // White has a passed pawn on e5, black pawns on a7/b7 only
        let passed = Board::from_fen("8/pp6/8/4P3/8/8/8/4K2k w - - 0 1").unwrap();
        let blocked = Board::from_fen("8/pp2p3/8/4P3/8/8/8/4K2k w - - 0 1").unwrap();
        assert!(evaluate(&passed) > evaluate(&blocked), "Passed pawn should score higher");
    }

    #[test]
    fn doubled_pawn_penalty() {
        crate::attacks::init();
        let doubled = Board::from_fen("4k3/8/4P3/8/4P3/8/8/4K3 w - - 0 1").unwrap();
        let normal = Board::from_fen("4k3/8/4P3/8/3P4/8/8/4K3 w - - 0 1").unwrap();
        assert!(evaluate(&normal) > evaluate(&doubled), "Doubled pawns should be penalized");
    }

    #[test]
    fn rook_on_open_file_bonus() {
        crate::attacks::init();
        // Same material — rook on open d-file vs rook on closed a-file
        // Both sides have pawns on b,c,e,f,g,h — d-file is open, a-file has white pawn
        let open = Board::from_fen("4k3/1pp1pppp/8/8/8/8/PPP1PPPP/3R1K2 w - - 0 1").unwrap();
        let closed = Board::from_fen("4k3/1pp1pppp/8/8/8/8/PPP1PPPP/R4K2 w - - 0 1").unwrap();
        let open_comps = eval_components(&open);
        let closed_comps = eval_components(&closed);
        assert!(
            open_comps.rook_placement > closed_comps.rook_placement,
            "Rook on open file should have higher rook_placement: open={}, closed={}",
            open_comps.rook_placement, closed_comps.rook_placement
        );
    }

    #[test]
    fn knight_outpost_bonus() {
        crate::attacks::init();
        // Knight on d5 supported by pawn on c4, no black pawns on c/e files
        let outpost = Board::from_fen("4k3/pp4pp/8/3N4/2P5/8/PP4PP/4K3 w - - 0 1").unwrap();
        let no_outpost = Board::from_fen("4k3/pp4pp/8/8/2PN4/8/PP4PP/4K3 w - - 0 1").unwrap();
        assert!(evaluate(&outpost) > evaluate(&no_outpost), "Knight outpost should score higher");
    }
}
