use std::time::{Duration, Instant};

use crate::attacks;
use crate::board::Board;
use crate::eval::{evaluate_with_pht, Score, INFINITY, MATE_SCORE};
use crate::movegen::{generate_legal_moves, make_move};
use crate::moves::*;
use crate::nnue;
use crate::tt::*;
use crate::types::*;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best_move: Move,
    pub score: Score,
    pub depth: u32,
    pub nodes: u64,
    /// Principal variation starting with `best_move`. Empty if no full
    /// PV could be reconstructed (e.g. early termination).
    pub pv: Vec<Move>,
}

const MAX_PLY: usize = 128;
const MAX_DEPTH: u32 = 64;

/// Late move pruning thresholds by depth
const LMP_THRESHOLD: [usize; 5] = [0, 5, 8, 12, 16];

/// SEE piece values (king = high to avoid capturing it as "cheapest")
const SEE_VALUE: [Score; Piece::COUNT] = [100, 320, 330, 500, 900, 20000];

pub struct Searcher {
    pub nodes: u64,
    pub tt: TranspositionTable,
    pawn_ht: crate::eval::PawnHashTable,
    nnue: nnue::NnueState,
    pub use_nnue: bool,
    pub eval_noise: i32,
    killers: [[Move; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; Color::COUNT],
    countermoves: [[Move; 64]; 64],
    /// Triangular PV table; `pv_table[ply][0..pv_length[ply]]` holds the PV
    /// from this ply downward. Boxed to keep the searcher off the stack.
    pv_table: Box<[[Move; MAX_PLY]; MAX_PLY]>,
    pv_length: [u8; MAX_PLY],
    position_history: Vec<u64>,
    search_hashes: [u64; MAX_PLY],
    stop_time: Option<Instant>,
    soft_limit: Option<Instant>,
    stopped: bool,
    prev_best_move: Move,
    best_move_stability: u32,
    search_start: Option<Instant>,
    pub silent: bool,
}

impl Searcher {
    pub fn new(tt_size_mb: usize) -> Self {
        Searcher {
            nodes: 0,
            tt: TranspositionTable::new(tt_size_mb),
            pawn_ht: crate::eval::PawnHashTable::new(1024), // 1MB pawn hash
            nnue: nnue::NnueState::new(),
            use_nnue: false,
            eval_noise: 0,
            killers: [[Move::NULL; 2]; MAX_PLY],
            history: [[[0i32; 64]; 64]; Color::COUNT],
            countermoves: [[Move::NULL; 64]; 64],
            pv_table: Box::new([[Move::NULL; MAX_PLY]; MAX_PLY]),
            pv_length: [0u8; MAX_PLY],
            position_history: Vec::new(),
            search_hashes: [0u64; MAX_PLY],
            stop_time: None,
            soft_limit: None,
            stopped: false,
            prev_best_move: Move::NULL,
            best_move_stability: 0,
            search_start: None,
            silent: false,
        }
    }

    /// Set the position history for repetition detection.
    /// Call this before each search with the hashes of all positions in the game so far.
    pub fn set_position_history(&mut self, hashes: Vec<u64>) {
        self.position_history = hashes;
    }

    /// Search to a fixed depth.
    pub fn search(&mut self, board: &Board, max_depth: u32) -> SearchResult {
        self.stop_time = None;
        self.soft_limit = None;
        self.stopped = false;
        self.search_internal(board, max_depth)
    }

    /// Search with a time limit (in milliseconds).
    pub fn search_timed(&mut self, board: &Board, time_ms: u64) -> SearchResult {
        self.stop_time = Some(Instant::now() + Duration::from_millis(time_ms));
        self.soft_limit = None;
        self.stopped = false;
        self.search_internal(board, MAX_DEPTH)
    }

    /// Search with soft/hard time limits for smart time management.
    pub fn search_with_time_management(
        &mut self,
        board: &Board,
        soft_ms: u64,
        hard_ms: u64,
    ) -> SearchResult {
        self.search_with_time_management_capped(board, soft_ms, hard_ms, None)
    }

    /// Search with soft/hard time limits and an optional depth cap. The soft
    /// limit enables best-move stability early-exit; the hard limit is the
    /// absolute stop. `max_depth = None` means unlimited (up to MAX_DEPTH).
    pub fn search_with_time_management_capped(
        &mut self,
        board: &Board,
        soft_ms: u64,
        hard_ms: u64,
        max_depth: Option<u32>,
    ) -> SearchResult {
        let now = Instant::now();
        self.soft_limit = Some(now + Duration::from_millis(soft_ms));
        self.stop_time = Some(now + Duration::from_millis(hard_ms));
        self.stopped = false;
        let depth = max_depth.unwrap_or(MAX_DEPTH).min(MAX_DEPTH);
        self.search_internal(board, depth)
    }

    fn check_time(&mut self) {
        if self.nodes & 2047 == 0 {
            if let Some(deadline) = self.stop_time {
                if Instant::now() >= deadline {
                    self.stopped = true;
                }
            }
        }
    }

    /// Evaluate a position using NNUE (if available) or HCE fallback.
    /// When `eval_noise > 0`, adds deterministic hash-based noise for
    /// strength limiting (weaker play at lower difficulty levels).
    #[inline]
    fn eval(&mut self, board: &Board, ply: u32) -> Score {
        let base = if self.use_nnue {
            self.nnue.evaluate_at_ply(board, ply as usize)
        } else {
            evaluate_with_pht(board, &mut self.pawn_ht)
        };
        if self.eval_noise > 0 {
            base + crate::strength::deterministic_noise(board.hash, self.eval_noise)
        } else {
            base
        }
    }

    /// Check if a position hash is a repetition (appears in game history or search tree).
    fn is_repetition(&self, hash: u64, ply: u32) -> bool {
        // Check game history (positions before the search started)
        for &h in &self.position_history {
            if h == hash {
                return true;
            }
        }
        // Check positions visited in this search (twofold within the search tree)
        for i in 0..ply as usize {
            if self.search_hashes[i] == hash {
                return true;
            }
        }
        false
    }

    fn search_internal(&mut self, board: &Board, max_depth: u32) -> SearchResult {
        let mut best_result = SearchResult {
            best_move: Move::NULL,
            score: 0,
            depth: 0,
            nodes: 0,
            pv: Vec::new(),
        };

        let mut prev_score = 0;
        self.best_move_stability = 0;
        self.prev_best_move = Move::NULL;
        self.search_start = Some(Instant::now());

        // Age history table between searches
        for c in 0..Color::COUNT {
            for f in 0..64 {
                for t in 0..64 {
                    self.history[c][f][t] /= 2;
                }
            }
        }

        // Initialize NNUE accumulator at root
        if self.use_nnue {
            self.nnue.refresh(board, 0);
        }

        let mut total_nodes = 0u64;

        for depth in 1..=max_depth {
            self.nodes = 0;
            self.stopped = false;

            // Check time between iterations
            if let Some(deadline) = self.stop_time {
                if Instant::now() >= deadline {
                    break;
                }
            }

            let (score, mv) = if depth <= 3 {
                self.negamax(board, depth, -INFINITY, INFINITY, 0, Move::NULL, Move::NULL)
            } else {
                let mut delta = 25;
                let mut alpha = prev_score - delta;
                let mut beta = prev_score + delta;
                loop {
                    let (score, mv) = self.negamax(board, depth, alpha, beta, 0, Move::NULL, Move::NULL);
                    if self.stopped { break (score, mv); }
                    if score <= alpha {
                        alpha = (alpha - delta).max(-INFINITY);
                        delta *= 2;
                    } else if score >= beta {
                        beta = (beta + delta).min(INFINITY);
                        delta *= 2;
                    } else {
                        break (score, mv);
                    }
                    if delta > 1000 {
                        break self.negamax(board, depth, -INFINITY, INFINITY, 0, Move::NULL, Move::NULL);
                    }
                }
            };

            total_nodes += self.nodes;

            if self.stopped && best_result.depth > 0 {
                break;
            }

            if !mv.is_null() {
                // Track move stability for time management
                if mv.raw() == self.prev_best_move.raw() {
                    self.best_move_stability += 1;
                } else {
                    self.best_move_stability = 0;
                    self.prev_best_move = mv;
                }

                let pv_len = self.pv_length[0] as usize;
                let mut pv = Vec::with_capacity(pv_len);
                for j in 0..pv_len {
                    pv.push(self.pv_table[0][j]);
                }
                if pv.is_empty() {
                    pv.push(mv);
                }

                best_result = SearchResult {
                    best_move: mv,
                    score,
                    depth,
                    nodes: total_nodes,
                    pv,
                };
            }

            prev_score = score;

            if !self.silent {
                eprintln!(
                    "info depth {} score cp {} nodes {} pv {}",
                    depth, score, self.nodes, best_result.best_move
                );
            }

            if score.abs() > MATE_SCORE - 100 {
                break;
            }

            // Smart early termination: stable best move + past soft limit.
            // Two tiers — strict at soft, lenient halfway from soft to hard,
            // so positions with naturally shallow stability (e.g. opening)
            // still get early-exit benefit.
            if let Some(soft) = self.soft_limit {
                let now = Instant::now();
                if now >= soft {
                    if self.best_move_stability >= 3 {
                        break;
                    }
                    if (score - prev_score).abs() <= 30 && self.best_move_stability >= 2 {
                        break;
                    }
                    if let Some(hard) = self.stop_time {
                        let window = hard.saturating_duration_since(soft);
                        let elapsed_in_window = now.saturating_duration_since(soft);
                        if elapsed_in_window * 2 >= window {
                            if self.best_move_stability >= 1 {
                                break;
                            }
                            if (score - prev_score).abs() <= 60 {
                                break;
                            }
                        }
                    }
                }
            }
        }

        best_result
    }

    fn negamax(
        &mut self,
        board: &Board,
        depth: u32,
        mut alpha: Score,
        beta: Score,
        ply: u32,
        prev_move: Move,
        excluded_move: Move, // for singular extensions
    ) -> (Score, Move) {
        self.nodes += 1;
        self.check_time();
        if self.stopped {
            return (0, Move::NULL);
        }

        let hash = board.hash;
        let ply_idx = (ply as usize).min(MAX_PLY - 1);

        // Reset PV length at this ply; it will be filled as we improve alpha.
        self.pv_length[ply_idx] = 0;

        // Record position for repetition detection
        self.search_hashes[ply_idx] = hash;

        // ── Draw detection ─────────────────────────────────────────────
        if ply > 0 {
            // 50-move rule
            if board.halfmove_clock >= 100 {
                return (0, Move::NULL);
            }
            // Repetition
            if self.is_repetition(hash, ply) {
                return (0, Move::NULL);
            }
        }

        // ── TT probe ───────────────────────────────────────────────────
        let mut tt_move = Move::NULL;
        if excluded_move.is_null() {
            if let Some(entry) = self.tt.probe(hash) {
                tt_move = entry.best_move;
                if entry.depth as u32 >= depth && ply > 0 {
                    let tt_score = entry.score;
                    match entry.flag {
                        TTFlag::Exact => return (tt_score, entry.best_move),
                        TTFlag::LowerBound => {
                            if tt_score >= beta { return (tt_score, entry.best_move); }
                        }
                        TTFlag::UpperBound => {
                            if tt_score <= alpha { return (tt_score, entry.best_move); }
                        }
                    }
                }
            }
        }

        if depth == 0 {
            return (self.quiescence(board, alpha, beta, ply), Move::NULL);
        }

        // ── IID ────────────────────────────────────────────────────────
        if tt_move.is_null() && depth >= 4 && ply > 0 && excluded_move.is_null() {
            let (_, iid_mv) = self.negamax(board, depth - 2, alpha, beta, ply, prev_move, Move::NULL);
            if !iid_mv.is_null() {
                tt_move = iid_mv;
            }
        }

        let king_sq = board.piece_bb(board.side_to_move, Piece::King).lsb();
        let in_check = attacks::is_square_attacked(board, king_sq, board.side_to_move.flip());
        let effective_depth = if in_check { depth + 1 } else { depth };

        // ── Pruning (not in check, not root, not singular search) ──────
        if !in_check && ply > 0 && excluded_move.is_null() {
            let static_eval = self.eval(board, ply);

            // Reverse futility pruning
            if depth <= 3 {
                let margin = 120 * depth as Score;
                if static_eval - margin >= beta {
                    return (static_eval - margin, Move::NULL);
                }
            }

            // Razoring
            if depth <= 2 {
                let margin = 300 + 200 * (depth as Score - 1);
                if static_eval + margin <= alpha {
                    let q_score = self.quiescence(board, alpha, beta, ply);
                    if q_score <= alpha {
                        return (q_score, Move::NULL);
                    }
                }
            }
        }

        // ── Null move pruning ──────────────────────────────────────────
        const NULL_MOVE_REDUCTION: u32 = 3;
        if !in_check
            && depth >= 3
            && ply > 0
            && excluded_move.is_null()
            && board.has_non_pawn_material(board.side_to_move)
        {
            if self.use_nnue {
                self.nnue.push_null(ply as usize);
            }
            let mut null_board = board.clone();
            null_board.make_null_move();

            let null_depth = effective_depth.saturating_sub(1 + NULL_MOVE_REDUCTION);
            let (null_score, _) = self.negamax(
                &null_board, null_depth, -beta, -beta + 1, ply + 1, Move::NULL, Move::NULL,
            );
            let null_score = -null_score;

            if null_score >= beta {
                return (beta, Move::NULL);
            }
        }

        // ── Singular extension check ───────────────────────────────────
        let mut singular_extension = 0i32;
        if depth >= 6
            && ply > 0
            && !tt_move.is_null()
            && excluded_move.is_null()
        {
            if let Some(entry) = self.tt.probe(hash) {
                if entry.depth as u32 >= depth - 3
                    && matches!(entry.flag, TTFlag::Exact | TTFlag::LowerBound)
                {
                    let s_beta = entry.score - (depth as Score * 2);
                    let s_depth = (depth - 1) / 2;
                    let (s_score, _) = self.negamax(
                        board, s_depth, s_beta - 1, s_beta, ply, prev_move, tt_move,
                    );
                    if s_score < s_beta {
                        singular_extension = 1;
                    }
                }
            }
        }

        let moves = generate_legal_moves(board);

        if moves.is_empty() {
            if in_check {
                return (-MATE_SCORE + ply as Score, Move::NULL);
            } else {
                return (0, Move::NULL);
            }
        }

        // ── Move ordering ──────────────────────────────────────────────
        let us = board.side_to_move;
        let killers = self.killers[ply_idx];
        let countermove = if !prev_move.is_null() {
            self.countermoves[prev_move.from_sq().0 as usize][prev_move.to_sq().0 as usize]
        } else {
            Move::NULL
        };
        let history = &self.history;
        let ordered = order_moves(board, &moves, tt_move, &killers, countermove, history);

        let mut best_move = ordered[0];
        let mut best_score = -INFINITY;
        let orig_alpha = alpha;

        let futility_eval = if !in_check && depth <= 2 {
            Some(self.eval(board, ply))
        } else {
            None
        };

        for i in 0..ordered.len() {
            if self.stopped { break; }

            let mv = ordered[i];

            // Skip excluded move (singular extension search)
            if mv.raw() == excluded_move.raw() {
                continue;
            }

            let is_cap = is_capture(board, mv);
            let is_quiet = !is_cap && !matches!(mv.flag(), MoveFlag::Promotion);

            // ── Futility pruning ───────────────────────────────────────
            if is_quiet && i > 0 && !in_check {
                if let Some(fe) = futility_eval {
                    let margin = 150 * depth as Score;
                    if fe + margin <= alpha {
                        continue;
                    }
                }
            }

            // ── LMP ────────────────────────────────────────────────────
            if is_quiet && !in_check && depth <= 4 && i >= LMP_THRESHOLD[depth as usize] {
                continue;
            }

            // ── SEE pruning for bad captures in main search ────────────
            if is_cap && i > 0 && depth <= 3 && !in_check && see(board, mv) < 0 {
                continue;
            }

            if self.use_nnue {
                self.nnue.push_and_update(board, mv, ply as usize);
            }
            let mut new_board = board.clone();
            make_move(&mut new_board, mv);

            // Extension for the TT/singular move
            let ext = if i == 0 && singular_extension > 0 {
                singular_extension as u32
            } else {
                0
            };
            let search_depth = effective_depth - 1 + ext;

            let score;

            // ── LMR (also reduce bad captures) ─────────────────────────
            let should_reduce = i >= 3
                && depth >= 3
                && !in_check
                && (is_quiet || (is_cap && see(board, mv) < 0));

            if should_reduce {
                let reduction = 1 + (i as u32 / 8) + (depth / 6);
                let reduced_depth = search_depth.saturating_sub(reduction);

                let (reduced_score, _) = self.negamax(
                    &new_board, reduced_depth, -alpha - 1, -alpha, ply + 1, mv, Move::NULL,
                );
                let reduced_score = -reduced_score;

                if reduced_score <= alpha {
                    if reduced_score > best_score {
                        best_score = reduced_score;
                        best_move = mv;
                    }
                    continue;
                }
            }

            // ── PVS ────────────────────────────────────────────────────
            if i == 0 {
                let (s, _) = self.negamax(
                    &new_board, search_depth, -beta, -alpha, ply + 1, mv, Move::NULL,
                );
                score = -s;
            } else {
                let (zw, _) = self.negamax(
                    &new_board, search_depth, -alpha - 1, -alpha, ply + 1, mv, Move::NULL,
                );
                let zw = -zw;

                if zw > alpha && zw < beta {
                    let (s, _) = self.negamax(
                        &new_board, search_depth, -beta, -alpha, ply + 1, mv, Move::NULL,
                    );
                    score = -s;
                } else {
                    score = zw;
                }
            }

            if score > best_score {
                best_score = score;
                best_move = mv;
                // Update PV: this move plus the child's PV.
                if ply_idx + 1 < MAX_PLY {
                    self.pv_table[ply_idx][0] = mv;
                    let child_len = self.pv_length[ply_idx + 1] as usize;
                    let copy_len = child_len.min(MAX_PLY - 1);
                    for j in 0..copy_len {
                        self.pv_table[ply_idx][j + 1] = self.pv_table[ply_idx + 1][j];
                    }
                    self.pv_length[ply_idx] = (copy_len + 1) as u8;
                }
            }

            if score >= beta {
                if is_quiet {
                    self.killers[ply_idx][1] = self.killers[ply_idx][0];
                    self.killers[ply_idx][0] = mv;

                    let bonus = (depth * depth) as i32;
                    let c = us as usize;
                    let h = &mut self.history[c][mv.from_sq().0 as usize][mv.to_sq().0 as usize];
                    *h += bonus - *h * bonus.abs() / 16384;

                    if !prev_move.is_null() {
                        self.countermoves[prev_move.from_sq().0 as usize][prev_move.to_sq().0 as usize] = mv;
                    }
                }

                self.tt.store(hash, depth as u8, TTFlag::LowerBound, score, mv);
                return (score, mv);
            }

            if score > alpha {
                alpha = score;
            }
        }

        if self.stopped {
            return (best_score, best_move);
        }

        let flag = if alpha > orig_alpha { TTFlag::Exact } else { TTFlag::UpperBound };
        self.tt.store(hash, depth as u8, flag, best_score, best_move);

        (best_score, best_move)
    }

    /// Quiescence search with SEE ordering and pruning.
    fn quiescence(&mut self, board: &Board, mut alpha: Score, beta: Score, ply: u32) -> Score {
        self.nodes += 1;
        self.check_time();
        if self.stopped { return 0; }

        let stand_pat = self.eval(board, ply);
        if stand_pat >= beta {
            return beta;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let delta = 1000;
        if stand_pat + delta < alpha {
            return alpha;
        }

        let moves = generate_legal_moves(board);

        // Collect and order captures by SEE
        let mut captures: Vec<(Move, Score)> = Vec::new();
        for i in 0..moves.len() {
            let mv = moves[i];
            if !is_capture(board, mv) && !matches!(mv.flag(), MoveFlag::Promotion) {
                continue;
            }
            let see_val = see(board, mv);
            if see_val < 0 { continue; } // SEE pruning: skip losing captures
            captures.push((mv, see_val));
        }
        captures.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        for (mv, _) in captures {
            if self.use_nnue {
                self.nnue.push_and_update(board, mv, ply as usize);
            }
            let mut new_board = board.clone();
            make_move(&mut new_board, mv);

            let score = -self.quiescence(&new_board, -beta, -alpha, ply + 1);

            if score >= beta {
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }

        alpha
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Static Exchange Evaluation (SEE)
// ════════════════════════════════════════════════════════════════════════════

/// Evaluate a capture sequence on the target square without searching.
/// Returns the material gain/loss from the moving side's perspective.
fn see(board: &Board, mv: Move) -> Score {
    let from = mv.from_sq();
    let to = mv.to_sq();

    // Get the initial attacker's value
    let attacker_piece = match board.piece_type_on(from) {
        Some(p) => p,
        None => return 0,
    };

    // Get the initial victim's value
    let victim_value = if matches!(mv.flag(), MoveFlag::EnPassant) {
        SEE_VALUE[Piece::Pawn as usize]
    } else {
        match board.piece_type_on(to) {
            Some(p) => SEE_VALUE[p as usize],
            None => {
                if matches!(mv.flag(), MoveFlag::Promotion) {
                    return SEE_VALUE[mv.promotion_piece() as usize] - SEE_VALUE[Piece::Pawn as usize];
                }
                return 0;
            }
        }
    };

    // Build the gain array: gain[d] = value of the piece captured at depth d
    let mut gain = [0i32; 33];
    let mut d = 0usize;
    gain[0] = victim_value;

    let mut side = board.side_to_move.flip(); // opponent gets to recapture first
    let mut occ = board.all_occupied();
    occ.0 &= !(1u64 << from.0); // remove initial attacker
    let mut current_piece_value = SEE_VALUE[attacker_piece as usize];

    loop {
        d += 1;
        gain[d] = current_piece_value - gain[d - 1]; // what we stand to win

        // Pruning: if even capturing can't beat the running total, stop
        if gain[d].max(-gain[d - 1]) < 0 {
            break;
        }

        // Find cheapest attacker for the recapturing side
        match cheapest_attacker(board, to, side, occ) {
            Some((sq, piece)) => {
                current_piece_value = SEE_VALUE[piece as usize];
                occ.0 &= !(1u64 << sq.0); // remove this piece (reveals x-rays)
                side = side.flip();
            }
            None => break,
        }

        if d >= 32 { break; }
    }

    // Negamax the gain array: each side can choose to not recapture
    while d > 0 {
        d -= 1;
        gain[d] = -((-gain[d]).max(gain[d + 1]));
    }

    gain[0]
}

/// Find the cheapest attacker of a square for a given side.
fn cheapest_attacker(board: &Board, sq: Square, side: Color, occ: Bitboard) -> Option<(Square, Piece)> {
    // Check in order of value: Pawn, Knight, Bishop, Rook, Queen, King

    // Pawns
    let pawn_attacks = attacks::pawn_attacks(side.flip(), sq); // squares that attack sq
    let pawns = board.piece_bb(side, Piece::Pawn) & pawn_attacks & occ;
    if pawns.is_not_empty() {
        return Some((pawns.lsb(), Piece::Pawn));
    }

    // Knights
    let knight_attacks = attacks::knight_attacks(sq);
    let knights = board.piece_bb(side, Piece::Knight) & knight_attacks & occ;
    if knights.is_not_empty() {
        return Some((knights.lsb(), Piece::Knight));
    }

    // Bishops (also finds x-ray through removed pieces)
    let bishop_attacks = attacks::bishop_attacks(sq, occ);
    let bishops = board.piece_bb(side, Piece::Bishop) & bishop_attacks;
    if bishops.is_not_empty() {
        return Some((bishops.lsb(), Piece::Bishop));
    }

    // Rooks
    let rook_attacks = attacks::rook_attacks(sq, occ);
    let rooks = board.piece_bb(side, Piece::Rook) & rook_attacks;
    if rooks.is_not_empty() {
        return Some((rooks.lsb(), Piece::Rook));
    }

    // Queens (bishop + rook attacks, discovers x-rays naturally)
    let queen_attacks = bishop_attacks | rook_attacks;
    let queens = board.piece_bb(side, Piece::Queen) & queen_attacks;
    if queens.is_not_empty() {
        return Some((queens.lsb(), Piece::Queen));
    }

    // King
    let king_attacks = attacks::king_attacks(sq);
    let kings = board.piece_bb(side, Piece::King) & king_attacks & occ;
    if kings.is_not_empty() {
        return Some((kings.lsb(), Piece::King));
    }

    None
}

// ════════════════════════════════════════════════════════════════════════════
// Move ordering
// ════════════════════════════════════════════════════════════════════════════

fn is_capture(board: &Board, mv: Move) -> bool {
    if matches!(mv.flag(), MoveFlag::EnPassant) {
        return true;
    }
    board.color_bb(board.side_to_move.flip()).contains(mv.to_sq())
}

fn order_moves(
    board: &Board,
    moves: &MoveList,
    tt_move: Move,
    killers: &[Move; 2],
    countermove: Move,
    history: &[[[i32; 64]; 64]; Color::COUNT],
) -> Vec<Move> {
    let mut scored: Vec<(Move, Score)> = Vec::with_capacity(moves.len());
    for i in 0..moves.len() {
        let mv = moves[i];
        scored.push((mv, score_move(board, mv, tt_move, killers, countermove, history)));
    }
    scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    scored.into_iter().map(|(mv, _)| mv).collect()
}

fn score_move(
    board: &Board,
    mv: Move,
    tt_move: Move,
    killers: &[Move; 2],
    countermove: Move,
    history: &[[[i32; 64]; 64]; Color::COUNT],
) -> Score {
    if mv.raw() == tt_move.raw() { return 100_000; }

    if matches!(mv.flag(), MoveFlag::Promotion) {
        return 80_000 + match mv.promotion_piece() {
            Piece::Queen => 900, Piece::Rook => 500,
            Piece::Bishop => 330, Piece::Knight => 320, _ => 0,
        };
    }

    if is_capture(board, mv) {
        let see_val = see(board, mv);
        return if see_val >= 0 {
            60_000 + see_val // good captures: above quiets
        } else {
            see_val - 10_000 // bad captures: below everything
        };
    }

    if mv.raw() == killers[0].raw() { return 50_000; }
    if mv.raw() == killers[1].raw() { return 49_000; }
    if mv.raw() == countermove.raw() { return 48_000; }

    history[board.side_to_move as usize][mv.from_sq().0 as usize][mv.to_sq().0 as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attacks;

    #[test]
    fn finds_mate_in_one() {
        attacks::init();
        let board = Board::from_fen(
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        ).unwrap();
        let mut searcher = Searcher::new(16);
        let result = searcher.search(&board, 3);
        assert!(result.score > MATE_SCORE - 10, "Should find mate, got {}", result.score);
    }

    #[test]
    fn does_not_blunder_queen() {
        attacks::init();
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1").unwrap();
        let mut searcher = Searcher::new(16);
        let result = searcher.search(&board, 4);
        assert!(result.score > -100, "Should not blunder, got {}", result.score);
    }

    #[test]
    fn captures_free_piece() {
        attacks::init();
        let board = Board::from_fen("rnb1kbnr/pppppppp/8/8/3q4/4P3/PPPP1PPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut searcher = Searcher::new(16);
        let result = searcher.search(&board, 4);
        assert_eq!(result.best_move.to_sq().to_algebraic(), "d4", "Got {}", result.best_move);
    }

    #[test]
    fn tt_speeds_up_search() {
        attacks::init();
        let board = Board::startpos();
        let mut searcher = Searcher::new(16);
        searcher.search(&board, 5);
        let first = searcher.nodes;
        searcher.nodes = 0;
        searcher.search(&board, 5);
        assert!(searcher.nodes < first, "TT should reduce nodes: first={first}, second={}", searcher.nodes);
    }

    #[test]
    fn time_limited_search_stops() {
        attacks::init();
        let board = Board::startpos();
        let mut searcher = Searcher::new(16);
        let start = Instant::now();
        let result = searcher.search_timed(&board, 100);
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_millis(500), "Should stop in time: {:?}", elapsed);
        assert!(!result.best_move.is_null(), "Should find a move");
    }

    #[test]
    fn repetition_detection_draw() {
        attacks::init();
        // Test that the repetition check works by verifying the engine
        // avoids positions it has seen before.
        let board = Board::from_fen("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1").unwrap();
        let mut searcher = Searcher::new(16);

        // Search without history — should get a normal positive score
        let result_no_rep = searcher.search(&board, 6);
        assert!(result_no_rep.score > 0, "White should be winning without repetition");

        // Now search with the same position in history — deeper searches should
        // find that many continuations lead to repetitions and score lower
        searcher.set_position_history(vec![board.hash]);
        let result_with_rep = searcher.search(&board, 6);

        // The score should be lower (possibly 0 if all good lines repeat)
        assert!(
            result_with_rep.score <= result_no_rep.score,
            "Score should not increase with repetition history: without={}, with={}",
            result_no_rep.score, result_with_rep.score
        );
    }

    #[test]
    fn see_free_capture_positive() {
        attacks::init();
        // Knight on d4 undefended, pawn on e3 can capture
        let board = Board::from_fen("4k3/8/8/8/3n4/4P3/8/4K3 w - - 0 1").unwrap();
        let mv = Move::new(Square::from_algebraic("e3").unwrap(), Square::from_algebraic("d4").unwrap());
        let score = see(&board, mv);
        assert!(score > 0, "PxN undefended should be positive, got {score}");
    }

    #[test]
    fn see_defended_capture() {
        attacks::init();
        // Pawn on d5 defended by pawn on e6, queen captures d5
        let board = Board::from_fen("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1").unwrap();
        let mv = Move::new(Square::from_algebraic("d1").unwrap(), Square::from_algebraic("d5").unwrap());
        let score = see(&board, mv);
        assert!(score < 0, "QxP defended by pawn should be negative, got {score}");
    }

    #[test]
    fn smart_time_management() {
        attacks::init();
        let board = Board::startpos();
        let mut searcher = Searcher::new(16);
        let start = Instant::now();
        // Soft limit 200ms, hard limit 5000ms — should stop well before hard limit
        let result = searcher.search_with_time_management(&board, 200, 5000);
        let elapsed = start.elapsed();
        assert!(!result.best_move.is_null());
        // Should stop well before the hard limit due to move stability.
        // The generous threshold accounts for unoptimized debug builds on slow CI runners.
        assert!(elapsed < Duration::from_millis(3000), "Should use smart early stop: {:?}", elapsed);
    }

    #[test]
    fn capped_time_management_respects_depth_cap() {
        attacks::init();
        let board = Board::startpos();
        let mut searcher = Searcher::new(16);
        // Generous time bounds, tight depth cap — depth must dominate.
        let result = searcher.search_with_time_management_capped(&board, 5000, 10000, Some(3));
        assert!(!result.best_move.is_null());
        assert!(result.depth <= 3, "Depth cap not respected: got {}", result.depth);
    }

    #[test]
    fn nnue_search_completes_without_crash() {
        attacks::init();
        let _ = crate::nnue::init(None); // loads embedded default net
        let board = Board::startpos();
        let mut searcher = Searcher::new(16);
        searcher.use_nnue = true;
        let result = searcher.search(&board, 4);
        assert!(!result.best_move.is_null(), "Should find a move with NNUE");
        assert!(result.nodes > 0, "Should search some nodes");
    }
}
