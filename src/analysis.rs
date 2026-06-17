use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;

use crate::attacks;
use crate::board::Board;
use crate::eval::{self, EvalBreakdown, Score};
use crate::movegen::{generate_legal_moves, make_move};
use crate::moves::{Move, MoveFlag};
use crate::eval::MATE_SCORE;
use crate::search::{SearchResult, Searcher, see};
use crate::types::*;

// ════════════════════════════════════════════════════════════════════════════
// Move classification
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveClass {
    Best,
    Good,
    Inaccuracy,
    Mistake,
    Blunder,
    Brilliant,
    Forced, // only one legal move
}

impl MoveClass {
    pub fn symbol(self) -> &'static str {
        match self {
            MoveClass::Best => "✓",
            MoveClass::Good => "",
            MoveClass::Inaccuracy => "?!",
            MoveClass::Mistake => "?",
            MoveClass::Blunder => "??",
            MoveClass::Brilliant => "!!",
            MoveClass::Forced => "□",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            MoveClass::Best => "Best",
            MoveClass::Good => "Good",
            MoveClass::Inaccuracy => "Inaccuracy",
            MoveClass::Mistake => "Mistake",
            MoveClass::Blunder => "Blunder",
            MoveClass::Brilliant => "Brilliant",
            MoveClass::Forced => "Forced",
        }
    }

    pub fn to_db_str(self) -> &'static str {
        match self {
            MoveClass::Best => "best",
            MoveClass::Good => "good",
            MoveClass::Inaccuracy => "inaccuracy",
            MoveClass::Mistake => "mistake",
            MoveClass::Blunder => "blunder",
            MoveClass::Brilliant => "brilliant",
            MoveClass::Forced => "forced",
        }
    }

    pub fn from_db_str(s: &str) -> Option<MoveClass> {
        match s {
            "best" => Some(MoveClass::Best),
            "good" => Some(MoveClass::Good),
            "inaccuracy" => Some(MoveClass::Inaccuracy),
            "mistake" => Some(MoveClass::Mistake),
            "blunder" => Some(MoveClass::Blunder),
            "brilliant" => Some(MoveClass::Brilliant),
            "forced" => Some(MoveClass::Forced),
            _ => None,
        }
    }

    pub fn from_cpl(cpl: Score, was_sacrifice: bool) -> Self {
        if was_sacrifice && cpl <= 20 {
            return MoveClass::Brilliant;
        }
        // Industry-standard bands (close to chess.com / lichess). The previous
        // thresholds (Mistake at 51 cp) were absurdly tight — a 76-cp opening
        // move like c4 vs Nc3 would get flagged as a mistake when it's a
        // perfectly normal alternative.
        match cpl {
            0..=15 => MoveClass::Best,         // 0-15 cp: nailed it (or within noise)
            16..=50 => MoveClass::Good,        // 16-50 cp: fine move, small loss
            51..=120 => MoveClass::Inaccuracy, // 51-120 cp: noticeable but recoverable
            121..=250 => MoveClass::Mistake,   // 121-250 cp: clear error
            _ => MoveClass::Blunder,           // 251+ cp: catastrophic
        }
    }
}

/// Whether a move is a genuine material sacrifice — the gate for a Brilliant
/// classification. A real sacrifice gives up *at least a minor piece* of
/// material (not merely a pawn) while the move is still among the best and the
/// position is not lost.
///
/// The previous check fired whenever the opponent had *any* pawn-winning
/// capture available after the move (`opp_best_see >= 100`), which is true on
/// almost every move in a middlegame — so nearly every accurate move was
/// labelled "Brilliant". Requiring ~a minor piece of net material plus a
/// non-losing position makes brilliancies rare and meaningful again.
pub(crate) fn is_sacrifice(
    board: &Board,
    played_mv: Move,
    board_after: &Board,
    cpl: Score,
    is_forced: bool,
    best_eval_for_side: Score,
) -> bool {
    // Net-material threshold for "a real sacrifice". Note SEE here nets the
    // capturing piece's value, so a hung knight taken by a pawn reads ~220
    // (320 - 100), a hung rook ~400, while a free pawn reads ~0 and winning
    // the exchange reads ~180. 200 cp therefore catches genuine piece
    // sacrifices while excluding pawn-grabs and exchange-level noise.
    const SAC_CP: Score = 200;

    // Brilliant requires a top, non-forced move in a position that is still at
    // least roughly equal. A desperate sacrifice while already lost is not it.
    if is_forced || cpl > 20 || best_eval_for_side < -50 {
        return false;
    }

    // (a) the move is itself a capture that loses >= that much by SEE.
    if see(board, played_mv) <= -SAC_CP {
        return true;
    }

    // (b) the move leaves >= that much of our material en prise: the
    //     opponent's best capture in the resulting position wins it.
    //     (see() returns 0 for non-captures, so the max is over captures.)
    let opp_moves = generate_legal_moves(board_after);
    let mut opp_best = 0;
    for j in 0..opp_moves.len() {
        let s = see(board_after, opp_moves[j]);
        if s > opp_best {
            opp_best = s;
        }
    }
    opp_best >= SAC_CP
}

/// Convert a centipawn eval (white's perspective) to white's winning
/// percentage in [0, 100]. Uses the lichess sigmoid mapping.
fn win_percentage(cp: Score) -> f64 {
    // Clamp to avoid numerical issues at extreme mate scores. ±2000 cp
    // already maps to >99.96% / <0.04% so the cap is invisible in practice.
    let cp = cp.clamp(-2000, 2000) as f64;
    100.0 / (1.0 + (-0.004 * cp).exp())
}

/// Lichess per-move accuracy formula. Takes win-percentage loss (from the
/// player's perspective, in [0, 100]) and returns accuracy in [0, 100].
/// A 0% loss → 100% accuracy; 50% loss → ~8.5% accuracy. Saturates near
/// zero for catastrophic moves rather than going linearly to zero.
fn accuracy_from_wp_loss(wp_loss: f64) -> f64 {
    let acc = 103.1668 * (-0.04354 * wp_loss).exp() - 3.1669;
    acc.clamp(0.0, 100.0)
}

// ════════════════════════════════════════════════════════════════════════════
// Analysis result per move
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct MoveAnalysis {
    pub move_number: usize,     // 1-based full move number
    pub side: Color,
    pub move_san: String,
    pub eval_before: Score,     // centipawns, from White's perspective
    pub eval_after: Score,      // centipawns, from White's perspective
    pub best_move_uci: String,
    pub best_eval: Score,       // centipawns, from White's perspective
    pub cpl: Score,             // centipawn loss (always >= 0)
    pub classification: MoveClass,
    pub explanation: Option<String>,
}

// ════════════════════════════════════════════════════════════════════════════
// Full game analysis result
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct GameAnalysis {
    pub moves: Vec<MoveAnalysis>,
    pub user_color: Color,
    pub user_accuracy: f64,     // 0-100
    pub eval_history: Vec<Score>, // eval at each position (from White's perspective)
}

// ════════════════════════════════════════════════════════════════════════════
// Analysis runner
// ════════════════════════════════════════════════════════════════════════════

/// Per-position state collected sequentially before parallel search.
struct PreparedPosition {
    board: Board,
    board_after: Board,
    played_mv: Move,
    is_forced: bool,
    move_number: usize,
    side: Color,
    san: String,
    /// Static eval of board_after (white POV) — fallback only; the real
    /// played-eval comes from the next position's depth-N search.
    eval_after: Score,
}

/// Per-thread TT size for parallel analysis. Total memory grows linearly with
/// thread count; 16 MB per worker is enough for high TT hit rates within a
/// single position search and keeps a 16-thread machine under 256 MB total.
const ANALYSIS_TT_MB_PER_THREAD: usize = 16;

/// Analyze a complete game. Searches positions in parallel across worker
/// threads (one `Searcher` per worker, no shared mutable state). The
/// `progress_fn` callback is invoked from the main thread after each
/// completion — the `current` count is monotonic, but the underlying
/// positions may finish out of order.
pub fn analyze_game(
    uci_moves: &[String],
    user_color: Color,
    depth: u32,
    use_nnue: bool,
    progress_fn: &mut dyn FnMut(usize, usize),
) -> GameAnalysis {
    let total = uci_moves.len();
    let mut board = Board::startpos();

    // ── Phase 1: sequential pre-pass — collect everything cheap ────────
    let mut prepared: Vec<Option<PreparedPosition>> = Vec::with_capacity(total);
    for uci_move in uci_moves.iter() {
        let side = board.side_to_move;
        let move_number = prepared.len() / 2 + 1;

        let legal_moves = generate_legal_moves(&board);
        let played_mv = match crate::uci::parse_move(&board, uci_move) {
            Some(m) => m,
            None => {
                prepared.push(None);
                continue;
            }
        };

        let is_forced = legal_moves.len() == 1;
        let san = crate::db::uci_to_san(&board, uci_move);
        let mut board_after = board.clone();
        make_move(&mut board_after, played_mv);
        // Static eval kept only as a FALLBACK — the real played-eval is
        // the depth-N search of the next position (phase 3).
        let eval_after = to_white_perspective(eval::evaluate(&board_after), board_after.side_to_move);

        prepared.push(Some(PreparedPosition {
            board: board.clone(),
            board_after: board_after.clone(),
            played_mv,
            is_forced,
            move_number,
            side,
            san,
            eval_after,
        }));

        board = board_after;
    }

    // ── Phase 2: parallel search ───────────────────────────────────────
    let work_count = prepared.iter().filter(|p| p.is_some()).count();
    let n_workers = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(work_count.max(1));

    let cursor = Arc::new(AtomicUsize::new(0));
    let prepared_arc: Arc<Vec<Option<PreparedPosition>>> = Arc::new(prepared);
    let mut search_results: Vec<Option<SearchResult>> = (0..total).map(|_| None).collect();
    let (tx, rx) = mpsc::channel::<(usize, Option<SearchResult>)>();

    thread::scope(|s| {
        for _ in 0..n_workers {
            let cursor = Arc::clone(&cursor);
            let prepared = Arc::clone(&prepared_arc);
            let tx = tx.clone();
            s.spawn(move || {
                let mut searcher = Searcher::new(ANALYSIS_TT_MB_PER_THREAD);
                searcher.use_nnue = use_nnue;
                searcher.silent = true;
                loop {
                    let i = cursor.fetch_add(1, Ordering::Relaxed);
                    if i >= prepared.len() {
                        break;
                    }
                    let result = prepared[i]
                        .as_ref()
                        .map(|p| searcher.search(&p.board, depth));
                    if tx.send((i, result)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx); // close the original sender so rx terminates when workers exit

        let mut completed = 0usize;
        while let Ok((i, result)) = rx.recv() {
            search_results[i] = result;
            completed += 1;
            progress_fn(completed, total);
        }
    });

    let prepared = Arc::into_inner(prepared_arc)
        .expect("worker threads should have dropped all Arc clones before scope exit");

    // White-POV searched eval of each PRE-move position. For move i the
    // position after the played move is prepared[i+1].board, so
    // searched_white[i+1] doubles as move i's searched played-eval —
    // comparing depth-N best against depth-N played instead of the old
    // depth-N-vs-static mix that labeled engine-best tactical moves as
    // blunders (and mates as cpl-28000 "Blunders").
    let searched_white: Vec<Option<Score>> = prepared
        .iter()
        .enumerate()
        .map(|(i, p)| match (p, &search_results[i]) {
            (Some(p), Some(r)) => Some(to_white_perspective(r.score, p.side)),
            _ => None,
        })
        .collect();

    // The LAST move has no next-position search — run one. Terminal
    // positions (mate/stalemate) get their exact score directly.
    let last_some_idx = prepared.iter().rposition(|p| p.is_some());
    let final_white: Option<Score> = last_some_idx.map(|i| {
        let b = &prepared[i].as_ref().unwrap().board_after;
        let legal = generate_legal_moves(b);
        let stm_score = if legal.is_empty() {
            let ksq = b.piece_bb(b.side_to_move, Piece::King).lsb();
            if attacks::is_square_attacked(b, ksq, b.side_to_move.flip()) {
                -MATE_SCORE
            } else {
                0
            }
        } else {
            let mut searcher = Searcher::new(ANALYSIS_TT_MB_PER_THREAD);
            searcher.use_nnue = use_nnue;
            searcher.silent = true;
            searcher.search(b, depth).score
        };
        to_white_perspective(stm_score, b.side_to_move)
    });

    // ── Phase 3: sequential post-processing ────────────────────────────
    let mut analysis = Vec::with_capacity(total);
    for (i, prep_opt) in prepared.into_iter().enumerate() {
        let prep = match prep_opt {
            Some(p) => p,
            None => continue,
        };
        let search_result = match search_results[i].take() {
            Some(r) => r,
            None => continue,
        };

        let best_eval = to_white_perspective(search_result.score, prep.side);
        let best_move_uci = search_result.best_move.to_uci();

        // Searched played-eval: next position's search, or the extra
        // final-position search for the last move. Static eval only as
        // a last-resort fallback (e.g. unparseable next move).
        let played_eval_white = if Some(i) == last_some_idx {
            final_white
        } else {
            searched_white.get(i + 1).copied().flatten()
        }
        .unwrap_or(prep.eval_after);

        let played_eval_for_side = to_side_perspective(played_eval_white, prep.side);
        let best_eval_for_side = to_side_perspective(best_eval, prep.side);
        let cpl = (best_eval_for_side - played_eval_for_side).max(0);

        // Brilliant gate: a genuine material sacrifice (>= a minor piece) that
        // is still a top move in a non-losing position. See is_sacrifice — the
        // old inline check fired on any pawn-winning capture the opponent had,
        // which made almost every accurate move "Brilliant".
        let was_sacrifice = is_sacrifice(
            &prep.board,
            prep.played_mv,
            &prep.board_after,
            cpl,
            prep.is_forced,
            best_eval_for_side,
        );

        let classification = if prep.is_forced {
            MoveClass::Forced
        } else {
            MoveClass::from_cpl(cpl, was_sacrifice)
        };

        let explanation = if matches!(classification, MoveClass::Best | MoveClass::Forced) {
            None
        } else {
            let bb = eval::eval_components(&prep.board);
            let ba = eval::eval_components(&prep.board_after);
            Some(generate_explanation(
                &prep.board,
                &prep.board_after,
                &bb,
                &ba,
                prep.side,
                &prep.san,
                prep.played_mv,
                &best_move_uci,
                &search_result.pv,
                cpl,
                classification,
            ))
        };

        analysis.push(MoveAnalysis {
            move_number: prep.move_number,
            side: prep.side,
            move_san: prep.san,
            // Searched values — the eval graph and accuracy formula
            // consume these, so they must match what the classification
            // was computed from.
            eval_before: best_eval,
            eval_after: played_eval_white,
            best_move_uci,
            best_eval,
            cpl,
            classification,
            explanation,
        });
    }

    // Eval graph from the same searched values the classifications use.
    let initial_white = searched_white
        .first()
        .copied()
        .flatten()
        .unwrap_or_else(|| eval::evaluate(&Board::startpos()));
    let mut eval_history = Vec::with_capacity(analysis.len() + 1);
    eval_history.push(initial_white);
    for m in &analysis {
        eval_history.push(m.eval_after);
    }

    // Compute accuracy
    let user_moves: Vec<&MoveAnalysis> = analysis
        .iter()
        .filter(|m| m.side == user_color && !matches!(m.classification, MoveClass::Forced))
        .collect();

    // Accuracy: per-move win-percentage loss → lichess sigmoid → mean.
    // Replaces the previous `100 - avg_cpl` linear formula which produced
    // absurdly low numbers (e.g. ~5% accuracy on a normal game) because it
    // treated every centipawn loss as equally costly regardless of position.
    let user_accuracy = if user_moves.is_empty() {
        100.0
    } else {
        let total: f64 = user_moves
            .iter()
            .map(|m| {
                let wp_before_white = win_percentage(m.eval_before);
                let wp_after_white  = win_percentage(m.eval_after);
                // Convert to player POV: black's win % is the complement.
                let (wp_before, wp_after) = match m.side {
                    Color::White => (wp_before_white, wp_after_white),
                    Color::Black => (100.0 - wp_before_white, 100.0 - wp_after_white),
                };
                // Negative wp_loss (the move "improved" win chances per the
                // analysis search) clamps to 0 — i.e. counts as 100% accurate.
                let wp_loss = (wp_before - wp_after).max(0.0);
                accuracy_from_wp_loss(wp_loss)
            })
            .sum();
        total / user_moves.len() as f64
    };

    GameAnalysis {
        moves: analysis,
        user_color,
        user_accuracy,
        eval_history,
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════════

/// Convert a score from side-to-move perspective to White's perspective.
fn to_white_perspective(score: Score, side: Color) -> Score {
    match side {
        Color::White => score,
        Color::Black => -score,
    }
}

/// Convert a score from White's perspective to a specific side's perspective.
fn to_side_perspective(white_score: Score, side: Color) -> Score {
    match side {
        Color::White => white_score,
        Color::Black => -white_score,
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Human-readable explanations
// ════════════════════════════════════════════════════════════════════════════

/// Build a human-readable explanation for a single classified move. Combines:
///   - a tone-appropriate intro for the classification,
///   - a static check for hanging material the opponent can win next move,
///   - the top three signed component swings (always rendered for consistency),
///   - the engine's preferred line as PV-SAN (`Nf3 → Bxe4 Nxe4`).
fn generate_explanation(
    before_board: &Board,
    after_board: &Board,
    before: &EvalBreakdown,
    after: &EvalBreakdown,
    side: Color,
    played_san: &str,
    _played_mv: Move,
    best_uci: &str,
    pv: &[Move],
    cpl: Score,
    class: MoveClass,
) -> String {
    let mut parts = Vec::new();
    parts.push(class_intro(class, played_san, cpl));

    // For mistakes/blunders, surface the concrete punishment the opponent has.
    if matches!(class, MoveClass::Mistake | MoveClass::Blunder)
        && let Some(loss) = describe_material_loss(after_board)
    {
        parts.push(format!("  • {loss}"));
    }

    // Top-3 component swings by magnitude, always rendered for consistency.
    let labels: [(&str, Score, Score); 12] = [
        ("Material", before.material, after.material),
        ("Piece activity", before.pst, after.pst),
        ("Mobility", before.mobility, after.mobility),
        ("Pawn structure", before.pawn_structure, after.pawn_structure),
        ("Passed pawns", before.passed_pawns, after.passed_pawns),
        ("King safety", before.king_safety, after.king_safety),
        ("Bishop pair", before.bishop_pair, after.bishop_pair),
        ("Rook placement", before.rook_placement, after.rook_placement),
        ("Knight outpost", before.knight_outpost, after.knight_outpost),
        ("Connected passers", before.connected_passers, after.connected_passers),
        ("King-pawn proximity", before.king_pawn_proximity, after.king_pawn_proximity),
        ("Tempo", before.tempo, after.tempo),
    ];
    let mut swings: Vec<(&str, Score)> = labels
        .iter()
        .map(|(name, b, a)| (*name, component_swing(*b, *a, side)))
        .filter(|(_, s)| *s != 0)
        .collect();
    swings.sort_by_key(|(_, v)| -v.abs());
    for (name, swing) in swings.iter().take(3) {
        let verb = if *swing < 0 { "worsened" } else { "improved" };
        parts.push(format!("  • {name} {verb} by {} cp.", swing.abs()));
    }

    // Engine line: render as SAN for the first 4 plies of the PV.
    let pv_san = pv_to_san(before_board, pv, 4);
    if !pv_san.is_empty() {
        parts.push(format!("Engine line: {}", pv_san.join(" → ")));
    } else if !best_uci.is_empty() {
        parts.push(format!("Engine preferred: {best_uci}."));
    }

    parts.join("\n")
}

fn class_intro(class: MoveClass, san: &str, cpl: Score) -> String {
    match class {
        MoveClass::Brilliant => format!("{san}!! is a brilliant find."),
        MoveClass::Good => format!("{san} is solid (only {cpl} cp loss)."),
        MoveClass::Inaccuracy => format!("{san} is an inaccuracy ({cpl} cp loss)."),
        MoveClass::Mistake => format!("{san} is a mistake (lost {cpl} cp)."),
        MoveClass::Blunder => format!("{san} is a blunder (lost {cpl} cp)."),
        MoveClass::Best | MoveClass::Forced => String::new(),
    }
}

/// How much a component changed from the moving side's perspective.
/// Negative = got worse for the moving side.
fn component_swing(before: Score, after: Score, side: Color) -> Score {
    let before_s = to_side_perspective(before, side);
    let after_s = to_side_perspective(after, side);
    after_s - before_s
}

fn piece_value(p: Piece) -> Score {
    match p {
        Piece::Pawn => 100,
        Piece::Knight => 320,
        Piece::Bishop => 330,
        Piece::Rook => 500,
        Piece::Queen => 900,
        Piece::King => 20000,
    }
}

fn piece_name(p: Piece) -> &'static str {
    match p {
        Piece::Pawn => "pawn",
        Piece::Knight => "knight",
        Piece::Bishop => "bishop",
        Piece::Rook => "rook",
        Piece::Queen => "queen",
        Piece::King => "king",
    }
}

fn square_name(sq: Square) -> String {
    let file = (b'a' + (sq.0 % 8)) as char;
    let rank = (b'1' + (sq.0 / 8)) as char;
    format!("{file}{rank}")
}

/// Render the first `max_plies` of a PV as SAN by replaying on a clone of `start`.
fn pv_to_san(start: &Board, pv: &[Move], max_plies: usize) -> Vec<String> {
    let mut board = start.clone();
    let mut out = Vec::with_capacity(max_plies);
    for &mv in pv.iter().take(max_plies) {
        if mv.is_null() {
            break;
        }
        let san = crate::db::uci_to_san(&board, &mv.to_uci());
        out.push(san);
        make_move(&mut board, mv);
    }
    out
}

/// Static "hanging piece" detector: look at the strongest opponent capture
/// available on `after_board`. If a clearly winning capture exists (victim
/// undefended, or victim worth more than attacker), describe it in plain
/// English. Returns `None` if no obvious material punishment is available.
fn describe_material_loss(after_board: &Board) -> Option<String> {
    let opponent = after_board.side_to_move;
    let our_side = opponent.flip();
    let moves = generate_legal_moves(after_board);

    let mut best: Option<(Score, Move, Piece, Piece, Square)> = None;
    for i in 0..moves.len() {
        let mv = moves[i];
        let from = mv.from_sq();
        let to = mv.to_sq();

        let attacker = match after_board.piece_type_on(from) {
            Some(p) => p,
            None => continue,
        };
        let victim = if matches!(mv.flag(), MoveFlag::EnPassant) {
            Piece::Pawn
        } else {
            match after_board.piece_on(to) {
                Some((c, p)) if c == our_side => p,
                _ => continue,
            }
        };

        let mut after_capture = after_board.clone();
        make_move(&mut after_capture, mv);
        let defended =
            attacks::is_square_attacked(&after_capture, to, after_capture.side_to_move);
        let attacker_val = piece_value(attacker);
        let victim_val = piece_value(victim);
        let net = if defended {
            victim_val - attacker_val
        } else {
            victim_val
        };

        if net > 50
            && best
                .as_ref()
                .map_or(true, |(prev_net, ..)| net > *prev_net)
        {
            best = Some((net, mv, attacker, victim, to));
        }
    }

    let (_net, mv, _attacker, victim, sq) = best?;
    let mv_san = crate::db::uci_to_san(after_board, &mv.to_uci());
    Some(format!(
        "Your {} on {} is hanging — {} wins it.",
        piece_name(victim),
        square_name(sq),
        mv_san,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_ranges() {
        // Industry-standard bands (close to chess.com/lichess).
        assert_eq!(MoveClass::from_cpl(0, false), MoveClass::Best);
        assert_eq!(MoveClass::from_cpl(15, false), MoveClass::Best);
        assert_eq!(MoveClass::from_cpl(30, false), MoveClass::Good);
        assert_eq!(MoveClass::from_cpl(50, false), MoveClass::Good);
        assert_eq!(MoveClass::from_cpl(80, false), MoveClass::Inaccuracy);
        assert_eq!(MoveClass::from_cpl(120, false), MoveClass::Inaccuracy);
        assert_eq!(MoveClass::from_cpl(180, false), MoveClass::Mistake);
        assert_eq!(MoveClass::from_cpl(250, false), MoveClass::Mistake);
        assert_eq!(MoveClass::from_cpl(400, false), MoveClass::Blunder);
    }

    #[test]
    fn brilliant_sacrifice_detection() {
        // Low CPL + sacrifice = brilliant
        assert_eq!(MoveClass::from_cpl(5, true), MoveClass::Brilliant);
        assert_eq!(MoveClass::from_cpl(20, true), MoveClass::Brilliant);
        // Above the sacrifice threshold, the cpl falls through to the normal
        // ranking. 200 cp is a Mistake (121..=250); 400 cp is a Blunder.
        assert_eq!(MoveClass::from_cpl(200, true), MoveClass::Mistake);
        assert_eq!(MoveClass::from_cpl(400, true), MoveClass::Blunder);
    }

    fn mv(board: &Board, uci: &str) -> Move {
        let from = Square::from_algebraic(&uci[0..2]).unwrap();
        let to = Square::from_algebraic(&uci[2..4]).unwrap();
        let moves = generate_legal_moves(board);
        (0..moves.len())
            .map(|i| moves[i])
            .find(|m| m.from_sq() == from && m.to_sq() == to)
            .unwrap_or_else(|| panic!("no legal move {uci}"))
    }

    #[test]
    fn sacrifice_requires_real_material_not_a_pawn() {
        crate::attacks::init();

        // (A) The regression: a quiet, accurate move after which the opponent
        // can only win a PAWN must NOT count as a sacrifice. White pushes
        // e3-e4; Black can take the undefended e4 pawn (SEE 100). Under the old
        // `>= 100` rule this was flagged a sacrifice -> Brilliant. It must not.
        let before = Board::from_fen("4k3/8/8/3p4/8/4P3/8/4K3 w - - 0 1").unwrap();
        let push = mv(&before, "e3e4");
        let mut after = before.clone();
        make_move(&mut after, push);
        assert!(
            !is_sacrifice(&before, push, &after, 10, false, 0),
            "winning only a pawn is not a sacrifice"
        );

        // (B) A real piece sacrifice: white plays Nf2-e4, leaving the knight en
        // prise to ...dxe4 (SEE 320 >= minor). Top move, equal-ish -> sacrifice.
        let before = Board::from_fen("4k3/8/8/3p4/8/8/5N2/4K3 w - - 0 1").unwrap();
        let nf2e4 = mv(&before, "f2e4");
        let mut after = before.clone();
        make_move(&mut after, nf2e4);
        assert!(
            is_sacrifice(&before, nf2e4, &after, 10, false, 50),
            "leaving a minor piece en prise while still best is a sacrifice"
        );

        // (C) Same hanging-piece position, but the player is already lost: not
        // brilliant (a desperate sac doesn't earn it).
        assert!(
            !is_sacrifice(&before, nf2e4, &after, 10, false, -200),
            "a sacrifice in a losing position is not brilliant"
        );

        // (D) Forced moves are never brilliant.
        assert!(
            !is_sacrifice(&before, nf2e4, &after, 10, true, 50),
            "forced moves are not brilliant"
        );
    }

    #[test]
    fn win_percentage_known_values() {
        // Sigmoid maps cp → win %.
        assert!((win_percentage(0) - 50.0).abs() < 0.01, "0 cp = 50% win");
        assert!(win_percentage(2000) > 99.0, "2000 cp = >99% win");
        assert!(win_percentage(-2000) < 1.0, "-2000 cp = <1% win");
        // 100 cp (one pawn) corresponds to ~60% win.
        let p = win_percentage(100);
        assert!(p > 55.0 && p < 65.0, "100 cp win % should be ~60%, got {p}");
    }

    #[test]
    fn accuracy_from_wp_loss_known_values() {
        // 0% loss → ~100% accurate (formula peaks at ~100).
        assert!(accuracy_from_wp_loss(0.0) > 99.0);
        // ~50% loss → very low accuracy (~8.5%).
        let a = accuracy_from_wp_loss(50.0);
        assert!(a > 5.0 && a < 15.0, "50% wp loss should give ~8.5% accuracy, got {a}");
        // Catastrophic losses clamp to 0.
        assert!(accuracy_from_wp_loss(100.0) < 5.0);
    }

    #[test]
    fn analyze_short_game() {
        crate::attacks::init();
        // Scholar's mate: 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6?? 4. Qxf7#
        let moves = vec![
            "e2e4".into(), "e7e5".into(),
            "f1c4".into(), "b8c6".into(),
            "d1h5".into(), "g8f6".into(),
            "h5f7".into(),
        ];

        let mut progress = Vec::new();
        // Pin use_nnue=false so this test isn't sensitive to whether another
        // parallel test loaded the NNUE net first via the global OnceLock.
        let result = analyze_game(&moves, Color::Black, 10, false, &mut |cur, tot| {
            progress.push((cur, tot));
        });

        assert_eq!(result.moves.len(), 7);
        assert_eq!(progress.len(), 7);
        assert_eq!(progress.last(), Some(&(7, 7)));
        assert_eq!(result.eval_history.len(), 8); // 7 moves + initial position

        // Nf6 should be classified as at least an inaccuracy (allows Qxf7#)
        let nf6 = &result.moves[5]; // index 5 = move 6 (0-based)
        assert!(
            matches!(nf6.classification, MoveClass::Inaccuracy | MoveClass::Mistake | MoveClass::Blunder),
            "Nf6 should be bad, got {:?} (cpl={})",
            nf6.classification, nf6.cpl
        );

        // Regression: the mate-delivering move must NOT be a blunder.
        // The old depth-N-vs-static CPL labeled checkmating moves as
        // Blunder with cpl ~28000 because the static eval of the mated
        // position knew nothing about mate. With searched played-evals
        // the cpl is ~0.
        let qxf7 = result.moves.last().unwrap();
        assert!(
            matches!(qxf7.classification, MoveClass::Best | MoveClass::Good | MoveClass::Brilliant),
            "mate-delivering move must be Best/Good, got {:?} (cpl={})",
            qxf7.classification, qxf7.cpl
        );
        assert!(qxf7.cpl <= 50, "mate move cpl should be ~0, got {}", qxf7.cpl);

        // The eval graph's final point must reflect the mate, not a
        // static material count of the final position.
        let last_eval = *result.eval_history.last().unwrap();
        assert!(
            last_eval > MATE_SCORE - 1000,
            "final eval-history entry should be mate-magnitude for white, got {last_eval}"
        );
    }
}
