use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;

use crate::attacks;
use crate::board::Board;
use crate::eval::{self, EvalBreakdown, Score};
use crate::movegen::{generate_legal_moves, make_move};
use crate::moves::{Move, MoveFlag};
use crate::search::{SearchResult, Searcher};
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

    pub fn from_cpl(cpl: Score, was_sacrifice: bool) -> Self {
        if was_sacrifice && cpl <= 20 {
            return MoveClass::Brilliant;
        }
        match cpl {
            0 => MoveClass::Best,
            1..=20 => MoveClass::Good,
            21..=50 => MoveClass::Inaccuracy,
            51..=150 => MoveClass::Mistake,
            _ => MoveClass::Blunder,
        }
    }
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
    eval_before: Score,
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
    let mut eval_history = Vec::with_capacity(total + 1);
    eval_history.push(eval::evaluate(&board));

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
        let eval_before = to_white_perspective(eval::evaluate(&board), side);
        let san = crate::db::uci_to_san(&board, uci_move);
        let mut board_after = board.clone();
        make_move(&mut board_after, played_mv);
        let eval_after = to_white_perspective(eval::evaluate(&board_after), board_after.side_to_move);
        eval_history.push(eval_after);

        prepared.push(Some(PreparedPosition {
            board: board.clone(),
            board_after: board_after.clone(),
            played_mv,
            is_forced,
            move_number,
            side,
            san,
            eval_before,
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

        let played_eval_for_side = to_side_perspective(prep.eval_after, prep.side);
        let best_eval_for_side = to_side_perspective(best_eval, prep.side);
        let cpl = (best_eval_for_side - played_eval_for_side).max(0);

        let mat_before = material_count(&prep.board, prep.side);
        let mat_after = material_count(&prep.board_after, prep.side);
        let was_sacrifice = mat_after < mat_before - 100 && cpl <= 20;

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
            eval_before: prep.eval_before,
            eval_after: prep.eval_after,
            best_move_uci,
            best_eval,
            cpl,
            classification,
            explanation,
        });
    }

    // Compute accuracy
    let user_moves: Vec<&MoveAnalysis> = analysis
        .iter()
        .filter(|m| m.side == user_color && !matches!(m.classification, MoveClass::Forced))
        .collect();

    let user_accuracy = if user_moves.is_empty() {
        100.0
    } else {
        let avg_cpl: f64 =
            user_moves.iter().map(|m| m.cpl as f64).sum::<f64>() / user_moves.len() as f64;
        (100.0 - avg_cpl).max(0.0).min(100.0)
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

/// Simple material count for one side.
fn material_count(board: &Board, color: Color) -> Score {
    let mut total = 0;
    total += board.piece_bb(color, Piece::Pawn).popcount() as Score * 100;
    total += board.piece_bb(color, Piece::Knight).popcount() as Score * 320;
    total += board.piece_bb(color, Piece::Bishop).popcount() as Score * 330;
    total += board.piece_bb(color, Piece::Rook).popcount() as Score * 500;
    total += board.piece_bb(color, Piece::Queen).popcount() as Score * 900;
    total
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
        assert_eq!(MoveClass::from_cpl(0, false), MoveClass::Best);
        assert_eq!(MoveClass::from_cpl(10, false), MoveClass::Good);
        assert_eq!(MoveClass::from_cpl(35, false), MoveClass::Inaccuracy);
        assert_eq!(MoveClass::from_cpl(100, false), MoveClass::Mistake);
        assert_eq!(MoveClass::from_cpl(200, false), MoveClass::Blunder);
    }

    #[test]
    fn brilliant_sacrifice_detection() {
        // Low CPL + sacrifice = brilliant
        assert_eq!(MoveClass::from_cpl(5, true), MoveClass::Brilliant);
        // High CPL + sacrifice = still a blunder
        assert_eq!(MoveClass::from_cpl(200, true), MoveClass::Blunder);
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
    }
}
