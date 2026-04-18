use crate::board::Board;
use crate::eval::{self, EvalBreakdown, Score};
use crate::movegen::{generate_legal_moves, make_move};
use crate::search::Searcher;
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

/// Analyze a complete game. Calls `progress_fn` with (current_move, total_moves)
/// after each position is analyzed.
pub fn analyze_game(
    uci_moves: &[String],
    user_color: Color,
    depth: u32,
    progress_fn: &mut dyn FnMut(usize, usize),
) -> GameAnalysis {
    let total = uci_moves.len();
    let mut searcher = Searcher::new(64); // reuse TT across all positions
    searcher.use_nnue = crate::nnue::network::get_network().is_some();
    let mut board = Board::startpos();
    let mut analysis = Vec::with_capacity(total);
    let mut eval_history = Vec::with_capacity(total + 1);

    // Initial position eval
    let init_eval = eval::evaluate(&board);
    eval_history.push(init_eval);

    for (i, uci_move) in uci_moves.iter().enumerate() {
        let side = board.side_to_move;
        let move_number = i / 2 + 1;

        let legal_moves = generate_legal_moves(&board);

        // Parse the played move
        let played_mv = match crate::uci::parse_move(&board, uci_move) {
            Some(m) => m,
            None => {
                // Can't parse move — skip
                progress_fn(i + 1, total);
                continue;
            }
        };

        // Check if forced (only one legal move)
        let is_forced = legal_moves.len() == 1;

        // Eval before this move (from White's perspective)
        let eval_before = to_white_perspective(eval::evaluate(&board), side);

        // Find engine's best move and eval
        let search_result = searcher.search(&board, depth);
        let best_eval = to_white_perspective(search_result.score, side);
        let best_move_uci = search_result.best_move.to_uci();

        // Apply the played move and get eval after
        let san = crate::db::uci_to_san(&board, uci_move);
        let mut board_after = board.clone();
        make_move(&mut board_after, played_mv);
        let eval_after = to_white_perspective(eval::evaluate(&board_after), board_after.side_to_move);

        eval_history.push(eval_after);

        // Centipawn loss: how much worse was the played move vs the best?
        // From the moving side's perspective
        let played_eval_for_side = to_side_perspective(eval_after, side);
        let best_eval_for_side = to_side_perspective(best_eval, side);
        let cpl = (best_eval_for_side - played_eval_for_side).max(0);

        // Detect sacrifice: material went down but eval stayed stable
        let mat_before = material_count(&board, side);
        let mat_after = material_count(&board_after, side);
        let was_sacrifice = mat_after < mat_before - 100 && cpl <= 20;

        let classification = if is_forced {
            MoveClass::Forced
        } else {
            MoveClass::from_cpl(cpl, was_sacrifice)
        };

        // Compute explanation for mistakes/blunders using eval component breakdown
        let explanation =
            if matches!(classification, MoveClass::Mistake | MoveClass::Blunder) {
                let bb = eval::eval_components(&board);
                let ba = eval::eval_components(&board_after);
                Some(generate_explanation(&bb, &ba, side, &san, &best_move_uci, cpl))
            } else {
                None
            };

        analysis.push(MoveAnalysis {
            move_number,
            side,
            move_san: san,
            eval_before,
            eval_after,
            best_move_uci,
            best_eval,
            cpl,
            classification,
            explanation,
        });

        // Advance the board
        board = board_after;
        progress_fn(i + 1, total);
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

fn generate_explanation(
    before: &EvalBreakdown,
    after: &EvalBreakdown,
    side: Color,
    played_san: &str,
    best_uci: &str,
    cpl: Score,
) -> String {
    let mut parts = Vec::new();

    let class = if cpl > 150 { "Blunder" } else { "Mistake" };
    parts.push(format!(
        "{played_san} was a {class} (lost {} centipawns).",
        cpl
    ));

    // Find the biggest component swings
    let mut swings: Vec<(&str, Score)> = vec![
        ("Material", component_swing(before.material, after.material, side)),
        ("Piece activity", component_swing(before.pst, after.pst, side)),
        ("Mobility", component_swing(before.mobility, after.mobility, side)),
        ("Pawn structure", component_swing(before.pawn_structure, after.pawn_structure, side)),
        ("Passed pawns", component_swing(before.passed_pawns, after.passed_pawns, side)),
        ("King safety", component_swing(before.king_safety, after.king_safety, side)),
        ("Bishop pair", component_swing(before.bishop_pair, after.bishop_pair, side)),
        ("Rook placement", component_swing(before.rook_placement, after.rook_placement, side)),
        ("Knight outpost", component_swing(before.knight_outpost, after.knight_outpost, side)),
        ("Connected passers", component_swing(before.connected_passers, after.connected_passers, side)),
        ("King-pawn proximity", component_swing(before.king_pawn_proximity, after.king_pawn_proximity, side)),
        ("Tempo", component_swing(before.tempo, after.tempo, side)),
    ];

    // Sort by magnitude of loss (most negative first)
    swings.sort_by_key(|(_, v)| *v);

    for (name, swing) in &swings {
        if *swing < -15 {
            parts.push(format!("  • {name} worsened by {} cp.", -swing));
        }
    }

    // Mention what the best move was
    if !best_uci.is_empty() {
        parts.push(format!("Engine preferred: {best_uci}."));
    }

    parts.join("\n")
}

/// How much a component changed from the moving side's perspective.
/// Negative = got worse for the moving side.
fn component_swing(before: Score, after: Score, side: Color) -> Score {
    // Components are stored from White's perspective
    let before_s = to_side_perspective(before, side);
    let after_s = to_side_perspective(after, side);
    after_s - before_s
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
        let result = analyze_game(&moves, Color::Black, 10, &mut |cur, tot| {
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
