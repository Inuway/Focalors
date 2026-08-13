//! Small curated opening book for local games.
//!
//! The GUI consults this before starting an engine search: in a known
//! opening position the coach replies instantly with a weighted-random
//! book move instead of burning clock on well-known theory (and instead
//! of deterministically playing the same first move every game).
//!
//! Scope is deliberately narrow: the engine (search/eval/UCI mode) never
//! sees the book, and selfmatch/selfplay use their own opening
//! randomization, so strength measurements stay book-free. Every lookup
//! is validated against legal move generation, and any miss or anomaly
//! simply falls through to a normal search. The worst case is exactly
//! the pre-book behavior.
//!
//! Data lives in `book_data.txt`, one position per line:
//!
//!   <history in UCI from the start position>|<reply>:<weight> ...
//!
//! Positions are keyed by Zobrist hash, so games that transpose into a
//! listed position also hit. The loader rejects illegal moves, malformed
//! rows, and two rows reaching the same position; `cargo test` fails on
//! any data error.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::board::Board;
use crate::movegen::make_move;
use crate::moves::Move;

static BOOK_DATA: &str = include_str!("book_data.txt");

struct Book {
    /// Zobrist hash of a position -> weighted UCI replies.
    positions: HashMap<u64, Vec<(String, u32)>>,
}

static BOOK: OnceLock<Book> = OnceLock::new();

fn book() -> &'static Book {
    BOOK.get_or_init(|| build_book(BOOK_DATA).0)
}

/// Parse the book text into a position table. Data errors are collected
/// (and the offending row skipped) rather than panicking, so a bad edit
/// can never take the app down; the unit tests assert the list is empty.
fn build_book(data: &str) -> (Book, Vec<String>) {
    let mut positions: HashMap<u64, Vec<(String, u32)>> = HashMap::new();
    let mut errors = Vec::new();

    'row: for (idx, raw) in data.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let lineno = idx + 1;
        let Some((history, replies)) = line.split_once('|') else {
            errors.push(format!("line {lineno}: missing '|' separator"));
            continue;
        };

        // Replay the history from the start position to find the row's
        // position; parse_move only accepts legal moves.
        let mut board = Board::startpos();
        for uci in history.split_whitespace() {
            let Some(mv) = crate::uci::parse_move(&board, uci) else {
                errors.push(format!("line {lineno}: illegal history move '{uci}'"));
                continue 'row;
            };
            make_move(&mut board, mv);
        }

        let mut parsed: Vec<(String, u32)> = Vec::new();
        for token in replies.split_whitespace() {
            let Some((uci, weight_str)) = token.split_once(':') else {
                errors.push(format!("line {lineno}: malformed reply '{token}'"));
                continue 'row;
            };
            let Ok(weight) = weight_str.parse::<u32>() else {
                errors.push(format!("line {lineno}: bad weight in '{token}'"));
                continue 'row;
            };
            if weight == 0 {
                errors.push(format!("line {lineno}: zero weight in '{token}'"));
                continue 'row;
            }
            if crate::uci::parse_move(&board, uci).is_none() {
                errors.push(format!("line {lineno}: illegal reply '{uci}'"));
                continue 'row;
            }
            parsed.push((uci.to_string(), weight));
        }
        if parsed.is_empty() {
            errors.push(format!("line {lineno}: no replies"));
            continue;
        }
        if positions.insert(board.hash, parsed).is_some() {
            errors.push(format!(
                "line {lineno}: duplicate position (another row transposes here)"
            ));
        }
    }

    (Book { positions }, errors)
}

/// How many plies of book "memory" each difficulty level gets, like a human:
/// a beginner knows a first move, a club player a couple of moves of theory,
/// a master the whole (small) book. Counted in total game plies, both sides,
/// so the engine leaves book at the same point of the game regardless of
/// color. Beyond the budget the engine is on its own, exactly as pre-book.
fn max_book_plies(level: u32) -> u32 {
    match level {
        0..=4 => 2,    // Beginner band: knows a first move
        5..=9 => 4,    // Club band: two moves of theory
        10..=15 => 8,  // Tournament band: four moves
        _ => u32::MAX, // Master band and Custom: the full book
    }
}

/// Weighted-random book reply for this position, or `None` when off book or
/// past the level's book-ply budget. The returned move is re-validated
/// against the live board's legal moves (via `parse_move`), so a stale or
/// corrupt entry degrades to `None`.
pub fn pick_book_move(board: &Board, level: u32) -> Option<Move> {
    let ply = (u32::from(board.fullmove_number).saturating_sub(1)) * 2
        + u32::from(board.side_to_move == crate::types::Color::Black);
    if ply >= max_book_plies(level) {
        return None;
    }
    let entries = book().positions.get(&board.hash)?;
    let total: u64 = entries.iter().map(|(_, w)| u64::from(*w)).sum();
    if total == 0 {
        return None;
    }
    let mut roll = time_entropy() % total;
    for (uci, weight) in entries {
        let w = u64::from(*weight);
        if roll < w {
            return crate::uci::parse_move(board, uci);
        }
        roll -= w;
    }
    None
}

/// True when `uci` is one of the book's listed replies in the position with
/// this Zobrist hash. Game analysis uses this to label opening theory as
/// "Book"; it always consults the full book (theory is theory in review,
/// whatever difficulty the game was played at), unlike `pick_book_move`,
/// which budgets book depth by level.
pub fn is_book_reply(position_hash: u64, uci: &str) -> bool {
    book()
        .positions
        .get(&position_hash)
        .is_some_and(|entries| entries.iter().any(|(reply, _)| reply == uci))
}

/// Randomized "thinking" pause (in ms) for an instant book reply, so the
/// coach appears to glance at the board instead of teleporting a piece.
pub fn humanized_delay_ms() -> u64 {
    500 + time_entropy() % 701
}

/// One random draw per engine move: splitmix64 over the wall clock.
/// No external RNG dependency, and quality is ample for picking among
/// a handful of weighted opening moves.
fn time_entropy() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::from(d.subsec_nanos()) ^ d.as_secs())
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::generate_legal_moves;

    #[test]
    fn book_data_has_no_errors() {
        crate::attacks::init();
        let (book, errors) = build_book(BOOK_DATA);
        assert!(
            errors.is_empty(),
            "book data errors:\n{}",
            errors.join("\n")
        );
        assert!(
            book.positions.len() >= 60,
            "suspiciously small book: {} positions",
            book.positions.len()
        );
    }

    #[test]
    fn startpos_always_yields_a_legal_book_move() {
        crate::attacks::init();
        let board = Board::startpos();
        let legal = generate_legal_moves(&board);
        for _ in 0..50 {
            let mv = pick_book_move(&board, 20).expect("startpos must be in book");
            let found = (0..legal.len()).any(|i| legal[i] == mv);
            assert!(found, "book returned illegal move {}", mv.to_uci());
        }
    }

    #[test]
    fn book_covers_main_replies_and_transpositions() {
        crate::attacks::init();
        // 1.e4 c5 is a book position for White's second move.
        let mut board = Board::startpos();
        for uci in ["e2e4", "c7c5"] {
            let mv = crate::uci::parse_move(&board, uci).unwrap();
            make_move(&mut board, mv);
        }
        assert!(pick_book_move(&board, 20).is_some(), "Sicilian should be in book");

        // Transposition: 1.d4 e6 2.e4 reaches the French (1.e4 e6 2.d4)
        // by a move order the book never lists explicitly.
        let mut board = Board::startpos();
        for uci in ["d2d4", "e7e6", "e2e4"] {
            let mv = crate::uci::parse_move(&board, uci).unwrap();
            make_move(&mut board, mv);
        }
        assert!(
            pick_book_move(&board, 20).is_some(),
            "transposed French position should hit the book"
        );
    }

    #[test]
    fn book_depth_scales_with_level() {
        crate::attacks::init();

        // Ply 0 (startpos): every level may use its "known" first move.
        let board = Board::startpos();
        assert!(pick_book_move(&board, 1).is_some(), "beginner knows a first move");

        // Ply 2 (after 1.e4 e5): beginner budget (2 plies) is spent,
        // club (4) and master still have theory.
        let mut board = Board::startpos();
        for uci in ["e2e4", "e7e5"] {
            let mv = crate::uci::parse_move(&board, uci).unwrap();
            make_move(&mut board, mv);
        }
        assert!(pick_book_move(&board, 1).is_none(), "beginner is out of book");
        assert!(pick_book_move(&board, 5).is_some(), "club still in book");
        assert!(pick_book_move(&board, 20).is_some(), "master still in book");

        // Ply 4 (after 1.e4 e5 2.Nf3 Nc6): club budget (4) is spent,
        // tournament (8) and master still have theory.
        for uci in ["g1f3", "b8c6"] {
            let mv = crate::uci::parse_move(&board, uci).unwrap();
            make_move(&mut board, mv);
        }
        assert!(pick_book_move(&board, 5).is_none(), "club is out of book");
        assert!(pick_book_move(&board, 12).is_some(), "tournament still in book");
        assert!(pick_book_move(&board, 20).is_some(), "master still in book");
    }

    #[test]
    fn is_book_reply_matches_listed_replies_only() {
        crate::attacks::init();
        let board = Board::startpos();
        assert!(is_book_reply(board.hash, "e2e4"));
        assert!(!is_book_reply(board.hash, "a2a3")); // legal but not book
        assert!(!is_book_reply(0xDEAD_BEEF, "e2e4")); // unknown position
    }

    #[test]
    fn humanized_delay_stays_in_range() {
        for _ in 0..100 {
            let ms = humanized_delay_ms();
            assert!((500..=1200).contains(&ms), "delay {ms}ms out of range");
        }
    }

    #[test]
    fn off_book_position_returns_none() {
        crate::attacks::init();
        let mut board = Board::startpos();
        for uci in ["a2a3", "a7a6"] {
            let mv = crate::uci::parse_move(&board, uci).unwrap();
            make_move(&mut board, mv);
        }
        assert!(pick_book_move(&board, 20).is_none());
    }
}
