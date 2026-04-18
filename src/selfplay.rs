//! Self-play data generator for NNUE training.
//!
//! Plays Focalors vs itself using HCE, generating positions with evaluations
//! and game results in bullet's ChessBoard format (32 bytes per record).
//!
//! Usage: `cargo run --release -- selfplay <num_games> <output_file>`

use crate::attacks;
use crate::board::Board;
use crate::eval::Score;
use crate::movegen::{generate_legal_moves, make_move};
use crate::search::Searcher;
use crate::types::*;

/// bullet ChessBoard format: 32 bytes per position, STM-relative.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ChessBoard {
    occ: u64,         // occupancy bitboard
    pcs: [u8; 16],    // piece data, 2 per byte (4-bit nibbles)
    score: i16,       // STM-relative eval in centipawns
    result: u8,       // 0=STM loss, 1=draw, 2=STM win
    ksq: u8,          // STM king square
    opp_ksq: u8,      // opponent king square XOR 56
    extra: [u8; 3],   // reserved
}

const _: () = assert!(std::mem::size_of::<ChessBoard>() == 32);

/// Piece type encoding for bullet nibbles: 0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King
fn piece_to_nibble(piece: Piece) -> u8 {
    match piece {
        Piece::Pawn => 0,
        Piece::Knight => 1,
        Piece::Bishop => 2,
        Piece::Rook => 3,
        Piece::Queen => 4,
        Piece::King => 5,
    }
}

/// Convert a Board position to a ChessBoard record.
/// The board is stored STM-relative: if Black to move, flip everything.
fn board_to_record(board: &Board, eval: Score, result: u8) -> ChessBoard {
    let stm = board.side_to_move;
    let flip = stm == Color::Black;

    // Build occupancy and piece nibbles
    let mut pcs = [0u8; 16];
    let mut piece_idx = 0usize;

    // We need to iterate squares in LSB order of the occupancy bitboard.
    // First build the occupancy, then iterate it.
    let all_occ = board.all_occupied();

    // If flip, we need to vertically mirror the bitboard
    let occ = if flip {
        all_occ.0.swap_bytes() // swap_bytes on u64 reverses byte order = vertical flip
    } else {
        all_occ.0
    };

    // Now iterate the occupancy in LSB order and fill piece nibbles
    let mut remaining = Bitboard(occ);
    while remaining.is_not_empty() {
        let sq = remaining.pop_lsb(); // square in the (possibly flipped) board

        // Map back to the original board's square
        let orig_sq = if flip {
            Square(sq.0 ^ 56) // un-flip to get original square
        } else {
            sq
        };

        let (color, piece) = board.piece_on(orig_sq).expect("Occupied square has no piece");

        // Determine if this is STM or non-STM
        let is_stm = color == stm;
        let nibble = piece_to_nibble(piece) | if is_stm { 0 } else { 0x08 };

        // Pack into pcs array: piece_idx/2 byte, piece_idx%2 nibble position
        let byte_idx = piece_idx / 2;
        let shift = 4 * (piece_idx & 1);
        pcs[byte_idx] |= nibble << shift;
        piece_idx += 1;
    }

    // STM-relative score. The search already returns scores from STM's
    // perspective (negamax negates going up), so we store it directly.
    // (Previously this negated for black-to-move, producing white-relative
    // scores — which poisoned every trained net with wrong-sign targets
    // for half the data. That bug caused bizarre play like queen sacrifices
    // as black, even with low training loss.)
    let score = eval;

    // STM-relative result (flip if black was STM)
    let stm_result = if flip { 2 - result } else { result };

    // King squares
    let stm_king = board.piece_bb(stm, Piece::King).lsb();
    let opp_king = board.piece_bb(stm.flip(), Piece::King).lsb();

    let ksq = if flip { stm_king.0 ^ 56 } else { stm_king.0 };
    let opp_ksq = if flip { opp_king.0 ^ 56 } else { opp_king.0 } ^ 56; // opponent always XOR 56

    ChessBoard {
        occ,
        pcs,
        score: score as i16,
        result: stm_result,
        ksq,
        opp_ksq,
        extra: [0; 3],
    }
}

/// Game result: 2=white win, 1=draw, 0=black win (white-relative).
#[derive(Clone, Copy)]
enum GameResult {
    WhiteWin,
    Draw,
    BlackWin,
}

impl GameResult {
    fn to_white_relative(self) -> u8 {
        match self {
            GameResult::WhiteWin => 2,
            GameResult::Draw => 1,
            GameResult::BlackWin => 0,
        }
    }
}

/// A tiny xorshift PRNG — no external deps, deterministic per seed.
struct ThreadRng {
    state: u64,
}

impl ThreadRng {
    fn new(seed: u64) -> Self {
        // Ensure non-zero state
        let state = if seed == 0 { 0xDEAD_BEEF } else { seed };
        ThreadRng { state }
    }
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
    fn next_index(&mut self, len: usize) -> usize {
        (self.next_u64() as usize) % len
    }
}

/// Play one self-play game with random opening.
///
/// Plays `random_plies` uniformly-random legal moves first (no search),
/// then verifies the position is still reasonably balanced (|eval| < 200cp).
/// If unbalanced, returns `None` to signal the game should be discarded.
///
/// After the random opening, plays normal fixed-depth search to completion,
/// recording every non-noisy position (not in check, prior move not a capture,
/// |eval| < 3000).
fn play_game(
    search_depth: u32,
    max_moves: usize,
    random_plies: u32,
    balance_threshold: Score,
    use_nnue: bool,
    rng: &mut ThreadRng,
) -> Option<(Vec<(Board, Score)>, GameResult)> {
    let mut board = Board::startpos();
    let mut searcher = Searcher::new(4); // small TT for self-play speed
    searcher.silent = true;
    searcher.use_nnue = use_nnue;

    // ── Random opening ────────────────────────────────────────────────
    // Pick `random_plies` uniformly-random legal moves. If a side runs
    // out of moves (stalemate/checkmate) during the random phase, discard.
    for _ in 0..random_plies {
        let moves = generate_legal_moves(&board);
        if moves.is_empty() {
            return None; // game ended during random opening
        }
        if board.halfmove_clock >= 100 || board.is_insufficient_material() {
            return None;
        }
        let idx = rng.next_index(moves.len());
        make_move(&mut board, moves[idx]);
    }

    // ── Balance check ─────────────────────────────────────────────────
    // After random moves, reject games that started in an already-decided position.
    // Use a shallow search for speed; this is a filter, not a training signal.
    let balance_eval = searcher.search(&board, 4).score;
    if balance_eval.abs() > balance_threshold {
        return None;
    }

    // ── Main game loop ────────────────────────────────────────────────
    let mut positions: Vec<(Board, Score)> = Vec::new();
    let mut move_count = random_plies as usize;
    let mut prev_was_capture = false;

    loop {
        let moves = generate_legal_moves(&board);

        let king_sq = board.piece_bb(board.side_to_move, Piece::King).lsb();
        let in_check = crate::attacks::is_square_attacked(
            &board, king_sq, board.side_to_move.flip(),
        );

        // Game-ending conditions
        if moves.is_empty() {
            let result = if in_check {
                match board.side_to_move {
                    Color::White => GameResult::BlackWin,
                    Color::Black => GameResult::WhiteWin,
                }
            } else {
                GameResult::Draw
            };
            return Some((positions, result));
        }

        if board.halfmove_clock >= 100 || board.is_insufficient_material() {
            return Some((positions, GameResult::Draw));
        }

        if move_count >= max_moves {
            return Some((positions, GameResult::Draw));
        }

        // Engine evaluation at full depth
        let result = searcher.search(&board, search_depth);
        let eval = result.score;

        // Filter noisy positions:
        // - extreme evals (decided positions)
        // - STM in check (forced moves)
        // - prior move was a capture (material swing noise)
        let usable = eval.abs() < 3000 && !in_check && !prev_was_capture;
        if usable {
            positions.push((board.clone(), eval));
        }

        if result.best_move.is_null() {
            return Some((positions, GameResult::Draw));
        }

        prev_was_capture = is_capture(&board, result.best_move);

        make_move(&mut board, result.best_move);
        move_count += 1;
    }
}

/// Returns true if the move captures a piece (including en passant).
fn is_capture(board: &Board, mv: crate::moves::Move) -> bool {
    use crate::moves::MoveFlag;
    if matches!(mv.flag(), MoveFlag::EnPassant) {
        return true;
    }
    let to = mv.to_sq();
    let them = board.side_to_move.flip();
    matches!(board.piece_on(to), Some((c, _)) if c == them)
}

pub fn run_selfplay(
    num_games: usize,
    output_path: &str,
    nnue_path: Option<&str>,
    search_depth: u32,
    num_threads: usize,
    random_plies: u32,
) {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::Mutex;

    attacks::init();

    // Optionally load NNUE net for stronger self-play (gen-2+)
    let use_nnue = if let Some(path) = nnue_path {
        match crate::nnue::init(Some(path)) {
            Ok(()) => {
                eprintln!("Loaded NNUE net from {path}");
                true
            }
            Err(e) => {
                eprintln!("Failed to load NNUE net '{path}': {e}");
                eprintln!("Falling back to HCE.");
                false
            }
        }
    } else {
        false
    };

    let max_moves = 300;
    let balance_threshold: Score = 200; // reject games unbalanced after random opening

    // Append if file exists, otherwise create
    let existing_positions = std::fs::metadata(output_path)
        .map(|m| m.len() / 32)
        .unwrap_or(0);

    let threads = num_threads.max(1);

    eprintln!("Self-play data generation");
    eprintln!("  Games: {num_games}");
    eprintln!("  Search depth: {search_depth}");
    eprintln!("  Random opening plies: {random_plies}");
    eprintln!("  Balance threshold: {balance_threshold}cp");
    eprintln!("  Evaluator: {}", if use_nnue { "NNUE" } else { "HCE" });
    eprintln!("  Threads: {threads}");
    eprintln!("  Output: {output_path}");
    if existing_positions > 0 {
        eprintln!("  Appending to existing file ({existing_positions} positions already)");
    }
    eprintln!();

    // Open file in append mode (safe to Ctrl+C, accumulates across runs)
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)
        .unwrap_or_else(|e| panic!("Failed to open '{output_path}': {e}"));
    let file = Mutex::new(file);

    let next_game = AtomicUsize::new(0);
    let games_done = AtomicUsize::new(0);
    let total_positions = AtomicU64::new(0);
    let start = std::time::Instant::now();

    // Seed the master RNG once from wall-clock time so each run is different.
    // Each thread derives its own seed from this + thread_id.
    let master_seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xBADC_0FFEE_DEAD);

    let discarded_games = AtomicU64::new(0);

    std::thread::scope(|scope| {
        for thread_id in 0..threads {
            let next_game = &next_game;
            let games_done = &games_done;
            let total_positions = &total_positions;
            let discarded_games = &discarded_games;
            let file = &file;
            // Derive a unique seed per thread so games don't collide
            let thread_seed = master_seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(thread_id as u64 * 0xDEAD_C0DE_1234_5678);

            scope.spawn(move || {
                let mut rng = ThreadRng::new(thread_seed);

                loop {
                    let g = next_game.fetch_add(1, Ordering::Relaxed);
                    if g >= num_games {
                        break;
                    }

                    // Play with random opening; retry on discard (up to 10 attempts
                    // to avoid infinite loops if everything keeps rejecting).
                    let mut attempt = 0;
                    let game_result = loop {
                        attempt += 1;
                        if let Some(result) = play_game(
                            search_depth,
                            max_moves,
                            random_plies,
                            balance_threshold,
                            use_nnue,
                            &mut rng,
                        ) {
                            break Some(result);
                        }
                        discarded_games.fetch_add(1, Ordering::Relaxed);
                        if attempt >= 10 {
                            break None;
                        }
                    };

                    let Some((positions, result)) = game_result else {
                        // Couldn't produce a valid game; count it as done anyway
                        games_done.fetch_add(1, Ordering::Relaxed);
                        continue;
                    };
                    let white_result = result.to_white_relative();

                    // Buffer all records for this game, then write under one lock
                    let mut buf = Vec::with_capacity(positions.len() * 32);
                    for (board, eval) in &positions {
                        let record = board_to_record(board, *eval, white_result);
                        let bytes: &[u8; 32] =
                            unsafe { &*(std::ptr::from_ref(&record) as *const [u8; 32]) };
                        buf.extend_from_slice(bytes);
                    }

                    if !buf.is_empty() {
                        let mut f = file.lock().unwrap();
                        f.write_all(&buf)
                            .unwrap_or_else(|e| panic!("Failed to write: {e}"));
                    }

                    total_positions.fetch_add(positions.len() as u64, Ordering::Relaxed);
                    let done = games_done.fetch_add(1, Ordering::Relaxed) + 1;

                    // Thread 0 prints progress (avoids interleaved output)
                    if thread_id == 0 && (done % 50 == 0 || done == num_games) {
                        let elapsed = start.elapsed().as_secs_f64();
                        let pct = 100.0 * done as f64 / num_games as f64;
                        let total_pos = total_positions.load(Ordering::Relaxed);
                        let games_per_sec = done as f64 / elapsed;
                        let remaining = if games_per_sec > 0.0 {
                            (num_games - done) as f64 / games_per_sec
                        } else {
                            0.0
                        };

                        eprint!(
                            "\r  [{:>5.1}%] {}/{} games | {} positions | {:.1} games/sec | ~{} remaining   ",
                            pct, done, num_games, total_pos,
                            games_per_sec, format_duration(remaining),
                        );
                    }
                }
            });
        }
    });

    let final_positions = total_positions.load(Ordering::Relaxed);
    let discarded = discarded_games.load(Ordering::Relaxed);
    eprintln!();
    eprintln!();
    let grand_total = existing_positions + final_positions;
    eprintln!("Done! +{final_positions} new positions ({grand_total} total) saved to {output_path}");
    if discarded > 0 {
        eprintln!("(Discarded {discarded} games due to unbalanced openings after random plies)");
    }
    eprintln!("Train with: cargo run --release -- train {output_path} --output nets/focalors.nnue");
}

fn format_duration(secs: f64) -> String {
    let s = secs as u64;
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m {}s", s / 60, s % 60)
    } else {
        format!("{}h {}m", s / 3600, (s % 3600) / 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Simulate just the random-opening phase and verify it produces diverse
    /// positions. Before this fix, every game reached identical positions.
    #[test]
    fn random_opening_produces_diverse_positions() {
        attacks::init();

        let mut hashes = HashSet::new();
        let mut rng_seed = 0xF00D_CAFE_0123_u64;

        // Play 100 random openings of 8 plies each and collect board hashes
        for _ in 0..100 {
            let mut rng = ThreadRng::new(rng_seed);
            rng_seed = rng_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);

            let mut board = Board::startpos();
            for _ in 0..8 {
                let moves = generate_legal_moves(&board);
                if moves.is_empty() {
                    break;
                }
                let idx = rng.next_index(moves.len());
                make_move(&mut board, moves[idx]);
            }
            hashes.insert(board.hash);
        }

        // With 8 random plies, we should easily see >50 distinct positions.
        // Before the fix (deterministic play), every game had the same hash.
        assert!(
            hashes.len() > 50,
            "Expected >50 unique positions from 100 random openings, got {}",
            hashes.len()
        );
    }

    #[test]
    fn thread_rng_is_deterministic() {
        let mut rng1 = ThreadRng::new(42);
        let mut rng2 = ThreadRng::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    /// Regression test: board_to_record must NOT flip the sign of the eval
    /// for black-to-move positions. The score is already STM-relative
    /// (returned by negamax from current STM's perspective).
    #[test]
    fn board_to_record_preserves_stm_relative_score_for_black() {
        attacks::init();

        // Black to move; STM-relative eval says black is winning (+500cp).
        let board = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
        ).unwrap();
        assert_eq!(board.side_to_move, Color::Black);

        // STM (black) thinks it's winning: +500cp
        let stm_eval: Score = 500;
        let white_result: u8 = 0; // Black win in white-relative

        let record = board_to_record(&board, stm_eval, white_result);

        // In the record, score must be STM-relative: +500 (black is STM and winning).
        // If the code incorrectly negated for black-to-move, this would be -500.
        assert_eq!(
            record.score, 500,
            "score should be STM-relative; +500 means STM winning"
        );

        // Result should also be STM-relative: 2 (STM won).
        assert_eq!(record.result, 2, "result should be STM-relative: 2=STM won");
    }

    #[test]
    fn board_to_record_stm_relative_score_for_white() {
        attacks::init();
        let board = Board::startpos();  // white to move
        let record = board_to_record(&board, 25, 1);
        assert_eq!(record.score, 25);  // unchanged for white-to-move
        assert_eq!(record.result, 1);  // draw
    }

    #[test]
    fn thread_rng_different_seeds_diverge() {
        let mut rng1 = ThreadRng::new(1);
        let mut rng2 = ThreadRng::new(2);
        let mut matches = 0;
        for _ in 0..100 {
            if rng1.next_u64() == rng2.next_u64() {
                matches += 1;
            }
        }
        assert!(matches < 5, "Different seeds should produce different sequences");
    }
}
