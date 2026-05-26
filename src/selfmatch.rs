//! Engine-vs-engine self-match for in-tree strength benchmarking.
//!
//! Runs N games of focalors-vs-focalors at fixed depth and reports W/L/D,
//! elo delta vs the standard logistic, 95% confidence interval, and LOS
//! (likelihood of superiority). Both engines run in the same process, with
//! one optionally using an alternate NNUE net loaded via `nnue::init_alt`.
//!
//! Usage:
//!   `focalors selfmatch <games> [--depth N] [--challenger-net PATH]`
//!
//! The match uses matched-pair openings (same random opening played from
//! both sides) for variance reduction. This is the standard cutechess-cli
//! pattern.

use crate::attacks;
use crate::board::Board;
use crate::movegen::{generate_legal_moves, make_move};
use crate::moves::Move;
use crate::nnue;
use crate::search::Searcher;
use crate::types::{Color, Piece};

// ════════════════════════════════════════════════════════════════════════════
// Public API
// ════════════════════════════════════════════════════════════════════════════

pub fn run_selfmatch(
    num_games: usize,
    search_depth: u32,
    challenger_net: Option<&str>,
    seed: Option<u64>,
    random_plies: u32,
    max_moves: usize,
    threads: usize,
) {
    use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

    if num_games == 0 {
        eprintln!("No games requested (use a positive integer).");
        return;
    }

    attacks::init();

    if let Err(e) = nnue::init(None) {
        eprintln!("Failed to initialize primary NNUE net: {e}");
        eprintln!("Selfmatch requires NNUE; aborting.");
        return;
    }

    let has_challenger = challenger_net.is_some();
    if let Some(path) = challenger_net {
        if let Err(e) = nnue::init_alt(path) {
            eprintln!("Failed to load challenger NNUE net '{path}': {e}");
            return;
        }
    }

    let master_seed = seed.unwrap_or_else(time_seed);
    let pairs = num_games / 2;
    let leftover = num_games % 2;
    let threads_used = threads.max(1).min(pairs.max(1));

    eprintln!("Self-match");
    eprintln!("  Games:    {num_games}");
    eprintln!("  Depth:    {search_depth}");
    eprintln!("  Threads:  {threads_used}");
    eprintln!(
        "  Engine A: current (embedded default net)"
    );
    eprintln!(
        "  Engine B: {}",
        if let Some(path) = challenger_net {
            format!("challenger ({path})")
        } else {
            "current (embedded default net)".to_string()
        }
    );
    eprintln!("  Seed:     {master_seed}");
    eprintln!("  Openings: {random_plies} random plies, max {max_moves} moves/game");
    eprintln!();

    // Atomic counters: workers fetch_add to claim work and report results.
    let next_pair = AtomicUsize::new(0);
    let games_done = AtomicU32::new(0);
    let wins = AtomicU32::new(0);
    let losses = AtomicU32::new(0);
    let draws = AtomicU32::new(0);

    let record = |r: GameResult| match r {
        GameResult::EngineAWin => { wins.fetch_add(1, Ordering::Relaxed); }
        GameResult::EngineBWin => { losses.fetch_add(1, Ordering::Relaxed); }
        GameResult::Draw => { draws.fetch_add(1, Ordering::Relaxed); }
    };

    let snapshot = || MatchStats {
        wins: wins.load(Ordering::Relaxed),
        losses: losses.load(Ordering::Relaxed),
        draws: draws.load(Ordering::Relaxed),
    };

    std::thread::scope(|scope| {
        for thread_id in 0..threads_used {
            let next_pair = &next_pair;
            let games_done = &games_done;
            let record = &record;
            let snapshot = &snapshot;
            let thread_seed = master_seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(thread_id as u64 * 0xDEAD_C0DE_1234_5678);

            scope.spawn(move || {
                // Each worker has its own Searchers — CH/history/killers
                // are per-instance and don't share across threads.
                let mut a = Searcher::new(8);
                a.silent = true;
                a.use_nnue = true;

                let mut b = if has_challenger {
                    let mut s = Searcher::with_alt_nnue(8);
                    s.silent = true;
                    s.use_nnue = true;
                    s
                } else {
                    let mut s = Searcher::new(8);
                    s.silent = true;
                    s.use_nnue = true;
                    s
                };

                let mut rng = ThreadRng::new(thread_seed);

                loop {
                    let p = next_pair.fetch_add(1, Ordering::Relaxed);
                    if p >= pairs { break; }

                    let opening = generate_opening(&mut rng, random_plies);

                    let r1 = play_one_game(
                        &mut a, &mut b, true, &opening, search_depth, max_moves,
                    );
                    record(r1);
                    let done = games_done.fetch_add(1, Ordering::Relaxed) + 1;
                    if thread_id == 0 {
                        print_progress(&snapshot(), done, num_games as u32);
                    }

                    let r2 = play_one_game(
                        &mut a, &mut b, false, &opening, search_depth, max_moves,
                    );
                    record(r2);
                    let done = games_done.fetch_add(1, Ordering::Relaxed) + 1;
                    if thread_id == 0 {
                        print_progress(&snapshot(), done, num_games as u32);
                    }
                }
            });
        }
    });

    // Odd-game-out: play one extra solo game on the main thread so the
    // result count exactly matches num_games. Rare path (only when caller
    // asks for an odd number of games).
    if leftover == 1 {
        let mut a = Searcher::new(8);
        a.silent = true;
        a.use_nnue = true;
        let mut b = if has_challenger {
            let mut s = Searcher::with_alt_nnue(8);
            s.silent = true;
            s.use_nnue = true;
            s
        } else {
            let mut s = Searcher::new(8);
            s.silent = true;
            s.use_nnue = true;
            s
        };

        let mut rng = ThreadRng::new(
            master_seed.wrapping_mul(0xCAFE_F00D_1234_5678),
        );
        let opening = generate_opening(&mut rng, random_plies);
        let r = play_one_game(&mut a, &mut b, true, &opening, search_depth, max_moves);
        record(r);
        let done = games_done.fetch_add(1, Ordering::Relaxed) + 1;
        print_progress(&snapshot(), done, num_games as u32);
    }

    eprintln!();
    eprintln!();
    print_final_results(&snapshot(), challenger_net, num_games);
}

// ════════════════════════════════════════════════════════════════════════════
// Statistics
// ════════════════════════════════════════════════════════════════════════════

#[derive(Default, Clone, Copy, Debug)]
struct MatchStats {
    wins: u32,
    losses: u32,
    draws: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GameResult {
    EngineAWin,
    EngineBWin,
    Draw,
}

impl MatchStats {
    fn n(&self) -> u32 {
        self.wins + self.losses + self.draws
    }

    fn score(&self) -> f64 {
        let n = self.n() as f64;
        if n == 0.0 {
            return 0.5;
        }
        (self.wins as f64 + 0.5 * self.draws as f64) / n
    }

    /// Elo delta from score% using the standard logistic.
    /// Returns 0 at 50%, positive when A is winning, negative when B is.
    fn elo_delta(&self) -> f64 {
        let p = self.score();
        if p <= 0.0 {
            return -999.0;
        }
        if p >= 1.0 {
            return 999.0;
        }
        -400.0 * (1.0 / p - 1.0).log10()
    }

    /// 95% CI on the elo delta via Wald on score%, with asymmetric
    /// back-conversion through the logistic. Uses sample variance over
    /// per-game outcomes {1.0, 0.5, 0.0} rather than naive p(1-p) — more
    /// accurate when draws are common.
    fn elo_ci95(&self) -> (f64, f64) {
        let n = self.n() as f64;
        if n < 1.0 {
            return (-999.0, 999.0);
        }
        let p = self.score();
        let var = (self.wins as f64 * (1.0 - p).powi(2)
            + self.draws as f64 * (0.5 - p).powi(2)
            + self.losses as f64 * (0.0 - p).powi(2))
            / n;
        let se = (var / n).sqrt();
        let p_lo = (p - 1.96 * se).max(0.001);
        let p_hi = (p + 1.96 * se).min(0.999);
        let to_elo = |x: f64| -400.0 * (1.0 / x - 1.0).log10();
        (to_elo(p_lo), to_elo(p_hi))
    }

    /// Likelihood of Superiority: P(A truly stronger | observed WLD), treating
    /// draws as uninformative. Z = (W - L) / sqrt(W + L); LOS = Phi(Z).
    /// Standard cutechess-cli / Chessprogramming wiki formula.
    fn los(&self) -> f64 {
        let w = self.wins as f64;
        let l = self.losses as f64;
        if w + l == 0.0 {
            return 0.5;
        }
        let z = (w - l) / (w + l).sqrt();
        0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
    }

}

/// Abramowitz & Stegun 7.1.26 approximation of the error function.
/// Max absolute error ~1.5e-7 across the real line. No external deps.
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592_f64;
    let a2 = -0.284496736_f64;
    let a3 = 1.421413741_f64;
    let a4 = -1.453152027_f64;
    let a5 = 1.061405429_f64;
    let p = 0.3275911_f64;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

// ════════════════════════════════════════════════════════════════════════════
// Game loop
// ════════════════════════════════════════════════════════════════════════════

fn play_one_game(
    a: &mut Searcher,
    b: &mut Searcher,
    a_plays_white: bool,
    opening: &[Move],
    depth: u32,
    max_moves: usize,
) -> GameResult {
    let mut board = Board::startpos();
    let mut history: Vec<u64> = vec![board.hash];

    for &mv in opening {
        make_move(&mut board, mv);
        history.push(board.hash);
    }
    let mut move_count = opening.len();

    loop {
        let moves = generate_legal_moves(&board);
        let king_sq = board.piece_bb(board.side_to_move, Piece::King).lsb();
        let in_check =
            attacks::is_square_attacked(&board, king_sq, board.side_to_move.flip());

        // Game-end checks before searching.
        if moves.is_empty() {
            return if in_check {
                // Side-to-move is mated; the OTHER color won.
                winner_for_color(board.side_to_move.flip(), a_plays_white)
            } else {
                GameResult::Draw
            };
        }
        if board.halfmove_clock >= 100 {
            return GameResult::Draw;
        }
        if board.is_insufficient_material() {
            return GameResult::Draw;
        }
        if is_threefold(&history, board.hash) {
            return GameResult::Draw;
        }
        if move_count >= max_moves {
            return GameResult::Draw;
        }

        let searcher: &mut Searcher =
            if (board.side_to_move == Color::White) == a_plays_white {
                &mut *a
            } else {
                &mut *b
            };

        searcher.set_position_history(history.clone());
        let result = searcher.search(&board, depth);

        if result.best_move.is_null() {
            // Defensive: shouldn't happen since we already checked that
            // moves is non-empty, but never let a buggy search loop forever.
            return GameResult::Draw;
        }

        make_move(&mut board, result.best_move);
        history.push(board.hash);
        move_count += 1;
    }
}

fn winner_for_color(winning: Color, a_plays_white: bool) -> GameResult {
    let a_color = if a_plays_white {
        Color::White
    } else {
        Color::Black
    };
    if winning == a_color {
        GameResult::EngineAWin
    } else {
        GameResult::EngineBWin
    }
}

fn is_threefold(history: &[u64], current_hash: u64) -> bool {
    // `history` already contains `current_hash` (pushed after the move that
    // reached this position), so the threshold for threefold is 3 occurrences.
    history.iter().filter(|&&h| h == current_hash).count() >= 3
}

// ════════════════════════════════════════════════════════════════════════════
// Opening generation
// ════════════════════════════════════════════════════════════════════════════

/// Generate a random opening by playing `plies` uniformly-random legal moves
/// from startpos. No balance filter — the matched-pair structure cancels
/// any positional imbalance across each pair.
fn generate_opening(rng: &mut ThreadRng, plies: u32) -> Vec<Move> {
    for _ in 0..10 {
        if let Some(opening) = try_generate_opening(rng, plies) {
            return opening;
        }
    }
    panic!("Failed to generate a {plies}-ply random opening after 10 attempts");
}

fn try_generate_opening(rng: &mut ThreadRng, plies: u32) -> Option<Vec<Move>> {
    let mut board = Board::startpos();
    let mut moves_played = Vec::with_capacity(plies as usize);
    for _ in 0..plies {
        let legal = generate_legal_moves(&board);
        if legal.is_empty() {
            return None; // ran into stalemate/checkmate during random walk
        }
        let idx = rng.next_index(legal.len());
        let mv = legal[idx];
        make_move(&mut board, mv);
        moves_played.push(mv);
    }
    Some(moves_played)
}

// ════════════════════════════════════════════════════════════════════════════
// Output
// ════════════════════════════════════════════════════════════════════════════

fn print_progress(stats: &MatchStats, played: u32, total: u32) {
    let pct = 100.0 * played as f64 / total as f64;
    let elo = stats.elo_delta();
    let (lo, hi) = stats.elo_ci95();
    let half_width = (hi - lo) / 2.0;
    let los = 100.0 * stats.los();
    eprint!(
        "\r[{:>5.1}%] {}/{} games | W:{} L:{} D:{} | elo: {:+.0} \u{00b1} {:.0} | LOS: {:.1}%   ",
        pct, played, total, stats.wins, stats.losses, stats.draws, elo, half_width, los
    );
}

fn print_final_results(stats: &MatchStats, challenger_net: Option<&str>, num_games: usize) {
    let n = stats.n() as f64;
    let pct_w = 100.0 * stats.wins as f64 / n;
    let pct_l = 100.0 * stats.losses as f64 / n;
    let pct_d = 100.0 * stats.draws as f64 / n;
    let score_pct = 100.0 * stats.score();
    let elo = stats.elo_delta();
    let (lo, hi) = stats.elo_ci95();
    let los = 100.0 * stats.los();

    println!("Self-match results");
    println!("  Games:    {num_games}");
    println!("  Engine A: current (embedded default net)");
    match challenger_net {
        Some(path) => println!("  Engine B: challenger ({path})"),
        None => println!("  Engine B: current (embedded default net)"),
    }
    println!();
    println!("  Wins:     {} ({:.1}%)", stats.wins, pct_w);
    println!("  Losses:   {} ({:.1}%)", stats.losses, pct_l);
    println!("  Draws:    {} ({:.1}%)", stats.draws, pct_d);
    println!("  Score:    {:.1}%", score_pct);
    println!("  Elo:      {:+.0}   [{:+.0}, {:+.0}]  (95% CI)", elo, lo, hi);
    println!("  LOS:      {:.1}%             (chance A is truly stronger than B)", los);

    if challenger_net.is_none() {
        println!();
        println!(
            "Note: both engines used the embedded default net. Use --challenger-net <path>"
        );
        println!("to compare against a different NNUE net.");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// PRNG (xorshift; copied from selfplay.rs to keep this module self-contained)
// ════════════════════════════════════════════════════════════════════════════

struct ThreadRng {
    state: u64,
}

impl ThreadRng {
    fn new(seed: u64) -> Self {
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

fn time_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xBADC_0FFEE_DEAD)
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(w: u32, l: u32, d: u32) -> MatchStats {
        MatchStats { wins: w, losses: l, draws: d }
    }

    #[test]
    fn stats_score_basics() {
        // Wins, losses, draws → score 0.625 (>50%, so elo > 0)
        let s = stats(10, 5, 5);
        assert!((s.score() - 0.625).abs() < 1e-9);
        assert!(s.elo_delta() > 0.0);

        // All losses → score 0, elo at the floor sentinel
        let s = stats(0, 10, 0);
        assert!((s.score() - 0.0).abs() < 1e-9);
        assert_eq!(s.elo_delta(), -999.0);

        // Even W/L, no draws → score 0.5, elo ≈ 0
        let s = stats(5, 5, 0);
        assert!((s.score() - 0.5).abs() < 1e-9);
        assert!(s.elo_delta().abs() < 1e-9);
    }

    #[test]
    fn stats_los_known_values() {
        // All wins → LOS very close to 1
        assert!(stats(10, 0, 0).los() > 0.99);
        // Even split → LOS = 0.5
        assert!((stats(5, 5, 0).los() - 0.5).abs() < 1e-6);
        // All losses → LOS very close to 0
        assert!(stats(0, 10, 0).los() < 0.01);
        // Draws don't shift LOS — same W and L gives same LOS regardless of draws
        let a = stats(5, 5, 0).los();
        let b = stats(5, 5, 100).los();
        assert!((a - b).abs() < 1e-9);
    }

    #[test]
    fn stats_ci_shrinks_with_sqrt_n() {
        // Same score%, 10x the games → CI half-width ~1/sqrt(10) of the original.
        let small = stats(50, 50, 100);
        let big = stats(500, 500, 1000);

        let (slo, shi) = small.elo_ci95();
        let (blo, bhi) = big.elo_ci95();
        let small_half = (shi - slo) / 2.0;
        let big_half = (bhi - blo) / 2.0;

        let ratio = small_half / big_half;
        let expected = (10.0_f64).sqrt();
        // Allow 10% tolerance — Wald + back-conversion isn't perfectly linear.
        assert!(
            (ratio - expected).abs() / expected < 0.1,
            "CI ratio {ratio:.3} should be ~{expected:.3}"
        );
    }

    #[test]
    fn stats_ci_contains_point_estimate() {
        // Use samples that aren't extreme so the CI actually has width.
        for &(w, l, d) in &[(15u32, 5, 10), (10, 10, 10), (3, 2, 5), (8, 12, 30)] {
            let s = stats(w, l, d);
            let elo = s.elo_delta();
            let (lo, hi) = s.elo_ci95();
            assert!(
                lo <= elo + 1e-6 && elo <= hi + 1e-6,
                "CI [{lo:.2}, {hi:.2}] must contain elo {elo:.2} for ({w},{l},{d})"
            );
        }
    }

    #[test]
    fn threefold_detection_works() {
        let h = 0xCAFE_BABE_u64;
        let other = 0xDEAD_BEEF_u64;

        // Three occurrences → true
        assert!(is_threefold(&[h, other, h, other, h], h));
        // Two occurrences → false
        assert!(!is_threefold(&[h, other, h], h));
        // Hash not in history → false
        assert!(!is_threefold(&[other, other], h));
    }

    #[test]
    fn elo_logistic_round_trips() {
        // Convert elo → score → elo and ensure we're back where we started.
        for &target_elo in &[-200.0_f64, -100.0, -50.0, 0.0, 50.0, 100.0, 200.0] {
            let p = 1.0 / (1.0 + 10f64.powf(-target_elo / 400.0));
            // Build stats that reproduce this exact p (use a large N to keep
            // rounding off our radar).
            let n = 10_000_u32;
            let wins = (p * n as f64).round() as u32;
            let s = MatchStats {
                wins,
                losses: n - wins,
                draws: 0,
            };
            let elo_back = s.elo_delta();
            assert!(
                (elo_back - target_elo).abs() < 1.0,
                "Round-trip {target_elo} elo → {elo_back} elo"
            );
        }
    }

    #[test]
    fn erf_known_values() {
        assert!((erf(0.0)).abs() < 1e-7);
        assert!((erf(1.0) - 0.8427007929).abs() < 1e-5);
        assert!((erf(-1.0) + 0.8427007929).abs() < 1e-5);
        // erf(infinity) → 1; check large positive
        assert!((erf(5.0) - 1.0).abs() < 1e-6);
        assert!((erf(-5.0) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn opening_replay_deterministic() {
        attacks::init();
        let mut rng1 = ThreadRng::new(0x1234_5678);
        let mut rng2 = ThreadRng::new(0x1234_5678);
        let o1 = generate_opening(&mut rng1, 8);
        let o2 = generate_opening(&mut rng2, 8);
        assert_eq!(o1.len(), 8);
        assert_eq!(o1, o2);
    }

    #[test]
    fn winner_for_color_mapping() {
        // A is white, white wins → A wins
        assert_eq!(
            winner_for_color(Color::White, true),
            GameResult::EngineAWin
        );
        // A is white, black wins → B wins
        assert_eq!(
            winner_for_color(Color::Black, true),
            GameResult::EngineBWin
        );
        // A is black, white wins → B wins
        assert_eq!(
            winner_for_color(Color::White, false),
            GameResult::EngineBWin
        );
        // A is black, black wins → A wins
        assert_eq!(
            winner_for_color(Color::Black, false),
            GameResult::EngineAWin
        );
    }
}
