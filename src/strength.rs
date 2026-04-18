//! Adaptive difficulty system for Focalors.
//!
//! Maps numeric levels 1-20 to concrete engine parameters (depth, time, eval noise,
//! NNUE toggle, move selection temperature). Also provides Elo estimation and
//! auto-adjust logic.

use crate::eval::Score;

// ════════════════════════════════════════════════════════════════════════════
// Strength configuration
// ════════════════════════════════════════════════════════════════════════════

/// Engine strength parameters for a given numeric level (1-20).
#[derive(Debug, Clone)]
pub struct StrengthConfig {
    pub level: u32,
    pub max_depth: Option<u32>,
    pub time_scale: f32,
    pub eval_noise_cp: i32,
    pub use_nnue: bool,
    pub top_n_moves: usize,
    pub temperature: f64,
    pub min_think_ms: u64,
}

/// Key calibration points. Intermediate levels interpolate linearly.
///   (level, depth, time_scale, noise, nnue, top_n, temperature, min_think_ms)
const CALIBRATION: &[(u32, u32, f32, i32, bool, usize, f64, u64)] = &[
    //  lvl  depth  time   noise  nnue  topN  temp   think
    (1, 3, 0.15, 200, false, 5, 2.0, 300),
    (3, 5, 0.25, 120, false, 4, 1.5, 250),
    (5, 6, 0.30, 80, false, 3, 1.2, 200),
    (7, 7, 0.40, 60, false, 3, 1.0, 150),
    (8, 8, 0.50, 50, true, 3, 0.8, 100),
    (10, 10, 0.60, 30, true, 2, 0.5, 50),
    (12, 12, 0.75, 15, true, 2, 0.3, 0),
    (14, 14, 0.85, 10, true, 1, 0.0, 0),
    (16, 16, 0.95, 5, true, 1, 0.0, 0),
    (18, 18, 1.0, 0, true, 1, 0.0, 0),
    (20, 64, 1.35, 0, true, 1, 0.0, 0), // 64 = effectively unlimited
];

impl StrengthConfig {
    /// Build a config for the given numeric level (clamped to 1-20).
    pub fn from_level(level: u32) -> Self {
        let level = level.clamp(1, 20);

        // Find the two calibration points we're between
        let (lo, hi) = find_bracket(level);

        if lo.0 == hi.0 {
            // Exact match
            return Self {
                level,
                max_depth: if lo.1 >= 64 { None } else { Some(lo.1) },
                time_scale: lo.2,
                eval_noise_cp: lo.3,
                use_nnue: lo.4,
                top_n_moves: lo.5,
                temperature: lo.6,
                min_think_ms: lo.7,
            };
        }

        let t = (level - lo.0) as f64 / (hi.0 - lo.0) as f64;

        let depth = lerp_u32(lo.1, hi.1, t);
        let time_scale = lerp_f32(lo.2, hi.2, t);
        let noise = lerp_i32(lo.3, hi.3, t);
        let top_n = lerp_usize(lo.5, hi.5, t);
        let temp = lerp_f64(lo.6, hi.6, t);
        let think = lerp_u64(lo.7, hi.7, t);
        // NNUE: use the higher bracket's setting (switch at level 8)
        let use_nnue = if t >= 0.5 { hi.4 } else { lo.4 };

        Self {
            level,
            max_depth: if depth >= 64 { None } else { Some(depth) },
            time_scale,
            eval_noise_cp: noise,
            use_nnue,
            top_n_moves: top_n.max(1),
            temperature: temp,
            min_think_ms: think,
        }
    }
}

type CalRow = (u32, u32, f32, i32, bool, usize, f64, u64);

fn find_bracket(level: u32) -> (&'static CalRow, &'static CalRow) {
    let mut lo = &CALIBRATION[0];
    let mut hi = &CALIBRATION[CALIBRATION.len() - 1];
    for pair in CALIBRATION.windows(2) {
        if level >= pair[0].0 && level <= pair[1].0 {
            lo = &pair[0];
            hi = &pair[1];
            break;
        }
    }
    (lo, hi)
}

fn lerp_u32(a: u32, b: u32, t: f64) -> u32 {
    (a as f64 + (b as f64 - a as f64) * t).round() as u32
}
fn lerp_f32(a: f32, b: f32, t: f64) -> f32 {
    a + (b - a) * t as f32
}
fn lerp_i32(a: i32, b: i32, t: f64) -> i32 {
    (a as f64 + (b as f64 - a as f64) * t).round() as i32
}
fn lerp_usize(a: usize, b: usize, t: f64) -> usize {
    (a as f64 + (b as f64 - a as f64) * t).round() as usize
}
fn lerp_f64(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}
fn lerp_u64(a: u64, b: u64, t: f64) -> u64 {
    (a as f64 + (b as f64 - a as f64) * t).round() as u64
}

// ════════════════════════════════════════════════════════════════════════════
// Elo estimation
// ════════════════════════════════════════════════════════════════════════════

/// Approximate Elo for a given engine level (linear: level 1 ≈ 400, level 20 ≈ 2400).
pub fn estimated_elo(level: u32) -> i32 {
    let level = level.clamp(1, 20) as i32;
    // 400 + (level-1) * (2000/19) ≈ 400 + (level-1)*105
    400 + (level - 1) * 105
}

/// Standard Elo rating update.
/// `score`: 1.0 = win, 0.5 = draw, 0.0 = loss.
pub fn elo_update(player_rating: i32, opponent_rating: i32, score: f64, k: i32) -> i32 {
    let expected = 1.0 / (1.0 + 10.0_f64.powf((opponent_rating - player_rating) as f64 / 400.0));
    let delta = (k as f64 * (score - expected)).round() as i32;
    player_rating + delta
}

/// K-factor: 40 for new players (< 30 rated games), 20 for established.
pub fn k_factor(rated_games: u32) -> i32 {
    if rated_games < 30 { 40 } else { 20 }
}

// ════════════════════════════════════════════════════════════════════════════
// Deterministic eval noise
// ════════════════════════════════════════════════════════════════════════════

/// Deterministic noise for a position hash. Returns a value in [-max_noise, +max_noise]
/// with a pseudo-Gaussian distribution (sum of 3 uniform samples / 3).
/// Hash-based: same position always gets the same noise → no search instability.
pub fn deterministic_noise(hash: u64, max_noise: i32) -> Score {
    if max_noise == 0 {
        return 0;
    }
    let max = max_noise as i64;
    let range = 2 * max + 1;
    // Three independent samples from different bit ranges of the hash
    let s1 = ((hash % range as u64) as i64 - max) as i32;
    let s2 = (((hash >> 21) % range as u64) as i64 - max) as i32;
    let s3 = (((hash >> 42) % range as u64) as i64 - max) as i32;
    // Average → pseudo-Gaussian, peaks near 0
    (s1 + s2 + s3) / 3
}

// ════════════════════════════════════════════════════════════════════════════
// Weighted move selection
// ════════════════════════════════════════════════════════════════════════════

use crate::board::Board;
use crate::movegen::{generate_legal_moves, make_move};
use crate::moves::Move;
use crate::search::Searcher;

/// After the engine completes its search, optionally pick a weaker move
/// based on the strength config. At high levels (top_n=1 or temp=0) this
/// just returns the best move unchanged.
pub fn select_move(
    searcher: &mut Searcher,
    board: &Board,
    best_move: Move,
    best_score: Score,
    search_depth: u32,
    config: &StrengthConfig,
) -> Move {
    if config.top_n_moves <= 1 || config.temperature <= 0.0 {
        return best_move;
    }

    let legal_moves = generate_legal_moves(board);
    if legal_moves.len() <= 1 {
        return best_move;
    }

    // Score the top-N candidate moves with a shallow verification search
    let verify_depth = search_depth.saturating_sub(2).max(1);
    let n = config.top_n_moves.min(legal_moves.len());

    let mut candidates: Vec<(Move, Score)> = Vec::with_capacity(n);
    candidates.push((best_move, best_score));

    // Silence the searcher during verification searches
    let was_silent = searcher.silent;
    searcher.silent = true;

    for i in 0..legal_moves.len() {
        if candidates.len() >= n {
            break;
        }
        let mv = legal_moves[i];
        if mv == best_move {
            continue;
        }
        // Play the move, search from opponent's perspective, negate
        let mut child = board.clone();
        make_move(&mut child, mv);
        let result = searcher.search(&child, verify_depth);
        let score = -result.score; // negate: opponent's score → our score
        candidates.push((mv, score));
    }

    searcher.silent = was_silent;

    // Sort by score descending (best first)
    candidates.sort_by(|a, b| b.1.cmp(&a.1));

    // Softmax selection with temperature
    let max_score = candidates[0].1 as f64;
    let weights: Vec<f64> = candidates
        .iter()
        .map(|(_, s)| ((*s as f64 - max_score) / (config.temperature * 100.0)).exp())
        .collect();
    let total: f64 = weights.iter().sum();

    // Deterministic RNG from position hash
    let rng = xorshift64(board.hash ^ config.level as u64);
    let r = (rng % 10000) as f64 / 10000.0;

    let mut cumulative = 0.0;
    for (i, w) in weights.iter().enumerate() {
        cumulative += w / total;
        if r < cumulative {
            return candidates[i].0;
        }
    }

    // Fallback (rounding edge case)
    candidates.last().unwrap().0
}

/// Simple xorshift64 PRNG — one step.
fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

// ════════════════════════════════════════════════════════════════════════════
// Auto-adjust
// ════════════════════════════════════════════════════════════════════════════

/// Evaluate whether the adaptive level should change based on recent results.
/// Returns `Some(new_level)` if adjustment is needed.
///
/// `recent_results`: slice of ("win"/"loss"/"draw") strings from most recent
/// games played on Adaptive, newest first.
pub fn evaluate_auto_adjust(current_level: u32, recent_results: &[String]) -> Option<u32> {
    if recent_results.len() < 5 {
        return None; // not enough data yet
    }

    let window = &recent_results[..recent_results.len().min(10)];
    let wins = window.iter().filter(|r| r.as_str() == "win").count();
    let total = window.len();
    let win_rate = wins as f64 / total as f64;

    if win_rate > 0.60 && current_level < 20 {
        Some(current_level + 1)
    } else if win_rate < 0.30 && current_level > 1 {
        Some(current_level - 1)
    } else {
        None
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Named preset mapping
// ════════════════════════════════════════════════════════════════════════════

/// Map the legacy named presets to numeric levels.
pub const BEGINNER_LEVEL: u32 = 3;
pub const CLUB_LEVEL: u32 = 8;
pub const TOURNAMENT_LEVEL: u32 = 14;
pub const MASTER_LEVEL: u32 = 20;

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_level_1_is_weak() {
        let c = StrengthConfig::from_level(1);
        assert_eq!(c.max_depth, Some(3));
        assert!(c.eval_noise_cp >= 150);
        assert!(!c.use_nnue);
        assert!(c.top_n_moves >= 4);
        assert!(c.temperature > 1.0);
    }

    #[test]
    fn config_level_20_is_full_strength() {
        let c = StrengthConfig::from_level(20);
        assert_eq!(c.max_depth, None); // unlimited
        assert_eq!(c.eval_noise_cp, 0);
        assert!(c.use_nnue);
        assert_eq!(c.top_n_moves, 1);
        assert_eq!(c.temperature, 0.0);
    }

    #[test]
    fn config_interpolation_monotonic() {
        // Noise should decrease as level increases
        let mut prev_noise = i32::MAX;
        for level in 1..=20 {
            let c = StrengthConfig::from_level(level);
            assert!(c.eval_noise_cp <= prev_noise, "noise not monotonic at level {level}");
            prev_noise = c.eval_noise_cp;
        }
    }

    #[test]
    fn elo_estimation() {
        assert_eq!(estimated_elo(1), 400);
        assert_eq!(estimated_elo(20), 2395); // 400 + 19*105
        assert!(estimated_elo(10) > estimated_elo(5));
    }

    #[test]
    fn elo_update_win_increases_rating() {
        let new = elo_update(1200, 1200, 1.0, 20);
        assert!(new > 1200);
    }

    #[test]
    fn elo_update_loss_decreases_rating() {
        let new = elo_update(1200, 1200, 0.0, 20);
        assert!(new < 1200);
    }

    #[test]
    fn elo_update_draw_equal_ratings() {
        let new = elo_update(1200, 1200, 0.5, 20);
        assert_eq!(new, 1200);
    }

    #[test]
    fn deterministic_noise_is_stable() {
        let n1 = deterministic_noise(0xDEADBEEF, 100);
        let n2 = deterministic_noise(0xDEADBEEF, 100);
        assert_eq!(n1, n2);
        assert!(n1.abs() <= 100);
    }

    #[test]
    fn deterministic_noise_zero_max() {
        assert_eq!(deterministic_noise(12345, 0), 0);
    }

    #[test]
    fn auto_adjust_not_enough_games() {
        assert_eq!(evaluate_auto_adjust(10, &vec!["win".into(); 3]), None);
    }

    #[test]
    fn auto_adjust_level_up_on_winning() {
        let results: Vec<String> = vec!["win".into(); 10];
        assert_eq!(evaluate_auto_adjust(10, &results), Some(11));
    }

    #[test]
    fn auto_adjust_level_down_on_losing() {
        let results: Vec<String> = vec!["loss".into(); 10];
        assert_eq!(evaluate_auto_adjust(10, &results), Some(9));
    }

    #[test]
    fn auto_adjust_no_change_mixed() {
        let results: Vec<String> = vec![
            "win".into(), "loss".into(), "win".into(), "loss".into(), "win".into(),
            "loss".into(), "win".into(), "loss".into(), "draw".into(), "draw".into(),
        ];
        assert_eq!(evaluate_auto_adjust(10, &results), None);
    }

    #[test]
    fn auto_adjust_capped_at_20() {
        let results: Vec<String> = vec!["win".into(); 10];
        assert_eq!(evaluate_auto_adjust(20, &results), None);
    }

    #[test]
    fn auto_adjust_floored_at_1() {
        let results: Vec<String> = vec!["loss".into(); 10];
        assert_eq!(evaluate_auto_adjust(1, &results), None);
    }
}
