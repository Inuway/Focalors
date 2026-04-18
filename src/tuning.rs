//! Texel tuning: automatically optimize eval weights against a dataset of
//! positions with known game outcomes.
//!
//! Usage: `cargo run --release -- tune <dataset_file>`
//!
//! Dataset format (one position per line):
//!   <FEN> [1.0]    — White won
//!   <FEN> [0.5]    — Draw
//!   <FEN> [0.0]    — Black won

use crate::attacks;
use crate::board::Board;
use crate::types::*;

// ════════════════════════════════════════════════════════════════════════════
// Tunable weights
// ════════════════════════════════════════════════════════════════════════════

/// All tunable eval parameters as a flat array.
/// Maps 1:1 to the constants in eval.rs.
pub struct TuningWeights {
    pub values: Vec<f64>,
    pub names: Vec<String>,
}

impl TuningWeights {
    /// Initialize with the current hand-tuned values from eval.rs.
    pub fn from_defaults() -> Self {
        let mut values = Vec::new();
        let mut names = Vec::new();

        // Piece values (index 0-4: N, B, R, Q, K — pawn fixed at 100)
        for (name, val) in [("Knight", 320.0), ("Bishop", 330.0), ("Rook", 500.0),
                            ("Queen", 900.0), ("King", 0.0)] {
            values.push(val);
            names.push(format!("piece_{name}"));
        }

        // Mobility weights MG (index 5-8: N, B, R, Q)
        for (name, val) in [("N", 4.0), ("B", 5.0), ("R", 2.0), ("Q", 1.0)] {
            values.push(val);
            names.push(format!("mob_mg_{name}"));
        }

        // Mobility weights EG (index 9-12)
        for (name, val) in [("N", 4.0), ("B", 5.0), ("R", 4.0), ("Q", 2.0)] {
            values.push(val);
            names.push(format!("mob_eg_{name}"));
        }

        // Mobility baselines (index 13-16)
        for (name, val) in [("N", 4.0), ("B", 6.0), ("R", 7.0), ("Q", 14.0)] {
            values.push(val);
            names.push(format!("mob_base_{name}"));
        }

        // Pawn structure penalties (index 17-22)
        for (name, val) in [("doubled_mg", 10.0), ("doubled_eg", 20.0),
                            ("isolated_mg", 15.0), ("isolated_eg", 20.0),
                            ("backward_mg", 10.0), ("backward_eg", 15.0)] {
            values.push(val);
            names.push(format!("pawn_{name}"));
        }

        // Passed pawn bonuses (index 23-34, ranks 1-6 × MG/EG)
        let pass_mg = [5.0, 10.0, 20.0, 35.0, 55.0, 80.0];
        let pass_eg = [10.0, 20.0, 40.0, 70.0, 110.0, 160.0];
        for r in 0..6 {
            values.push(pass_mg[r]);
            names.push(format!("pass_mg_r{}", r + 1));
            values.push(pass_eg[r]);
            names.push(format!("pass_eg_r{}", r + 1));
        }

        // King safety weights (index 35-40)
        for (name, val) in [("shield_missing", 15.0), ("shield_second", 10.0),
                            ("open_file", 20.0), ("semi_open_file", 10.0),
                            ("attacker_sq_coeff", 3.0), ("attacker_threshold", 2.0)] {
            values.push(val);
            names.push(format!("ks_{name}"));
        }

        // Bishop pair (index 41-42)
        values.push(30.0); names.push("bishop_pair_mg".into());
        values.push(50.0); names.push("bishop_pair_eg".into());

        // New eval terms (index 43-56)
        for (name, val) in [
            ("rook_open_mg", 20.0), ("rook_open_eg", 10.0),
            ("rook_semi_mg", 10.0), ("rook_semi_eg", 5.0),
            ("rook_7th_mg", 20.0), ("rook_7th_eg", 40.0),
            ("rook_behind_mg", 15.0), ("rook_behind_eg", 25.0),
            ("knight_outpost_mg", 20.0), ("knight_outpost_eg", 15.0),
            ("connected_pass_mg", 10.0), ("connected_pass_eg", 20.0),
            ("king_pawn_dist_eg", 5.0), ("tempo_mg", 10.0),
        ] {
            values.push(val);
            names.push(name.into());
        }

        TuningWeights { values, names }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Simplified eval with tunable weights
// ════════════════════════════════════════════════════════════════════════════

/// Evaluate a position using tunable weights (slow — for tuning only).
fn evaluate_with_weights(board: &Board, w: &TuningWeights) -> f64 {
    let piece_values = [100.0, w.values[0], w.values[1], w.values[2], w.values[3], w.values[4]];
    let phase = game_phase(board, &piece_values);

    let mut mg = 0.0f64;
    let mut eg = 0.0f64;

    // Material
    for (pi, &val) in piece_values.iter().enumerate() {
        let wc = board.piece_bb(Color::White, Piece::ALL[pi]).popcount() as f64;
        let bc = board.piece_bb(Color::Black, Piece::ALL[pi]).popcount() as f64;
        mg += val * (wc - bc);
        eg += val * (wc - bc);
    }

    // Bishop pair
    if board.piece_bb(Color::White, Piece::Bishop).popcount() >= 2 {
        mg += w.values[41]; eg += w.values[42];
    }
    if board.piece_bb(Color::Black, Piece::Bishop).popcount() >= 2 {
        mg -= w.values[41]; eg -= w.values[42];
    }

    // Tempo
    let tempo = w.values[56];
    match board.side_to_move {
        Color::White => mg += tempo,
        Color::Black => mg -= tempo,
    }

    // Taper
    let total_phase = 24.0f64;
    let score = (mg * phase + eg * (total_phase - phase)) / total_phase;

    // Return from White's perspective (not side-to-move)
    score
}

fn game_phase(board: &Board, _piece_values: &[f64; 6]) -> f64 {
    let phase_weights = [0.0, 1.0, 1.0, 2.0, 4.0, 0.0];
    let mut phase = 0.0f64;
    for pi in 0..6 {
        let count = board.piece_bb(Color::White, Piece::ALL[pi]).popcount()
            + board.piece_bb(Color::Black, Piece::ALL[pi]).popcount();
        phase += phase_weights[pi] * count as f64;
    }
    phase.min(24.0)
}

// ════════════════════════════════════════════════════════════════════════════
// Optimization
// ════════════════════════════════════════════════════════════════════════════

fn sigmoid(eval: f64, k: f64) -> f64 {
    1.0 / (1.0 + 10.0_f64.powf(-k * eval / 400.0))
}

fn mean_squared_error(dataset: &[(Board, f64)], weights: &TuningWeights, k: f64) -> f64 {
    let sum: f64 = dataset.iter().map(|(board, result)| {
        let eval = evaluate_with_weights(board, weights);
        let predicted = sigmoid(eval, k);
        (result - predicted).powi(2)
    }).sum();
    sum / dataset.len() as f64
}

fn find_optimal_k(dataset: &[(Board, f64)], weights: &TuningWeights) -> f64 {
    let mut best_k = 1.0;
    let mut best_error = f64::MAX;

    // Coarse search
    let mut k = 0.1;
    while k <= 5.0 {
        let error = mean_squared_error(dataset, weights, k);
        if error < best_error {
            best_error = error;
            best_k = k;
        }
        k += 0.1;
    }

    // Fine search around best
    k = best_k - 0.1;
    while k <= best_k + 0.1 {
        let error = mean_squared_error(dataset, weights, k);
        if error < best_error {
            best_error = error;
            best_k = k;
        }
        k += 0.01;
    }

    best_k
}

/// Coordinate descent tuning.
fn tune(dataset: &[(Board, f64)], weights: &mut TuningWeights) {
    let k = find_optimal_k(dataset, weights);
    eprintln!("Optimal K: {k:.3}");

    let mut iteration = 0;
    loop {
        iteration += 1;
        let base_error = mean_squared_error(dataset, weights, k);
        eprintln!("Iteration {iteration}: MSE = {base_error:.8}");
        let mut improved = false;

        for i in 0..weights.len() {
            // Try +1
            weights.values[i] += 1.0;
            let error_up = mean_squared_error(dataset, weights, k);

            if error_up < base_error {
                improved = true;
                eprintln!("  {} += 1 (MSE: {base_error:.8} -> {error_up:.8})", weights.names[i]);
                continue;
            }

            // Try -1
            weights.values[i] -= 2.0;
            let error_down = mean_squared_error(dataset, weights, k);

            if error_down < base_error {
                improved = true;
                eprintln!("  {} -= 1 (MSE: {base_error:.8} -> {error_down:.8})", weights.names[i]);
                continue;
            }

            // Neither improved — restore
            weights.values[i] += 1.0;
        }

        if !improved {
            eprintln!("Converged after {iteration} iterations.");
            break;
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Dataset loading
// ════════════════════════════════════════════════════════════════════════════

fn load_dataset(path: &str) -> Vec<(Board, f64)> {
    let contents = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read dataset file '{path}': {e}"));

    let mut dataset = Vec::new();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }

        // Format: <FEN> [result]
        if let Some(bracket_start) = line.rfind('[') {
            if let Some(bracket_end) = line.rfind(']') {
                let fen = line[..bracket_start].trim();
                let result_str = &line[bracket_start + 1..bracket_end];
                if let Ok(result) = result_str.parse::<f64>() {
                    if let Ok(board) = Board::from_fen(fen) {
                        dataset.push((board, result));
                    }
                }
            }
        }
    }

    dataset
}

// ════════════════════════════════════════════════════════════════════════════
// Entry point
// ════════════════════════════════════════════════════════════════════════════

pub fn run_tuning(dataset_path: &str) {
    attacks::init();

    eprintln!("Loading dataset from '{dataset_path}'...");
    let dataset = load_dataset(dataset_path);
    eprintln!("Loaded {} positions.", dataset.len());

    if dataset.is_empty() {
        eprintln!("Dataset is empty. Nothing to tune.");
        return;
    }

    let mut weights = TuningWeights::from_defaults();
    eprintln!("Tuning {} parameters...", weights.len());

    tune(&dataset, &mut weights);

    // Print results
    println!("\n=== Optimized Weights ===\n");
    for (i, name) in weights.names.iter().enumerate() {
        println!("{name}: {:.0}", weights.values[i]);
    }

    println!("\n=== Copy-paste into eval.rs ===\n");
    println!("// Piece values: N={:.0}, B={:.0}, R={:.0}, Q={:.0}",
        weights.values[0], weights.values[1], weights.values[2], weights.values[3]);
    println!("// Mobility MG: N={:.0}, B={:.0}, R={:.0}, Q={:.0}",
        weights.values[5], weights.values[6], weights.values[7], weights.values[8]);
    println!("// Mobility EG: N={:.0}, B={:.0}, R={:.0}, Q={:.0}",
        weights.values[9], weights.values[10], weights.values[11], weights.values[12]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_at_zero_is_half() {
        let s = sigmoid(0.0, 1.0);
        assert!((s - 0.5).abs() < 0.001, "sigmoid(0) should be 0.5, got {s}");
    }

    #[test]
    fn weights_have_correct_count() {
        let w = TuningWeights::from_defaults();
        assert_eq!(w.values.len(), w.names.len());
        assert!(w.len() > 50, "Should have 50+ tunable parameters, got {}", w.len());
    }

    #[test]
    fn evaluate_with_weights_produces_reasonable_score() {
        crate::attacks::init();
        let board = Board::startpos();
        let w = TuningWeights::from_defaults();
        let score = evaluate_with_weights(&board, &w);
        assert!(score.abs() < 100.0, "Starting position should be near 0 with default weights, got {score}");
    }
}
