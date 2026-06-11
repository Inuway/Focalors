mod analysis;
mod attacks;
mod board;
mod db;
mod eval;
mod gui;
mod movegen;
mod moves;
mod nnue;
mod search;
mod openings;
mod pgn;
mod puzzles;
mod selfmatch;
mod selfplay;
mod strength;
mod trainer;
#[cfg(feature = "gpu-training")]
mod trainer_gpu;
mod tt;
mod tuning;
mod types;
mod uci;
mod zobrist;

/// Exit with a usage message instead of a panic + abort on bad CLI input.
fn usage_exit(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}

/// Parse an optional `--flag value`. Missing flag → default;
/// present-but-unparseable value → usage error (a typo like
/// `--depth ten` silently falling back to the default could quietly
/// invalidate a long selfplay/selfmatch run).
fn parsed_flag<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    match args.iter().position(|a| a == flag) {
        None => default,
        Some(i) => match args.get(i + 1).map(|s| s.parse::<T>()) {
            Some(Ok(v)) => v,
            _ => usage_exit(&format!("Error: {flag} requires a valid value")),
        },
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("gui") | None => {
            // Desktop GUI mode (default when launched with no arguments,
            // e.g. double-clicking the released binary)
            gui::run_gui();
        }
        Some("tune") => {
            let Some(dataset) = args.get(2) else {
                usage_exit("Usage: focalors tune <dataset_file>");
            };
            tuning::run_tuning(dataset);
        }
        Some("train") => {
            let Some(data_path) = args.get(2) else {
                usage_exit("Usage: focalors train <data_file> [options]");
            };
            trainer::run_training(data_path, &args[3..]);
        }
        #[cfg(feature = "gpu-training")]
        Some("train-gpu") => {
            let Some(data_path) = args.get(2) else {
                eprintln!("Usage: focalors train-gpu <data_file> [options]");
                std::process::exit(1);
            };
            trainer_gpu::run_training_gpu(data_path, &args[3..]);
        }
        Some("promote") => {
            let Some(source) = args.get(2) else {
                usage_exit("Usage: focalors promote <path-to-net.nnue>");
            };
            let dest = "nets/current.nnue";

            // Verify the source is a valid NNUE file before copying
            match std::fs::read(source) {
                Ok(bytes) => {
                    match nnue::network::Network::from_bytes(&bytes) {
                        Ok(_) => {
                            std::fs::copy(source, dest)
                                .unwrap_or_else(|e| panic!("Failed to copy '{source}' -> '{dest}': {e}"));
                            println!("Promoted {source} to {dest} ({} bytes)", bytes.len());
                            println!("Run `cargo build --release` to embed it in the binary.");
                        }
                        Err(e) => {
                            eprintln!("Error: '{source}' is not a valid NNUE net: {e}");
                            std::process::exit(1);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error: Failed to read '{source}': {e}");
                    std::process::exit(1);
                }
            }
        }
        Some("selfplay") => {
            const USAGE: &str =
                "Usage: focalors selfplay <num_games> <output_file> [--nnue <net_file>] [--depth N] [--threads N] [--random-plies N]";
            let Some(num_games) = args.get(2).and_then(|s| s.parse::<usize>().ok()) else {
                usage_exit(USAGE);
            };
            let Some(output) = args.get(3) else {
                usage_exit(USAGE);
            };

            // Optional --nnue flag for gen-2+ training
            let nnue_path = args.iter().position(|a| a == "--nnue")
                .and_then(|i| args.get(i + 1))
                .map(|s| s.as_str());

            let default_threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            let depth: u32 = parsed_flag(&args, "--depth", 6);
            let threads: usize = parsed_flag(&args, "--threads", default_threads);
            let random_plies: u32 = parsed_flag(&args, "--random-plies", 8);

            selfplay::run_selfplay(num_games, output, nnue_path, depth, threads, random_plies);
        }
        Some("selfmatch") => {
            const USAGE: &str =
                "Usage: focalors selfmatch <games> [--depth N] [--challenger-net PATH] [--seed N] [--random-plies N] [--max-moves N] [--threads N]";
            let Some(num_games) = args.get(2).and_then(|s| s.parse::<usize>().ok()) else {
                usage_exit(USAGE);
            };

            let challenger_net: Option<&str> = args.iter().position(|a| a == "--challenger-net")
                .and_then(|i| args.get(i + 1))
                .map(|s| s.as_str());

            let seed: Option<u64> = match args.iter().position(|a| a == "--seed") {
                None => None,
                Some(i) => match args.get(i + 1).and_then(|s| s.parse().ok()) {
                    Some(v) => Some(v),
                    // A typo'd seed silently becoming "no seed" would make
                    // a long validation run non-reproducible.
                    None => usage_exit("Error: --seed requires a number"),
                },
            };

            let default_threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            let depth: u32 = parsed_flag(&args, "--depth", 8);
            let random_plies: u32 = parsed_flag(&args, "--random-plies", 8);
            let max_moves: usize = parsed_flag(&args, "--max-moves", 200);
            let threads: usize = parsed_flag(&args, "--threads", default_threads);

            selfmatch::run_selfmatch(num_games, depth, challenger_net, seed, random_plies, max_moves, threads);
        }
        Some("uci") => {
            attacks::init();
            // NNUE init is deferred into uci_loop (first `go`) so that
            // `setoption name EvalFile` can install a custom net before
            // the embedded default claims the one-shot global slot.
            uci::uci_loop();
        }
        Some(other) => {
            eprintln!("Unknown mode: {other}");
            eprintln!("Usage: focalors [gui|uci|tune|selfplay|selfmatch|train|promote]");
            eprintln!("  gui                    — Desktop GUI for local play, review, and stats (default)");
            eprintln!("  uci                    — UCI protocol mode (for chess GUIs)");
            eprintln!("  tune <dataset>         — Texel tuning (HCE weight optimization)");
            eprintln!("  train <data> [opts]    — Train NNUE net from self-play data");
            eprintln!("  selfplay <games> <out> — Generate NNUE training data via self-play");
            eprintln!("  selfmatch <games> [opts] — Run focalors-vs-focalors match (elo delta + LOS)");
            eprintln!("  promote <net.nnue>     — Set a .nnue file as the shipped default");
            #[cfg(feature = "gpu-training")]
            eprintln!("  train-gpu <data> [opts] — Experimental GPU NNUE training (see BRANCH.md)");
            std::process::exit(1);
        }
    }
}
