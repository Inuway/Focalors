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
mod selfplay;
mod strength;
mod trainer;
mod tt;
mod tuning;
mod types;
mod uci;
mod zobrist;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("gui") => {
            // Desktop GUI mode
            gui::run_gui();
        }
        Some("tune") => {
            let dataset = args.get(2).expect("Usage: focalors tune <dataset_file>");
            tuning::run_tuning(dataset);
        }
        Some("train") => {
            let data_path = args.get(2).expect("Usage: focalors train <data_file> [options]");
            trainer::run_training(data_path, &args[3..]);
        }
        Some("promote") => {
            let source = args.get(2).expect("Usage: focalors promote <path-to-net.nnue>");
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
            let num_games: usize = args.get(2)
                .expect("Usage: focalors selfplay <num_games> <output_file> [--nnue <net_file>]")
                .parse()
                .expect("num_games must be a number");
            let output = args.get(3).expect("Usage: focalors selfplay <num_games> <output_file> [--nnue <net_file>]");

            // Optional --nnue flag for gen-2+ training
            let nnue_path = args.iter().position(|a| a == "--nnue")
                .and_then(|i| args.get(i + 1))
                .map(|s| s.as_str());

            // Optional --depth flag (default 6)
            let depth: u32 = args.iter().position(|a| a == "--depth")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse().ok())
                .unwrap_or(6);

            // Optional --threads flag (default: available CPU parallelism)
            let threads: usize = args.iter().position(|a| a == "--threads")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| {
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(1)
                });

            // Optional --random-plies flag (default 8; creates opening diversity)
            let random_plies: u32 = args.iter().position(|a| a == "--random-plies")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse().ok())
                .unwrap_or(8);

            selfplay::run_selfplay(num_games, output, nnue_path, depth, threads, random_plies);
        }
        Some("uci") | None => {
            attacks::init();
            // Initialize NNUE (will use embedded net or fall back to HCE)
            match nnue::init(None) {
                Ok(()) => eprintln!("info string NNUE initialized"),
                Err(_) => eprintln!("info string NNUE not available, using HCE"),
            }
            uci::uci_loop();
        }
        Some(other) => {
            eprintln!("Unknown mode: {other}");
            eprintln!("Usage: focalors [uci|gui|tune|selfplay|train|promote]");
            eprintln!("  uci                    — UCI protocol mode (default, for chess GUIs)");
            eprintln!("  gui                    — Desktop GUI for local play, review, and stats");
            eprintln!("  tune <dataset>         — Texel tuning (HCE weight optimization)");
            eprintln!("  train <data> [opts]    — Train NNUE net from self-play data");
            eprintln!("  selfplay <games> <out> — Generate NNUE training data via self-play");
            eprintln!("  promote <net.nnue>     — Set a .nnue file as the shipped default");
            std::process::exit(1);
        }
    }
}
