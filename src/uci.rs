use std::io::{self, BufRead};

use crate::board::Board;
use crate::movegen::{generate_legal_moves, make_move};
use crate::moves::Move;
use crate::nnue;
use crate::search::{SearchResult, Searcher};
use crate::types::{Color, Square};

/// What the "go" command asked for.
enum SearchLimit {
    Depth(u32),
    MoveTime(u64),
    Clock { our_time: u64, our_inc: u64, moves_to_go: Option<u32> },
    Infinite,
}

pub fn uci_loop() {
    let nnue_available = nnue::network::get_network().is_some();

    let stdin = io::stdin();
    let mut board = Board::startpos();
    let mut position_hashes: Vec<u64> = vec![board.hash];
    let mut searcher = Searcher::new(64);
    searcher.use_nnue = nnue_available;

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() { continue; }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens[0] {
            "uci" => {
                println!("id name Focalors 0.3.0");
                println!("id author mPmprz");
                println!("option name UseNNUE type check default {nnue_available}");
                println!("option name EvalFile type string default <internal>");
                println!("uciok");
            }
            "isready" => println!("readyok"),
            "setoption" => {
                // setoption name <name> value <value>
                let name_idx = tokens.iter().position(|&t| t == "name");
                let value_idx = tokens.iter().position(|&t| t == "value");
                if let (Some(ni), Some(vi)) = (name_idx, value_idx) {
                    let name = tokens[ni + 1..vi].join(" ");
                    let value = tokens[vi + 1..].join(" ");
                    match name.to_lowercase().as_str() {
                        "usennue" => {
                            searcher.use_nnue = value == "true" && nnue_available;
                        }
                        "evalfile" => {
                            if value != "<internal>" {
                                match nnue::init(Some(&value)) {
                                    Ok(()) => {
                                        searcher.use_nnue = true;
                                        eprintln!("info string Loaded NNUE net from {value}");
                                    }
                                    Err(e) => eprintln!("info string Failed to load net: {e}"),
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            "ucinewgame" => {
                board = Board::startpos();
                position_hashes = vec![board.hash];
                searcher.tt.clear();
            }
            "position" => {
                let (b, hashes) = parse_position_with_hashes(&tokens);
                board = b;
                position_hashes = hashes;
            }
            "go" => {
                searcher.set_position_history(position_hashes.clone());
                let result = execute_go(&mut searcher, &board, &tokens);
                println!(
                    "info depth {} score cp {} nodes {}",
                    result.depth, result.score, result.nodes
                );
                println!("bestmove {}", result.best_move);
            }
            "quit" => break,
            "d" | "display" => {
                println!("{board}");
                println!("FEN: {}", board.to_fen());
            }
            "perft" => {
                if let Some(d) = tokens.get(1).and_then(|s| s.parse::<u32>().ok()) {
                    println!("Nodes searched: {}", perft(&board, d));
                }
            }
            _ => {}
        }
    }
}

fn execute_go(searcher: &mut Searcher, board: &Board, tokens: &[&str]) -> SearchResult {
    let limit = parse_go(board, tokens);
    match limit {
        SearchLimit::Depth(d) => searcher.search(board, d),
        SearchLimit::MoveTime(ms) => searcher.search_timed(board, ms),
        SearchLimit::Clock { our_time, our_inc, moves_to_go } => {
            let (soft, hard) = allocate_time(our_time, our_inc, moves_to_go);
            searcher.search_with_time_management(board, soft, hard)
        }
        SearchLimit::Infinite => searcher.search(board, 64),
    }
}

/// Allocate time for a single move. Returns (soft_limit_ms, hard_limit_ms).
/// Soft limit: normal allocation, engine may stop early if move is stable.
/// Hard limit: absolute maximum, never exceeded.
pub fn allocate_time(time_remaining: u64, increment: u64, moves_to_go: Option<u32>) -> (u64, u64) {
    let mtg = moves_to_go.unwrap_or(30) as u64;
    let mtg = mtg.max(1);

    let base = time_remaining / mtg;
    let inc_use = increment * 7 / 10;
    let allocated = base + inc_use;

    let max_soft = time_remaining * 3 / 10;
    let safe_max = time_remaining.saturating_sub(50);

    let soft = allocated.min(max_soft).min(safe_max);
    let soft = if soft == 0 { time_remaining.max(1) } else { soft.max(10) };

    // Hard limit: 3x soft, capped at 50% of remaining time
    let hard = (soft * 3).min(time_remaining / 2).min(safe_max).max(soft);

    (soft, hard)
}

fn parse_go(board: &Board, tokens: &[&str]) -> SearchLimit {
    let mut depth = None;
    let mut movetime = None;
    let mut wtime = None;
    let mut btime = None;
    let mut winc = 0u64;
    let mut binc = 0u64;
    let mut movestogo = None;
    let mut infinite = false;

    let mut i = 1;
    while i < tokens.len() {
        match tokens[i] {
            "depth" => { depth = tokens.get(i + 1).and_then(|s| s.parse().ok()); i += 2; }
            "movetime" => { movetime = tokens.get(i + 1).and_then(|s| s.parse().ok()); i += 2; }
            "wtime" => { wtime = tokens.get(i + 1).and_then(|s| s.parse().ok()); i += 2; }
            "btime" => { btime = tokens.get(i + 1).and_then(|s| s.parse().ok()); i += 2; }
            "winc" => { winc = tokens.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0); i += 2; }
            "binc" => { binc = tokens.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0); i += 2; }
            "movestogo" => { movestogo = tokens.get(i + 1).and_then(|s| s.parse().ok()); i += 2; }
            "infinite" => { infinite = true; i += 1; }
            _ => { i += 1; }
        }
    }

    if let Some(d) = depth {
        return SearchLimit::Depth(d);
    }
    if infinite {
        return SearchLimit::Infinite;
    }
    if let Some(ms) = movetime {
        return SearchLimit::MoveTime(ms);
    }

    // Clock-based time control
    let (our_time, our_inc) = match board.side_to_move {
        Color::White => (wtime.unwrap_or(60000), winc),
        Color::Black => (btime.unwrap_or(60000), binc),
    };

    SearchLimit::Clock { our_time, our_inc, moves_to_go: movestogo }
}

fn parse_position_with_hashes(tokens: &[&str]) -> (Board, Vec<u64>) {
    if tokens.len() < 2 {
        let b = Board::startpos();
        let h = vec![b.hash];
        return (b, h);
    }

    let (mut board, move_start_idx) = if tokens[1] == "startpos" {
        (Board::startpos(), 2)
    } else if tokens[1] == "fen" {
        let fen_end = tokens.iter().position(|&t| t == "moves").unwrap_or(tokens.len());
        let fen_str = tokens[2..fen_end].join(" ");
        (Board::from_fen(&fen_str).unwrap_or_else(|_| Board::startpos()), fen_end)
    } else {
        let b = Board::startpos();
        let h = vec![b.hash];
        return (b, h);
    };

    let mut hashes = vec![board.hash];

    if move_start_idx < tokens.len() && tokens[move_start_idx] == "moves" {
        for &mv_str in &tokens[move_start_idx + 1..] {
            if let Some(mv) = parse_uci_move(&board, mv_str) {
                make_move(&mut board, mv);
                hashes.push(board.hash);
            }
        }
    }

    (board, hashes)
}

fn parse_uci_move(board: &Board, mv_str: &str) -> Option<Move> {
    let bytes = mv_str.as_bytes();
    if bytes.len() < 4 { return None; }

    let from = Square::from_algebraic(&mv_str[0..2])?;
    let to = Square::from_algebraic(&mv_str[2..4])?;

    let moves = generate_legal_moves(board);
    for i in 0..moves.len() {
        let mv = moves[i];
        if mv.from_sq().0 != from.0 || mv.to_sq().0 != to.0 { continue; }

        if bytes.len() >= 5 {
            if matches!(mv.flag(), crate::moves::MoveFlag::Promotion) {
                let promo_char = bytes[4] as char;
                let expected = match mv.promotion_piece() {
                    crate::types::Piece::Queen => 'q', crate::types::Piece::Rook => 'r',
                    crate::types::Piece::Bishop => 'b', crate::types::Piece::Knight => 'n',
                    _ => return None,
                };
                if promo_char == expected { return Some(mv); }
                continue;
            }
        } else if matches!(mv.flag(), crate::moves::MoveFlag::Promotion) {
            if matches!(mv.promotion_piece(), crate::types::Piece::Queen) { return Some(mv); }
            continue;
        }
        return Some(mv);
    }
    None
}

fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 { return 1; }
    let moves = generate_legal_moves(board);
    if depth == 1 { return moves.len() as u64; }
    let mut nodes = 0u64;
    for i in 0..moves.len() {
        let mut clone = board.clone();
        make_move(&mut clone, moves[i]);
        nodes += perft(&clone, depth - 1);
    }
    nodes
}


pub fn parse_move(board: &Board, mv_str: &str) -> Option<Move> {
    parse_uci_move(board, mv_str)
}
