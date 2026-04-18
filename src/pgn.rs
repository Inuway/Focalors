use crate::board::Board;
use crate::movegen::{generate_legal_moves, make_move};
use crate::moves::{Move, MoveFlag};
use crate::types::{Color, Piece, Square};

#[derive(Debug, Clone, Default)]
pub struct ParsedPgn {
    pub white: Option<String>,
    pub black: Option<String>,
    pub result: Option<String>,
    pub event: Option<String>,
    pub uci_moves: Vec<String>,
}

pub fn parse_pgn(text: &str) -> Result<ParsedPgn, String> {
    let mut parsed = ParsedPgn::default();
    let mut movetext = String::with_capacity(text.len());

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            if let Some((tag, value)) = parse_header(trimmed) {
                match tag.as_str() {
                    "White" => parsed.white = Some(value),
                    "Black" => parsed.black = Some(value),
                    "Result" => parsed.result = Some(value),
                    "Event" => parsed.event = Some(value),
                    _ => {}
                }
            }
        } else {
            movetext.push_str(line);
            movetext.push(' ');
        }
    }

    let cleaned = strip_annotations(&movetext);
    let mut board = Board::startpos();

    for (ply, raw_token) in cleaned.split_whitespace().enumerate() {
        let token = raw_token.trim_end_matches(['!', '?', '+', '#']);
        if token.is_empty() {
            continue;
        }
        if is_move_number(token) || is_result_token(token) {
            if is_result_token(token) && parsed.result.is_none() {
                parsed.result = Some(token.to_string());
            }
            continue;
        }
        let mv = san_to_move(&board, token).map_err(|e| {
            format!("Ply {}: couldn't parse SAN \"{token}\" — {e}", ply + 1)
        })?;
        parsed.uci_moves.push(mv.to_uci());
        make_move(&mut board, mv);
    }

    if parsed.uci_moves.is_empty() {
        return Err("No moves found in PGN".to_string());
    }
    Ok(parsed)
}

fn parse_header(line: &str) -> Option<(String, String)> {
    let inner = line.strip_prefix('[')?.strip_suffix(']')?;
    let (tag, rest) = inner.split_once(char::is_whitespace)?;
    let value = rest.trim().trim_matches('"');
    Some((tag.to_string(), value.to_string()))
}

fn strip_annotations(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut brace_depth: u32 = 0;
    let mut paren_depth: u32 = 0;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if brace_depth > 0 {
            if c == '}' {
                brace_depth -= 1;
            } else if c == '{' {
                brace_depth += 1;
            }
            continue;
        }
        if paren_depth > 0 {
            if c == ')' {
                paren_depth -= 1;
            } else if c == '(' {
                paren_depth += 1;
            }
            continue;
        }
        match c {
            '{' => brace_depth = 1,
            '(' => paren_depth = 1,
            ';' => {
                while let Some(&nx) = chars.peek() {
                    chars.next();
                    if nx == '\n' {
                        break;
                    }
                }
            }
            '$' => {
                while let Some(&nx) = chars.peek() {
                    if nx.is_ascii_digit() {
                        chars.next();
                    } else {
                        break;
                    }
                }
            }
            '0' if chars.peek() == Some(&'-') => {
                let snapshot: String = chars.clone().collect();
                if snapshot.starts_with("-0-0") {
                    out.push_str("O-O-O");
                    for _ in 0..4 { chars.next(); }
                } else if snapshot.starts_with("-0") {
                    out.push_str("O-O");
                    for _ in 0..2 { chars.next(); }
                } else {
                    out.push(c);
                }
            }
            _ => out.push(c),
        }
    }
    out
}

fn is_move_number(token: &str) -> bool {
    let stripped = token.trim_end_matches('.');
    !stripped.is_empty() && stripped.chars().all(|c| c.is_ascii_digit())
        && token.len() > stripped.len()
}

fn is_result_token(token: &str) -> bool {
    matches!(token, "1-0" | "0-1" | "1/2-1/2" | "*")
}

pub fn san_to_move(board: &Board, san: &str) -> Result<Move, String> {
    if san == "O-O" || san == "O-O-O" {
        let kingside = san == "O-O";
        let target_file: u8 = if kingside { 6 } else { 2 };
        let legal = generate_legal_moves(board);
        for i in 0..legal.len() {
            let mv = legal[i];
            if mv.flag() == MoveFlag::Castling && mv.to_sq().file() == target_file {
                return Ok(mv);
            }
        }
        return Err("castling not legal".to_string());
    }

    let mut body = san;
    let mut promo: Option<Piece> = None;
    if let Some(eq_idx) = body.find('=') {
        let promo_char = body[eq_idx + 1..]
            .chars()
            .next()
            .ok_or_else(|| "missing promotion piece".to_string())?;
        promo = Some(piece_from_char(promo_char)
            .ok_or_else(|| format!("unknown promotion piece '{promo_char}'"))?);
        body = &body[..eq_idx];
    }

    let first = body.chars().next().ok_or_else(|| "empty token".to_string())?;
    let (piece, rest) = if "KQRBN".contains(first) {
        (piece_from_char(first).unwrap(), &body[1..])
    } else {
        (Piece::Pawn, body)
    };

    let rest_no_x: String = rest.chars().filter(|&c| c != 'x').collect();
    if rest_no_x.len() < 2 {
        return Err("destination square missing".to_string());
    }
    let dest_str = &rest_no_x[rest_no_x.len() - 2..];
    let dest = Square::from_algebraic(dest_str)
        .ok_or_else(|| format!("bad destination '{dest_str}'"))?;
    let disambig = &rest_no_x[..rest_no_x.len() - 2];

    let mut want_file: Option<u8> = None;
    let mut want_rank: Option<u8> = None;
    for c in disambig.chars() {
        if ('a'..='h').contains(&c) {
            want_file = Some((c as u8) - b'a');
        } else if ('1'..='8').contains(&c) {
            want_rank = Some((c as u8) - b'1');
        }
    }

    let legal = generate_legal_moves(board);
    let us = board.side_to_move;
    let mut matches: Vec<Move> = Vec::new();
    for i in 0..legal.len() {
        let mv = legal[i];
        if mv.to_sq() != dest {
            continue;
        }
        let from = mv.from_sq();
        let Some((color, ptype)) = board.piece_on(from) else { continue; };
        if color != us || ptype != piece {
            continue;
        }
        match (promo, mv.flag()) {
            (Some(p), MoveFlag::Promotion) if mv.promotion_piece() != p => continue,
            (Some(_), flag) if flag != MoveFlag::Promotion => continue,
            (None, MoveFlag::Promotion) => continue,
            _ => {}
        }
        if let Some(f) = want_file {
            if from.file() != f { continue; }
        }
        if let Some(r) = want_rank {
            if from.rank() != r { continue; }
        }
        matches.push(mv);
    }

    match matches.len() {
        1 => Ok(matches[0]),
        0 => Err("no legal move matches".to_string()),
        _ => Err("ambiguous SAN".to_string()),
    }
}

fn piece_from_char(c: char) -> Option<Piece> {
    match c {
        'K' => Some(Piece::King),
        'Q' => Some(Piece::Queen),
        'R' => Some(Piece::Rook),
        'B' => Some(Piece::Bishop),
        'N' => Some(Piece::Knight),
        _ => None,
    }
}

pub fn user_color_from_headers(parsed: &ParsedPgn, profile_name: Option<&str>) -> Color {
    if let Some(name) = profile_name {
        let lower = name.to_lowercase();
        if parsed.white.as_deref().map(|s| s.to_lowercase()) == Some(lower.clone()) {
            return Color::White;
        }
        if parsed.black.as_deref().map(|s| s.to_lowercase()) == Some(lower) {
            return Color::Black;
        }
    }
    Color::White
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;

    static INIT: Once = Once::new();
    fn init() {
        INIT.call_once(crate::attacks::init);
    }

    #[test]
    fn parses_simple_game() {
        init();
        let pgn = "[White \"a\"]\n[Black \"b\"]\n[Result \"1-0\"]\n\n1. e4 e5 2. Nf3 Nc6 1-0";
        let p = parse_pgn(pgn).unwrap();
        assert_eq!(p.uci_moves, vec!["e2e4", "e7e5", "g1f3", "b8c6"]);
        assert_eq!(p.result.as_deref(), Some("1-0"));
    }

    #[test]
    fn handles_castling() {
        init();
        let pgn = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. O-O Be7 5. d3 O-O";
        let p = parse_pgn(pgn).unwrap();
        assert_eq!(p.uci_moves[6], "e1g1");
        assert_eq!(p.uci_moves.last().unwrap(), "e8g8");
    }

    #[test]
    fn strips_brace_comments_and_variations() {
        init();
        let pgn = "1. e4 {best by test} (1. d4 d5) 1... e5 2. Nf3";
        let p = parse_pgn(pgn).unwrap();
        assert_eq!(p.uci_moves, vec!["e2e4", "e7e5", "g1f3"]);
    }

    #[test]
    fn promotion_and_disambiguation() {
        init();
        // White pawn on a7 ready to promote; SAN resolver needs to handle =Q.
        let board = Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let mv = san_to_move(&board, "a8=Q").unwrap();
        assert_eq!(mv.to_uci(), "a7a8q");

        // Two knights both attack d2 — Nbd2 must pick the b1 knight, not the f3 one.
        let board = Board::from_fen("4k3/8/8/8/8/5N2/8/1N2K3 w - - 0 1").unwrap();
        let mv = san_to_move(&board, "Nbd2").unwrap();
        assert_eq!(mv.to_uci(), "b1d2");
    }

    #[test]
    fn rejects_garbage() {
        init();
        assert!(parse_pgn("1. zz5").is_err());
    }
}
