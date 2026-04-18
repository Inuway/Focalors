//! Opening repertoire detection via embedded ECO lookup table.
//!
//! Given a game's UCI move list, find the longest prefix match against a curated
//! table of common openings. Returns the ECO code and name.

// ════════════════════════════════════════════════════════════════════════════
// ECO lookup table
// ════════════════════════════════════════════════════════════════════════════

struct EcoEntry {
    moves: &'static str, // space-separated UCI moves
    eco: &'static str,
    name: &'static str,
}

/// Detect the opening from a game's UCI move list.
/// Returns `Some((eco, name))` for the longest matching prefix, or `None`.
pub fn detect_opening(uci_moves: &[String]) -> Option<(&'static str, &'static str)> {
    let game_str = uci_moves.join(" ");
    let mut best: Option<&EcoEntry> = None;
    let mut best_len = 0;

    for entry in ECO_TABLE {
        if game_str.starts_with(entry.moves)
            && entry.moves.len() > best_len
            // Ensure we match on a move boundary (next char is space or end)
            && (game_str.len() == entry.moves.len()
                || game_str.as_bytes().get(entry.moves.len()) == Some(&b' '))
        {
            best = Some(entry);
            best_len = entry.moves.len();
        }
    }

    best.map(|e| (e.eco, e.name))
}

// Sorted by move count descending so longest match wins naturally,
// but we check all entries anyway for correctness.
static ECO_TABLE: &[EcoEntry] = &[
    // ── King's Pawn (e4) ─────────────────────────────────────────────

    // Ruy Lopez variations
    EcoEntry { moves: "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7", eco: "C84", name: "Ruy Lopez: Closed" },
    EcoEntry { moves: "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 b7b5", eco: "C80", name: "Ruy Lopez: Open" },
    EcoEntry { moves: "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1", eco: "C88", name: "Ruy Lopez: Closed" },
    EcoEntry { moves: "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6", eco: "C78", name: "Ruy Lopez: Morphy Defense" },
    EcoEntry { moves: "e2e4 e7e5 g1f3 b8c6 f1b5 g8f6", eco: "C65", name: "Ruy Lopez: Berlin Defense" },
    EcoEntry { moves: "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4", eco: "C70", name: "Ruy Lopez: Morphy Defense" },
    EcoEntry { moves: "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6", eco: "C70", name: "Ruy Lopez: Morphy Defense" },
    EcoEntry { moves: "e2e4 e7e5 g1f3 b8c6 f1b5", eco: "C60", name: "Ruy Lopez" },

    // Italian Game variations
    EcoEntry { moves: "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3", eco: "C54", name: "Italian Game: Giuoco Piano" },
    EcoEntry { moves: "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6", eco: "C55", name: "Italian Game: Two Knights Defense" },
    EcoEntry { moves: "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5", eco: "C50", name: "Italian Game: Giuoco Piano" },
    EcoEntry { moves: "e2e4 e7e5 g1f3 b8c6 f1c4", eco: "C50", name: "Italian Game" },

    // Scotch Game
    EcoEntry { moves: "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4", eco: "C45", name: "Scotch Game" },
    EcoEntry { moves: "e2e4 e7e5 g1f3 b8c6 d2d4", eco: "C44", name: "Scotch Game" },

    // Four Knights
    EcoEntry { moves: "e2e4 e7e5 g1f3 b8c6 b1c3 g8f6", eco: "C47", name: "Four Knights Game" },

    // King's Gambit
    EcoEntry { moves: "e2e4 e7e5 f2f4 e5f4", eco: "C33", name: "King's Gambit Accepted" },
    EcoEntry { moves: "e2e4 e7e5 f2f4 f8c5", eco: "C30", name: "King's Gambit Declined" },
    EcoEntry { moves: "e2e4 e7e5 f2f4", eco: "C30", name: "King's Gambit" },

    // Vienna Game
    EcoEntry { moves: "e2e4 e7e5 b1c3 g8f6", eco: "C26", name: "Vienna Game" },
    EcoEntry { moves: "e2e4 e7e5 b1c3", eco: "C25", name: "Vienna Game" },

    // Philidor Defense
    EcoEntry { moves: "e2e4 e7e5 g1f3 d7d6", eco: "C41", name: "Philidor Defense" },

    // Petrov Defense
    EcoEntry { moves: "e2e4 e7e5 g1f3 g8f6 d2d4", eco: "C43", name: "Petrov Defense: Steinitz" },
    EcoEntry { moves: "e2e4 e7e5 g1f3 g8f6 f3e5", eco: "C42", name: "Petrov Defense: Classical" },
    EcoEntry { moves: "e2e4 e7e5 g1f3 g8f6", eco: "C42", name: "Petrov Defense" },

    // ── Sicilian Defense ─────────────────────────────────────────────

    // Najdorf
    EcoEntry { moves: "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6", eco: "B90", name: "Sicilian: Najdorf" },
    // Dragon
    EcoEntry { moves: "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 g7g6", eco: "B70", name: "Sicilian: Dragon" },
    // Classical
    EcoEntry { moves: "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 b8c6", eco: "B56", name: "Sicilian: Classical" },
    // Open Sicilian
    EcoEntry { moves: "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3", eco: "B50", name: "Sicilian: Open" },
    EcoEntry { moves: "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4", eco: "B50", name: "Sicilian: Open" },
    EcoEntry { moves: "e2e4 c7c5 g1f3 d7d6 d2d4", eco: "B50", name: "Sicilian: Open" },
    // Sveshnikov
    EcoEntry { moves: "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 e7e5", eco: "B33", name: "Sicilian: Sveshnikov" },
    // Scheveningen
    EcoEntry { moves: "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 e7e6", eco: "B80", name: "Sicilian: Scheveningen" },
    // Alapin
    EcoEntry { moves: "e2e4 c7c5 c2c3", eco: "B22", name: "Sicilian: Alapin" },
    // Closed Sicilian
    EcoEntry { moves: "e2e4 c7c5 b1c3", eco: "B23", name: "Sicilian: Closed" },
    // Smith-Morra Gambit
    EcoEntry { moves: "e2e4 c7c5 d2d4 c5d4 c2c3", eco: "B21", name: "Sicilian: Smith-Morra Gambit" },
    // Sicilian base
    EcoEntry { moves: "e2e4 c7c5 g1f3 b8c6 d2d4", eco: "B30", name: "Sicilian: Open" },
    EcoEntry { moves: "e2e4 c7c5 g1f3 e7e6", eco: "B40", name: "Sicilian: French Variation" },
    EcoEntry { moves: "e2e4 c7c5 g1f3 d7d6", eco: "B50", name: "Sicilian Defense" },
    EcoEntry { moves: "e2e4 c7c5 g1f3 b8c6", eco: "B30", name: "Sicilian Defense" },
    EcoEntry { moves: "e2e4 c7c5 g1f3", eco: "B27", name: "Sicilian Defense" },
    EcoEntry { moves: "e2e4 c7c5", eco: "B20", name: "Sicilian Defense" },

    // ── French Defense ───────────────────────────────────────────────

    EcoEntry { moves: "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 c1g5", eco: "C13", name: "French: Classical" },
    EcoEntry { moves: "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4", eco: "C15", name: "French: Winawer" },
    EcoEntry { moves: "e2e4 e7e6 d2d4 d7d5 b1c3 d5e4", eco: "C10", name: "French: Rubinstein" },
    EcoEntry { moves: "e2e4 e7e6 d2d4 d7d5 e4e5", eco: "C02", name: "French: Advance" },
    EcoEntry { moves: "e2e4 e7e6 d2d4 d7d5 b1d2", eco: "C06", name: "French: Tarrasch" },
    EcoEntry { moves: "e2e4 e7e6 d2d4 d7d5 b1c3", eco: "C10", name: "French Defense" },
    EcoEntry { moves: "e2e4 e7e6 d2d4 d7d5", eco: "C00", name: "French Defense" },
    EcoEntry { moves: "e2e4 e7e6", eco: "C00", name: "French Defense" },

    // ── Caro-Kann ────────────────────────────────────────────────────

    EcoEntry { moves: "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 b8d7", eco: "B17", name: "Caro-Kann: Steinitz" },
    EcoEntry { moves: "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5", eco: "B18", name: "Caro-Kann: Classical" },
    EcoEntry { moves: "e2e4 c7c6 d2d4 d7d5 e4e5", eco: "B12", name: "Caro-Kann: Advance" },
    EcoEntry { moves: "e2e4 c7c6 d2d4 d7d5 e4d5 c6d5", eco: "B13", name: "Caro-Kann: Exchange" },
    EcoEntry { moves: "e2e4 c7c6 d2d4 d7d5 b1c3", eco: "B15", name: "Caro-Kann" },
    EcoEntry { moves: "e2e4 c7c6 d2d4 d7d5", eco: "B12", name: "Caro-Kann" },
    EcoEntry { moves: "e2e4 c7c6", eco: "B10", name: "Caro-Kann" },

    // ── Pirc / Modern ────────────────────────────────────────────────

    EcoEntry { moves: "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6", eco: "B07", name: "Pirc Defense" },
    EcoEntry { moves: "e2e4 d7d6 d2d4 g8f6", eco: "B07", name: "Pirc Defense" },
    EcoEntry { moves: "e2e4 g7g6 d2d4 f8g7", eco: "B06", name: "Modern Defense" },
    EcoEntry { moves: "e2e4 g7g6", eco: "B06", name: "Modern Defense" },

    // ── Scandinavian ─────────────────────────────────────────────────

    EcoEntry { moves: "e2e4 d7d5 e4d5 d8d5", eco: "B01", name: "Scandinavian: Queen Recapture" },
    EcoEntry { moves: "e2e4 d7d5 e4d5 g8f6", eco: "B01", name: "Scandinavian: Marshall" },
    EcoEntry { moves: "e2e4 d7d5", eco: "B01", name: "Scandinavian Defense" },

    // ── Alekhine Defense ─────────────────────────────────────────────

    EcoEntry { moves: "e2e4 g8f6 e4e5 f6d5", eco: "B03", name: "Alekhine Defense" },
    EcoEntry { moves: "e2e4 g8f6", eco: "B02", name: "Alekhine Defense" },

    // ── Queen's Pawn (d4) ────────────────────────────────────────────

    // Queen's Gambit Declined
    EcoEntry { moves: "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7", eco: "D53", name: "QGD: Classical" },
    EcoEntry { moves: "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 g1f3", eco: "D37", name: "Queen's Gambit Declined" },
    EcoEntry { moves: "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6", eco: "D35", name: "Queen's Gambit Declined" },
    EcoEntry { moves: "d2d4 d7d5 c2c4 e7e6 b1c3", eco: "D31", name: "Queen's Gambit Declined" },
    EcoEntry { moves: "d2d4 d7d5 c2c4 e7e6 g1f3 g8f6", eco: "D37", name: "Queen's Gambit Declined" },
    EcoEntry { moves: "d2d4 d7d5 c2c4 e7e6", eco: "D30", name: "Queen's Gambit Declined" },

    // Queen's Gambit Accepted
    EcoEntry { moves: "d2d4 d7d5 c2c4 d5c4 g1f3 g8f6", eco: "D27", name: "QGA: Classical" },
    EcoEntry { moves: "d2d4 d7d5 c2c4 d5c4 e2e4", eco: "D20", name: "QGA: Central Variation" },
    EcoEntry { moves: "d2d4 d7d5 c2c4 d5c4", eco: "D20", name: "Queen's Gambit Accepted" },

    // Slav Defense
    EcoEntry { moves: "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 d5c4", eco: "D43", name: "Slav: Semi-Slav" },
    EcoEntry { moves: "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 e7e6", eco: "D43", name: "Slav: Semi-Slav" },
    EcoEntry { moves: "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6", eco: "D10", name: "Slav Defense" },
    EcoEntry { moves: "d2d4 d7d5 c2c4 c7c6", eco: "D10", name: "Slav Defense" },

    // Queen's Gambit base
    EcoEntry { moves: "d2d4 d7d5 c2c4", eco: "D06", name: "Queen's Gambit" },

    // ── Indian Defenses ──────────────────────────────────────────────

    // King's Indian Defense
    EcoEntry { moves: "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8", eco: "E90", name: "King's Indian: Classical" },
    EcoEntry { moves: "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 f2f3", eco: "E81", name: "King's Indian: Samisch" },
    EcoEntry { moves: "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6", eco: "E70", name: "King's Indian Defense" },
    EcoEntry { moves: "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7", eco: "E60", name: "King's Indian Defense" },
    EcoEntry { moves: "d2d4 g8f6 c2c4 g7g6 b1c3", eco: "E60", name: "King's Indian Defense" },
    EcoEntry { moves: "d2d4 g8f6 c2c4 g7g6", eco: "E60", name: "King's Indian Defense" },

    // Nimzo-Indian Defense
    EcoEntry { moves: "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 d1c2", eco: "E32", name: "Nimzo-Indian: Classical" },
    EcoEntry { moves: "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3", eco: "E40", name: "Nimzo-Indian: Rubinstein" },
    EcoEntry { moves: "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4", eco: "E20", name: "Nimzo-Indian Defense" },

    // Queen's Indian Defense
    EcoEntry { moves: "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6", eco: "E15", name: "Queen's Indian Defense" },

    // Bogo-Indian Defense
    EcoEntry { moves: "d2d4 g8f6 c2c4 e7e6 g1f3 f8b4", eco: "E11", name: "Bogo-Indian Defense" },

    // Catalan Opening
    EcoEntry { moves: "d2d4 g8f6 c2c4 e7e6 g2g3 d7d5", eco: "E01", name: "Catalan Opening" },
    EcoEntry { moves: "d2d4 g8f6 c2c4 e7e6 g2g3", eco: "E01", name: "Catalan Opening" },

    // Indian base
    EcoEntry { moves: "d2d4 g8f6 c2c4 e7e6 g1f3", eco: "E10", name: "Indian Defense" },
    EcoEntry { moves: "d2d4 g8f6 c2c4 e7e6", eco: "E00", name: "Indian Defense" },
    EcoEntry { moves: "d2d4 g8f6 c2c4", eco: "A45", name: "Indian Defense" },

    // Grunfeld Defense
    EcoEntry { moves: "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5 f6d5", eco: "D85", name: "Grunfeld Defense: Exchange" },
    EcoEntry { moves: "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5", eco: "D80", name: "Grunfeld Defense" },

    // Benoni Defense
    EcoEntry { moves: "d2d4 g8f6 c2c4 c7c5 d4d5 e7e6", eco: "A60", name: "Benoni Defense" },
    EcoEntry { moves: "d2d4 g8f6 c2c4 c7c5 d4d5", eco: "A56", name: "Benoni Defense" },
    EcoEntry { moves: "d2d4 g8f6 c2c4 c7c5", eco: "A56", name: "Benoni Defense" },

    // Dutch Defense
    EcoEntry { moves: "d2d4 f7f5 c2c4 g8f6 g2g3 e7e6", eco: "A81", name: "Dutch: Leningrad" },
    EcoEntry { moves: "d2d4 f7f5 c2c4 g8f6", eco: "A80", name: "Dutch Defense" },
    EcoEntry { moves: "d2d4 f7f5", eco: "A80", name: "Dutch Defense" },

    // ── London / System Openings ─────────────────────────────────────

    EcoEntry { moves: "d2d4 d7d5 c1f4", eco: "D00", name: "London System" },
    EcoEntry { moves: "d2d4 g8f6 c1f4", eco: "D00", name: "London System" },
    EcoEntry { moves: "d2d4 d7d5 g1f3 g8f6 c1f4", eco: "D00", name: "London System" },

    // Torre Attack
    EcoEntry { moves: "d2d4 g8f6 g1f3 e7e6 c1g5", eco: "A46", name: "Torre Attack" },

    // Colle System
    EcoEntry { moves: "d2d4 d7d5 g1f3 g8f6 e2e3", eco: "D05", name: "Colle System" },

    // ── Flank Openings ───────────────────────────────────────────────

    // English Opening
    EcoEntry { moves: "c2c4 e7e5 b1c3 g8f6", eco: "A22", name: "English: Reversed Sicilian" },
    EcoEntry { moves: "c2c4 e7e5 g2g3", eco: "A20", name: "English Opening" },
    EcoEntry { moves: "c2c4 g8f6 b1c3 e7e5", eco: "A22", name: "English: Reversed Sicilian" },
    EcoEntry { moves: "c2c4 c7c5", eco: "A30", name: "English: Symmetrical" },
    EcoEntry { moves: "c2c4 e7e5", eco: "A20", name: "English Opening" },
    EcoEntry { moves: "c2c4 g8f6", eco: "A15", name: "English Opening" },
    EcoEntry { moves: "c2c4", eco: "A10", name: "English Opening" },

    // Reti Opening
    EcoEntry { moves: "g1f3 d7d5 g2g3 g8f6", eco: "A05", name: "Reti Opening" },
    EcoEntry { moves: "g1f3 d7d5 c2c4", eco: "A09", name: "Reti Opening" },
    EcoEntry { moves: "g1f3 d7d5", eco: "A04", name: "Reti Opening" },
    EcoEntry { moves: "g1f3 g8f6", eco: "A04", name: "Reti Opening" },
    EcoEntry { moves: "g1f3", eco: "A04", name: "Reti Opening" },

    // Bird's Opening
    EcoEntry { moves: "f2f4 d7d5", eco: "A02", name: "Bird's Opening" },
    EcoEntry { moves: "f2f4", eco: "A02", name: "Bird's Opening" },

    // ── Base moves (catch-all) ───────────────────────────────────────

    EcoEntry { moves: "d2d4 d7d5 g1f3 g8f6", eco: "D02", name: "Queen's Pawn Game" },
    EcoEntry { moves: "d2d4 d7d5 g1f3", eco: "D02", name: "Queen's Pawn Game" },
    EcoEntry { moves: "d2d4 d7d5", eco: "D00", name: "Queen's Pawn Game" },
    EcoEntry { moves: "d2d4 g8f6 g1f3", eco: "A46", name: "Indian Defense" },
    EcoEntry { moves: "d2d4 g8f6", eco: "A45", name: "Indian Defense" },
    EcoEntry { moves: "d2d4", eco: "A40", name: "Queen's Pawn Game" },
    EcoEntry { moves: "e2e4 e7e5 g1f3 b8c6", eco: "C40", name: "King's Knight Opening" },
    EcoEntry { moves: "e2e4 e7e5 g1f3", eco: "C40", name: "King's Knight Opening" },
    EcoEntry { moves: "e2e4 e7e5", eco: "C20", name: "King's Pawn Game" },
    EcoEntry { moves: "e2e4", eco: "B00", name: "King's Pawn Game" },
];

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_sicilian_najdorf() {
        let moves: Vec<String> = "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1e3"
            .split_whitespace().map(String::from).collect();
        let (eco, name) = detect_opening(&moves).unwrap();
        assert_eq!(eco, "B90");
        assert_eq!(name, "Sicilian: Najdorf");
    }

    #[test]
    fn detect_italian_game() {
        let moves: Vec<String> = "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 d2d3"
            .split_whitespace().map(String::from).collect();
        let (eco, name) = detect_opening(&moves).unwrap();
        assert_eq!(eco, "C50");
        assert!(name.contains("Italian"));
    }

    #[test]
    fn detect_queens_gambit_declined() {
        let moves: Vec<String> = "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5"
            .split_whitespace().map(String::from).collect();
        let (_, name) = detect_opening(&moves).unwrap();
        assert!(name.contains("Queen's Gambit Declined"));
    }

    #[test]
    fn detect_kings_pawn_base() {
        let moves: Vec<String> = "e2e4 b7b6".split_whitespace().map(String::from).collect();
        let (eco, _) = detect_opening(&moves).unwrap();
        assert_eq!(eco, "B00"); // only e4 matches
    }

    #[test]
    fn detect_london_system() {
        let moves: Vec<String> = "d2d4 d7d5 c1f4 g8f6 e2e3"
            .split_whitespace().map(String::from).collect();
        let (_, name) = detect_opening(&moves).unwrap();
        assert_eq!(name, "London System");
    }

    #[test]
    fn empty_game_returns_none() {
        let moves: Vec<String> = vec![];
        assert!(detect_opening(&moves).is_none());
    }

    #[test]
    fn single_random_move() {
        let moves: Vec<String> = vec!["a2a3".into()];
        assert!(detect_opening(&moves).is_none());
    }

    #[test]
    fn longest_match_wins() {
        // This should match Ruy Lopez, not just King's Knight Opening
        let moves: Vec<String> = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6"
            .split_whitespace().map(String::from).collect();
        let (eco, name) = detect_opening(&moves).unwrap();
        assert!(name.contains("Ruy Lopez"));
        assert_eq!(eco, "C78");
    }
}
