use rusqlite::{params, Connection, Result as SqlResult};
use std::path::PathBuf;

// ════════════════════════════════════════════════════════════════════════════
// Data types
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct UserProfile {
    pub name: String,
    pub rating: i32,
    pub games_played: i32,
}

#[derive(Debug, Clone)]
pub struct SavedGame {
    pub id: i64,
    pub played_at: String,
    pub user_color: String,
    pub result: String,
    pub result_reason: Option<String>,
    pub time_control: Option<String>,
    pub engine_level: Option<String>,
    pub pgn: String,
    pub move_count: Option<i32>,
}

#[derive(Debug, Clone)]
pub struct SavedPuzzle {
    pub id: i64,
    pub fen: String,
    pub solution: String,
    pub theme: Option<String>,
    pub rating: Option<i32>,
}

// ════════════════════════════════════════════════════════════════════════════
// Database handle
// ════════════════════════════════════════════════════════════════════════════

pub struct Database {
    conn: Connection,
}

impl Database {
    /// Open (or create) the database. Uses the platform data directory:
    ///   Linux:   ~/.local/share/focalors/focalors.db
    ///   macOS:   ~/Library/Application Support/focalors/focalors.db
    ///   Windows: C:\Users\<user>\AppData\Roaming\focalors\focalors.db
    ///
    /// Falls back to `./focalors.db` next to the binary if the data dir isn't available.
    pub fn open() -> SqlResult<Self> {
        let path = db_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        eprintln!("Database: {}", path.display());
        let conn = Connection::open(&path)?;
        let db = Database { conn };
        db.init_schema()?;
        db.migrate_phase4()?;
        Ok(db)
    }

    fn init_schema(&self) -> SqlResult<()> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS user_profile (
                id           INTEGER PRIMARY KEY DEFAULT 1,
                name         TEXT    NOT NULL DEFAULT 'Player',
                rating       INTEGER NOT NULL DEFAULT 1200,
                games_played INTEGER NOT NULL DEFAULT 0,
                created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at   TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS games (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                played_at      TEXT    NOT NULL DEFAULT (datetime('now')),
                user_color     TEXT    NOT NULL CHECK(user_color IN ('white', 'black')),
                result         TEXT    NOT NULL CHECK(result IN ('win', 'loss', 'draw')),
                result_reason  TEXT,
                time_control   TEXT,
                engine_level   TEXT,
                pgn            TEXT    NOT NULL,
                move_count     INTEGER,
                user_accuracy  REAL,
                opening_name   TEXT
            );

            CREATE TABLE IF NOT EXISTS move_analysis (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id         INTEGER NOT NULL REFERENCES games(id),
                move_number     INTEGER NOT NULL,
                side            TEXT    NOT NULL CHECK(side IN ('white', 'black')),
                move_uci        TEXT    NOT NULL,
                move_san        TEXT,
                eval_before     INTEGER,
                eval_after      INTEGER,
                best_move       TEXT,
                best_eval       INTEGER,
                classification  TEXT,
                eval_components TEXT
            );

            CREATE TABLE IF NOT EXISTS puzzles (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id    INTEGER REFERENCES games(id),
                fen        TEXT    NOT NULL,
                solution   TEXT    NOT NULL,
                theme      TEXT,
                rating     INTEGER,
                attempts   INTEGER DEFAULT 0,
                solved     INTEGER DEFAULT 0,
                created_at TEXT    NOT NULL DEFAULT (datetime('now'))
            );
            ",
        )
    }

    // ── User Profile ───────────────────────────────────────────────────

    /// Get the user profile, creating a default one if none exists.
    pub fn get_or_create_profile(&self) -> SqlResult<UserProfile> {
        let exists: bool = self.conn.query_row(
            "SELECT COUNT(*) > 0 FROM user_profile WHERE id = 1",
            [],
            |row| row.get(0),
        )?;

        if !exists {
            self.conn.execute(
                "INSERT INTO user_profile (id, name, rating, games_played) VALUES (1, 'Player', 1200, 0)",
                [],
            )?;
        }

        self.conn.query_row(
            "SELECT name, rating, games_played FROM user_profile WHERE id = 1",
            [],
            |row| {
                Ok(UserProfile {
                    name: row.get(0)?,
                    rating: row.get(1)?,
                    games_played: row.get(2)?,
                })
            },
        )
    }

    /// Update the user's display name and rating.
    pub fn update_profile(&self, name: &str, rating: i32) -> SqlResult<()> {
        self.conn.execute(
            "UPDATE user_profile SET name = ?1, rating = ?2, updated_at = datetime('now') WHERE id = 1",
            params![name, rating],
        )?;
        Ok(())
    }

    /// Increment games_played counter.
    pub fn increment_games_played(&self) -> SqlResult<()> {
        self.conn.execute(
            "UPDATE user_profile SET games_played = games_played + 1, updated_at = datetime('now') WHERE id = 1",
            [],
        )?;
        Ok(())
    }

    // ── Games ──────────────────────────────────────────────────────────

    /// Save a completed game. Returns the new game ID.
    pub fn save_game(
        &self,
        user_color: &str,
        result: &str,
        result_reason: Option<&str>,
        time_control: Option<&str>,
        engine_level: Option<&str>,
        pgn: &str,
        move_count: Option<i32>,
    ) -> SqlResult<i64> {
        self.conn.execute(
            "INSERT INTO games (user_color, result, result_reason, time_control, engine_level, pgn, move_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                user_color,
                result,
                result_reason,
                time_control,
                engine_level,
                pgn,
                move_count,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Get recent games, newest first. Limit 0 = all.
    pub fn get_recent_games(&self, limit: u32) -> SqlResult<Vec<SavedGame>> {
        let sql = if limit > 0 {
            format!(
                "SELECT id, played_at, user_color, result, result_reason, time_control,
                        engine_level, pgn, move_count
                 FROM games ORDER BY played_at DESC LIMIT {limit}"
            )
        } else {
            "SELECT id, played_at, user_color, result, result_reason, time_control,
                    engine_level, pgn, move_count
             FROM games ORDER BY played_at DESC"
                .to_string()
        };

        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([], |row| {
            Ok(SavedGame {
                id: row.get(0)?,
                played_at: row.get(1)?,
                user_color: row.get(2)?,
                result: row.get(3)?,
                result_reason: row.get(4)?,
                time_control: row.get(5)?,
                engine_level: row.get(6)?,
                pgn: row.get(7)?,
                move_count: row.get(8)?,
            })
        })?;

        rows.collect()
    }

    /// Get a single game by ID.
    pub fn get_game(&self, game_id: i64) -> SqlResult<SavedGame> {
        self.conn.query_row(
            "SELECT id, played_at, user_color, result, result_reason, time_control,
                    engine_level, pgn, move_count
             FROM games WHERE id = ?1",
            params![game_id],
            |row| {
                Ok(SavedGame {
                    id: row.get(0)?,
                    played_at: row.get(1)?,
                    user_color: row.get(2)?,
                    result: row.get(3)?,
                    result_reason: row.get(4)?,
                    time_control: row.get(5)?,
                    engine_level: row.get(6)?,
                    pgn: row.get(7)?,
                    move_count: row.get(8)?,
                })
            },
        )
    }

    // ── Phase 4: Adaptive Difficulty ─────────────────────────────────

    /// Schema migration for Phase 4. Adds columns if they don't already exist.
    /// ALTER TABLE ... ADD COLUMN is a no-op error if the column exists — we
    /// just ignore those errors.
    fn migrate_phase4(&self) -> SqlResult<()> {
        // user_profile additions
        let _ = self.conn.execute(
            "ALTER TABLE user_profile ADD COLUMN rating_games INTEGER NOT NULL DEFAULT 0",
            [],
        );
        let _ = self.conn.execute(
            "ALTER TABLE user_profile ADD COLUMN adaptive_level INTEGER NOT NULL DEFAULT 10",
            [],
        );
        // games table additions
        let _ = self.conn.execute(
            "ALTER TABLE games ADD COLUMN engine_numeric_level INTEGER",
            [],
        );
        let _ = self.conn.execute(
            "ALTER TABLE games ADD COLUMN rating_before INTEGER",
            [],
        );
        let _ = self.conn.execute(
            "ALTER TABLE games ADD COLUMN rating_after INTEGER",
            [],
        );
        Ok(())
    }

    /// Update the user's rating after a game using Elo formula.
    /// Returns (old_rating, new_rating).
    pub fn update_rating_after_game(
        &self,
        result: &str,
        engine_elo: i32,
    ) -> SqlResult<(i32, i32)> {
        let (old_rating, rating_games): (i32, u32) = self.conn.query_row(
            "SELECT rating, rating_games FROM user_profile WHERE id = 1",
            [],
            |row| Ok((row.get(0)?, row.get::<_, u32>(1)?)),
        )?;

        let score = match result {
            "win" => 1.0,
            "draw" => 0.5,
            _ => 0.0,
        };

        let k = crate::strength::k_factor(rating_games);
        let new_rating = crate::strength::elo_update(old_rating, engine_elo, score, k);

        self.conn.execute(
            "UPDATE user_profile SET rating = ?1, rating_games = rating_games + 1, updated_at = datetime('now') WHERE id = 1",
            params![new_rating],
        )?;

        Ok((old_rating, new_rating))
    }

    /// Save the rating snapshot to the most recent game row.
    pub fn update_game_rating(
        &self,
        game_id: i64,
        engine_numeric_level: u32,
        rating_before: i32,
        rating_after: i32,
    ) -> SqlResult<()> {
        self.conn.execute(
            "UPDATE games SET engine_numeric_level = ?1, rating_before = ?2, rating_after = ?3 WHERE id = ?4",
            params![engine_numeric_level as i32, rating_before, rating_after, game_id],
        )?;
        Ok(())
    }

    /// Get recent game results (for auto-adjust). Returns list of result strings
    /// ("win"/"loss"/"draw"), newest first.
    pub fn get_recent_results(&self, n: u32) -> SqlResult<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT result FROM games ORDER BY played_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![n], |row| row.get::<_, String>(0))?;
        rows.collect()
    }

    /// Get the adaptive difficulty level from the profile.
    pub fn get_adaptive_level(&self) -> SqlResult<u32> {
        self.conn.query_row(
            "SELECT adaptive_level FROM user_profile WHERE id = 1",
            [],
            |row| row.get::<_, u32>(0),
        )
    }

    /// Set the adaptive difficulty level.
    pub fn set_adaptive_level(&self, level: u32) -> SqlResult<()> {
        self.conn.execute(
            "UPDATE user_profile SET adaptive_level = ?1, updated_at = datetime('now') WHERE id = 1",
            params![level],
        )?;
        Ok(())
    }

    /// Get win/loss/draw counts.
    pub fn get_result_counts(&self) -> SqlResult<(i32, i32, i32)> {
        let wins: i32 = self.conn.query_row(
            "SELECT COUNT(*) FROM games WHERE result = 'win'",
            [],
            |row| row.get(0),
        )?;
        let losses: i32 = self.conn.query_row(
            "SELECT COUNT(*) FROM games WHERE result = 'loss'",
            [],
            |row| row.get(0),
        )?;
        let draws: i32 = self.conn.query_row(
            "SELECT COUNT(*) FROM games WHERE result = 'draw'",
            [],
            |row| row.get(0),
        )?;
        Ok((wins, losses, draws))
    }

    // ── Puzzles (Phase 5) ─────────────────────────────────────────────

    /// Save a puzzle extracted from post-game analysis. Returns the new puzzle ID.
    pub fn save_puzzle(
        &self,
        game_id: Option<i64>,
        fen: &str,
        solution: &str,
        theme: &str,
        rating: i32,
    ) -> SqlResult<i64> {
        self.conn.execute(
            "INSERT INTO puzzles (game_id, fen, solution, theme, rating) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![game_id, fen, solution, theme, rating],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Get training puzzles, prioritizing themes the user struggles with.
    /// Returns unsolved or low-success puzzles first.
    pub fn get_training_puzzles(&self, limit: u32) -> SqlResult<Vec<SavedPuzzle>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, fen, solution, theme, rating
             FROM puzzles
             ORDER BY
                CASE WHEN attempts = 0 THEN 0 ELSE CAST(solved AS REAL) / attempts END ASC,
                attempts ASC
             LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit], |row| {
            Ok(SavedPuzzle {
                id: row.get(0)?,
                fen: row.get(1)?,
                solution: row.get(2)?,
                theme: row.get(3)?,
                rating: row.get(4)?,
            })
        })?;
        rows.collect()
    }

    /// Record a puzzle attempt.
    pub fn record_puzzle_attempt(&self, puzzle_id: i64, solved: bool) -> SqlResult<()> {
        if solved {
            self.conn.execute(
                "UPDATE puzzles SET attempts = attempts + 1, solved = solved + 1 WHERE id = ?1",
                params![puzzle_id],
            )?;
        } else {
            self.conn.execute(
                "UPDATE puzzles SET attempts = attempts + 1 WHERE id = ?1",
                params![puzzle_id],
            )?;
        }
        Ok(())
    }

    /// Get solve rate per theme: (theme, attempts, solved).
    pub fn get_theme_stats(&self) -> SqlResult<Vec<(String, i32, i32)>> {
        let mut stmt = self.conn.prepare(
            "SELECT theme, SUM(attempts), SUM(solved)
             FROM puzzles
             WHERE theme IS NOT NULL
             GROUP BY theme
             ORDER BY theme",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i32>(1)?,
                row.get::<_, i32>(2)?,
            ))
        })?;
        rows.collect()
    }

    /// Get total puzzle counts: (total, solved_at_least_once).
    pub fn get_puzzle_counts(&self) -> SqlResult<(i32, i32)> {
        let total: i32 = self.conn.query_row(
            "SELECT COUNT(*) FROM puzzles",
            [],
            |row| row.get(0),
        )?;
        let solved: i32 = self.conn.query_row(
            "SELECT COUNT(*) FROM puzzles WHERE solved > 0",
            [],
            |row| row.get(0),
        )?;
        Ok((total, solved))
    }

    // ── Move Analysis Persistence (Phase 5.5) ─────────────────────────

    /// Save per-move analysis data for a game. `uci_moves` provides the UCI
    /// strings since MoveAnalysis stores SAN only.
    pub fn save_move_analysis(
        &self,
        game_id: i64,
        uci_moves: &[String],
        moves: &[crate::analysis::MoveAnalysis],
    ) -> SqlResult<()> {
        let mut stmt = self.conn.prepare(
            "INSERT INTO move_analysis (game_id, move_number, side, move_uci, move_san,
             eval_before, eval_after, best_move, best_eval, classification)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        )?;
        for (i, ma) in moves.iter().enumerate() {
            let side = if ma.side == crate::types::Color::White { "white" } else { "black" };
            let uci = uci_moves.get(i).map(|s| s.as_str()).unwrap_or("");
            stmt.execute(params![
                game_id,
                ma.move_number as i32,
                side,
                uci,
                ma.move_san,
                ma.eval_before,
                ma.eval_after,
                ma.best_move_uci,
                ma.best_eval,
                ma.classification.to_db_str(),
            ])?;
        }
        Ok(())
    }

    /// Get the saved accuracy for a game, or None if it hasn't been
    /// analyzed yet.
    pub fn get_game_accuracy(&self, game_id: i64) -> SqlResult<Option<f64>> {
        self.conn
            .query_row(
                "SELECT user_accuracy FROM games WHERE id = ?1",
                params![game_id],
                |row| row.get::<_, Option<f64>>(0),
            )
    }

    /// Load saved per-move analysis for a game. Returns an empty Vec if the
    /// game has never been analyzed (no rows in move_analysis). Rows are
    /// returned in move order. Explanation is left as `None` — explanations
    /// are not persisted; callers that want them can regenerate from the
    /// board state.
    pub fn get_move_analysis(
        &self,
        game_id: i64,
    ) -> SqlResult<Vec<crate::analysis::MoveAnalysis>> {
        let mut stmt = self.conn.prepare(
            "SELECT move_number, side, move_san, eval_before, eval_after,
                    best_move, best_eval, classification
             FROM move_analysis
             WHERE game_id = ?1
             ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![game_id], |row| {
            let side_str: String = row.get(1)?;
            let class_str: String = row.get(7)?;
            let side = if side_str == "white" {
                crate::types::Color::White
            } else {
                crate::types::Color::Black
            };
            let classification = crate::analysis::MoveClass::from_db_str(&class_str)
                .unwrap_or(crate::analysis::MoveClass::Good);
            let eval_before: i32 = row.get(3)?;
            let eval_after: i32 = row.get(4)?;
            let best_eval: i32 = row.get(6)?;
            // CPL = max(0, best_eval - played_eval) from the moving side's
            // perspective. Since stored evals are White-perspective, flip
            // for Black before comparing.
            let played_for_side = if side == crate::types::Color::White { eval_after } else { -eval_after };
            let best_for_side = if side == crate::types::Color::White { best_eval } else { -best_eval };
            let cpl = (best_for_side - played_for_side).max(0);
            Ok(crate::analysis::MoveAnalysis {
                move_number: row.get::<_, i32>(0)? as usize,
                side,
                move_san: row.get(2)?,
                eval_before,
                eval_after,
                best_move_uci: row.get(5)?,
                best_eval,
                cpl,
                classification,
                explanation: None,
            })
        })?;
        rows.collect()
    }

    /// Update the user_accuracy field on a game after analysis completes.
    pub fn update_game_accuracy(&self, game_id: i64, accuracy: f64) -> SqlResult<()> {
        self.conn.execute(
            "UPDATE games SET user_accuracy = ?1 WHERE id = ?2",
            params![accuracy, game_id],
        )?;
        Ok(())
    }

    // ── Coaching Report Queries (Phase 5.5) ───────────────────────────

    /// Get accuracy for the last N analyzed games (user_accuracy IS NOT NULL).
    /// Returns (played_at, accuracy) pairs, oldest first.
    pub fn get_accuracy_history(&self, n: u32) -> SqlResult<Vec<(String, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT played_at, user_accuracy FROM games
             WHERE user_accuracy IS NOT NULL
             ORDER BY played_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![n], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })?;
        let mut result: Vec<_> = rows.collect::<SqlResult<_>>()?;
        result.reverse(); // oldest first
        Ok(result)
    }

    /// Get classification counts for the user's moves across the last N analyzed games.
    /// Returns (classification, count) pairs.
    pub fn get_classification_stats(&self, n: u32) -> SqlResult<Vec<(String, i32)>> {
        let mut stmt = self.conn.prepare(
            "SELECT ma.classification, COUNT(*)
             FROM move_analysis ma
             JOIN games g ON ma.game_id = g.id
             WHERE g.user_accuracy IS NOT NULL
               AND ma.game_id IN (
                   SELECT id FROM games WHERE user_accuracy IS NOT NULL
                   ORDER BY played_at DESC LIMIT ?1
               )
             GROUP BY ma.classification
             ORDER BY COUNT(*) DESC",
        )?;
        let rows = stmt.query_map(params![n], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i32>(1)?))
        })?;
        rows.collect()
    }

    /// Get blunder counts by game phase for the last N analyzed games.
    /// Returns (opening, middlegame, endgame) blunder counts.
    /// Opening = move 1-15, middlegame = 16-35, endgame = 36+.
    pub fn get_phase_weakness(&self, n: u32) -> SqlResult<(i32, i32, i32)> {
        let mut stmt = self.conn.prepare(
            "SELECT
                SUM(CASE WHEN move_number <= 15 THEN 1 ELSE 0 END),
                SUM(CASE WHEN move_number BETWEEN 16 AND 35 THEN 1 ELSE 0 END),
                SUM(CASE WHEN move_number > 35 THEN 1 ELSE 0 END)
             FROM move_analysis
             WHERE classification IN ('blunder', 'mistake')
               AND game_id IN (
                   SELECT id FROM games WHERE user_accuracy IS NOT NULL
                   ORDER BY played_at DESC LIMIT ?1
               )",
        )?;
        stmt.query_row(params![n], |row| {
            Ok((
                row.get::<_, i32>(0).unwrap_or(0),
                row.get::<_, i32>(1).unwrap_or(0),
                row.get::<_, i32>(2).unwrap_or(0),
            ))
        })
    }

    // ── Opening Repertoire (Phase 5.3) ────────────────────────────────

    /// Set the opening name for a game.
    pub fn update_game_opening(&self, game_id: i64, opening_name: &str) -> SqlResult<()> {
        self.conn.execute(
            "UPDATE games SET opening_name = ?1 WHERE id = ?2",
            params![opening_name, game_id],
        )?;
        Ok(())
    }

    /// Get opening stats: (opening_name, games_played, wins, losses, draws).
    pub fn get_opening_stats(&self) -> SqlResult<Vec<(String, i32, i32, i32, i32)>> {
        let mut stmt = self.conn.prepare(
            "SELECT opening_name,
                    COUNT(*) as total,
                    SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN result = 'draw' THEN 1 ELSE 0 END)
             FROM games
             WHERE opening_name IS NOT NULL
             GROUP BY opening_name
             ORDER BY COUNT(*) DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i32>(1)?,
                row.get::<_, i32>(2)?,
                row.get::<_, i32>(3)?,
                row.get::<_, i32>(4)?,
            ))
        })?;
        rows.collect()
    }

    // ── Statistics Dashboard (Phase 6) ────────────────────────────────

    /// Get rating progression: (played_at, rating_before, rating_after) for games
    /// that have rating data, oldest first.
    pub fn get_rating_history(&self, n: u32) -> SqlResult<Vec<(String, i32, i32)>> {
        let mut stmt = self.conn.prepare(
            "SELECT played_at, rating_before, rating_after FROM games
             WHERE rating_before IS NOT NULL AND rating_after IS NOT NULL
             ORDER BY played_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![n], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i32>(1)?,
                row.get::<_, i32>(2)?,
            ))
        })?;
        let mut result: Vec<_> = rows.collect::<SqlResult<_>>()?;
        result.reverse(); // oldest first
        Ok(result)
    }

    /// Get W/L/D counts split by user color.
    /// Returns ((white_w, white_l, white_d), (black_w, black_l, black_d)).
    pub fn get_results_by_color(&self) -> SqlResult<((i32, i32, i32), (i32, i32, i32))> {
        let query = |color: &str, result: &str| -> SqlResult<i32> {
            self.conn.query_row(
                "SELECT COUNT(*) FROM games WHERE user_color = ?1 AND result = ?2",
                params![color, result],
                |row| row.get(0),
            )
        };
        Ok((
            (query("white", "win")?, query("white", "loss")?, query("white", "draw")?),
            (query("black", "win")?, query("black", "loss")?, query("black", "draw")?),
        ))
    }

    /// Get W/L/D counts by time control.
    /// Returns (time_control, wins, losses, draws) per distinct TC.
    pub fn get_results_by_time_control(&self) -> SqlResult<Vec<(String, i32, i32, i32)>> {
        let mut stmt = self.conn.prepare(
            "SELECT time_control,
                    SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN result = 'draw' THEN 1 ELSE 0 END)
             FROM games
             WHERE time_control IS NOT NULL
             GROUP BY time_control
             ORDER BY COUNT(*) DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i32>(1)?,
                row.get::<_, i32>(2)?,
                row.get::<_, i32>(3)?,
            ))
        })?;
        rows.collect()
    }
}

/// Determine the database file path.
fn db_path() -> PathBuf {
    if let Some(data_dir) = dirs::data_dir() {
        data_dir.join("focalors").join("focalors.db")
    } else {
        PathBuf::from("focalors.db")
    }
}

// ════════════════════════════════════════════════════════════════════════════
// PGN generation
// ════════════════════════════════════════════════════════════════════════════

use crate::board::Board;
use crate::movegen::{generate_legal_moves, make_move};
use crate::types::*;

/// Generate a PGN string from a list of UCI moves and game metadata.
pub fn generate_pgn(
    uci_moves: &[String],
    white_name: &str,
    black_name: &str,
    result: &str,
    time_control: Option<&str>,
    date: Option<&str>,
) -> String {
    let mut pgn = String::new();

    // Headers
    pgn.push_str(&format!("[Event \"Focalors Game\"]\n"));
    pgn.push_str(&format!("[Site \"Focalors Desktop\"]\n"));
    pgn.push_str(&format!(
        "[Date \"{}\"]\n",
        date.unwrap_or("????.??.??")
    ));
    pgn.push_str(&format!("[White \"{}\"]\n", white_name));
    pgn.push_str(&format!("[Black \"{}\"]\n", black_name));
    pgn.push_str(&format!("[Result \"{}\"]\n", result));
    if let Some(tc) = time_control {
        pgn.push_str(&format!("[TimeControl \"{}\"]\n", tc));
    }
    pgn.push('\n');

    // Moves in SAN notation
    let mut board = Board::startpos();
    for (i, uci_move) in uci_moves.iter().enumerate() {
        if i % 2 == 0 {
            pgn.push_str(&format!("{}. ", i / 2 + 1));
        }

        let san = uci_to_san(&board, uci_move);
        pgn.push_str(&san);
        pgn.push(' ');

        // Apply the move
        if let Some(mv) = crate::uci::parse_move(&board, uci_move) {
            make_move(&mut board, mv);
        }
    }

    pgn.push_str(result);
    pgn.push('\n');
    pgn
}

/// Convert a UCI move to Standard Algebraic Notation (SAN).
pub fn uci_to_san(board: &Board, uci: &str) -> String {
    let mv = match crate::uci::parse_move(board, uci) {
        Some(m) => m,
        None => return uci.to_string(), // fallback to UCI if parsing fails
    };

    let from = mv.from_sq();
    let to = mv.to_sq();
    let flag = mv.flag();

    // Castling
    if matches!(flag, crate::moves::MoveFlag::Castling) {
        return if to.file() > from.file() {
            "O-O".to_string()
        } else {
            "O-O-O".to_string()
        };
    }

    let piece = match board.piece_type_on(from) {
        Some(p) => p,
        None => return uci.to_string(),
    };

    let is_capture = board.piece_on(to).is_some()
        || matches!(flag, crate::moves::MoveFlag::EnPassant);

    let mut san = String::new();

    if piece == Piece::Pawn {
        if is_capture {
            san.push((b'a' + from.file()) as char);
            san.push('x');
        }
        san.push_str(&to.to_algebraic());
    } else {
        // Piece letter
        san.push(match piece {
            Piece::Knight => 'N',
            Piece::Bishop => 'B',
            Piece::Rook => 'R',
            Piece::Queen => 'Q',
            Piece::King => 'K',
            _ => '?',
        });

        // Disambiguation: check if another piece of the same type can move to the same square
        let moves = generate_legal_moves(board);
        let mut ambiguous_file = false;
        let mut ambiguous_rank = false;
        let mut ambiguous_count = 0;

        for i in 0..moves.len() {
            let other = moves[i];
            if other.to_sq().0 == to.0
                && other.from_sq().0 != from.0
                && board.piece_type_on(other.from_sq()) == Some(piece)
            {
                ambiguous_count += 1;
                if other.from_sq().file() == from.file() {
                    ambiguous_rank = true; // same file → need rank
                }
                if other.from_sq().rank() == from.rank() {
                    ambiguous_file = true; // same rank → need file
                }
            }
        }

        if ambiguous_count > 0 {
            if ambiguous_file || (!ambiguous_file && !ambiguous_rank) {
                san.push((b'a' + from.file()) as char);
            }
            if ambiguous_rank {
                san.push((b'1' + from.rank()) as char);
            }
        }

        if is_capture {
            san.push('x');
        }
        san.push_str(&to.to_algebraic());
    }

    // Promotion
    if matches!(flag, crate::moves::MoveFlag::Promotion) {
        san.push('=');
        san.push(match mv.promotion_piece() {
            Piece::Queen => 'Q',
            Piece::Rook => 'R',
            Piece::Bishop => 'B',
            Piece::Knight => 'N',
            _ => '?',
        });
    }

    // Check / checkmate indicator
    if let Some(parsed) = crate::uci::parse_move(board, uci) {
        let mut test_board = board.clone();
        make_move(&mut test_board, parsed);
        let legal = generate_legal_moves(&test_board);
        let king_sq = test_board
            .piece_bb(test_board.side_to_move, Piece::King)
            .lsb();
        let in_check = crate::attacks::is_square_attacked(
            &test_board,
            king_sq,
            test_board.side_to_move.flip(),
        );
        if in_check {
            if legal.is_empty() {
                san.push('#');
            } else {
                san.push('+');
            }
        }
    }

    san
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn db_opens_and_creates_profile() {
        let db = Database {
            conn: Connection::open_in_memory().unwrap(),
        };
        db.init_schema().unwrap();
        let profile = db.get_or_create_profile().unwrap();
        assert_eq!(profile.name, "Player");
        assert_eq!(profile.rating, 1200);
        assert_eq!(profile.games_played, 0);
    }

    #[test]
    fn db_updates_profile() {
        let db = Database {
            conn: Connection::open_in_memory().unwrap(),
        };
        db.init_schema().unwrap();
        db.get_or_create_profile().unwrap();
        db.update_profile("Nick", 1500).unwrap();
        let p = db.get_or_create_profile().unwrap();
        assert_eq!(p.name, "Nick");
        assert_eq!(p.rating, 1500);
    }

    #[test]
    fn db_saves_and_retrieves_games() {
        let db = Database {
            conn: Connection::open_in_memory().unwrap(),
        };
        db.init_schema().unwrap();

        let id = db
            .save_game("white", "win", Some("checkmate"), Some("10+0"), Some("Club"), "1. e4 e5 1-0", Some(2))
            .unwrap();
        assert!(id > 0);

        let games = db.get_recent_games(10).unwrap();
        assert_eq!(games.len(), 1);
        assert_eq!(games[0].result, "win");
        assert_eq!(games[0].result_reason.as_deref(), Some("checkmate"));
    }

    #[test]
    fn db_counts_results() {
        let db = Database {
            conn: Connection::open_in_memory().unwrap(),
        };
        db.init_schema().unwrap();

        db.save_game("white", "win", None, None, None, "1-0", None).unwrap();
        db.save_game("black", "loss", None, None, None, "1-0", None).unwrap();
        db.save_game("white", "draw", None, None, None, "1/2", None).unwrap();
        db.save_game("white", "win", None, None, None, "1-0", None).unwrap();

        let (w, l, d) = db.get_result_counts().unwrap();
        assert_eq!(w, 2);
        assert_eq!(l, 1);
        assert_eq!(d, 1);
    }

    #[test]
    fn pgn_generation_basic() {
        crate::attacks::init();
        let moves = vec!["e2e4".to_string(), "e7e5".to_string()];
        let pgn = generate_pgn(&moves, "Player", "Focalors", "1-0", Some("10+0"), Some("2026.04.13"));
        assert!(pgn.contains("[White \"Player\"]"));
        assert!(pgn.contains("[Black \"Focalors\"]"));
        assert!(pgn.contains("1. e4 e5"));
        assert!(pgn.contains("1-0"));
    }

    #[test]
    fn san_conversion_castling() {
        crate::attacks::init();
        let board = Board::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1").unwrap();
        assert_eq!(uci_to_san(&board, "e1g1"), "O-O");
        assert_eq!(uci_to_san(&board, "e1c1"), "O-O-O");
    }
}
