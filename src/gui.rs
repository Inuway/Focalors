use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use eframe::egui;

use crate::attacks;
use crate::board::Board;
use crate::movegen::{generate_legal_moves, make_move};
use crate::moves::{Move, MoveFlag};
use crate::search::Searcher;
use crate::types::*;
use crate::uci;

// ════════════════════════════════════════════════════════════════════════════
// Shared state between GUI and background threads
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub struct EngineSettings {
    pub max_depth: u32,
    pub think_time_ms: u64,
    pub tt_size_mb: usize,
    pub use_time_limit: bool, // true = time-based, false = depth-based
    pub analysis_depth: u32,
}

impl Default for EngineSettings {
    fn default() -> Self {
        Self {
            max_depth: 12,
            think_time_ms: 5000,
            tt_size_mb: 64,
            use_time_limit: true,
            analysis_depth: 14,
        }
    }
}

#[derive(Clone)]
struct SearchInfo {
    depth: u32,
    score: i32,
    nodes: u64,
    best_move: String,
    searching: bool,
}

impl Default for SearchInfo {
    fn default() -> Self {
        Self {
            depth: 0,
            score: 0,
            nodes: 0,
            best_move: String::new(),
            searching: false,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SideChoice {
    White,
    Black,
    Random,
}

impl SideChoice {
    fn resolve(self) -> Color {
        match self {
            SideChoice::White => Color::White,
            SideChoice::Black => Color::Black,
            SideChoice::Random => {
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.subsec_nanos())
                    .unwrap_or(0);
                if nanos % 2 == 0 {
                    Color::White
                } else {
                    Color::Black
                }
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TimePreset {
    Bullet1_0,
    Bullet2_1,
    Blitz3_0,
    Blitz5_0,
    Blitz5_3,
    Rapid10_0,
    Rapid15_10,
    Classical30_0,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LocalDifficulty {
    Beginner,
    Club,
    Tournament,
    Master,
    Adaptive,
    Custom,
}

impl LocalDifficulty {
    const ALL: [LocalDifficulty; 6] = [
        LocalDifficulty::Beginner,
        LocalDifficulty::Club,
        LocalDifficulty::Tournament,
        LocalDifficulty::Master,
        LocalDifficulty::Adaptive,
        LocalDifficulty::Custom,
    ];

    fn label(self) -> &'static str {
        match self {
            LocalDifficulty::Beginner => "Beginner",
            LocalDifficulty::Club => "Club",
            LocalDifficulty::Tournament => "Tournament",
            LocalDifficulty::Master => "Master",
            LocalDifficulty::Adaptive => "Adaptive",
            LocalDifficulty::Custom => "Custom",
        }
    }

    fn numeric_level(self, adaptive_level: u32) -> u32 {
        match self {
            LocalDifficulty::Beginner => crate::strength::BEGINNER_LEVEL,
            LocalDifficulty::Club => crate::strength::CLUB_LEVEL,
            LocalDifficulty::Tournament => crate::strength::TOURNAMENT_LEVEL,
            LocalDifficulty::Master => crate::strength::MASTER_LEVEL,
            LocalDifficulty::Adaptive => adaptive_level,
            LocalDifficulty::Custom => crate::strength::MASTER_LEVEL,
        }
    }

    fn description(self, adaptive_level: u32) -> String {
        match self {
            LocalDifficulty::Beginner => "Fast, forgiving replies with shallow calculation.".into(),
            LocalDifficulty::Club => "Solid club-level play with restrained search depth.".into(),
            LocalDifficulty::Tournament => "Balanced default strength for serious local games.".into(),
            LocalDifficulty::Master => "Focalors' strongest local profile with deeper search.".into(),
            LocalDifficulty::Adaptive => format!(
                "Auto-adjusts to your level. Currently Level {} (~{} Elo).",
                adaptive_level,
                crate::strength::estimated_elo(adaptive_level),
            ),
            LocalDifficulty::Custom => "Uses the values from Advanced engine settings (depth, time, TT).".into(),
        }
    }
}

impl Default for LocalDifficulty {
    fn default() -> Self {
        Self::Tournament
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HomePage {
    Overview,
    Analyze,
    Progress,
    Statistics,
    History,
    Puzzles,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum UiTheme {
    Dark,
    Light,
}

impl UiTheme {
    fn toggle(self) -> Self {
        match self {
            UiTheme::Dark => UiTheme::Light,
            UiTheme::Light => UiTheme::Dark,
        }
    }

    fn index(self) -> u8 {
        match self {
            UiTheme::Dark => 0,
            UiTheme::Light => 1,
        }
    }

    /// String form persisted to the SQLite user_profile.ui_theme column.
    fn as_db_str(self) -> &'static str {
        match self {
            UiTheme::Dark => "dark",
            UiTheme::Light => "light",
        }
    }

    /// Inverse of `as_db_str`. Unknown values fall back to Light so a
    /// corrupted or older DB row doesn't break startup.
    fn from_db_str(s: &str) -> Self {
        match s {
            "dark" => UiTheme::Dark,
            _ => UiTheme::Light,
        }
    }
}

static ACTIVE_THEME: AtomicU8 = AtomicU8::new(0);

impl TimePreset {
    const ALL: [TimePreset; 8] = [
        TimePreset::Bullet1_0,
        TimePreset::Bullet2_1,
        TimePreset::Blitz3_0,
        TimePreset::Blitz5_0,
        TimePreset::Blitz5_3,
        TimePreset::Rapid10_0,
        TimePreset::Rapid15_10,
        TimePreset::Classical30_0,
    ];

    fn label(self) -> &'static str {
        match self {
            TimePreset::Bullet1_0 => "1+0",
            TimePreset::Bullet2_1 => "2+1",
            TimePreset::Blitz3_0 => "3+0",
            TimePreset::Blitz5_0 => "5+0",
            TimePreset::Blitz5_3 => "5+3",
            TimePreset::Rapid10_0 => "10+0",
            TimePreset::Rapid15_10 => "15+10",
            TimePreset::Classical30_0 => "30+0",
        }
    }

    fn initial_ms(self) -> u64 {
        match self {
            TimePreset::Bullet1_0 => 60_000,
            TimePreset::Bullet2_1 => 120_000,
            TimePreset::Blitz3_0 => 180_000,
            TimePreset::Blitz5_0 => 300_000,
            TimePreset::Blitz5_3 => 300_000,
            TimePreset::Rapid10_0 => 600_000,
            TimePreset::Rapid15_10 => 900_000,
            TimePreset::Classical30_0 => 1_800_000,
        }
    }

    fn increment_ms(self) -> u64 {
        match self {
            TimePreset::Bullet2_1 => 1_000,
            TimePreset::Blitz5_3 => 3_000,
            TimePreset::Rapid15_10 => 10_000,
            _ => 0,
        }
    }
}

#[derive(Clone)]
struct TimeControl {
    label: &'static str,
    initial_time_ms: u64,
    increment_ms: u64,
    white_remaining_ms: u64,
    black_remaining_ms: u64,
    active_since: Option<Instant>,
}

impl TimeControl {
    fn from_preset(preset: TimePreset) -> Self {
        let initial_ms = preset.initial_ms();
        Self {
            label: preset.label(),
            initial_time_ms: initial_ms,
            increment_ms: preset.increment_ms(),
            white_remaining_ms: initial_ms,
            black_remaining_ms: initial_ms,
            active_since: None,
        }
    }

    fn clear_active(&mut self) {
        self.active_since = None;
    }

    fn start_turn_now(&mut self) {
        self.active_since = Some(Instant::now());
    }

    fn remaining_ms(&self, side: Color) -> u64 {
        match side {
            Color::White => self.white_remaining_ms,
            Color::Black => self.black_remaining_ms,
        }
    }

    fn displayed_remaining_ms(&self, side: Color, active_side: Option<Color>) -> u64 {
        let remaining = self.remaining_ms(side);
        if active_side == Some(side) {
            if let Some(active_since) = self.active_since {
                return remaining.saturating_sub(elapsed_ms(active_since));
            }
        }
        remaining
    }

    fn consume_running_time(&mut self, side: Color) -> u64 {
        let elapsed = self.active_since.take().map(elapsed_ms).unwrap_or(0);
        let remaining = match side {
            Color::White => &mut self.white_remaining_ms,
            Color::Black => &mut self.black_remaining_ms,
        };
        *remaining = remaining.saturating_sub(elapsed);
        *remaining
    }

    fn add_increment(&mut self, side: Color) {
        let remaining = match side {
            Color::White => &mut self.white_remaining_ms,
            Color::Black => &mut self.black_remaining_ms,
        };
        *remaining = remaining.saturating_add(self.increment_ms);
    }

    fn flag_side(&mut self, side: Color) {
        match side {
            Color::White => self.white_remaining_ms = 0,
            Color::Black => self.black_remaining_ms = 0,
        }
        self.clear_active();
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GameOutcome {
    Checkmate(Color),
    Stalemate,
    FiftyMoveRule,
    ThreefoldRepetition,
    InsufficientMaterial,
    Timeout(Color),
    Resignation(Color),
}

#[derive(Clone)]
struct LocalGameState {
    active: bool,
    human_color: Color,
    difficulty: LocalDifficulty,
    numeric_level: u32,
    outcome: Option<GameOutcome>,
    time_control: TimeControl,
}

impl LocalGameState {
    fn idle(human_color: Color, time_preset: TimePreset, difficulty: LocalDifficulty, level: u32) -> Self {
        Self {
            active: false,
            human_color,
            difficulty,
            numeric_level: level,
            outcome: None,
            time_control: TimeControl::from_preset(time_preset),
        }
    }

    fn new(human_color: Color, time_preset: TimePreset, difficulty: LocalDifficulty, level: u32) -> Self {
        let mut time_control = TimeControl::from_preset(time_preset);
        time_control.start_turn_now();

        Self {
            active: true,
            human_color,
            difficulty,
            numeric_level: level,
            outcome: None,
            time_control,
        }
    }
}

impl Default for LocalGameState {
    fn default() -> Self {
        Self::idle(Color::White, TimePreset::Rapid10_0, LocalDifficulty::default(), crate::strength::TOURNAMENT_LEVEL)
    }
}

#[derive(Clone)]
struct LocalSnapshot {
    board: Board,
    white_remaining_ms: u64,
    black_remaining_ms: u64,
    move_uci: Option<String>,
}

#[derive(Clone)]
struct PendingPromotion {
    moves: Vec<Move>,
}

#[derive(Clone, Copy)]
struct DragState {
    from_sq: u8,
    piece_color: Color,
    piece: Piece,
    pointer_pos: egui::Pos2,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AnnotationColor {
    Green,
    Red,
    Yellow,
    Blue,
}

impl AnnotationColor {
    fn from_modifiers(m: egui::Modifiers) -> Self {
        // Shift > command/ctrl > alt > plain. Matches lichess/chess.com convention.
        if m.shift { Self::Red }
        else if m.command { Self::Yellow }
        else if m.alt { Self::Blue }
        else { Self::Green }
    }

    fn fill(self) -> egui::Color32 {
        // ~63% alpha — visible over both light and dark squares without
        // obliterating the piece beneath.
        match self {
            Self::Green  => egui::Color32::from_rgba_premultiplied(70, 150, 70, 160),
            Self::Red    => egui::Color32::from_rgba_premultiplied(200, 70, 70, 160),
            Self::Yellow => egui::Color32::from_rgba_premultiplied(210, 165, 60, 160),
            Self::Blue   => egui::Color32::from_rgba_premultiplied(70, 130, 200, 160),
        }
    }

    fn arrow(self) -> egui::Color32 {
        // Slightly more opaque for arrows so the shape reads cleanly over pieces.
        match self {
            Self::Green  => egui::Color32::from_rgba_premultiplied(70, 150, 70, 210),
            Self::Red    => egui::Color32::from_rgba_premultiplied(200, 70, 70, 210),
            Self::Yellow => egui::Color32::from_rgba_premultiplied(210, 165, 60, 210),
            Self::Blue   => egui::Color32::from_rgba_premultiplied(70, 130, 200, 210),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Annotation {
    Highlight { sq: u8, color: AnnotationColor },
    Arrow { from: u8, to: u8, color: AnnotationColor },
}

#[derive(Clone)]
struct LocalSearchRequest {
    soft_time_ms: u64,
    hard_time_ms: u64,
    depth_cap: Option<u32>,
    generation: u64,
    strength_config: crate::strength::StrengthConfig,
}

/// State for replaying a saved game.
#[derive(Clone)]
struct ReplayState {
    game: crate::db::SavedGame,
    /// Parsed UCI move list, lazily populated on first open.
    uci_moves: Vec<String>,
    /// Cached positions: `boards[i]` is the position before ply `i`.
    /// `boards[0]` is the start position; `boards[uci_moves.len()]` is the final position.
    boards: Vec<Board>,
    /// Move objects parallel to `uci_moves`, used for last-move highlighting.
    moves: Vec<Move>,
    /// Currently displayed ply index, in `0..=uci_moves.len()`.
    cursor: usize,
}

/// Read-only view of a board position, used by `draw_board` when rendering
/// something other than the live game (e.g. replay).
struct BoardView<'a> {
    board: &'a Board,
    last_move: Option<Move>,
    interactive: bool,
}

/// Puzzle trainer state — active when the user is solving puzzles.
struct PuzzleTrainerState {
    puzzle: crate::db::SavedPuzzle,
    board: Board,
    solution_move: Move,
    user_attempted: bool,
    user_solved: bool,
    show_hint: bool,
    show_answer: bool,
    wrong_attempts: u32,
}

/// Analysis state — shared between the GUI and the background analysis thread.
#[derive(Clone)]
enum AnalysisState {
    Idle,
    Running {
        progress: usize, // moves analyzed so far
        total: usize,
    },
    Complete {
        analysis: crate::analysis::GameAnalysis,
        puzzles: Vec<crate::puzzles::PuzzleCandidate>,
        uci_moves: Vec<String>,
        /// The game this analysis belongs to, captured when the worker
        /// was SPAWNED — never read from current UI state at persist
        /// time, or a stale worker finishing late would save game A's
        /// moves under whatever game the user has navigated to since.
        game_id: Option<i64>,
    },
}

struct SharedState {
    board: Board,
    move_history: Vec<String>,
    local_history: Vec<LocalSnapshot>,
    local_history_cursor: usize,
    local_search_generation: u64,
    engine_settings: EngineSettings,
    search_info: SearchInfo,
    local_game: LocalGameState,
    game_saved: bool,
    /// Persistent searcher for TT reuse across moves
    persistent_searcher: Arc<Mutex<Searcher>>,
    status_message: String,
}

impl Default for SharedState {
    fn default() -> Self {
        Self {
            board: Board::startpos(),
            move_history: Vec::new(),
            local_history: Vec::new(),
            local_history_cursor: 0,
            local_search_generation: 0,
            engine_settings: EngineSettings::default(),
            search_info: SearchInfo::default(),
            local_game: LocalGameState::default(),
            game_saved: false,
            persistent_searcher: {
                let mut s = Searcher::new(64);
                s.use_nnue = crate::nnue::network::get_network().is_some();
                Arc::new(Mutex::new(s))
            },
            status_message: "Idle. Choose local play to begin.".to_string(),
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Main GUI app
// ════════════════════════════════════════════════════════════════════════════

pub struct FocalorsApp {
    state: Arc<Mutex<SharedState>>,
    db: Option<crate::db::Database>,
    profile: Option<crate::db::UserProfile>,
    recent_games: Vec<crate::db::SavedGame>,
    result_counts: (i32, i32, i32), // wins, losses, draws
    show_welcome: bool,
    welcome_name: String,
    welcome_rating_choice: i32,
    home_page: HomePage,
    ui_theme: UiTheme,
    replay_game: Option<ReplayState>,
    analysis_state: Arc<Mutex<AnalysisState>>,
    analysis_review_cursor: usize, // which move is selected in review
    /// If set, completed analysis should be persisted against this game id
    /// instead of the most-recent-game fallback.
    analysis_target_game_id: Option<i64>,
    /// Monotonic stamp for analysis workers; see start_analysis.
    analysis_generation: Arc<AtomicU64>,
    /// Cache of (game_id → final-position board) for History thumbnails.
    /// Computed lazily on first display, never invalidated since saved game
    /// PGNs don't change after persistence.
    history_thumbnails: std::collections::HashMap<i64, Option<Board>>,
    selected_square: Option<u8>,
    drag_state: Option<DragState>,
    /// User-drawn annotations (right-click highlights + arrows for tactical
    /// thinking). Cleared on left-click, on move played, and on board change.
    annotations: Vec<Annotation>,
    /// Square where a right-click drag began. Some while the right button is
    /// held down; resolved into an annotation on release.
    right_drag_origin: Option<u8>,
    /// Last-seen board hash. When this changes between frames, annotations
    /// are wiped (a new position invalidates whatever the user drew before).
    last_annotation_board_hash: Option<u64>,
    flipped: bool,
    local_side_choice: SideChoice,
    local_time_preset: TimePreset,
    local_difficulty: LocalDifficulty,
    adaptive_level: u32,
    auto_adjust_message: Option<(String, std::time::Instant)>,
    show_eval_panel: bool,
    session_start_rating: i32,
    session_start_games: i32,
    session_best_accuracy: Option<f64>,
    puzzle_trainer: Option<PuzzleTrainerState>,
    puzzle_message: Option<(String, egui::Color32, std::time::Instant)>,
    show_advanced_engine_settings: bool,
    pending_promotion: Option<PendingPromotion>,
    piece_textures: HashMap<(Color, Piece), egui::TextureHandle>,
    pgn_import_text: String,
    pgn_import_parsed: Option<crate::pgn::ParsedPgn>,
    pgn_import_error: Option<String>,
    pgn_import_user_color: Color,
    /// Cache for the live PGN parse, keyed on the exact import text.
    /// parse_pgn replays the whole game (legal movegen per SAN token),
    /// so re-running it every frame while text sits in the box wastes
    /// milliseconds per frame on long games.
    pgn_parse_cache: Option<(String, Result<crate::pgn::ParsedPgn, String>)>,
}

/// Replay the saved PGN once, building parallel `boards`/`moves` vectors so the
/// review UI can step through positions without re-parsing on every frame.
/// Falls back to a start-position-only state if the PGN is unparseable.
fn build_replay_state(game: crate::db::SavedGame) -> ReplayState {
    let parsed = crate::pgn::parse_pgn(&game.pgn).ok();
    let mut board = Board::startpos();
    let mut boards = vec![board.clone()];
    let mut moves = Vec::new();
    let mut uci_moves = Vec::new();

    if let Some(p) = parsed {
        for uci in p.uci_moves {
            let mv = match crate::uci::parse_move(&board, &uci) {
                Some(m) => m,
                None => break,
            };
            make_move(&mut board, mv);
            uci_moves.push(uci);
            moves.push(mv);
            boards.push(board.clone());
        }
    }

    ReplayState {
        game,
        uci_moves,
        boards,
        moves,
        cursor: 0,
    }
}

fn load_piece_texture(
    ctx: &egui::Context,
    name: &str,
    data: &[u8],
) -> egui::TextureHandle {
    let img = image::load_from_memory(data)
        .expect("Failed to load piece image")
        .into_rgba8();
    let size = [img.width() as _, img.height() as _];
    let pixels = img.into_raw();
    let color_image = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);
    ctx.load_texture(name, color_image, egui::TextureOptions::LINEAR)
}

impl FocalorsApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Theme is configured below once the DB has been opened and the
        // persisted preference loaded — so the very first frame already
        // reflects the user's saved choice instead of always being Light.

        let state = Arc::new(Mutex::new(SharedState::default()));

        // Load piece textures from embedded PNGs
        let ctx = &cc.egui_ctx;
        let mut piece_textures = HashMap::new();
        piece_textures.insert(
            (Color::White, Piece::King),
            load_piece_texture(ctx, "wK", include_bytes!("../assets/pieces/wK.png")),
        );
        piece_textures.insert(
            (Color::White, Piece::Queen),
            load_piece_texture(ctx, "wQ", include_bytes!("../assets/pieces/wQ.png")),
        );
        piece_textures.insert(
            (Color::White, Piece::Rook),
            load_piece_texture(ctx, "wR", include_bytes!("../assets/pieces/wR.png")),
        );
        piece_textures.insert(
            (Color::White, Piece::Bishop),
            load_piece_texture(ctx, "wB", include_bytes!("../assets/pieces/wB.png")),
        );
        piece_textures.insert(
            (Color::White, Piece::Knight),
            load_piece_texture(ctx, "wN", include_bytes!("../assets/pieces/wN.png")),
        );
        piece_textures.insert(
            (Color::White, Piece::Pawn),
            load_piece_texture(ctx, "wP", include_bytes!("../assets/pieces/wP.png")),
        );
        piece_textures.insert(
            (Color::Black, Piece::King),
            load_piece_texture(ctx, "bK", include_bytes!("../assets/pieces/bK.png")),
        );
        piece_textures.insert(
            (Color::Black, Piece::Queen),
            load_piece_texture(ctx, "bQ", include_bytes!("../assets/pieces/bQ.png")),
        );
        piece_textures.insert(
            (Color::Black, Piece::Rook),
            load_piece_texture(ctx, "bR", include_bytes!("../assets/pieces/bR.png")),
        );
        piece_textures.insert(
            (Color::Black, Piece::Bishop),
            load_piece_texture(ctx, "bB", include_bytes!("../assets/pieces/bB.png")),
        );
        piece_textures.insert(
            (Color::Black, Piece::Knight),
            load_piece_texture(ctx, "bN", include_bytes!("../assets/pieces/bN.png")),
        );
        piece_textures.insert(
            (Color::Black, Piece::Pawn),
            load_piece_texture(ctx, "bP", include_bytes!("../assets/pieces/bP.png")),
        );

        // Initialize database
        let db = match crate::db::Database::open() {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("Warning: Could not open database: {e}");
                None
            }
        };

        let (profile, show_welcome) = if let Some(ref db) = db {
            match db.get_or_create_profile() {
                Ok(p) => {
                    let first_time = p.name == "Player" && p.games_played == 0;
                    (Some(p), first_time)
                }
                Err(_) => (None, true),
            }
        } else {
            (None, false)
        };

        let recent_games = db
            .as_ref()
            .and_then(|db| db.get_recent_games(20).ok())
            .unwrap_or_default();

        let result_counts = db
            .as_ref()
            .and_then(|db| db.get_result_counts().ok())
            .unwrap_or((0, 0, 0));

        let adaptive_level = db
            .as_ref()
            .and_then(|db| db.get_adaptive_level().ok())
            .unwrap_or(10);

        // Load the persisted theme (falls back to Light if absent / unparsable)
        // and apply it to egui's visuals before the first frame.
        let ui_theme = db
            .as_ref()
            .and_then(|db| db.get_ui_theme().ok())
            .map(|s| UiTheme::from_db_str(&s))
            .unwrap_or(UiTheme::Light);
        configure_theme(&cc.egui_ctx, ui_theme);

        let session_start_rating = profile.as_ref().map_or(1200, |p| p.rating);
        let session_start_games = profile.as_ref().map_or(0, |p| p.games_played);

        Self {
            state,
            db,
            profile,
            recent_games,
            result_counts,
            show_welcome,
            welcome_name: String::new(),
            welcome_rating_choice: 1200,
            home_page: HomePage::Overview,
            ui_theme,
            replay_game: None,
            analysis_state: Arc::new(Mutex::new(AnalysisState::Idle)),
            analysis_review_cursor: 0,
            analysis_target_game_id: None,
            analysis_generation: Arc::new(AtomicU64::new(0)),
            history_thumbnails: std::collections::HashMap::new(),
            selected_square: None,
            drag_state: None,
            annotations: Vec::new(),
            right_drag_origin: None,
            last_annotation_board_hash: None,
            flipped: false,
            local_side_choice: SideChoice::White,
            local_time_preset: TimePreset::Rapid10_0,
            local_difficulty: LocalDifficulty::default(),
            adaptive_level,
            auto_adjust_message: None,
            show_eval_panel: false,
            session_start_rating,
            session_start_games,
            session_best_accuracy: None,
            puzzle_trainer: None,
            puzzle_message: None,
            show_advanced_engine_settings: false,
            pending_promotion: None,
            piece_textures,
            pgn_import_text: String::new(),
            pgn_import_parsed: None,
            pgn_import_error: None,
            pgn_import_user_color: Color::White,
            pgn_parse_cache: None,
        }
    }

    fn set_ui_theme(&mut self, ctx: &egui::Context, theme: UiTheme) {
        self.ui_theme = theme;
        configure_theme(ctx, theme);
        // Persist so the choice survives restart. Best-effort: ignore DB
        // write errors so a transient I/O hiccup never breaks the toggle.
        if let Some(ref db) = self.db {
            let _ = db.set_ui_theme(theme.as_db_str());
        }
    }

    fn local_input_enabled(&self) -> bool {
        if self.pending_promotion.is_some() {
            return false;
        }

        // Puzzle mode: allow input if puzzle is active and not solved
        if let Some(ref t) = self.puzzle_trainer {
            return !t.user_solved && !t.show_answer;
        }

        let state = self.state.lock().unwrap();
        state.local_game.active
            && state.local_game.outcome.is_none()
            && !state.search_info.searching
            && state.board.side_to_move == state.local_game.human_color
            && state
                .local_game
                .time_control
                .displayed_remaining_ms(state.board.side_to_move, Some(state.board.side_to_move))
                > 0
    }

    /// Toggle/replace an annotation. If the exact same annotation (same
    /// color + shape) already exists, remove it. If a same-shape annotation
    /// exists with a different color, replace it. Otherwise add.
    fn toggle_annotation(&mut self, new_annot: Annotation) {
        let existing_idx = self.annotations.iter().position(|&a| {
            match (a, new_annot) {
                (Annotation::Highlight { sq: s1, .. }, Annotation::Highlight { sq: s2, .. }) => {
                    s1 == s2
                }
                (
                    Annotation::Arrow { from: f1, to: t1, .. },
                    Annotation::Arrow { from: f2, to: t2, .. },
                ) => f1 == f2 && t1 == t2,
                _ => false,
            }
        });

        match existing_idx {
            Some(idx) if self.annotations[idx] == new_annot => {
                self.annotations.remove(idx);
            }
            Some(idx) => {
                self.annotations[idx] = new_annot;
            }
            None => {
                self.annotations.push(new_annot);
            }
        }
    }

    fn has_running_local_clock(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.local_game.active
            && state.local_game.outcome.is_none()
            && state.local_game.time_control.active_since.is_some()
    }

    fn update_local_clock_state(&mut self) {
        let mut state = self.state.lock().unwrap();
        if handle_active_side_timeout(&mut state) {
            self.pending_promotion = None;
            self.selected_square = None;
            self.drag_state = None;
        }
    }

    fn begin_local_game(&mut self, human_color: Color) {
        let time_label = self.local_time_preset.label();
        let difficulty_label = self.local_difficulty.label();
        let level = self.local_difficulty.numeric_level(self.adaptive_level);
        let config = crate::strength::StrengthConfig::from_level(level);
        {
            let mut state = self.state.lock().unwrap();
            state.board = Board::startpos();
            state.move_history.clear();
            state.local_history.clear();
            state.search_info = SearchInfo::default();
            state.local_search_generation = state.local_search_generation.wrapping_add(1);
            state.local_game = LocalGameState::new(
                human_color,
                self.local_time_preset,
                self.local_difficulty,
                level,
            );
            state.game_saved = false;
            // Configure searcher for this difficulty level
            {
                let mut searcher = state.persistent_searcher.lock().unwrap();
                searcher.tt.clear();
                searcher.use_nnue = config.use_nnue
                    && crate::nnue::network::get_network().is_some();
                searcher.eval_noise = config.eval_noise_cp;
            }
            seed_local_history(&mut state);
            state.status_message = format!(
                "Local game started. You play {}. Time control: {}. Difficulty: {} (Level {}, ~{} Elo).",
                color_name(human_color),
                time_label,
                difficulty_label,
                level,
                crate::strength::estimated_elo(level),
            );
        }

        self.selected_square = None;
        self.drag_state = None;
        self.pending_promotion = None;
        self.flipped = human_color == Color::Black;

        if human_color == Color::Black {
            self.start_engine_search();
        }
    }

    fn navigate_local_history(&mut self, target_index: usize) {
        let history_label;

        {
            let mut state = self.state.lock().unwrap();
            if !state.local_game.active || target_index >= state.local_history.len() {
                return;
            }

            invalidate_local_search(&mut state, true);
            state.local_history_cursor = target_index;

            let Some(snapshot) = state.local_history.get(target_index).cloned() else {
                return;
            };

            restore_local_snapshot(&mut state, &snapshot);
            state.local_game.outcome = None;
            sync_local_move_history(&mut state);

            update_local_game_outcome(&mut state);

            history_label = local_history_label(target_index, snapshot.move_uci.as_deref());
            if let Some(outcome) = state.local_game.outcome {
                state.status_message = local_outcome_status(outcome, state.local_game.human_color);
            } else {
                state.status_message = format!("Reviewing {history_label}. Use the arrows or resume from here.");
            }
        }

        self.pending_promotion = None;
        self.selected_square = None;
        self.drag_state = None;
    }

    fn resume_local_game(&mut self) {
        let mut should_start_engine = false;

        {
            let mut state = self.state.lock().unwrap();
            if !state.local_game.active || state.local_game.outcome.is_some() {
                return;
            }

            state.local_game.time_control.start_turn_now();
            if state.board.side_to_move == state.local_game.human_color {
                state.status_message = "Resumed local game. Your turn.".to_string();
            } else {
                state.status_message = "Resumed local game. Focalors is thinking...".to_string();
                should_start_engine = true;
            }
        }

        if should_start_engine {
            self.start_engine_search();
        }
    }

    fn abort_local_game(&mut self) {
        let local_game = {
            let mut state = self.state.lock().unwrap();
            if !state.local_game.active {
                return;
            }

            invalidate_local_search(&mut state, true);
            let local_game = LocalGameState::idle(
                state.local_game.human_color,
                self.local_time_preset,
                self.local_difficulty,
                state.local_game.numeric_level,
            );
            state.board = Board::startpos();
            state.move_history.clear();
            state.local_history.clear();
            state.local_history_cursor = 0;
            state.local_game = local_game.clone();
            state.status_message = "Local game ended. Back on the home screen.".to_string();
            local_game
        };

        self.pending_promotion = None;
        self.selected_square = None;
        self.drag_state = None;
        self.flipped = local_game.human_color == Color::Black;
    }

    // ── Welcome dialog (first launch) ────────────────────────────────

    fn draw_welcome_dialog(&mut self, ctx: &egui::Context) {
        if !self.show_welcome {
            return;
        }

        egui::Window::new("Welcome to Focalors")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .fixed_size([360.0, 280.0])
            .show(ctx, |ui| {
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new("Set up your profile")
                        .size(16.0)
                        .strong(),
                );
                ui.add_space(4.0);
                    ui.label("This helps Focalors track your progress over time.");
                ui.add_space(12.0);

                ui.horizontal(|ui| {
                    ui.label("Your name:");
                    ui.text_edit_singleline(&mut self.welcome_name);
                });

                ui.add_space(8.0);
                ui.label("Approximate rating:");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.welcome_rating_choice, 800, "Beginner (~800)");
                    ui.selectable_value(&mut self.welcome_rating_choice, 1200, "Intermediate (~1200)");
                });
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.welcome_rating_choice, 1600, "Advanced (~1600)");
                    ui.selectable_value(&mut self.welcome_rating_choice, 2000, "Expert (~2000)");
                });

                ui.add_space(16.0);
                let name_valid = !self.welcome_name.trim().is_empty();
                ui.add_enabled_ui(name_valid, |ui| {
                    if ui.button(egui::RichText::new("Get Started").strong().size(14.0)).clicked() {
                        if let Some(ref db) = self.db {
                            let name = self.welcome_name.trim().to_string();
                            if db.update_profile(&name, self.welcome_rating_choice).is_ok() {
                                self.profile = db.get_or_create_profile().ok();
                            }
                        }
                        self.show_welcome = false;
                    }
                });
                if !name_valid {
                    ui.label(
                        egui::RichText::new("Enter your name to continue")
                            .size(11.0)
                            .color(hydra_subtle_text()),
                    );
                }
            });
    }

    // ── Profile display (on home screen) ───────────────────────────────

    // ── Statistics dashboard ─────────────────────────────────────────

    fn draw_statistics_panel(&mut self, ui: &mut egui::Ui) {
        let db = match &self.db {
            Some(d) => d,
            None => return,
        };

        // Gather all data upfront
        let rating_history = db.get_rating_history(50).ok().unwrap_or_default();
        let accuracy_history = db.get_accuracy_history(50).ok().unwrap_or_default();
        let (total_w, total_l, total_d) = db.get_result_counts().ok().unwrap_or((0, 0, 0));
        let by_color = db.get_results_by_color().ok().unwrap_or(((0,0,0),(0,0,0)));
        let by_tc = db.get_results_by_time_control().ok().unwrap_or_default();
        let (phase_o, phase_m, phase_e) = db.get_phase_weakness(20).ok().unwrap_or((0, 0, 0));
        let theme_stats = db.get_theme_stats().ok().unwrap_or_default();
        let (puzzle_total, puzzle_solved) = db.get_puzzle_counts().ok().unwrap_or((0, 0));

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Statistics").size(22.0).strong());
            ui.label(egui::RichText::new("Last 50 games").color(hydra_subtle_text()));
        });
        ui.add_space(16.0);

        // ── KPI strip ────────────────────────────────────────────────
        let total_games = total_w + total_l + total_d;
        let current_rating = rating_history
            .last()
            .map(|(_, _, r)| *r)
            .unwrap_or_else(|| self.profile.as_ref().map_or(1200, |p| p.rating));
        let rating_delta = if rating_history.len() >= 2 {
            let earlier_idx = rating_history.len().saturating_sub(10);
            current_rating - rating_history[earlier_idx].2
        } else {
            0
        };
        let rating_spark: Vec<f64> = rating_history.iter().map(|(_, _, r)| *r as f64).collect();

        let (recent_acc, prior_acc, acc_spark) = if accuracy_history.is_empty() {
            (0.0_f64, 0.0_f64, Vec::<f64>::new())
        } else {
            let recent: Vec<f64> = accuracy_history.iter().rev().take(10).map(|(_, a)| *a).collect();
            let recent_avg = recent.iter().sum::<f64>() / recent.len() as f64;
            let prior: Vec<f64> = accuracy_history.iter().rev().skip(10).take(10).map(|(_, a)| *a).collect();
            let prior_avg = if prior.is_empty() {
                recent_avg
            } else {
                prior.iter().sum::<f64>() / prior.len() as f64
            };
            let spark: Vec<f64> = accuracy_history.iter().map(|(_, a)| *a).collect();
            (recent_avg, prior_avg, spark)
        };
        let acc_delta = recent_acc - prior_acc;

        let win_rate = if total_games > 0 {
            total_w as f64 / total_games as f64 * 100.0
        } else {
            0.0
        };
        let puzzle_rate = if puzzle_total > 0 {
            puzzle_solved as f64 / puzzle_total as f64 * 100.0
        } else {
            0.0
        };

        // KPI row — no per-tile cards. Just naked stat columns; the page
        // spacing separates them.
        ui.columns(4, |cols| {
            // RATING
            {
                let ui = &mut cols[0];
                ui.label(egui::RichText::new("RATING").size(10.0).color(hydra_subtle_text()).strong());
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(format!("{current_rating}")).size(24.0).strong());
                    if rating_delta != 0 {
                        let color = if rating_delta > 0 { hydra_success() } else { hydra_danger() };
                        let arrow = if rating_delta > 0 { "▲" } else { "▼" };
                        ui.label(
                            egui::RichText::new(format!("{arrow} {}", rating_delta.abs()))
                                .size(11.0).color(color).strong(),
                        );
                    }
                });
                if rating_spark.len() >= 2 {
                    sparkline(ui, "rating_kpi", &rating_spark, hydra_accent(), 32.0);
                }
            }
            // ACCURACY
            {
                let ui = &mut cols[1];
                ui.label(egui::RichText::new("ACCURACY").size(10.0).color(hydra_subtle_text()).strong());
                ui.horizontal(|ui| {
                    if acc_spark.is_empty() {
                        ui.label(egui::RichText::new("—").size(24.0).strong());
                    } else {
                        ui.label(
                            egui::RichText::new(format!("{recent_acc:.1}%"))
                                .size(24.0).strong().color(accuracy_color(recent_acc)),
                        );
                        if acc_delta.abs() > 0.5 {
                            let color = if acc_delta > 0.0 { hydra_success() } else { hydra_danger() };
                            let arrow = if acc_delta > 0.0 { "▲" } else { "▼" };
                            ui.label(
                                egui::RichText::new(format!("{arrow} {:.1}", acc_delta.abs()))
                                    .size(11.0).color(color).strong(),
                            );
                        }
                    }
                });
                if acc_spark.len() >= 2 {
                    sparkline(ui, "acc_kpi", &acc_spark, hydra_success(), 32.0);
                }
            }
            // WIN RATE
            {
                let ui = &mut cols[2];
                ui.label(egui::RichText::new("WIN RATE").size(10.0).color(hydra_subtle_text()).strong());
                if total_games > 0 {
                    ui.label(egui::RichText::new(format!("{win_rate:.0}%")).size(24.0).strong());
                } else {
                    ui.label(egui::RichText::new("—").size(24.0).strong());
                }
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(format!("{total_w}W  {total_d}D  {total_l}L"))
                        .size(11.0).color(hydra_subtle_text()),
                );
            }
            // PUZZLES
            {
                let ui = &mut cols[3];
                ui.label(egui::RichText::new("PUZZLES").size(10.0).color(hydra_subtle_text()).strong());
                if puzzle_total > 0 {
                    ui.label(egui::RichText::new(format!("{puzzle_rate:.0}%")).size(24.0).strong());
                } else {
                    ui.label(egui::RichText::new("—").size(24.0).strong());
                }
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(format!("{puzzle_solved}/{puzzle_total} solved"))
                        .size(11.0).color(hydra_subtle_text()),
                );
            }
        });
        subtle_row_separator(ui);

        // ── Row 2: Rating chart + Results (naked sections) ────────────
        ui.columns(2, |cols| {
            // Rating Over Time
            {
                let ui = &mut cols[0];
                ui.label(egui::RichText::new("Rating Over Time").size(14.0).strong());
                ui.add_space(6.0);
                if rating_history.len() >= 2 {
                    let points: Vec<[f64; 2]> = rating_history
                        .iter()
                        .enumerate()
                        .map(|(i, (_, _, after))| [i as f64 + 1.0, *after as f64])
                        .collect();
                    let line = egui_plot::Line::new("rating", egui_plot::PlotPoints::new(points))
                        .color(hydra_accent());
                    let min_r = rating_history.iter().map(|(_, _, r)| *r).min().unwrap_or(800) - 50;
                    let max_r = rating_history.iter().map(|(_, _, r)| *r).max().unwrap_or(1600) + 50;
                    egui_plot::Plot::new("rating_chart")
                        .height(190.0)
                        .include_y(min_r as f64)
                        .include_y(max_r as f64)
                        .allow_drag(false)
                        .allow_zoom(false)
                        .allow_scroll(false)
                        .show_axes(true)
                        .y_axis_label("Elo")
                        .show(ui, |plot_ui| {
                            plot_ui.line(line);
                        });
                } else {
                    ui.add_space(80.0);
                    ui.label(
                        egui::RichText::new("Play a few games to see your rating trend.")
                            .color(hydra_subtle_text()),
                    );
                }
            }
            // Results
            {
                let ui = &mut cols[1];
                ui.label(egui::RichText::new("Results").size(14.0).strong());
                ui.add_space(8.0);
                if total_games > 0 {
                    let ((ww, wl, wd), (bw, bl, bd)) = by_color;
                    draw_result_bar(ui, "Overall", total_w as u32, total_l as u32, total_d as u32);
                    if (ww + wl + wd) > 0 {
                        draw_result_bar(ui, "As White", ww as u32, wl as u32, wd as u32);
                    }
                    if (bw + bl + bd) > 0 {
                        draw_result_bar(ui, "As Black", bw as u32, bl as u32, bd as u32);
                    }
                    if !by_tc.is_empty() {
                        ui.add_space(8.0);
                        ui.label(
                            egui::RichText::new("By Time Control")
                                .size(11.0).color(hydra_subtle_text()).strong(),
                        );
                        for (tc, w, l, d) in &by_tc {
                            draw_result_bar(ui, tc, *w as u32, *l as u32, *d as u32);
                        }
                    }
                } else {
                    ui.add_space(80.0);
                    ui.label(
                        egui::RichText::new("No completed games yet.").color(hydra_subtle_text()),
                    );
                }
            }
        });
        subtle_row_separator(ui);

        // ── Row 3: Accuracy chart + Phase weakness (naked sections) ───
        ui.columns(2, |cols| {
            // Accuracy chart
            {
                let ui = &mut cols[0];
                ui.label(egui::RichText::new("Accuracy Trends").size(14.0).strong());
                ui.add_space(6.0);
                if accuracy_history.len() >= 2 {
                    let acc_points: Vec<[f64; 2]> = accuracy_history
                        .iter()
                        .enumerate()
                        .map(|(i, (_, acc))| [i as f64 + 1.0, *acc])
                        .collect();
                    let rolling: Vec<[f64; 2]> = accuracy_history
                        .iter()
                        .enumerate()
                        .map(|(i, _)| {
                            let start = i.saturating_sub(4);
                            let window = &accuracy_history[start..=i];
                            let avg = window.iter().map(|(_, a)| a).sum::<f64>() / window.len() as f64;
                            [(i + 1) as f64, avg]
                        })
                        .collect();
                    let line_acc = egui_plot::Line::new("accuracy", egui_plot::PlotPoints::new(acc_points))
                        .color(egui::Color32::from_rgb(100, 180, 255))
                        .name("Per game");
                    let line_avg = egui_plot::Line::new("rolling_avg", egui_plot::PlotPoints::new(rolling))
                        .color(egui::Color32::from_rgb(255, 180, 50))
                        .name("5-game avg");
                    egui_plot::Plot::new("accuracy_chart")
                        .height(170.0)
                        .include_y(0.0)
                        .include_y(100.0)
                        .allow_drag(false)
                        .allow_zoom(false)
                        .allow_scroll(false)
                        .show_axes(true)
                        .legend(egui_plot::Legend::default())
                        .show(ui, |plot_ui| {
                            plot_ui.line(line_acc);
                            plot_ui.line(line_avg);
                        });
                    if let Some(best) = accuracy_history
                        .iter().map(|(_, a)| *a).max_by(|a, b| a.partial_cmp(b).unwrap())
                    {
                        ui.label(
                            egui::RichText::new(format!("Personal best: {best:.1}%"))
                                .size(11.0).color(class_best()),
                        );
                    }
                } else {
                    ui.add_space(80.0);
                    ui.label(
                        egui::RichText::new("Analyze games to see accuracy trends.")
                            .color(hydra_subtle_text()),
                    );
                }
            }
            // Phase weakness
            {
                let ui = &mut cols[1];
                ui.label(egui::RichText::new("Phase Weakness").size(14.0).strong());
                ui.add_space(8.0);
                let total_phase_errors = phase_o + phase_m + phase_e;
                if total_phase_errors > 0 {
                    let max_count = phase_o.max(phase_m).max(phase_e);
                    let phases = [
                        ("Opening", phase_o, "moves 1-15"),
                        ("Middlegame", phase_m, "moves 16-35"),
                        ("Endgame", phase_e, "moves 36+"),
                    ];
                    for (label, count, sub) in &phases {
                        let pct = *count as f64 / total_phase_errors as f64;
                        let color = if *count == max_count {
                            class_blunder()
                        } else {
                            hydra_accent()
                        };
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(*label).size(12.0).strong());
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(
                                    egui::RichText::new(format!("{count} errors"))
                                        .size(11.0).color(hydra_subtle_text()),
                                );
                            });
                        });
                        ui.label(egui::RichText::new(*sub).size(10.0).color(hydra_subtle_text()));
                        let bar_w = ui.available_width();
                        let (rect, _) = ui.allocate_exact_size(
                            egui::vec2(bar_w, 10.0),
                            egui::Sense::hover(),
                        );
                        ui.painter().rect_filled(rect, 5.0, hydra_panel_alt_fill());
                        let fill_w = bar_w * pct as f32;
                        if fill_w > 0.0 {
                            let fill_rect = egui::Rect::from_min_size(
                                rect.min,
                                egui::vec2(fill_w, 10.0),
                            );
                            ui.painter().rect_filled(fill_rect, 5.0, color);
                        }
                    }
                } else {
                    ui.add_space(80.0);
                    ui.label(
                        egui::RichText::new("Analyze games to see phase weaknesses.")
                            .color(hydra_subtle_text()),
                    );
                }
            }
        });
        subtle_row_separator(ui);

        // ── Row 4: Puzzle theme heatmap (naked section) ────────────────
        if !theme_stats.is_empty() {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Puzzle Themes").size(14.0).strong());
                ui.label(
                    egui::RichText::new("(sorted weakest first)")
                        .size(10.0).color(hydra_subtle_text()),
                );
            });
            ui.add_space(8.0);
            let mut themes: Vec<_> = theme_stats.iter().filter(|(_, a, _)| *a >= 1).collect();
            themes.sort_by(|a, b| {
                let ra = a.2 as f64 / a.1 as f64;
                let rb = b.2 as f64 / b.1 as f64;
                ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
            });
            ui.horizontal_wrapped(|ui| {
                for (theme, attempts, solved) in &themes {
                    let rate = *solved as f64 / *attempts as f64;
                    let color = if rate < 0.4 {
                        class_blunder()
                    } else if rate < 0.6 {
                        class_mistake()
                    } else if rate < 0.75 {
                        class_inaccuracy()
                    } else {
                        class_best()
                    };
                    let label = crate::puzzles::PuzzleTheme::from_db_str(theme).label();
                    let tile = egui::Frame::new()
                        .fill(color.gamma_multiply(0.18))
                        .stroke(egui::Stroke::new(1.0, color))
                        .corner_radius(6)
                        .inner_margin(egui::Margin::same(8));
                    tile.show(ui, |ui| {
                        ui.vertical(|ui| {
                            ui.label(egui::RichText::new(label).size(11.0).strong());
                            ui.label(
                                egui::RichText::new(format!(
                                    "{:.0}% · {solved}/{attempts}",
                                    rate * 100.0
                                ))
                                .size(10.0).color(hydra_subtle_text()),
                            );
                        });
                    });
                }
            });
            subtle_row_separator(ui);
        }

        // ── Session Summary (compact naked footer) ────────────────────
        let current_games = self.profile.as_ref().map_or(0, |p| p.games_played);
        let current_rating_p = self.profile.as_ref().map_or(1200, |p| p.rating);
        let session_games = current_games - self.session_start_games;
        let session_rating_delta = current_rating_p - self.session_start_rating;
        if session_games > 0 || self.session_best_accuracy.is_some() {
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("This Session").size(13.0).strong());
                ui.label(
                    egui::RichText::new(format!("· Games: {session_games}"))
                        .size(11.0).color(hydra_subtle_text()),
                );
                if session_rating_delta != 0 {
                    let sign = if session_rating_delta >= 0 { "+" } else { "" };
                    let color = if session_rating_delta > 0 {
                        hydra_success()
                    } else if session_rating_delta < 0 {
                        hydra_danger()
                    } else {
                        hydra_text()
                    };
                    ui.label(
                        egui::RichText::new(format!(
                            "· Rating: {current_rating_p} ({sign}{session_rating_delta})"
                        ))
                        .size(11.0).color(color),
                    );
                }
                if let Some(best) = self.session_best_accuracy {
                    ui.label(
                        egui::RichText::new(format!("· Best accuracy: {best:.1}%"))
                            .size(11.0).color(accuracy_color(best)),
                    );
                }
            });
        }
    }

    // ── Opening stats ───────────────────────────────────────────────

    fn draw_opening_stats(&mut self, ui: &mut egui::Ui) {
        let stats = match &self.db {
            Some(db) => db.get_opening_stats().ok().unwrap_or_default(),
            None => return,
        };
        if stats.is_empty() {
            return;
        }

        // Naked section header + list; no card wrap.
        ui.label(egui::RichText::new("Your Openings").size(14.0).strong());
        ui.add_space(4.0);
        for (name, total, wins, losses, draws) in &stats {
            let win_rate = if *total > 0 { *wins as f64 / *total as f64 * 100.0 } else { 0.0 };
            let color = if win_rate >= 60.0 {
                hydra_success()
            } else if win_rate <= 35.0 {
                hydra_danger()
            } else {
                hydra_text()
            };
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(name).size(11.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "{total}g  W{wins}/L{losses}/D{draws}  ({win_rate:.0}%)"
                        ))
                        .size(10.0)
                        .color(color),
                    );
                });
            });
        }
    }

    // ── Eval explanation panel ────────────────────────────────────────

    fn draw_eval_panel(&mut self, ui: &mut egui::Ui) {
        let board = self.state.lock().unwrap().board.clone();
        let breakdown = crate::eval::eval_components(&board);

        hydra_card_frame().show(ui, |ui| {
            let total_pawns = breakdown.total as f64 / 100.0;
            let sign = if total_pawns >= 0.0 { "+" } else { "" };
            ui.label(
                egui::RichText::new(format!("Eval: {sign}{total_pawns:.2}"))
                    .size(14.0)
                    .strong()
                    .color(if breakdown.total > 50 {
                        egui::Color32::from_rgb(100, 200, 100)
                    } else if breakdown.total < -50 {
                        egui::Color32::from_rgb(220, 100, 80)
                    } else {
                        hydra_text()
                    }),
            );
            ui.add_space(4.0);

            let components = [
                ("Material", breakdown.material),
                ("Piece Placement", breakdown.pst),
                ("Mobility", breakdown.mobility),
                ("Pawn Structure", breakdown.pawn_structure),
                ("Passed Pawns", breakdown.passed_pawns),
                ("King Safety", breakdown.king_safety),
                ("Bishop Pair", breakdown.bishop_pair),
                ("Rook Placement", breakdown.rook_placement),
                ("Knight Outposts", breakdown.knight_outpost),
                ("Connected Passers", breakdown.connected_passers),
                ("King-Pawn Prox.", breakdown.king_pawn_proximity),
                ("Tempo", breakdown.tempo),
            ];

            for (name, value) in &components {
                if *value == 0 {
                    continue; // skip zero components for cleanliness
                }
                let pawns = *value as f64 / 100.0;
                let sign = if pawns >= 0.0 { "+" } else { "" };
                let color = if *value > 30 {
                    egui::Color32::from_rgb(100, 200, 100)
                } else if *value < -30 {
                    egui::Color32::from_rgb(220, 100, 80)
                } else {
                    hydra_subtle_text()
                };

                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!("{name}:"))
                            .size(10.0)
                            .color(hydra_subtle_text()),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let mut text = format!("{sign}{pawns:.2}");
                        // Human-readable annotation for significant values
                        if name == &"Material" && value.abs() > 200 {
                            let pieces = value.abs() / 100;
                            text = format!("{sign}{pawns:.2} ({pieces} pawn{})", if pieces > 1 { "s" } else { "" });
                        } else if name == &"King Safety" && *value < -100 {
                            text = format!("{sign}{pawns:.2} (exposed)");
                        } else if name == &"Bishop Pair" && *value > 30 {
                            text = format!("{sign}{pawns:.2} (advantage)");
                        }
                        ui.label(egui::RichText::new(text).size(10.0).color(color));
                    });
                });
            }
        });
    }

    // ── Coaching report ──────────────────────────────────────────────

    fn draw_coaching_report(&mut self, ui: &mut egui::Ui) {
        let db = match &self.db {
            Some(d) => d,
            None => return,
        };

        let accuracy_history = db.get_accuracy_history(20).ok().unwrap_or_default();
        let classification_stats = db.get_classification_stats(10).ok().unwrap_or_default();
        let phase_weakness = db.get_phase_weakness(10).ok().unwrap_or((0, 0, 0));
        let theme_stats = db.get_theme_stats().ok().unwrap_or_default();

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Progress").size(22.0).strong());
            ui.label(egui::RichText::new("Last 10 games").color(hydra_subtle_text()));
        });
        ui.add_space(16.0);

        // Empty-state for the whole tab — no analysis yet.
        if accuracy_history.is_empty() {
            hydra_card_frame().show(ui, |ui| {
                ui.set_min_height(140.0);
                ui.vertical_centered(|ui| {
                    ui.add_space(30.0);
                    ui.label(egui::RichText::new("No analyzed games yet").size(16.0).strong());
                    ui.label(
                        egui::RichText::new("Play a game and click \"Analyze Game\" to see your progress.")
                            .color(hydra_subtle_text()),
                    );
                });
            });
            return;
        }

        // Accuracy stats from the most-recent vs prior 10 analyzed games.
        let recent_avg = {
            let recent: Vec<f64> = accuracy_history.iter().rev().take(10).map(|(_, a)| *a).collect();
            recent.iter().sum::<f64>() / recent.len() as f64
        };
        let prior_avg = {
            let prior: Vec<f64> = accuracy_history.iter().rev().skip(10).take(10).map(|(_, a)| *a).collect();
            if prior.is_empty() {
                recent_avg
            } else {
                prior.iter().sum::<f64>() / prior.len() as f64
            }
        };
        let acc_delta = recent_avg - prior_avg;

        // ── Row 1: Accuracy gauge + Error breakdown (naked sections) ──
        ui.columns(2, |cols| {
            // Accuracy hero with radial gauge.
            {
                let ui = &mut cols[0];
                ui.label(egui::RichText::new("Recent Accuracy").size(14.0).strong());
                ui.add_space(8.0);
                ui.vertical_centered(|ui| {
                    draw_radial_gauge(ui, 160.0, recent_avg, accuracy_color(recent_avg));
                    ui.add_space(6.0);
                    if accuracy_history.len() >= 11 && acc_delta.abs() > 0.5 {
                        let color = if acc_delta > 0.0 { hydra_success() } else { hydra_danger() };
                        let arrow = if acc_delta > 0.0 { "▲" } else { "▼" };
                        ui.label(
                            egui::RichText::new(format!("{arrow} {:.1}% vs prior 10", acc_delta.abs()))
                                .size(12.0).color(color).strong(),
                        );
                    } else if accuracy_history.len() >= 11 {
                        ui.label(
                            egui::RichText::new("≈ stable vs prior 10")
                                .size(12.0).color(hydra_subtle_text()),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new(format!("{} analyzed games", accuracy_history.len()))
                                .size(12.0).color(hydra_subtle_text()),
                        );
                    }
                });
            }

            // Error breakdown with per-class stacked bars.
            {
                let ui = &mut cols[1];
                ui.label(egui::RichText::new("Move Breakdown").size(14.0).strong());
                ui.label(
                    egui::RichText::new("Across the last 10 analyzed games")
                        .size(10.0).color(hydra_subtle_text()),
                );
                ui.add_space(12.0);
                if classification_stats.is_empty() {
                    ui.add_space(60.0);
                    ui.label(
                        egui::RichText::new("No classified moves yet.").color(hydra_subtle_text()),
                    );
                } else {
                    let lookup = |class: &str| -> i32 {
                        classification_stats
                            .iter()
                            .find(|(c, _)| c == class)
                            .map_or(0, |(_, n)| *n)
                    };
                    let rows = [
                        ("Book", lookup("book"), class_book()),
                        ("Best", lookup("best"), class_best()),
                        ("Good", lookup("good"), class_good()),
                        ("Inaccuracy", lookup("inaccuracy"), class_inaccuracy()),
                        ("Mistake", lookup("mistake"), class_mistake()),
                        ("Blunder", lookup("blunder"), class_blunder()),
                    ];
                    let total: i32 = rows.iter().map(|(_, n, _)| *n).sum();
                    if total > 0 {
                        for (label, n, color) in &rows {
                            if *n == 0 {
                                continue;
                            }
                            let pct = *n as f32 / total as f32;
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(*label).size(11.0).color(*color).strong(),
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        ui.label(
                                            egui::RichText::new(format!("{n}"))
                                                .size(11.0).color(hydra_subtle_text()),
                                        );
                                    },
                                );
                            });
                            let bar_w = ui.available_width();
                            let (rect, _) = ui.allocate_exact_size(
                                egui::vec2(bar_w, 6.0),
                                egui::Sense::hover(),
                            );
                            ui.painter().rect_filled(rect, 3.0, hydra_panel_alt_fill());
                            let fill_w = bar_w * pct;
                            if fill_w > 0.0 {
                                let fill_rect = egui::Rect::from_min_size(
                                    rect.min,
                                    egui::vec2(fill_w, 6.0),
                                );
                                ui.painter().rect_filled(fill_rect, 3.0, *color);
                            }
                            ui.add_space(6.0);
                        }
                    }
                }
            }
        });
        subtle_row_separator(ui);

        // ── Row 2: Phase Breakdown (naked section with colored tiles) ─
        let (opening, middle, endgame) = phase_weakness;
        let total_phase_errors = opening + middle + endgame;
        ui.label(egui::RichText::new("Phase Breakdown").size(14.0).strong());
        ui.label(
            egui::RichText::new("Where your errors happen")
                .size(10.0).color(hydra_subtle_text()),
        );
        ui.add_space(12.0);
            if total_phase_errors > 0 {
                let max_count = opening.max(middle).max(endgame);
                let phases = [
                    ("Opening", opening, "moves 1-15"),
                    ("Middlegame", middle, "moves 16-35"),
                    ("Endgame", endgame, "moves 36+"),
                ];
                ui.columns(3, |cols| {
                    for (i, (label, count, sub)) in phases.iter().enumerate() {
                        let pct = *count as f64 / total_phase_errors as f64;
                        let color = if *count == max_count && max_count > 0 {
                            class_blunder()
                        } else {
                            hydra_accent()
                        };
                        let phase_tile = egui::Frame::new()
                            .fill(color.gamma_multiply(0.10))
                            .stroke(egui::Stroke::new(1.0, color))
                            .corner_radius(6)
                            .inner_margin(egui::Margin::same(12));
                        phase_tile.show(&mut cols[i], |ui| {
                            ui.set_min_height(110.0);
                            ui.vertical_centered(|ui| {
                                ui.label(egui::RichText::new(*label).size(12.0).strong());
                                ui.label(
                                    egui::RichText::new(*sub).size(10.0).color(hydra_subtle_text()),
                                );
                                ui.add_space(8.0);
                                ui.label(
                                    egui::RichText::new(format!("{count}"))
                                        .size(28.0).strong().color(color),
                                );
                                ui.label(
                                    egui::RichText::new(format!("{:.0}% of errors", pct * 100.0))
                                        .size(10.0).color(hydra_subtle_text()),
                                );
                            });
                        });
                    }
                });
        } else {
            ui.label(
                egui::RichText::new("No phase errors recorded yet.")
                    .color(hydra_subtle_text()),
            );
        }
        subtle_row_separator(ui);

        // ── Row 3: Coach tips (color-coded callouts) ───────────────────
        let weak_themes: Vec<_> = theme_stats
            .iter()
            .filter(|(_, attempts, solved)| *attempts >= 3 && (*solved as f64 / *attempts as f64) < 0.5)
            .collect();
        let blunders = classification_stats
            .iter()
            .find(|(c, _)| c == "blunder")
            .map_or(0, |(_, n)| *n);

        let mut tips: Vec<(String, egui::Color32)> = Vec::new();

        if total_phase_errors > 0 {
            let (weakest_name, action) = if opening >= middle && opening >= endgame {
                ("opening", "Focus on piece development and central control in the first 15 moves.")
            } else if middle >= endgame {
                ("middlegame", "Work on tactical awareness and piece coordination through moves 16-35.")
            } else {
                ("endgame", "Practice king activity and passed-pawn technique in the endgame.")
            };
            tips.push((
                format!("Your {weakest_name} is your weakest phase. {action}"),
                class_mistake(),
            ));
        }

        if blunders > 5 {
            tips.push((
                "You're blundering frequently — slow down and double-check captures and checks before moving."
                    .into(),
                class_blunder(),
            ));
        }

        if !weak_themes.is_empty() {
            let names: Vec<_> = weak_themes
                .iter()
                .map(|(t, _, _)| crate::puzzles::PuzzleTheme::from_db_str(t).label())
                .collect();
            tips.push((
                format!(
                    "Weak puzzle themes: {}. Use the puzzle trainer to practice.",
                    names.join(", ")
                ),
                hydra_accent(),
            ));
        }

        if let Some(ref p) = self.profile {
            if p.games_played >= 10 {
                let (w, l, _) = self.result_counts;
                if l > w * 2 {
                    tips.push((
                        "You're losing more than two-thirds of your games — consider Adaptive mode or a lower difficulty."
                            .into(),
                        class_inaccuracy(),
                    ));
                }
            }
        }

        if tips.is_empty() {
            tips.push((
                "Keep playing and analyzing games to unlock more insights.".into(),
                hydra_accent(),
            ));
        }

        // Unified "Tips" section — single header, then each tip as a
        // colored ▶ marker plus plain text. The marker carries the
        // semantic color signal (warning / mistake / suggestion) without
        // each tip needing its own border. Same principle as the rest of
        // the page: spacing + colored cues, no boxes.
        ui.label(egui::RichText::new("Tips").size(14.0).strong());
        ui.add_space(8.0);
        for (text, accent) in tips {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("▶").size(13.0).color(accent).strong());
                ui.label(egui::RichText::new(text).size(12.0));
            });
            ui.add_space(6.0);
        }
    }

    // ── Puzzle trainer ───────────────────────────────────────────────

    fn start_puzzle_trainer(&mut self) {
        let db = match &self.db {
            Some(d) => d,
            None => return,
        };
        let puzzles = match db.get_training_puzzles(1) {
            Ok(p) if !p.is_empty() => p,
            _ => return,
        };
        let puzzle = puzzles.into_iter().next().unwrap();
        self.home_page = HomePage::Puzzles;
        self.load_puzzle(puzzle);
    }

    fn load_puzzle(&mut self, puzzle: crate::db::SavedPuzzle) {
        let board = match Board::from_fen(&puzzle.fen) {
            Ok(b) => b,
            Err(_) => return,
        };
        let solution_move = match crate::uci::parse_move(&board, &puzzle.solution) {
            Some(m) => m,
            None => return,
        };
        // Set the shared board to the puzzle position
        self.state.lock().unwrap().board = board.clone();
        let board_copy = board.clone();
        self.puzzle_trainer = Some(PuzzleTrainerState {
            puzzle,
            board: board_copy,
            solution_move,
            user_attempted: false,
            user_solved: false,
            show_hint: false,
            show_answer: false,
            wrong_attempts: 0,
        });
        self.puzzle_message = None;
        self.flipped = board.side_to_move == Color::Black;
    }

    fn draw_puzzle_trainer(&mut self, ui: &mut egui::Ui) {
        // Extract display data upfront to avoid borrow conflicts
        let (theme_str, rating, stm_white, solved, show_answer, show_hint,
             wrong_attempts, solution_uci) = {
            let t = match &self.puzzle_trainer {
                Some(t) => t,
                None => return,
            };
            (
                t.puzzle.theme.clone(),
                t.puzzle.rating,
                t.board.side_to_move == Color::White,
                t.user_solved,
                t.show_answer,
                t.show_hint,
                t.wrong_attempts,
                t.solution_move.to_uci(),
            )
        };

        let theme_stats = self.db.as_ref()
            .and_then(|db| db.get_theme_stats().ok())
            .unwrap_or_default();

        let mut exit_clicked = false;
        let mut hint_clicked = false;
        let mut answer_clicked = false;
        let mut next_clicked = false;

        hydra_card_frame().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Puzzle Trainer").size(15.0).strong());
                if let Some(ref ts) = theme_str {
                    let theme = crate::puzzles::PuzzleTheme::from_db_str(ts);
                    ui.separator();
                    ui.label(
                        egui::RichText::new(theme.label())
                            .size(12.0)
                            .color(egui::Color32::from_rgb(100, 180, 255)),
                    );
                }
                if let Some(r) = rating {
                    ui.separator();
                    ui.label(
                        egui::RichText::new(format!("Rating: {r}"))
                            .size(11.0)
                            .color(hydra_subtle_text()),
                    );
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Exit").clicked() {
                        exit_clicked = true;
                    }
                });
            });

            let stm = if stm_white { "White" } else { "Black" };
            ui.label(
                egui::RichText::new(format!("{stm} to move. Find the best move!"))
                    .size(12.0),
            );

            if let Some((ref msg, color, when)) = self.puzzle_message {
                if when.elapsed().as_secs() < 5 {
                    ui.label(egui::RichText::new(msg).size(13.0).strong().color(color));
                }
            }

            if !solved && !show_answer {
                ui.horizontal(|ui| {
                    if !show_hint {
                        if ui.small_button("Show Hint").clicked() {
                            hint_clicked = true;
                        }
                    }
                    if wrong_attempts >= 1 {
                        if ui.small_button("Show Answer").clicked() {
                            answer_clicked = true;
                        }
                    }
                });
            }

            if solved || show_answer {
                if ui.button("Next Puzzle").clicked() {
                    next_clicked = true;
                }
            }

            if !theme_stats.is_empty() {
                ui.add_space(6.0);
                ui.label(egui::RichText::new("Solve Rates").size(11.0).strong());
                for (theme, attempts, solved) in &theme_stats {
                    let label = crate::puzzles::PuzzleTheme::from_db_str(theme).label();
                    let rate = if *attempts > 0 {
                        format!("{}/{} ({:.0}%)", solved, attempts, *solved as f64 / *attempts as f64 * 100.0)
                    } else {
                        "0/0".to_string()
                    };
                    ui.label(
                        egui::RichText::new(format!("  {label}: {rate}"))
                            .size(10.0)
                            .color(hydra_subtle_text()),
                    );
                }
            }
        });

        // Process deferred actions
        if exit_clicked {
            self.puzzle_trainer = None;
            self.puzzle_message = None;
            self.state.lock().unwrap().board = Board::startpos();
        }
        if hint_clicked {
            if let Some(ref ts) = theme_str {
                let theme = crate::puzzles::PuzzleTheme::from_db_str(ts);
                self.puzzle_message = Some((
                    theme.hint().to_string(),
                    egui::Color32::from_rgb(200, 200, 100),
                    std::time::Instant::now(),
                ));
            }
            if let Some(ref mut t) = self.puzzle_trainer {
                t.show_hint = true;
            }
        }
        if answer_clicked {
            self.puzzle_message = Some((
                format!("Answer: {solution_uci}"),
                egui::Color32::from_rgb(200, 150, 100),
                std::time::Instant::now(),
            ));
            if let Some(ref mut t) = self.puzzle_trainer {
                t.show_answer = true;
            }
        }
        if next_clicked {
            self.start_puzzle_trainer();
        }

        // Render the chess board below the metadata card. The board reads
        // self.state.board, which load_puzzle / handle_puzzle_move keep in
        // sync with the puzzle's current position. Skip on the exit frame
        // since puzzle_trainer was just cleared and the user is leaving
        // this surface anyway.
        if !exit_clicked {
            ui.add_space(12.0);
            self.draw_board(ui, None);
        }
    }

    /// Handle a move made during puzzle solving.
    fn handle_puzzle_move(&mut self, played_move: Move) {
        // Extract what we need before mutating
        let (solution_move, puzzle_id, puzzle_board, is_solved, is_shown) = {
            let t = match &self.puzzle_trainer {
                Some(t) => t,
                None => return,
            };
            (t.solution_move, t.puzzle.id, t.board.clone(), t.user_solved, t.show_answer)
        };
        if is_solved || is_shown {
            return;
        }

        let is_correct = played_move == solution_move;

        if is_correct {
            if let Some(ref db) = self.db {
                let _ = db.record_puzzle_attempt(puzzle_id, true);
            }
            self.puzzle_message = Some((
                "Correct!".to_string(),
                egui::Color32::from_rgb(80, 200, 80),
                std::time::Instant::now(),
            ));
            let mut board = puzzle_board;
            make_move(&mut board, played_move);
            self.state.lock().unwrap().board = board;
            if let Some(ref mut t) = self.puzzle_trainer {
                t.user_solved = true;
                t.user_attempted = true;
            }
        } else {
            if let Some(ref mut t) = self.puzzle_trainer {
                t.wrong_attempts += 1;
                t.user_attempted = true;
            }
            if let Some(ref db) = self.db {
                let _ = db.record_puzzle_attempt(puzzle_id, false);
            }
            self.puzzle_message = Some((
                "Not quite. Try again!".to_string(),
                egui::Color32::from_rgb(220, 100, 80),
                std::time::Instant::now(),
            ));
            self.state.lock().unwrap().board = puzzle_board;
        }
    }

    // ── Game history panel ──────────────────────────────────────────────

    fn draw_game_history(&mut self, ui: &mut egui::Ui) {
        hydra_card_frame().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Recent Games").size(14.0).strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Hide").clicked() {
                        self.home_page = HomePage::Overview;
                    }
                });
            });
            ui.add_space(4.0);

            if self.recent_games.is_empty() {
                ui.label(
                    egui::RichText::new("No games yet. Play your first game!")
                        .size(12.0)
                        .color(hydra_subtle_text()),
                );
                return;
            }

            // Take a snapshot of game ids and metadata so we can borrow self
            // mutably for the thumbnail cache inside the loop.
            let games_snapshot: Vec<crate::db::SavedGame> = self.recent_games.clone();

            let mut replay_id = None;
            egui::ScrollArea::vertical()
                .max_height(560.0)
                .show(ui, |ui| {
                    for game in &games_snapshot {
                        let result_icon = match game.result.as_str() {
                            "win" => "W",
                            "loss" => "L",
                            "draw" => "D",
                            _ => "?",
                        };
                        let result_color = match game.result.as_str() {
                            "win" => egui::Color32::from_rgb(100, 200, 100),
                            "loss" => egui::Color32::from_rgb(200, 100, 100),
                            _ => egui::Color32::from_rgb(200, 200, 100),
                        };
                        let reason = game.result_reason.as_deref().unwrap_or("");
                        let tc = game.time_control.as_deref().unwrap_or("");
                        let level = game.engine_level.as_deref().unwrap_or("");
                        let moves = game.move_count.unwrap_or(0);
                        let date = &game.played_at[..10.min(game.played_at.len())];

                        ui.horizontal(|ui| {
                            // Final-position thumbnail (left of the row)
                            let thumb = self.thumbnail_for_game(game);
                            if let Some(ref board) = thumb {
                                draw_board_thumbnail(ui, board, 120.0);
                            } else {
                                ui.allocate_space(egui::vec2(120.0, 120.0));
                            }

                            ui.add_space(8.0);

                            // Game metadata on the right
                            ui.vertical(|ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new(result_icon)
                                            .strong()
                                            .size(14.0)
                                            .color(result_color)
                                            .monospace(),
                                    );
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "{} as {}",
                                            date, game.user_color
                                        ))
                                        .size(12.0)
                                        .strong(),
                                    );
                                });
                                if !reason.is_empty() {
                                    ui.label(
                                        egui::RichText::new(reason)
                                            .size(11.0)
                                            .color(hydra_subtle_text()),
                                    );
                                }
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{}{} ({} moves)",
                                        if tc.is_empty() { "" } else { tc },
                                        if level.is_empty() {
                                            String::new()
                                        } else {
                                            format!(" vs {level}")
                                        },
                                        moves,
                                    ))
                                    .size(11.0)
                                    .color(hydra_subtle_text()),
                                );
                                ui.add_space(4.0);
                                if ui.small_button("Open in Analyze").clicked() {
                                    replay_id = Some(game.id);
                                }
                            });
                        });
                        ui.add_space(4.0);
                        ui.separator();
                    }
                });

            if let Some(id) = replay_id {
                self.start_replay(id);
                self.home_page = HomePage::Analyze;
            }
        });
    }

    /// Get (or compute and cache) the final-position board for a game,
    /// used to render History thumbnails. Returns None if the game's PGN
    /// can't be parsed.
    fn thumbnail_for_game(&mut self, game: &crate::db::SavedGame) -> Option<Board> {
        if !self.history_thumbnails.contains_key(&game.id) {
            let board = build_replay_state(game.clone())
                .boards
                .last()
                .cloned();
            self.history_thumbnails.insert(game.id, board);
        }
        self.history_thumbnails.get(&game.id).cloned().flatten()
    }

    fn start_replay(&mut self, game_id: i64) {
        let game = match self.db.as_ref().and_then(|db| db.get_game(game_id).ok()) {
            Some(g) => g,
            None => return,
        };

        // If this game has saved analysis, hydrate the in-memory state so
        // the replay panel shows it immediately instead of pretending the
        // game has never been analyzed.
        if let Some(ref db) = self.db {
            let saved_accuracy = db.get_game_accuracy(game_id).ok().flatten();
            if let Some(accuracy) = saved_accuracy {
                if let Ok(moves) = db.get_move_analysis(game_id) {
                    if !moves.is_empty() {
                        let user_color = if game.user_color == "white" {
                            Color::White
                        } else {
                            Color::Black
                        };
                        let mut eval_history = Vec::with_capacity(moves.len() + 1);
                        eval_history.push(moves[0].eval_before);
                        for m in &moves {
                            eval_history.push(m.eval_after);
                        }
                        let analysis = crate::analysis::GameAnalysis {
                            moves,
                            user_color,
                            user_accuracy: accuracy,
                            eval_history,
                        };
                        // Empty puzzles + uci_moves so persist_completed_analysis
                        // knows this snapshot is already saved and skips re-saving.
                        *self.analysis_state.lock().unwrap() =
                            AnalysisState::Complete {
                                analysis,
                                puzzles: Vec::new(),
                                uci_moves: Vec::new(),
                                game_id: Some(game_id),
                            };
                        self.analysis_target_game_id = Some(game_id);
                        self.analysis_review_cursor = 0;
                    }
                }
            }
        }

        let replay = build_replay_state(game);
        self.replay_game = Some(replay);
    }

    // ── Save game to database ──────────────────────────────────────────

    fn save_completed_game(&mut self) {
        let db = match &self.db {
            Some(d) => d,
            None => return,
        };

        let state = self.state.lock().unwrap();
        let local_game = &state.local_game;
        let outcome = match &local_game.outcome {
            Some(o) => o,
            None => return,
        };

        let human_color = local_game.human_color;
        let user_color = if human_color == Color::White { "white" } else { "black" };

        let (result, result_reason) = match outcome {
            GameOutcome::Checkmate(winner) => {
                if *winner == human_color {
                    ("win", "checkmate")
                } else {
                    ("loss", "checkmate")
                }
            }
            GameOutcome::Timeout(winner) => {
                if *winner == human_color {
                    ("win", "timeout")
                } else {
                    ("loss", "timeout")
                }
            }
            GameOutcome::Resignation(winner) => {
                if *winner == human_color {
                    ("win", "resignation")
                } else {
                    ("loss", "resignation")
                }
            }
            GameOutcome::Stalemate => ("draw", "stalemate"),
            GameOutcome::FiftyMoveRule => ("draw", "fifty-move rule"),
            GameOutcome::ThreefoldRepetition => ("draw", "threefold repetition"),
            GameOutcome::InsufficientMaterial => ("draw", "insufficient material"),
        };

        let tc = &local_game.time_control;
        let time_control_str = format!(
            "{}+{}",
            tc.initial_time_ms / 1000,
            tc.increment_ms / 1000
        );

        let engine_level = local_game.difficulty.label().to_string();
        let numeric_level = local_game.numeric_level;
        let was_adaptive = local_game.difficulty == LocalDifficulty::Adaptive;

        // Collect UCI moves from history
        let uci_moves: Vec<String> = state
            .local_history
            .iter()
            .filter_map(|snap| snap.move_uci.clone())
            .collect();

        let move_count = uci_moves.len() as i32;

        let (white_name, black_name) = if human_color == Color::White {
            (
                self.profile.as_ref().map_or("Player", |p| &p.name).to_string(),
                format!("Focalors ({})", engine_level),
            )
        } else {
            (
                format!("Focalors ({})", engine_level),
                self.profile.as_ref().map_or("Player", |p| &p.name).to_string(),
            )
        };

        let pgn_result = match result {
            "win" if human_color == Color::White => "1-0",
            "win" => "0-1",
            "loss" if human_color == Color::White => "0-1",
            "loss" => "1-0",
            _ => "1/2-1/2",
        };

        let today = chrono_today();
        let pgn = crate::db::generate_pgn(
            &uci_moves,
            &white_name,
            &black_name,
            pgn_result,
            Some(&time_control_str),
            Some(&today),
        );

        // Save to DB
        drop(state); // release lock before DB operations
        if let Ok(game_id) = db.save_game(
            user_color,
            result,
            Some(result_reason),
            Some(&time_control_str),
            Some(&engine_level),
            &pgn,
            Some(move_count),
        ) {
            let _ = db.increment_games_played();

            // Detect and save opening name
            if let Some((_, opening_name)) = crate::openings::detect_opening(&uci_moves) {
                let _ = db.update_game_opening(game_id, opening_name);
            }

            // Update Elo rating
            let engine_elo = crate::strength::estimated_elo(numeric_level);
            if let Ok((old_rating, new_rating)) = db.update_rating_after_game(result, engine_elo) {
                let _ = db.update_game_rating(game_id, numeric_level, old_rating, new_rating);
                let delta = new_rating - old_rating;
                let sign = if delta >= 0 { "+" } else { "" };
                let mut s = self.state.lock().unwrap();
                s.status_message = format!(
                    "Game saved. Rating: {} -> {} ({}{}).",
                    old_rating, new_rating, sign, delta
                );
            }

            // Auto-adjust: if playing on Adaptive, check if level should change
            if was_adaptive {
                if let Ok(recent) = db.get_recent_results(10) {
                    if let Some(new_level) = crate::strength::evaluate_auto_adjust(self.adaptive_level, &recent) {
                        let old = self.adaptive_level;
                        self.adaptive_level = new_level;
                        let _ = db.set_adaptive_level(new_level);
                        let direction = if new_level > old { "up" } else { "down" };
                        self.auto_adjust_message = Some((
                            format!(
                                "Difficulty adjusted {} to Level {} (~{} Elo).",
                                direction,
                                new_level,
                                crate::strength::estimated_elo(new_level),
                            ),
                            std::time::Instant::now(),
                        ));
                    }
                }
            }

            // Refresh cached data
            self.profile = db.get_or_create_profile().ok();
            self.recent_games = db.get_recent_games(20).ok().unwrap_or_default();
            self.result_counts = db.get_result_counts().ok().unwrap_or((0, 0, 0));

            // Auto-show coaching report every 10 games
            if let Some(ref p) = self.profile {
                if p.games_played > 0 && p.games_played % 10 == 0 {
                    self.home_page = HomePage::Progress;
                }
            }
        }
    }

    // ── Analysis ────────────────────────────────────────────────────────

    fn start_analysis(&mut self, uci_moves: Vec<String>, user_color: Color) {
        let analysis_state = self.analysis_state.clone();

        // Generation stamp: starting a new analysis invalidates any
        // still-running older worker. Stale workers compare their stamp
        // against the counter before every state write and discard their
        // results — otherwise a slow analysis of game A finishing late
        // would overwrite game B's state and persist A's moves under
        // B's id (permanent DB corruption).
        let my_gen = self.analysis_generation.fetch_add(1, Ordering::SeqCst) + 1;
        let gen_counter = Arc::clone(&self.analysis_generation);
        let target_game_id = self.analysis_target_game_id;

        {
            let mut a = analysis_state.lock().unwrap();
            *a = AnalysisState::Running {
                progress: 0,
                total: uci_moves.len(),
            };
        }

        self.analysis_review_cursor = 0;

        let user_rating = self.profile.as_ref().map_or(1200, |p| p.rating);
        let analysis_depth = self.state.lock().unwrap().engine_settings.analysis_depth;

        thread::spawn(move || {
            // RAII guard: if this worker exits via panic before the Complete
            // assignment below, AnalysisState would stay Running forever and
            // the UI spinner would hang. The guard runs on any unwind and
            // resets state to Idle so the user can retry. On the success path
            // the state has already moved to Complete, so the guard sees a
            // non-Running state and does nothing. A STALE worker's guard
            // must not touch state owned by a newer run, hence the
            // generation check.
            struct ResetGuard {
                state: Arc<Mutex<AnalysisState>>,
                gen_counter: Arc<AtomicU64>,
                my_gen: u64,
            }
            impl Drop for ResetGuard {
                fn drop(&mut self) {
                    if self.gen_counter.load(Ordering::SeqCst) != self.my_gen {
                        return;
                    }
                    let mut a = match self.state.lock() {
                        Ok(g) => g,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    if matches!(*a, AnalysisState::Running { .. }) {
                        *a = AnalysisState::Idle;
                        eprintln!(
                            "focalors: analysis worker exited unexpectedly; \
                             UI reset to Idle"
                        );
                    }
                }
            }
            let _reset_guard = ResetGuard {
                state: analysis_state.clone(),
                gen_counter: gen_counter.clone(),
                my_gen,
            };

            crate::attacks::init();
            let use_nnue = crate::nnue::network::get_network().is_some();
            let result = crate::analysis::analyze_game(
                &uci_moves,
                user_color,
                analysis_depth,
                use_nnue,
                &mut |cur, tot| {
                    if gen_counter.load(Ordering::SeqCst) != my_gen {
                        return; // superseded — don't clobber the new run's progress
                    }
                    let mut a = analysis_state.lock().unwrap();
                    *a = AnalysisState::Running {
                        progress: cur,
                        total: tot,
                    };
                },
            );
            // Extract puzzles from blunders
            let puzzles = crate::puzzles::extract_puzzles(
                &uci_moves, &result, user_color, user_rating,
            );
            let mut a = analysis_state.lock().unwrap();
            if gen_counter.load(Ordering::SeqCst) != my_gen {
                return; // superseded while finishing — discard silently
            }
            *a = AnalysisState::Complete {
                analysis: result,
                puzzles,
                uci_moves,
                game_id: target_game_id,
            };
        });
    }

    fn draw_analysis_button(&mut self, ui: &mut egui::Ui) {
        let state = self.state.lock().unwrap();
        let analysis = self.analysis_state.lock().unwrap();

        // Show "Review this game" + "Analyze Game" buttons when the game is
        // over and not already analyzing.
        if state.local_game.active
            && state.local_game.outcome.is_some()
            && matches!(*analysis, AnalysisState::Idle)
        {
            let uci_moves: Vec<String> = state
                .local_history
                .iter()
                .filter_map(|s| s.move_uci.clone())
                .collect();
            let user_color = state.local_game.human_color;
            drop(state);
            drop(analysis);

            if uci_moves.is_empty() {
                return;
            }

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui
                    .button(
                        egui::RichText::new("Review this game")
                            .size(13.0)
                            .strong(),
                    )
                    .clicked()
                {
                    self.open_review_for_most_recent_game();
                }
                if ui.button("Analyze Game").clicked() {
                    // Load the just-saved game into Analyze, set the target
                    // game for persistence, navigate, then kick off the
                    // analysis. Progress + result render on the Analyze page.
                    let recent_id = self
                        .db
                        .as_ref()
                        .and_then(|db| db.get_recent_games(1).ok())
                        .and_then(|g| g.into_iter().next())
                        .map(|g| g.id);
                    if let Some(id) = recent_id {
                        self.start_replay(id);
                        self.analysis_target_game_id = Some(id);
                        self.home_page = HomePage::Analyze;
                    }
                    self.start_analysis(uci_moves, user_color);
                }
            });
            return;
        }
        drop(state);
        drop(analysis);
    }

    /// Look up the most recently saved game and open it in the replay panel.
    /// Used by the "Review this game" button shown right after a live game
    /// finishes (the just-saved game is the most recent one).
    fn open_review_for_most_recent_game(&mut self) {
        let game_id = self
            .db
            .as_ref()
            .and_then(|db| db.get_recent_games(1).ok())
            .and_then(|g| g.into_iter().next())
            .map(|g| g.id);
        if let Some(id) = game_id {
            self.start_replay(id);
            self.home_page = HomePage::Analyze;
        }
    }

    fn draw_analysis_progress(&self, ui: &mut egui::Ui) {
        let analysis = self.analysis_state.lock().unwrap();
        if let AnalysisState::Running { progress, total } = *analysis {
            ui.add_space(8.0);
            hydra_card_frame().show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(
                        egui::RichText::new(format!("Analyzing... {progress}/{total} moves"))
                            .size(13.0),
                    );
                });
                if total > 0 {
                    let frac = progress as f32 / total as f32;
                    ui.add(egui::ProgressBar::new(frac).show_percentage());
                }
            });
        }
    }

    /// One-shot persistence of completed analysis. Persists puzzles + move
    /// analysis against the chosen game id (preferring `analysis_target_game_id`
    /// when set, falling back to the most recent saved game). Clears `puzzles`
    /// and `uci_moves` on the `Complete` state to mark "already saved".
    fn persist_completed_analysis(&mut self) {
        let acc_opt: Option<f64> = {
            let analysis = self.analysis_state.lock().unwrap();
            if let AnalysisState::Complete { puzzles, analysis: ga, uci_moves, game_id } =
                &*analysis
            {
                if let Some(ref db) = self.db {
                    if !puzzles.is_empty() {
                        for p in puzzles {
                            let _ = db.save_puzzle(
                                p.game_id,
                                &p.fen,
                                &p.solution_uci,
                                p.theme.to_db_str(),
                                p.rating,
                            );
                        }
                    }
                    if !uci_moves.is_empty() {
                        // Use the id captured when the worker was spawned —
                        // NOT current UI state, which may point at a
                        // different game by the time a slow analysis lands.
                        let game_id = game_id.or_else(|| {
                            db.get_recent_games(1)
                                .ok()
                                .and_then(|g| g.into_iter().next())
                                .map(|g| g.id)
                        });
                        if let Some(gid) = game_id {
                            let _ = db.save_move_analysis(gid, uci_moves, &ga.moves);
                            let _ = db.update_game_accuracy(gid, ga.user_accuracy);
                        }
                    }
                }
                Some(ga.user_accuracy)
            } else {
                None
            }
        };
        if let Some(acc) = acc_opt {
            if self.session_best_accuracy.map_or(true, |best| acc > best) {
                self.session_best_accuracy = Some(acc);
            }
        }
        // Clear puzzles + uci_moves to mark "saved" so we don't re-save next frame.
        let mut analysis = self.analysis_state.lock().unwrap();
        if let AnalysisState::Complete { puzzles, uci_moves, .. } = &mut *analysis {
            puzzles.clear();
            uci_moves.clear();
        }
    }

    /// Dedicated full-page game-review surface. Composes the board, eval
    /// graph, classified move list, per-move detail, and navigation in a
    /// proper two-column layout instead of the cramped embedded card.
    /// Empty state when no game is loaded — user is told to pick from
    /// History or paste a PGN in Import.
    fn draw_analyze_page(&mut self, ui: &mut egui::Ui) {
        // Make sure a freshly-completed analysis gets persisted before we read it.
        self.persist_completed_analysis();

        // ── Empty state — no game loaded ────────────────────────────────
        if self.replay_game.is_none() {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Analyze a Game").size(22.0).strong());
                ui.label(
                    egui::RichText::new("Pick a recent game or paste a PGN")
                        .color(hydra_subtle_text()),
                );
            });
            ui.add_space(16.0);

            let mut replay_id: Option<i64> = None;
            let mut imported: Option<crate::pgn::ParsedPgn> = None;
            let mut clear_pgn = false;
            let mut goto_history = false;
            let games_snapshot: Vec<crate::db::SavedGame> = self.recent_games.clone();

            // Live PGN parse — drives the validation badge and the parsed
            // panel state without requiring an explicit "Parse" click.
            // Re-parsed only when the text actually changes.
            let pgn_validation = if self.pgn_import_text.trim().is_empty() {
                self.pgn_parse_cache = None;
                None
            } else {
                let stale = self
                    .pgn_parse_cache
                    .as_ref()
                    .is_none_or(|(text, _)| *text != self.pgn_import_text);
                if stale {
                    let result = crate::pgn::parse_pgn(&self.pgn_import_text);
                    self.pgn_parse_cache = Some((self.pgn_import_text.clone(), result));
                }
                self.pgn_parse_cache.as_ref().map(|(_, r)| r.clone())
            };
            // Resync the suggested user color whenever the parsed PGN
            // changes — first call after a paste, then leave it sticky so
            // the user can override their side and the selection survives.
            if let Some(Ok(ref parsed)) = pgn_validation {
                if self.pgn_import_parsed.as_ref().map(|p| &p.uci_moves) != Some(&parsed.uci_moves) {
                    let profile_name = self.profile.as_ref().map(|p| p.name.clone());
                    self.pgn_import_user_color = crate::pgn::user_color_from_headers(
                        parsed,
                        profile_name.as_deref(),
                    );
                    self.pgn_import_parsed = Some(parsed.clone());
                }
            } else {
                self.pgn_import_parsed = None;
            }

            ui.columns(2, |cols| {
                // ── LEFT: recent games list ────────────────────────────
                hydra_card_frame().show(&mut cols[0], |ui| {
                    ui.set_min_height(420.0);
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Your Recent Games").size(14.0).strong());
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("Open History").clicked() {
                                goto_history = true;
                            }
                        });
                    });
                    ui.add_space(8.0);

                    if games_snapshot.is_empty() {
                        ui.add_space(80.0);
                        ui.vertical_centered(|ui| {
                            ui.label(
                                egui::RichText::new("No games yet.")
                                    .size(13.0).color(hydra_subtle_text()),
                            );
                            ui.label(
                                egui::RichText::new("Play a game and it'll show up here.")
                                    .size(11.0).color(hydra_subtle_text()),
                            );
                        });
                    } else {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            for game in &games_snapshot {
                                let result_color = match game.result.as_str() {
                                    "win" => hydra_success(),
                                    "loss" => hydra_danger(),
                                    _ => class_inaccuracy(),
                                };
                                let result_label = match game.result.as_str() {
                                    "win" => "W",
                                    "loss" => "L",
                                    "draw" => "D",
                                    _ => "?",
                                };
                                let date = &game.played_at[..10.min(game.played_at.len())];
                                let moves = game.move_count.unwrap_or(0);
                                let tc = game.time_control.as_deref().unwrap_or("");
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new(format!(" {result_label} "))
                                            .color(hydra_text_on_accent())
                                            .background_color(result_color)
                                            .strong()
                                            .monospace(),
                                    );
                                    ui.label(
                                        egui::RichText::new(date).size(11.0).strong(),
                                    );
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "as {} · {moves} moves",
                                            game.user_color
                                        ))
                                        .size(10.0).color(hydra_subtle_text()),
                                    );
                                    if !tc.is_empty() {
                                        ui.label(
                                            egui::RichText::new(format!("· {tc}"))
                                                .size(10.0).color(hydra_subtle_text()),
                                        );
                                    }
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            if ui.small_button("Review").clicked() {
                                                replay_id = Some(game.id);
                                            }
                                        },
                                    );
                                });
                                ui.add_space(4.0);
                                ui.separator();
                            }
                        });
                    }
                });

                // ── RIGHT: PGN paste with live validation ──────────────
                hydra_card_frame().show(&mut cols[1], |ui| {
                    ui.set_min_height(420.0);
                    ui.label(egui::RichText::new("Import PGN").size(14.0).strong());
                    ui.label(
                        egui::RichText::new("Paste a game's PGN to run a full engine review.")
                            .size(11.0).color(hydra_subtle_text()),
                    );
                    ui.add_space(8.0);

                    ui.add(
                        egui::TextEdit::multiline(&mut self.pgn_import_text)
                            .font(egui::TextStyle::Monospace)
                            .desired_rows(12)
                            .desired_width(f32::INFINITY)
                            .hint_text(
                                "[Event \"…\"]\n[White \"…\"]\n…\n\n1. e4 e5 2. Nf3 Nc6 …",
                            ),
                    );
                    ui.add_space(8.0);

                    match &pgn_validation {
                        Some(Ok(parsed)) => {
                            ui.horizontal_wrapped(|ui| {
                                ui.label(
                                    egui::RichText::new("✓ Valid")
                                        .size(12.0).strong().color(hydra_success()),
                                );
                                ui.label(
                                    egui::RichText::new(format!(
                                        "· {} moves · {} vs {}",
                                        parsed.uci_moves.len(),
                                        parsed.white.as_deref().unwrap_or("?"),
                                        parsed.black.as_deref().unwrap_or("?"),
                                    ))
                                    .size(11.0).color(hydra_subtle_text()),
                                );
                            });
                            ui.add_space(6.0);
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new("Analyze as")
                                        .size(11.0).color(hydra_subtle_text()).strong(),
                                );
                                if ui
                                    .selectable_label(
                                        self.pgn_import_user_color == Color::White,
                                        "White",
                                    )
                                    .clicked()
                                {
                                    self.pgn_import_user_color = Color::White;
                                }
                                if ui
                                    .selectable_label(
                                        self.pgn_import_user_color == Color::Black,
                                        "Black",
                                    )
                                    .clicked()
                                {
                                    self.pgn_import_user_color = Color::Black;
                                }
                            });
                            ui.add_space(10.0);
                            if ui
                                .add_sized(
                                    [ui.available_width(), 40.0],
                                    primary_button("Save & Review"),
                                )
                                .clicked()
                            {
                                imported = Some(parsed.clone());
                            }
                        }
                        Some(Err(err)) => {
                            egui::Frame::new()
                                .fill(hydra_danger().gamma_multiply(0.10))
                                .stroke(egui::Stroke::new(1.0, hydra_danger()))
                                .corner_radius(6)
                                .inner_margin(egui::Margin::same(10))
                                .show(ui, |ui| {
                                    ui.horizontal_wrapped(|ui| {
                                        ui.label(
                                            egui::RichText::new("✗")
                                                .size(13.0).color(hydra_danger()).strong(),
                                        );
                                        ui.label(
                                            egui::RichText::new(err)
                                                .size(11.0).color(hydra_danger()),
                                        );
                                    });
                                });
                        }
                        None => {
                            ui.label(
                                egui::RichText::new("Paste a PGN above to validate.")
                                    .size(11.0).color(hydra_subtle_text()),
                            );
                        }
                    }

                    if !self.pgn_import_text.is_empty() {
                        ui.add_space(6.0);
                        if ui.small_button("Clear").clicked() {
                            clear_pgn = true;
                        }
                    }
                });
            });

            // Apply deferred actions outside the columns closure so we
            // don't fight with the mutable borrow of self inside it.
            if let Some(id) = replay_id {
                self.start_replay(id);
            }
            if let Some(parsed) = imported {
                self.save_imported_pgn_and_open_review(&parsed);
            }
            if clear_pgn {
                self.pgn_import_text.clear();
                self.pgn_import_parsed = None;
                self.pgn_import_error = None;
            }
            if goto_history {
                self.home_page = HomePage::History;
            }
            return;
        }

        // ── Snapshot state for this frame ──────────────────────────────
        let (cursor, num_plies, board, last_move, game_id, header_text, _user_color, uci_moves) = {
            let r = self.replay_game.as_ref().unwrap();
            let total_plies = r.moves.len();
            let cursor = r.cursor.min(total_plies);
            let board = r.boards[cursor].clone();
            let last_move = if cursor > 0 {
                r.moves.get(cursor - 1).copied()
            } else {
                None
            };
            let header = format!(
                "{} — {} ({})",
                r.game.result,
                r.game.result_reason.as_deref().unwrap_or(""),
                r.game.played_at.get(..10).unwrap_or(""),
            );
            let user_color = if r.game.user_color == "white" {
                Color::White
            } else {
                Color::Black
            };
            (
                cursor,
                total_plies,
                board,
                last_move,
                r.game.id,
                header,
                user_color,
                r.uci_moves.clone(),
            )
        };

        let analysis_for_this_game: Option<crate::analysis::GameAnalysis> = {
            let a = self.analysis_state.lock().unwrap();
            match &*a {
                AnalysisState::Complete { analysis, .. }
                    if self.analysis_target_game_id == Some(game_id) =>
                {
                    Some(analysis.clone())
                }
                _ => None,
            }
        };

        // Keyboard navigation
        let (key_left, key_right, key_home, key_end) = ui.ctx().input(|i| {
            (
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
                i.key_pressed(egui::Key::Home),
                i.key_pressed(egui::Key::End),
            )
        });
        let mut new_cursor = cursor;
        if key_left && new_cursor > 0 {
            new_cursor -= 1;
        }
        if key_right && new_cursor < num_plies {
            new_cursor += 1;
        }
        if key_home {
            new_cursor = 0;
        }
        if key_end {
            new_cursor = num_plies;
        }

        let mut want_close = false;
        let mut want_analyze = false;

        // ── Header strip ────────────────────────────────────────────────
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("Game Review")
                    .size(18.0)
                    .strong()
                    .color(hydra_accent()),
            );
            ui.label(
                egui::RichText::new(header_text)
                    .size(11.0)
                    .color(hydra_subtle_text()),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("Close").clicked() {
                    want_close = true;
                }
                if analysis_for_this_game.is_none()
                    && !uci_moves.is_empty()
                    && ui.small_button("Run Analysis").clicked()
                {
                    want_analyze = true;
                }
                if let Some(ref ga) = analysis_for_this_game {
                    ui.label(
                        egui::RichText::new(format!("Accuracy: {:.1}%", ga.user_accuracy))
                            .size(13.0)
                            .strong()
                            .color(accuracy_color(ga.user_accuracy)),
                    );
                }
            });
        });

        if num_plies == 0 {
            ui.add_space(6.0);
            ui.label(
                egui::RichText::new("Could not parse this game's PGN — nothing to review.")
                    .color(hydra_warning()),
            );
            if want_close {
                self.replay_game = None;
            }
            return;
        }

        ui.add_space(8.0);

        // ── Two-column body, centered horizontally ─────────────────────
        ui.horizontal_top(|ui| {
            // Center the board+sidebar block on wide screens by padding the
            // start of the horizontal layout. Left-pad alone shifts the
            // whole row right; combined with the fixed widths below it
            // produces visual centering.
            let board_w: f32 = 760.0;
            let gap_w: f32 = 12.0;
            let right_w: f32 = 420.0;
            let total = board_w + gap_w + right_w;
            let avail = ui.available_width();
            if avail > total {
                ui.add_space((avail - total) / 2.0);
            }

            // Left: big read-only board
            ui.vertical(|ui| {
                ui.set_max_width(board_w);
                let view = BoardView {
                    board: &board,
                    last_move,
                    interactive: false,
                };
                self.draw_board(ui, Some(&view));
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if ui.button("⏮").clicked() {
                        new_cursor = 0;
                    }
                    if ui.button("◀ Prev").clicked() && new_cursor > 0 {
                        new_cursor -= 1;
                    }
                    ui.label(
                        egui::RichText::new(format!("Move {} / {}", new_cursor, num_plies))
                            .size(12.0)
                            .color(hydra_subtle_text()),
                    );
                    if ui.button("Next ▶").clicked() && new_cursor < num_plies {
                        new_cursor += 1;
                    }
                    if ui.button("⏭").clicked() {
                        new_cursor = num_plies;
                    }
                });
            });

            ui.add_space(12.0);

            // Right: eval graph + move list + per-move detail. Capped so
            // it doesn't take over on wide screens — the board stays the
            // visual focus.
            ui.vertical(|ui| {
                ui.set_max_width(420.0);
                if let Some(ref ga) = analysis_for_this_game {
                    let points: Vec<[f64; 2]> = ga
                        .eval_history
                        .iter()
                        .enumerate()
                        .map(|(i, &e)| [i as f64, (e as f64 / 100.0).clamp(-5.0, 5.0)])
                        .collect();
                    let line = egui_plot::Line::new("eval", egui_plot::PlotPoints::new(points))
                        .color(hydra_accent());
                    let zero = egui_plot::HLine::new("zero", 0.0)
                        .color(egui::Color32::from_gray(80));
                    egui_plot::Plot::new("analyze_eval_graph")
                        .height(140.0)
                        .include_y(-3.0)
                        .include_y(3.0)
                        .allow_drag(false)
                        .allow_zoom(false)
                        .allow_scroll(false)
                        .show_axes(true)
                        .y_axis_label("Eval")
                        .show(ui, |plot_ui| {
                            plot_ui.line(line);
                            plot_ui.hline(zero);
                            if new_cursor > 0 {
                                let vline = egui_plot::VLine::new(
                                    "cursor",
                                    new_cursor as f64,
                                )
                                .color(egui::Color32::from_rgb(255, 200, 50));
                                plot_ui.vline(vline);
                            }
                        });

                    ui.add_space(6.0);

                    // Summary counts
                    let mut counts = [0u32; 5]; // best, good, inacc, mistake, blunder
                    for m in &ga.moves {
                        if m.side != ga.user_color {
                            continue;
                        }
                        match m.classification {
                            crate::analysis::MoveClass::Best => counts[0] += 1,
                            crate::analysis::MoveClass::Good => counts[1] += 1,
                            crate::analysis::MoveClass::Inaccuracy => counts[2] += 1,
                            crate::analysis::MoveClass::Mistake => counts[3] += 1,
                            crate::analysis::MoveClass::Blunder => counts[4] += 1,
                            _ => {}
                        }
                    }
                    ui.label(
                        egui::RichText::new(format!(
                            "Best:{} Good:{} Inaccuracy:{} Mistake:{} Blunder:{}",
                            counts[0], counts[1], counts[2], counts[3], counts[4],
                        ))
                        .size(11.0)
                        .color(hydra_subtle_text()),
                    );
                    ui.separator();
                } else if matches!(*self.analysis_state.lock().unwrap(), AnalysisState::Running { .. }) {
                    self.draw_analysis_progress(ui);
                    ui.separator();
                }

                ui.label(egui::RichText::new("Moves").size(12.0).strong());
                ui.add_space(2.0);

                // Grid-based move list: one row per full move, three columns
                // (number, white move, black move). Each move is a clickable
                // selectable label, color-coded by classification.
                let n_pairs = (num_plies + 1) / 2;
                let render_ply = |ui: &mut egui::Ui, ply: usize, new_cursor: &mut usize| {
                    if ply >= num_plies {
                        ui.label("");
                        return;
                    }
                    let san_or_uci = analysis_for_this_game
                        .as_ref()
                        .and_then(|ga| ga.moves.get(ply))
                        .map(|m| m.move_san.clone())
                        .unwrap_or_else(|| uci_moves.get(ply).cloned().unwrap_or_default());
                    let class_color = analysis_for_this_game
                        .as_ref()
                        .and_then(|ga| ga.moves.get(ply))
                        .map(|m| classification_color(m.classification))
                        .unwrap_or_else(hydra_text);
                    let symbol = analysis_for_this_game
                        .as_ref()
                        .and_then(|ga| ga.moves.get(ply))
                        .map(|m| m.classification.symbol())
                        .unwrap_or("");
                    let label = format!("{san_or_uci}{symbol}");
                    let selected = *new_cursor == ply + 1;
                    let resp = ui.selectable_label(
                        selected,
                        egui::RichText::new(label)
                            .size(12.0)
                            .color(class_color)
                            .monospace(),
                    );
                    if resp.clicked() {
                        *new_cursor = ply + 1;
                    }
                };

                egui::ScrollArea::vertical()
                    .id_salt("analyze_move_list")
                    .auto_shrink([false, false])
                    .max_height(360.0)
                    .show(ui, |ui| {
                        egui::Grid::new("analyze_moves_grid")
                            .num_columns(3)
                            .spacing([10.0, 4.0])
                            .striped(true)
                            .show(ui, |ui| {
                                for pair_idx in 0..n_pairs {
                                    ui.label(
                                        egui::RichText::new(format!("{}.", pair_idx + 1))
                                            .size(11.0)
                                            .color(hydra_subtle_text())
                                            .monospace(),
                                    );
                                    render_ply(ui, pair_idx * 2, &mut new_cursor);
                                    render_ply(ui, pair_idx * 2 + 1, &mut new_cursor);
                                    ui.end_row();
                                }
                            });
                    });

                // Per-move detail, OUTSIDE the scroll area so it stays
                // visible when the move list scrolls.
                if new_cursor > 0
                    && let Some(ga) = analysis_for_this_game.as_ref()
                    && let Some(ma) = ga.moves.get(new_cursor - 1)
                {
                    ui.add_space(6.0);
                    ui.separator();
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(format!(
                            "{} — CPL {} | Eval {:.2} → {:.2} | Best: {} ({:.2})",
                            ma.classification.label(),
                            ma.cpl,
                            ma.eval_before as f64 / 100.0,
                            ma.eval_after as f64 / 100.0,
                            ma.best_move_uci,
                            ma.best_eval as f64 / 100.0,
                        ))
                        .size(11.0)
                        .color(hydra_subtle_text()),
                    );
                    if let Some(ref expl) = ma.explanation {
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new(expl)
                                .size(11.0)
                                .color(hydra_subtle_text()),
                        );
                    }
                }
            });
        });

        // Apply state mutations after the closures
        if new_cursor != cursor {
            if let Some(r) = self.replay_game.as_mut() {
                r.cursor = new_cursor.min(r.moves.len());
            }
        }
        if want_close {
            self.replay_game = None;
        }
        if want_analyze {
            let user_color = if self.replay_game.as_ref().map(|r| r.game.user_color.as_str())
                == Some("white")
            {
                Color::White
            } else {
                Color::Black
            };
            self.analysis_target_game_id = Some(game_id);
            self.start_analysis(uci_moves, user_color);
        }
    }

    fn draw_idle_home(&mut self, ui: &mut egui::Ui) {
        let (searching, status_message) = {
            let state = self.state.lock().unwrap();
            (state.search_info.searching, state.status_message.clone())
        };

        let ctx = ui.ctx().clone();
        let status_label = "Offline local play ready".to_string();
        let has_puzzles = self
            .db
            .as_ref()
            .and_then(|db| db.get_puzzle_counts().ok())
            .map_or(false, |(total, _)| total > 0);
        let has_analyzed = self
            .db
            .as_ref()
            .and_then(|db| db.get_accuracy_history(1).ok())
            .map_or(false, |h| !h.is_empty());
        let has_games = !self.recent_games.is_empty();

        // Welcome strip on Overview — replaces the old generic "Focalors"
        // title + redundant Profile card. The KPI strip below already shows
        // Rating and Games, so a personal greeting with status badge is all
        // the orientation chrome the page needs.
        if self.home_page == HomePage::Overview {
            let profile_name = self
                .profile
                .as_ref()
                .map(|p| p.name.clone())
                .unwrap_or_else(|| "there".to_string());
            ui.add_space(12.0);
            ui.horizontal_wrapped(|ui| {
                ui.vertical(|ui| {
                    ui.label(
                        egui::RichText::new(format!("Welcome back, {profile_name}"))
                            .size(26.0)
                            .strong(),
                    );
                    ui.label(
                        egui::RichText::new("Train, play, and review chess in one focused workspace.")
                            .size(12.0)
                            .color(hydra_subtle_text()),
                    );
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    hydra_badge(ui, &status_label, hydra_panel_alt_fill());
                });
            });
        }
        ui.add_space(14.0);
        ui.horizontal_wrapped(|ui| {
            if ui.add(home_nav_button(self.home_page == HomePage::Overview, "Overview")).clicked() {
                self.home_page = HomePage::Overview;
            }

            if ui.add(home_nav_button(self.home_page == HomePage::Analyze, "Analyze")).clicked() {
                self.home_page = HomePage::Analyze;
            }

            let progress_response = ui.add_enabled(
                has_analyzed,
                home_nav_button(self.home_page == HomePage::Progress, "Progress"),
            );
            if progress_response.clicked() {
                self.home_page = HomePage::Progress;
            }

            let statistics_response = ui.add_enabled(
                has_games,
                home_nav_button(self.home_page == HomePage::Statistics, "Statistics"),
            );
            if statistics_response.clicked() {
                self.home_page = HomePage::Statistics;
            }

            let history_response = ui.add_enabled(
                has_games,
                home_nav_button(self.home_page == HomePage::History, "History"),
            );
            if history_response.clicked() {
                self.home_page = HomePage::History;
            }

            let puzzle_response = ui.add_enabled(
                has_puzzles || self.puzzle_trainer.is_some(),
                home_nav_button(self.home_page == HomePage::Puzzles, "Puzzles"),
            );
            if puzzle_response.clicked() {
                self.home_page = HomePage::Puzzles;
            }
        });
        ui.add_space(6.0);
        ui.separator();
        ui.add_space(14.0);

        match self.home_page {
            HomePage::Overview => {
                // ── KPI strip ──────────────────────────────────────────
                self.draw_home_kpi_strip(ui);
                ui.add_space(14.0);

                // ── Local play setup ───────────────────────────────────
                self.draw_local_setup_card(ui, searching);

                if !status_message.is_empty() {
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(status_message)
                            .size(11.0)
                            .color(hydra_subtle_text()),
                    );
                }

                if let Some((ref msg, when)) = self.auto_adjust_message {
                    if when.elapsed().as_secs() < 8 {
                        ui.add_space(6.0);
                        hydra_callout_frame().show(ui, |ui| {
                            ui.label(
                                egui::RichText::new(msg)
                                    .size(12.0)
                                    .strong()
                                    .color(hydra_success()),
                            );
                        });
                    } else {
                        self.auto_adjust_message = None;
                    }
                }

                // ── Recent games + Your openings ───────────────────────
                if has_games {
                    ui.add_space(14.0);
                    let total_w = ui.available_width();
                    if total_w > 760.0 {
                        ui.columns(2, |cols| {
                            self.draw_recent_games_compact(&mut cols[0]);
                            self.draw_opening_stats(&mut cols[1]);
                        });
                    } else {
                        self.draw_recent_games_compact(ui);
                        self.draw_opening_stats(ui);
                    }
                }
            }
            HomePage::Analyze => self.draw_analyze_page(ui),
            HomePage::Progress => self.draw_coaching_report(ui),
            HomePage::Statistics => self.draw_statistics_panel(ui),
            HomePage::History => self.draw_game_history(ui),
            HomePage::Puzzles => {
                if self.puzzle_trainer.is_some() {
                    self.draw_puzzle_trainer(ui);
                } else {
                    hydra_card_frame().show(ui, |ui| {
                        ui.label(
                            egui::RichText::new("Puzzle Trainer")
                                .size(16.0)
                                .strong()
                                .color(hydra_accent()),
                        );
                        ui.add_space(4.0);
                        if has_puzzles {
                            ui.label(
                                egui::RichText::new(
                                    "Drill positions saved from your analyzed games.",
                                )
                                .size(12.0)
                                .color(hydra_subtle_text()),
                            );
                            ui.add_space(12.0);
                            if ui
                                .add_sized([ui.available_width().min(240.0), 38.0], primary_button("Start Puzzle Session"))
                                .clicked()
                            {
                                self.start_puzzle_trainer();
                            }
                        } else {
                            ui.label(
                                egui::RichText::new(
                                    "No saved puzzles yet. Analyze finished games first so Focalors can extract training positions.",
                                )
                                .size(12.0)
                                .color(hydra_subtle_text()),
                            );
                        }
                    });
                }
            }
        }

        self.draw_home_engine_settings_popup(&ctx, searching);
    }

    /// Render a 4-card KPI strip at the top of the Overview: rating,
    /// accuracy, win rate, puzzles. Each tile shows a big value, a delta
    /// indicator vs. an earlier window, and a sparkline when there's
    /// enough history. Gracefully degrades on first-run / no-data state.
    fn draw_home_kpi_strip(&mut self, ui: &mut egui::Ui) {
        let (rating_history, accuracy_history, total_w, total_l, total_d, puzzle_total, puzzle_solved) = {
            let db = match &self.db {
                Some(d) => d,
                None => return,
            };
            (
                db.get_rating_history(50).ok().unwrap_or_default(),
                db.get_accuracy_history(50).ok().unwrap_or_default(),
                db.get_result_counts().ok().unwrap_or((0, 0, 0)).0,
                db.get_result_counts().ok().unwrap_or((0, 0, 0)).1,
                db.get_result_counts().ok().unwrap_or((0, 0, 0)).2,
                db.get_puzzle_counts().ok().unwrap_or((0, 0)).0,
                db.get_puzzle_counts().ok().unwrap_or((0, 0)).1,
            )
        };

        let total_games = total_w + total_l + total_d;
        let current_rating = rating_history
            .last()
            .map(|(_, _, r)| *r)
            .unwrap_or_else(|| self.profile.as_ref().map_or(1200, |p| p.rating));
        let rating_delta = if rating_history.len() >= 2 {
            let earlier_idx = rating_history.len().saturating_sub(10);
            current_rating - rating_history[earlier_idx].2
        } else {
            0
        };
        let rating_spark: Vec<f64> = rating_history.iter().map(|(_, _, r)| *r as f64).collect();

        let (recent_acc, prior_acc, acc_spark) = if accuracy_history.is_empty() {
            (0.0_f64, 0.0_f64, Vec::<f64>::new())
        } else {
            let recent: Vec<f64> = accuracy_history.iter().rev().take(10).map(|(_, a)| *a).collect();
            let recent_avg = recent.iter().sum::<f64>() / recent.len() as f64;
            let prior: Vec<f64> = accuracy_history.iter().rev().skip(10).take(10).map(|(_, a)| *a).collect();
            let prior_avg = if prior.is_empty() {
                recent_avg
            } else {
                prior.iter().sum::<f64>() / prior.len() as f64
            };
            let spark: Vec<f64> = accuracy_history.iter().map(|(_, a)| *a).collect();
            (recent_avg, prior_avg, spark)
        };
        let acc_delta = recent_acc - prior_acc;

        let win_rate = if total_games > 0 {
            total_w as f64 / total_games as f64 * 100.0
        } else {
            0.0
        };
        let puzzle_rate = if puzzle_total > 0 {
            puzzle_solved as f64 / puzzle_total as f64 * 100.0
        } else {
            0.0
        };

        // KPI row: no per-tile card chrome. Each column is just a label,
        // value, and (optionally) sparkline laid out vertically — the page
        // spacing is what separates them, not boxes.
        ui.columns(4, |cols| {
            // RATING
            {
                let ui = &mut cols[0];
                ui.label(
                    egui::RichText::new("RATING")
                        .size(10.0).color(hydra_subtle_text()).strong(),
                );
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(format!("{current_rating}")).size(24.0).strong());
                    if rating_delta != 0 {
                        let color = if rating_delta > 0 { hydra_success() } else { hydra_danger() };
                        let arrow = if rating_delta > 0 { "▲" } else { "▼" };
                        ui.label(
                            egui::RichText::new(format!("{arrow} {}", rating_delta.abs()))
                                .size(11.0).color(color).strong(),
                        );
                    }
                });
                if rating_spark.len() >= 2 {
                    sparkline(ui, "home_rating_kpi", &rating_spark, hydra_accent(), 32.0);
                }
            }
            // ACCURACY
            {
                let ui = &mut cols[1];
                ui.label(
                    egui::RichText::new("ACCURACY")
                        .size(10.0).color(hydra_subtle_text()).strong(),
                );
                ui.horizontal(|ui| {
                    if acc_spark.is_empty() {
                        ui.label(egui::RichText::new("—").size(24.0).strong());
                    } else {
                        ui.label(
                            egui::RichText::new(format!("{recent_acc:.1}%"))
                                .size(24.0).strong().color(accuracy_color(recent_acc)),
                        );
                        if acc_delta.abs() > 0.5 {
                            let color = if acc_delta > 0.0 { hydra_success() } else { hydra_danger() };
                            let arrow = if acc_delta > 0.0 { "▲" } else { "▼" };
                            ui.label(
                                egui::RichText::new(format!("{arrow} {:.1}", acc_delta.abs()))
                                    .size(11.0).color(color).strong(),
                            );
                        }
                    }
                });
                if acc_spark.len() >= 2 {
                    sparkline(ui, "home_acc_kpi", &acc_spark, hydra_success(), 32.0);
                }
            }
            // WIN RATE
            {
                let ui = &mut cols[2];
                ui.label(
                    egui::RichText::new("WIN RATE")
                        .size(10.0).color(hydra_subtle_text()).strong(),
                );
                if total_games > 0 {
                    ui.label(egui::RichText::new(format!("{win_rate:.0}%")).size(24.0).strong());
                } else {
                    ui.label(egui::RichText::new("—").size(24.0).strong());
                }
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(format!("{total_w}W  {total_d}D  {total_l}L"))
                        .size(11.0).color(hydra_subtle_text()),
                );
            }
            // PUZZLES
            {
                let ui = &mut cols[3];
                ui.label(
                    egui::RichText::new("PUZZLES")
                        .size(10.0).color(hydra_subtle_text()).strong(),
                );
                if puzzle_total > 0 {
                    ui.label(egui::RichText::new(format!("{puzzle_rate:.0}%")).size(24.0).strong());
                } else {
                    ui.label(egui::RichText::new("—").size(24.0).strong());
                }
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(format!("{puzzle_solved}/{puzzle_total} solved"))
                        .size(11.0).color(hydra_subtle_text()),
                );
            }
        });
    }

    /// Compact "last 5 games" feed for the Overview right-side. Each row is
    /// a colored result chip + date + game info + a small "Review" button.
    /// Clicking Review loads the game into the Analyze tab and routes there.
    fn draw_recent_games_compact(&mut self, ui: &mut egui::Ui) {
        let games_snapshot: Vec<crate::db::SavedGame> = self
            .recent_games
            .iter()
            .take(5)
            .cloned()
            .collect();
        let mut replay_id: Option<i64> = None;
        // Naked section — header + list, no card. The page bg does the
        // structural work via spacing rather than wrapping each block in a
        // box.
        ui.label(egui::RichText::new("Recent Games").size(14.0).strong());
        ui.add_space(4.0);
        if games_snapshot.is_empty() {
            ui.label(
                egui::RichText::new("No games yet — play one to fill this in.")
                    .size(11.0).color(hydra_subtle_text()),
            );
            return;
        }
        for game in &games_snapshot {
            let result_color = match game.result.as_str() {
                "win" => hydra_success(),
                "loss" => hydra_danger(),
                _ => class_inaccuracy(),
            };
            let result_label = match game.result.as_str() {
                "win" => "W",
                "loss" => "L",
                "draw" => "D",
                _ => "?",
            };
            let date = &game.played_at[..10.min(game.played_at.len())];
            let moves = game.move_count.unwrap_or(0);
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!(" {result_label} "))
                        .color(hydra_text_on_accent())
                        .background_color(result_color)
                        .strong()
                        .monospace(),
                );
                ui.label(egui::RichText::new(date).size(11.0));
                ui.label(
                    egui::RichText::new(format!("as {} · {moves}m", game.user_color))
                        .size(10.0).color(hydra_subtle_text()),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Review").clicked() {
                        replay_id = Some(game.id);
                    }
                });
            });
            ui.add_space(2.0);
        }
        if let Some(id) = replay_id {
            self.start_replay(id);
            self.home_page = HomePage::Analyze;
        }
    }

    fn draw_local_setup_card(&mut self, ui: &mut egui::Ui, searching: bool) {
        hydra_card_frame().show(ui, |ui| {
            ui.label(
                egui::RichText::new("Local Play")
                    .strong()
                    .size(16.0)
                    .color(hydra_accent()),
            );
            ui.label(
                egui::RichText::new(
                    "Choose your side, clock, and Focalors profile before entering a focused offline session.",
                )
                .size(12.0)
                .color(hydra_subtle_text()),
            );

            ui.add_space(10.0);
            egui::Grid::new("local_setup_form")
                .num_columns(2)
                .spacing([16.0, 10.0])
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new("Side")
                            .size(11.0)
                            .strong()
                            .color(hydra_subtle_text()),
                    );
                    ui.horizontal_wrapped(|ui| {
                        ui.radio_value(&mut self.local_side_choice, SideChoice::White, "White");
                        ui.radio_value(&mut self.local_side_choice, SideChoice::Black, "Black");
                        ui.radio_value(&mut self.local_side_choice, SideChoice::Random, "Random");
                    });
                    ui.end_row();

                    ui.label(
                        egui::RichText::new("Time Control")
                            .size(11.0)
                            .strong()
                            .color(hydra_subtle_text()),
                    );
                    egui::ComboBox::from_id_salt("home_time_preset")
                        .selected_text(self.local_time_preset.label())
                        .width(180.0)
                        .show_ui(ui, |ui| {
                            for preset in TimePreset::ALL {
                                ui.selectable_value(&mut self.local_time_preset, preset, preset.label());
                            }
                        });
                    ui.end_row();

                    ui.label(
                        egui::RichText::new("Focalors Profile")
                            .size(11.0)
                            .strong()
                            .color(hydra_subtle_text()),
                    );
                    ui.horizontal(|ui| {
                        let prev_difficulty = self.local_difficulty;
                        egui::ComboBox::from_id_salt("home_local_difficulty")
                            .selected_text(self.local_difficulty.label())
                            .width(180.0)
                            .show_ui(ui, |ui| {
                                for difficulty in LocalDifficulty::ALL {
                                    ui.selectable_value(
                                        &mut self.local_difficulty,
                                        difficulty,
                                        difficulty.label(),
                                    );
                                }
                            });
                        if self.local_difficulty != prev_difficulty
                            && self.local_difficulty == LocalDifficulty::Custom
                        {
                            self.show_advanced_engine_settings = true;
                        }
                        if self.local_difficulty == LocalDifficulty::Custom
                            && ui.add(secondary_button("Advanced")).clicked()
                        {
                            self.show_advanced_engine_settings = true;
                        }
                    });
                    ui.end_row();
                });

            ui.add_space(6.0);
            ui.label(
                egui::RichText::new(self.local_difficulty.description(self.adaptive_level))
                    .size(11.0)
                    .color(hydra_subtle_text()),
            );

            ui.add_space(10.0);
            let start_clicked = ui
                .add_enabled_ui(!searching, |ui| {
                    ui.add_sized([ui.available_width(), 38.0], primary_button("Start Local Game"))
                        .clicked()
                })
                .inner;
            if start_clicked {
                self.begin_local_game(self.local_side_choice.resolve());
            }
        });
    }

    /// Persist a parsed PGN as a `SavedGame` with `engine_level = "imported"`,
    /// then route the user into the History → replay panel for that game.
    /// Imports get the same review pipeline as locally played games.
    fn save_imported_pgn_and_open_review(&mut self, parsed: &crate::pgn::ParsedPgn) {
        let db = match self.db.as_ref() {
            Some(d) => d,
            None => return,
        };
        let user_color_str = if self.pgn_import_user_color == Color::White {
            "white"
        } else {
            "black"
        };
        let result_token = parsed.result.as_deref().unwrap_or("*");
        let result = match (result_token, self.pgn_import_user_color) {
            ("1-0", Color::White) | ("0-1", Color::Black) => "win",
            ("1-0", Color::Black) | ("0-1", Color::White) => "loss",
            ("1/2-1/2", _) => "draw",
            _ => "draw",
        };
        // Re-emit a clean PGN from the UCI moves so generated SAN matches what
        // the replay panel expects.
        let white = parsed.white.clone().unwrap_or_else(|| "?".to_string());
        let black = parsed.black.clone().unwrap_or_else(|| "?".to_string());
        let pgn = crate::db::generate_pgn(
            &parsed.uci_moves,
            &white,
            &black,
            result_token,
            None,
            Some(&chrono_today()),
        );
        let move_count = parsed.uci_moves.len() as i32;

        let game_id = match db.save_game(
            user_color_str,
            result,
            Some("imported"),
            None,
            Some("imported"),
            &pgn,
            Some(move_count),
        ) {
            Ok(id) => id,
            Err(_) => return,
        };

        if let Some((_, opening_name)) = crate::openings::detect_opening(&parsed.uci_moves) {
            let _ = db.update_game_opening(game_id, opening_name);
        }

        self.recent_games = db.get_recent_games(20).ok().unwrap_or_default();
        self.pgn_import_text.clear();
        self.pgn_import_parsed = None;
        self.pgn_import_error = None;

        self.start_replay(game_id);
        self.home_page = HomePage::Analyze;
    }

    fn draw_home_engine_settings_popup(&mut self, ctx: &egui::Context, searching: bool) {
        if !self.show_advanced_engine_settings {
            return;
        }

        let mut settings = self.state.lock().unwrap().engine_settings.clone();
        let mut open = self.show_advanced_engine_settings;

        egui::Window::new("Advanced Engine Settings")
            .open(&mut open)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .collapsible(false)
            .resizable(false)
            .default_width(420.0)
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new(
                        "Active for analysis runs and the Custom local-game profile. Other difficulty profiles ignore these settings.",
                    )
                    .size(11.0)
                    .color(hydra_subtle_text()),
                );
                ui.add_space(8.0);
                draw_engine_settings_controls(ui, &mut settings, searching);
            });

        self.state.lock().unwrap().engine_settings = settings;
        self.show_advanced_engine_settings = open;
    }

    fn apply_human_move(&mut self, mv: Move) -> bool {
        let mut state = self.state.lock().unwrap();
        if state.local_game.outcome.is_some() {
            return false;
        }

        if handle_active_side_timeout(&mut state) {
            return false;
        }

        let mover = state.board.side_to_move;
        if !consume_local_turn_time(&mut state, mover) {
            return false;
        }

        let uci_str = mv.to_uci();

        make_move(&mut state.board, mv);
        state.local_game.time_control.add_increment(mover);
        record_local_snapshot(&mut state, uci_str.clone());
        state.status_message = format!("You played: {uci_str}");
        update_local_game_outcome(&mut state);
        if state.local_game.outcome.is_none() {
            state.local_game.time_control.start_turn_now();
        }
        true
    }

    fn complete_promotion(&mut self, piece: Piece) {
        let Some(pending) = self.pending_promotion.take() else {
            return;
        };

        if let Some(mv) = pending
            .moves
            .into_iter()
            .find(|mv| mv.promotion_piece() == piece)
        {
            if self.apply_human_move(mv) {
                self.start_engine_search();
            }
        }
    }

    fn cancel_promotion(&mut self) {
        self.pending_promotion = None;
        self.selected_square = None;
        self.drag_state = None;
        self.state.lock().unwrap().status_message = "Promotion cancelled.".to_string();
    }

    fn draw_promotion_window(&mut self, ctx: &egui::Context) {
        if self.pending_promotion.is_none() {
            return;
        }

        let mut chosen_piece = None;
        let mut cancel = false;

        egui::Window::new("Choose Promotion")
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label("Promote pawn to:");
                ui.horizontal(|ui| {
                    for piece in [Piece::Queen, Piece::Rook, Piece::Bishop, Piece::Knight] {
                        if ui.button(piece_name(piece)).clicked() {
                            chosen_piece = Some(piece);
                        }
                    }
                });
                ui.add_space(4.0);
                if ui.button("Cancel").clicked() {
                    cancel = true;
                }
            });

        if let Some(piece) = chosen_piece {
            self.complete_promotion(piece);
        } else if cancel {
            self.cancel_promotion();
        }
    }

}

impl eframe::App for FocalorsApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.update_local_clock_state();

        // Auto-save completed games to DB
        {
            let should_save = {
                let state = self.state.lock().unwrap();
                state.local_game.active
                    && state.local_game.outcome.is_some()
                    && !state.game_saved
            };
            if should_save {
                self.save_completed_game();
                self.state.lock().unwrap().game_saved = true;
            }
        }

        let (show_home_screen, mode_label, searching) = {
            let state = self.state.lock().unwrap();
            let show_home_screen = !state.local_game.active;
            let mode_label = if show_home_screen { "Home" } else { "Local Play" };
            (show_home_screen, mode_label, state.search_info.searching)
        };

        // Request repaint periodically for live updates
        let repaint_after = if self.has_running_local_clock() { 100 } else { 200 };
        ctx.request_repaint_after(Duration::from_millis(repaint_after));

        // Welcome dialog (first launch) — skip all other content
        if self.show_welcome {
            egui::CentralPanel::default().show_inside(ui, |_ui| {});
            self.draw_welcome_dialog(&ctx);
            return;
        }

        // Top menu bar
        egui::Panel::top("menu_bar").show_inside(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.label(
                    egui::RichText::new("FOCALORS")
                        .strong()
                        .size(18.0)
                        .color(hydra_accent()),
                );
                ui.label(
                    egui::RichText::new(mode_label)
                        .size(12.0)
                        .color(hydra_subtle_text()),
                );
                if searching {
                    ui.separator();
                    ui.spinner();
                    hydra_badge(ui, "Thinking", hydra_warning());
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let toggle_label = match self.ui_theme {
                        UiTheme::Dark => "Light Mode",
                        UiTheme::Light => "Dark Mode",
                    };
                    if ui.add(theme_toggle_button(toggle_label)).clicked() {
                        self.set_ui_theme(&ctx, self.ui_theme.toggle());
                    }
                });
            });
        });

        // Right panel: settings and controls (only during play)
        if !show_home_screen {
            egui::Panel::right("controls")
                .min_size(300.0)
                .max_size(360.0)
                .show_inside(ui, |ui| {
                    self.draw_controls(ui);
                });

            // Bottom info strip: clocks + status + last move (compact, no scroll)
            egui::Panel::bottom("play_status_bar")
                .resizable(false)
                .default_size(56.0)
                .min_size(48.0)
                .max_size(96.0)
                .show_inside(ui, |ui| {
                    self.draw_play_status_bar(ui);
                });
        }

        // Central panel: home (scrollable) or board only (fills remaining space)
        egui::CentralPanel::default().show_inside(ui, |ui| {
            if show_home_screen {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        // Outer gutter: keeps dashboard content away from
                        // the window edge and the scrollbar so charts and
                        // bars don't run flush against the right side.
                        egui::Frame::new()
                            .inner_margin(egui::Margin {
                                left: 24,
                                right: 24,
                                top: 0,
                                bottom: 0,
                            })
                            .show(ui, |ui| {
                                self.draw_idle_home(ui);
                            });
                    });
            } else {
                ui.add_space(6.0);
                self.draw_board(ui, None);
            }
        });
    }
}

impl FocalorsApp {
    // ── Chess board ────────────────────────────────────────────────────

    fn draw_board(&mut self, ui: &mut egui::Ui, override_pos: Option<&BoardView<'_>>) {
        let interactive = override_pos.map(|v| v.interactive).unwrap_or(true);
        let board = match override_pos {
            Some(v) => v.board.clone(),
            None => self.state.lock().unwrap().board.clone(),
        };

        // Clear annotations when the (main) board changes. Covers move played,
        // engine reply, move-list navigation, new game — anything that shifts
        // the position invalidates user-drawn arrows/highlights.
        if interactive {
            if self.last_annotation_board_hash != Some(board.hash) {
                self.annotations.clear();
                self.right_drag_origin = None;
                self.last_annotation_board_hash = Some(board.hash);
            }
        }
        let last_move_squares: Option<(u8, u8)> = override_pos
            .and_then(|v| v.last_move)
            .filter(|m| !m.is_null())
            .map(|m| (m.from_sq().0, m.to_sq().0));
        let available = ui.available_size();
        let max_height = (available.y - 18.0).max(180.0);
        let board_size = available.x.min(max_height);
        let sq_size = board_size / 8.0;
        let board_total_height = board_size + 28.0;

        let light = egui::Color32::from_rgb(227, 218, 201);
        let dark = egui::Color32::from_rgb(120, 99, 81);
        let selected_color = egui::Color32::from_rgba_premultiplied(216, 179, 92, 180);
        let legal_color = egui::Color32::from_rgba_premultiplied(148, 126, 98, 110);
        let last_move_color = egui::Color32::from_rgba_premultiplied(212, 181, 96, 130);

        // Compute legal moves from selected square (interactive only)
        let legal_targets: Vec<u8> = if interactive
            && let Some(from_sq) = self.selected_square
        {
            let moves = generate_legal_moves(&board);
            let mut targets = Vec::new();
            for i in 0..moves.len() {
                let mv = moves[i];
                if mv.from_sq().0 == from_sq {
                    targets.push(mv.to_sq().0);
                }
            }
            targets
        } else {
            Vec::new()
        };

        // Draw board
        let (response, painter) = ui.allocate_painter(
            egui::vec2(available.x, board_total_height),
            egui::Sense::click_and_drag(),
        );
        let board_rect = egui::Rect::from_min_size(
            egui::pos2(response.rect.center().x - board_size / 2.0, response.rect.min.y),
            egui::vec2(board_size, board_size),
        );
        let board_shell = board_rect.expand(14.0);
        let allow_input = interactive && self.local_input_enabled();
        let dragged_from_sq = if interactive {
            self.drag_state.map(|drag| drag.from_sq)
        } else {
            None
        };
        let drag_hover_sq = if interactive {
            self.drag_state.and_then(|drag| {
                board_square_from_pos(board_rect, sq_size, self.flipped, drag.pointer_pos)
            })
        } else {
            None
        };
        let selected_square = if interactive { self.selected_square } else { None };

        painter.rect_filled(
            board_shell.translate(egui::vec2(0.0, 8.0)),
            24.0,
            egui::Color32::from_rgba_premultiplied(0, 0, 0, 62),
        );
        painter.rect_filled(board_shell, 24.0, hydra_panel_raised_fill());
        painter.rect_filled(board_shell.shrink(1.5), 22.0, hydra_panel_fill());

        for rank in 0..8u8 {
            for file in 0..8u8 {
                let display_rank = if self.flipped { rank } else { 7 - rank };
                let display_file = if self.flipped { 7 - file } else { file };
                let sq_idx = display_rank * 8 + display_file;

                let x = board_rect.min.x + file as f32 * sq_size;
                let y = board_rect.min.y + rank as f32 * sq_size;
                let rect = egui::Rect::from_min_size(
                    egui::pos2(x, y),
                    egui::vec2(sq_size, sq_size),
                );

                // Square color. a1 (file 0, rank 0) must be dark — light squares
                // are the odd-parity ones, matching the standard "light square on
                // the right" / white queen on her own (light) color. This was
                // inverted (== 0), which put the white queen on a dark d1.
                let is_light = (display_rank + display_file) % 2 == 1;
                let base_color = if is_light { light } else { dark };
                let is_last_move = last_move_squares
                    .map(|(f, t)| f == sq_idx || t == sq_idx)
                    .unwrap_or(false);
                let color = if selected_square == Some(sq_idx) {
                    selected_color
                } else if drag_hover_sq == Some(sq_idx) {
                    egui::Color32::from_rgba_premultiplied(201, 150, 74, 135)
                } else if legal_targets.contains(&sq_idx) {
                    legal_color
                } else if is_last_move {
                    last_move_color
                } else {
                    base_color
                };

                painter.rect_filled(rect, 0.0, color);

                // User-drawn square highlight (right-click annotation)
                if interactive {
                    for annot in &self.annotations {
                        if let Annotation::Highlight { sq: h_sq, color: h_color } = *annot
                            && h_sq == sq_idx
                        {
                            painter.rect_filled(rect, 0.0, h_color.fill());
                        }
                    }
                }

                // Draw piece
                if let Some((piece_color, piece)) = board.piece_on(Square(sq_idx)) {
                    if dragged_from_sq == Some(sq_idx) {
                        continue;
                    }

                    if let Some(tex) = self.piece_textures.get(&(piece_color, piece)) {
                        let padding = sq_size * 0.05;
                        let img_rect = egui::Rect::from_min_size(
                            egui::pos2(rect.min.x + padding, rect.min.y + padding),
                            egui::vec2(sq_size - 2.0 * padding, sq_size - 2.0 * padding),
                        );
                        draw_piece_image(&painter, tex, img_rect);
                    }
                }

                // Legal move dot
                if legal_targets.contains(&sq_idx) && board.piece_on(Square(sq_idx)).is_none() {
                    painter.circle_filled(
                        rect.center(),
                        sq_size * 0.15,
                        egui::Color32::from_rgba_premultiplied(244, 233, 210, 168),
                    );
                }
            }
        }

        // File labels
        let label_y = board_rect.max.y + 2.0;
        for file in 0..8u8 {
            let display_file = if self.flipped { 7 - file } else { file };
            let x = board_rect.min.x + file as f32 * sq_size + sq_size / 2.0;
            painter.text(
                egui::pos2(x, label_y),
                egui::Align2::CENTER_TOP,
                (b'a' + display_file) as char,
                egui::FontId::proportional(12.0),
                egui::Color32::GRAY,
            );
        }

        // Rank labels
        for rank in 0..8u8 {
            let display_rank = if self.flipped { rank } else { 7 - rank };
            let y = board_rect.min.y + rank as f32 * sq_size + sq_size / 2.0;
            painter.text(
                egui::pos2(board_rect.min.x - 12.0, y),
                egui::Align2::CENTER_CENTER,
                (b'1' + display_rank) as char,
                egui::FontId::proportional(12.0),
                egui::Color32::GRAY,
            );
        }

        // User-drawn arrows (right-click annotations) — rendered last so they
        // sit on top of pieces. Highlights were drawn earlier under the pieces.
        if interactive {
            for annot in &self.annotations {
                if let Annotation::Arrow { from, to, color } = *annot {
                    let from_pos = board_square_center(board_rect, sq_size, self.flipped, from);
                    let to_pos   = board_square_center(board_rect, sq_size, self.flipped, to);
                    draw_annotation_arrow(&painter, from_pos, to_pos, color.arrow(), sq_size);
                }
            }
        }

        // Right-click annotation handling — press records origin square,
        // release resolves into a highlight (no drag) or arrow (drag), with
        // modifier keys selecting the color.
        if interactive {
            let secondary_pressed = ui.ctx().input(|i| i.pointer.secondary_pressed());
            let secondary_released = ui.ctx().input(|i| i.pointer.secondary_released());
            let pointer_pos = ui.ctx().input(|i| i.pointer.interact_pos());
            let modifiers = ui.ctx().input(|i| i.modifiers);

            if secondary_pressed
                && let Some(pos) = pointer_pos
                && let Some(sq) = board_square_from_pos(board_rect, sq_size, self.flipped, pos)
            {
                self.right_drag_origin = Some(sq);
            }

            if secondary_released
                && let Some(from_sq) = self.right_drag_origin.take()
                && let Some(pos) = pointer_pos
                && let Some(to_sq) = board_square_from_pos(board_rect, sq_size, self.flipped, pos)
            {
                let color = AnnotationColor::from_modifiers(modifiers);
                let new_annot = if from_sq == to_sq {
                    Annotation::Highlight { sq: from_sq, color }
                } else {
                    Annotation::Arrow { from: from_sq, to: to_sq, color }
                };
                self.toggle_annotation(new_annot);
            }
        }

        if response.drag_started() && allow_input && self.pending_promotion.is_none() {
            if let Some(pos) = response.interact_pointer_pos() {
                if let Some(from_sq) = board_square_from_pos(board_rect, sq_size, self.flipped, pos)
                {
                    if let Some((piece_color, piece)) = board.piece_on(Square(from_sq)) {
                        if piece_color == board.side_to_move {
                            self.drag_state = Some(DragState {
                                from_sq,
                                piece_color,
                                piece,
                                pointer_pos: pos,
                            });
                            self.selected_square = Some(from_sq);
                        }
                    }
                }
            }
        }

        if interactive
            && let Some(drag_state) = &mut self.drag_state
        {
            if let Some(pos) = response.interact_pointer_pos() {
                drag_state.pointer_pos = pos;
            }
        }

        // A left-click anywhere on the board also clears annotations (matches
        // lichess/chess.com convention — you click to commit to the plan, the
        // arrows you drew while thinking go away). Fires even if `allow_input`
        // is false so reviewing the board still respects the clear.
        if interactive && response.clicked() {
            self.annotations.clear();
        }

        // Handle clicks
        if response.clicked() && allow_input && self.pending_promotion.is_none() {
            if let Some(pos) = response.interact_pointer_pos() {
                if let Some(sq_idx) = board_square_from_pos(board_rect, sq_size, self.flipped, pos)
                {
                    if let Some(from_sq) = self.selected_square {
                        // Try to make a move
                        let success = self.try_make_move(from_sq, sq_idx);
                        self.selected_square = None;
                        if success {
                            // Engine responds
                            self.start_engine_search();
                        }
                    } else {
                        // Select a piece
                        let b = self.state.lock().unwrap().board.clone();
                        if let Some((color, _)) = b.piece_on(Square(sq_idx)) {
                            if color == b.side_to_move {
                                self.selected_square = Some(sq_idx);
                            }
                        }
                    }
                }
            }
        }

        let pointer_down = ui.ctx().input(|input| input.pointer.primary_down());
        if interactive && let Some(drag_state) = self.drag_state {
            if let Some(texture) = self
                .piece_textures
                .get(&(drag_state.piece_color, drag_state.piece))
            {
                let piece_size = sq_size * 0.94;
                let piece_rect = egui::Rect::from_center_size(
                    drag_state.pointer_pos,
                    egui::vec2(piece_size, piece_size),
                );
                draw_piece_image(&painter, texture, piece_rect);
            }

            if !pointer_down {
                self.drag_state = None;
                self.selected_square = None;

                if allow_input {
                    if let Some(target_sq) = board_square_from_pos(
                        board_rect,
                        sq_size,
                        self.flipped,
                        drag_state.pointer_pos,
                    ) {
                        let success = self.try_make_move(drag_state.from_sq, target_sq);
                        if success {
                            self.start_engine_search();
                        }
                    }
                }
            }
        }

        if interactive {
            self.draw_promotion_window(ui.ctx());
        }
    }

    fn try_make_move(&mut self, from: u8, to: u8) -> bool {
        let state = self.state.lock().unwrap();
        let moves = generate_legal_moves(&state.board);
        let in_puzzle = self.puzzle_trainer.is_some();
        drop(state);

        let mut promotion_moves = Vec::new();

        for i in 0..moves.len() {
            let mv = moves[i];
            if mv.from_sq().0 == from && mv.to_sq().0 == to {
                if matches!(mv.flag(), MoveFlag::Promotion) {
                    promotion_moves.push(mv);
                    continue;
                }

                if in_puzzle {
                    self.handle_puzzle_move(mv);
                    return true;
                }
                return self.apply_human_move(mv);
            }
        }

        if !promotion_moves.is_empty() {
            self.pending_promotion = Some(PendingPromotion {
                moves: promotion_moves,
            });
            self.state.lock().unwrap().status_message = "Choose a promotion piece.".to_string();
        }

        false
    }

    fn start_engine_search(&self) {
        self.start_engine_search_inner(false);
    }

    fn start_engine_search_inner(&self, force: bool) {
        let state = self.state.clone();
        let local_request = {
            let mut s = state.lock().unwrap();

            if s.search_info.searching {
                return;
            }

            if !force {
                if !s.local_game.active
                    || s.local_game.outcome.is_some()
                    || s.board.side_to_move == s.local_game.human_color
                {
                    return;
                }
            }

            if handle_active_side_timeout(&mut s) {
                return;
            }

            let moves = generate_legal_moves(&s.board);
            if moves.is_empty() {
                return;
            }

            let config = crate::strength::StrengthConfig::from_level(s.local_game.numeric_level);
            local_engine_search_request(&s, &config)
        };

        thread::spawn(move || {
            // RAII guard: if this worker exits via panic, search_info.searching
            // would stay `true` forever and the GUI would show the "thinking"
            // state with no way to recover except restart. On unwind the guard
            // clears the flag and surfaces a status message — but ONLY if we
            // are still the active search (generation match). If a newer
            // search has taken over (e.g. invalidate_local_search bumped the
            // generation and started another search), our state belongs to
            // that newer search and we leave it alone. On the success path
            // the worker explicitly sets searching=false at the end, so the
            // guard sees a cleared flag and does nothing.
            struct SearchGuard {
                state: Arc<Mutex<SharedState>>,
                our_generation: Option<u64>,
            }
            impl Drop for SearchGuard {
                fn drop(&mut self) {
                    let mut s = match self.state.lock() {
                        Ok(g) => g,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    let still_current = match self.our_generation {
                        Some(our) => s.local_search_generation == our,
                        None => true,
                    };
                    if !still_current { return; }
                    if s.search_info.searching {
                        s.search_info.searching = false;
                        s.status_message =
                            "Search failed unexpectedly. Make your move or restart.".to_string();
                        eprintln!(
                            "focalors: search worker exited unexpectedly; \
                             UI reset"
                        );
                    }
                }
            }
            let _search_guard = SearchGuard {
                state: state.clone(),
                our_generation: local_request.as_ref().map(|r| r.generation),
            };

            let (board, settings) = {
                let mut s = state.lock().unwrap();
                s.search_info.searching = true;
                s.status_message = if force {
                    "Engine thinking...".to_string()
                } else {
                    "Focalors thinking...".to_string()
                };
                (s.board.clone(), s.engine_settings.clone())
            };

            // Build position history for repetition detection
            let position_hashes: Vec<u64> = {
                let s = state.lock().unwrap();
                s.local_history.iter().map(|snap| snap.board.hash).collect()
            };

            // Use persistent searcher for TT reuse across moves
            let searcher_arc = state.lock().unwrap().persistent_searcher.clone();
            let mut searcher = searcher_arc.lock().unwrap();
            searcher.set_position_history(position_hashes.clone());

            let result = if let Some(ref request) = local_request {
                if request.depth_cap.is_none() {
                    // Master / time-bound Custom: Lazy SMP across all cores.
                    // The TT is shared with the persistent searcher so cross-
                    // move TT reuse still works.
                    let tt = searcher.tt.clone();
                    let use_nnue = searcher.use_nnue;
                    let n_threads = std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(1);
                    crate::search::search_lazy_smp(
                        tt,
                        &board,
                        use_nnue,
                        n_threads,
                        request.soft_time_ms,
                        request.hard_time_ms,
                        request.depth_cap,
                        position_hashes,
                    )
                } else {
                    // Beginner / Club / Tournament / depth-bound Custom:
                    // single-thread is more efficient for shallow depth caps.
                    searcher.search_with_time_management_capped(
                        &board,
                        request.soft_time_ms,
                        request.hard_time_ms,
                        request.depth_cap,
                    )
                }
            } else if settings.use_time_limit {
                searcher.search_timed(&board, settings.think_time_ms)
            } else {
                searcher.search(&board, settings.max_depth)
            };

            // Apply weighted move selection for difficulty levels that use it
            let final_move = if let Some(ref request) = local_request {
                crate::strength::select_move(
                    &mut searcher,
                    &board,
                    result.best_move,
                    result.score,
                    result.depth,
                    &request.strength_config,
                )
            } else {
                result.best_move
            };

            let mut s = state.lock().unwrap();
            if let Some(ref request) = local_request {
                if s.local_search_generation != request.generation {
                    return;
                }
            }
            s.search_info.depth = result.depth;
            s.search_info.score = result.score;
            s.search_info.nodes = result.nodes;
            s.search_info.best_move = final_move.to_uci();
            s.search_info.searching = false;

            if s.local_game.active && s.local_game.outcome.is_some() {
                return;
            }

            if !final_move.is_null() {
                if s.local_game.active {
                    let mover = s.board.side_to_move;
                    if !consume_local_turn_time(&mut s, mover) {
                        return;
                    }
                    s.local_game.time_control.add_increment(mover);
                }

                let uci_str = final_move.to_uci();
                make_move(&mut s.board, final_move);
                if s.local_game.active {
                    record_local_snapshot(&mut s, uci_str.clone());
                    update_local_game_outcome(&mut s);
                    if s.local_game.outcome.is_none() {
                        s.local_game.time_control.start_turn_now();
                        s.status_message = format!("Focalors played: {uci_str}. Your move.");
                    }
                } else {
                    s.move_history.push(uci_str.clone());
                    s.status_message = format!(
                        "Engine played: {} (depth {}, {} cp, {} nodes)",
                        uci_str, result.depth, result.score, result.nodes
                    );
                }
            }
        });
    }

    // ── Compact bottom status bar (clocks + status + eval) ─────────────

    fn draw_play_status_bar(&self, ui: &mut egui::Ui) {
        let state = self.state.lock().unwrap();
        let active_side = if state.local_game.active && state.local_game.outcome.is_none() {
            Some(state.board.side_to_move)
        } else {
            None
        };

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if state.local_game.active {
                let white_ms = state
                    .local_game
                    .time_control
                    .displayed_remaining_ms(Color::White, active_side);
                let black_ms = state
                    .local_game
                    .time_control
                    .displayed_remaining_ms(Color::Black, active_side);

                ui.label(
                    egui::RichText::new("White")
                        .size(10.0)
                        .strong()
                        .color(hydra_subtle_text()),
                );
                ui.label(
                    egui::RichText::new(format_clock_ms(white_ms))
                        .size(15.0)
                        .strong()
                        .monospace()
                        .color(clock_color(white_ms, active_side == Some(Color::White))),
                );
                ui.add_space(12.0);
                ui.label(
                    egui::RichText::new("Black")
                        .size(10.0)
                        .strong()
                        .color(hydra_subtle_text()),
                );
                ui.label(
                    egui::RichText::new(format_clock_ms(black_ms))
                        .size(15.0)
                        .strong()
                        .monospace()
                        .color(clock_color(black_ms, active_side == Some(Color::Black))),
                );
                ui.add_space(14.0);
                ui.separator();
                ui.add_space(8.0);
            }

            ui.add(
                egui::Label::new(
                    egui::RichText::new(&state.status_message)
                        .size(12.0)
                        .color(hydra_text()),
                )
                .truncate(),
            );

            if state.search_info.depth > 0 {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // search_info.score comes from the engine's own search, so it
                    // is signed from the engine's perspective. The engine is always
                    // your opponent here, so negating it gives a "you-positive"
                    // reading: + means you are winning, - means you are losing.
                    let score = -state.search_info.score;
                    let score_text = if score.abs() > 20000 {
                        let mate_in = (29000 - score.abs() + 1) / 2;
                        if score > 0 {
                            format!("M{mate_in}")
                        } else {
                            format!("-M{mate_in}")
                        }
                    } else {
                        format!("{:+.2}", score as f64 / 100.0)
                    };

                    if !state.search_info.best_move.is_empty() {
                        ui.label(
                            egui::RichText::new(&state.search_info.best_move)
                                .size(11.0)
                                .strong()
                                .monospace()
                                .color(hydra_success()),
                        );
                        ui.label(
                            egui::RichText::new("best")
                                .size(10.0)
                                .color(hydra_subtle_text()),
                        );
                        ui.add_space(10.0);
                    }

                    ui.label(
                        egui::RichText::new(score_text)
                            .size(12.0)
                            .strong()
                            .monospace()
                            .color(hydra_accent()),
                    );
                    ui.label(
                        egui::RichText::new("eval")
                            .size(10.0)
                            .color(hydra_subtle_text()),
                    );
                    ui.add_space(10.0);

                    ui.label(
                        egui::RichText::new(format!(
                            "d{} · {}",
                            state.search_info.depth,
                            format_nodes(state.search_info.nodes)
                        ))
                        .size(11.0)
                        .monospace()
                        .color(hydra_subtle_text()),
                    );
                });
            } else if state.search_info.searching {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new("Focalors thinking…")
                            .size(11.0)
                            .color(hydra_warning()),
                    );
                });
            }
        });
    }

    // ── Controls panel ─────────────────────────────────────────────────

    fn draw_controls(&mut self, ui: &mut egui::Ui) {
        let local_active = self.state.lock().unwrap().local_game.active;

        if local_active {
            self.draw_local_game_controls(ui);
        }

        // Eval explanation panel toggle + display
        ui.add_space(8.0);
        let label = if self.show_eval_panel { "Hide Eval" } else { "Show Eval" };
        if ui.add_sized([ui.available_width(), 28.0], secondary_button(label)).clicked() {
            self.show_eval_panel = !self.show_eval_panel;
        }
        if self.show_eval_panel {
            self.draw_eval_panel(ui);
        }
    }

    fn draw_local_game_controls(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            let (
                searching,
                local_game,
                side_to_move,
                history_cursor,
                history_len,
                history_choices,
                history_paused,
            ) = {
                let state = self.state.lock().unwrap();
                let history_choices = state
                    .local_history
                    .iter()
                    .enumerate()
                    .map(|(index, snapshot)| {
                        (index, local_history_label(index, snapshot.move_uci.as_deref()))
                    })
                    .collect::<Vec<_>>();
                (
                    state.search_info.searching,
                    state.local_game.clone(),
                    state.board.side_to_move,
                    state.local_history_cursor,
                    state.local_history.len(),
                    history_choices,
                    is_reviewing_history(&state) || state.local_game.time_control.active_since.is_none(),
                )
            };

            let mut navigate_to = None;
            let mut resume = false;
            let mut abort = false;

            hydra_card_frame().show(ui, |ui| {
                ui.label(
                    egui::RichText::new("Game Session")
                        .strong()
                        .size(16.0)
                        .color(hydra_accent()),
                );
                ui.add_space(10.0);

                egui::Grid::new("local_session_summary")
                    .num_columns(2)
                    .spacing([10.0, 6.0])
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new("You")
                                .size(11.0)
                                .strong()
                                .color(hydra_subtle_text()),
                        );
                        ui.label(
                            egui::RichText::new(color_name(local_game.human_color))
                                .size(12.0)
                                .color(hydra_text()),
                        );
                        ui.end_row();

                        ui.label(
                            egui::RichText::new("Profile")
                                .size(11.0)
                                .strong()
                                .color(hydra_subtle_text()),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "{} / {}",
                                local_game.time_control.label,
                                local_game.difficulty.label()
                            ))
                            .size(12.0)
                            .color(hydra_text()),
                        );
                        ui.end_row();

                        ui.label(
                            egui::RichText::new("To move")
                                .size(11.0)
                                .strong()
                                .color(hydra_subtle_text()),
                        );
                        ui.label(
                            egui::RichText::new(color_name(side_to_move))
                                .size(12.0)
                                .color(hydra_text()),
                        );
                        ui.end_row();
                    });

                if history_paused && local_game.outcome.is_none() {
                    ui.add_space(8.0);
                    hydra_callout_frame().show(ui, |ui| {
                        ui.label(
                            egui::RichText::new("History view active. Clocks stay paused until you resume.")
                                .size(11.0)
                                .color(hydra_warning()),
                        );
                    });
                }

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(8.0);
                ui.label(egui::RichText::new("Quick Actions").strong().color(hydra_accent()));
                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    if ui.add_sized([138.0, 34.0], secondary_button("Flip Board")).clicked() {
                        self.flipped = !self.flipped;
                    }

                    let can_resign = local_game.outcome.is_none() && !searching;
                    let resign_clicked = ui
                        .add_enabled_ui(can_resign, |ui| {
                            ui.add_sized([138.0, 34.0], danger_button("Resign"))
                                .clicked()
                        })
                        .inner;
                    if resign_clicked {
                        let mut s = self.state.lock().unwrap();
                        self.pending_promotion = None;
                        self.selected_square = None;
                        self.drag_state = None;
                        set_local_game_outcome(
                            &mut s,
                            GameOutcome::Resignation(local_game.human_color.flip()),
                        );
                    }

                    if ui.add_sized([138.0, 34.0], danger_button("End Game")).clicked() {
                        abort = true;
                    }
                });

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(8.0);
                ui.label(egui::RichText::new("History").strong().color(hydra_accent()));
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(history_cursor > 0, secondary_button("< Prev"))
                        .clicked()
                    {
                        navigate_to = Some(history_cursor - 1);
                    }
                    if ui
                        .add_enabled(history_cursor + 1 < history_len, secondary_button("Next >"))
                        .clicked()
                    {
                        navigate_to = Some(history_cursor + 1);
                    }
                    ui.menu_button("Jump", |ui| {
                        for (index, label) in &history_choices {
                            let selected = *index == history_cursor;
                            if ui.selectable_label(selected, label).clicked() {
                                navigate_to = Some(*index);
                                ui.close();
                            }
                        }
                    });
                });

                let resume_enabled = local_game.outcome.is_none() && !searching && history_paused;
                let resume_clicked = ui
                    .add_enabled_ui(resume_enabled, |ui| {
                        ui.add_sized([ui.available_width(), 34.0], primary_button("Resume from Here"))
                            .clicked()
                    })
                    .inner;
                if resume_clicked {
                    resume = true;
                }

                if let Some((_, current_label)) = history_choices.get(history_cursor) {
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new(format!("Current position: {current_label}"))
                            .size(11.0)
                            .color(hydra_subtle_text()),
                    );
                }

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(8.0);
                let (status_text, status_color) = if let Some(outcome) = local_game.outcome {
                    local_outcome_banner(outcome, local_game.human_color)
                } else if searching {
                    ("Focalors is thinking...".to_string(), hydra_warning())
                } else if history_paused {
                    ("Reviewing the game history.".to_string(), hydra_subtle_text())
                } else if side_to_move == local_game.human_color {
                    ("Your turn.".to_string(), hydra_text())
                } else {
                    ("Focalors to move.".to_string(), hydra_subtle_text())
                };
                ui.label(
                    egui::RichText::new(status_text)
                        .color(status_color)
                        .strong()
                        .size(15.0),
                );
            });

            self.draw_analysis_button(ui);

            if abort {
                self.abort_local_game();
            }
            if let Some(target_index) = navigate_to {
                self.navigate_local_history(target_index);
            }
            if resume {
                self.resume_local_game();
            }
        });
    }

}

fn current_theme() -> UiTheme {
    match ACTIVE_THEME.load(Ordering::Relaxed) {
        1 => UiTheme::Light,
        _ => UiTheme::Dark,
    }
}

fn theme_rgb(dark: [u8; 3], light: [u8; 3]) -> egui::Color32 {
    match current_theme() {
        UiTheme::Dark => egui::Color32::from_rgb(dark[0], dark[1], dark[2]),
        UiTheme::Light => egui::Color32::from_rgb(light[0], light[1], light[2]),
    }
}

fn theme_rgba(dark: [u8; 4], light: [u8; 4]) -> egui::Color32 {
    match current_theme() {
        UiTheme::Dark => egui::Color32::from_rgba_premultiplied(dark[0], dark[1], dark[2], dark[3]),
        UiTheme::Light => egui::Color32::from_rgba_premultiplied(light[0], light[1], light[2], light[3]),
    }
}

fn hydra_accent() -> egui::Color32 {
    theme_rgb([118, 147, 255], [69, 102, 214])
}

fn hydra_accent_soft() -> egui::Color32 {
    theme_rgba([118, 147, 255, 34], [69, 102, 214, 24])
}

fn hydra_text() -> egui::Color32 {
    theme_rgb([232, 236, 242], [34, 39, 46])
}

fn hydra_text_on_accent() -> egui::Color32 {
    theme_rgb([246, 248, 252], [247, 249, 255])
}

fn hydra_subtle_text() -> egui::Color32 {
    theme_rgb([151, 160, 173], [82, 92, 108])
}

fn hydra_bg() -> egui::Color32 {
    theme_rgb([20, 24, 31], [237, 240, 244])
}

fn hydra_panel_fill() -> egui::Color32 {
    // Slightly brighter than the page background so borderless cards still
    // read as "elevated" surfaces. With the 1px border removed, this fill
    // delta is the primary differentiator between card and page.
    theme_rgb([36, 41, 50], [252, 253, 255])
}

fn hydra_panel_alt_fill() -> egui::Color32 {
    theme_rgb([38, 44, 53], [242, 245, 248])
}

fn hydra_panel_raised_fill() -> egui::Color32 {
    theme_rgb([47, 55, 65], [233, 237, 242])
}

fn hydra_border() -> egui::Color32 {
    theme_rgb([66, 76, 89], [208, 215, 224])
}

fn hydra_success() -> egui::Color32 {
    theme_rgb([111, 191, 143], [51, 126, 87])
}

fn hydra_warning() -> egui::Color32 {
    theme_rgb([203, 159, 92], [170, 121, 51])
}

fn hydra_danger() -> egui::Color32 {
    theme_rgb([199, 109, 102], [181, 84, 72])
}

// Chess move classification palette — used by the Game Review move list,
// the Progress error breakdown bars, and any eval-graph highlights. Matches
// the chess.com convention so users coming from that tool already recognize
// the meaning of each color. Allow dead code until per-tab rebuilds wire
// them in.
#[allow(dead_code)]
fn class_book() -> egui::Color32 {
    theme_rgb([170, 130, 90], [140, 100, 60])
}

#[allow(dead_code)]
fn class_best() -> egui::Color32 {
    theme_rgb([120, 200, 120], [70, 160, 70])
}

#[allow(dead_code)]
fn class_good() -> egui::Color32 {
    theme_rgb([110, 180, 220], [60, 140, 200])
}

#[allow(dead_code)]
fn class_inaccuracy() -> egui::Color32 {
    theme_rgb([230, 200, 100], [200, 160, 50])
}

#[allow(dead_code)]
fn class_mistake() -> egui::Color32 {
    theme_rgb([230, 150, 80], [200, 110, 40])
}

#[allow(dead_code)]
fn class_blunder() -> egui::Color32 {
    theme_rgb([220, 100, 100], [190, 70, 70])
}

#[allow(dead_code)]
fn class_brilliant() -> egui::Color32 {
    theme_rgb([100, 220, 220], [50, 180, 180])
}

/// Embed a tiny trend line inside a KPI tile. No axes, no grid, no
/// interaction — pure visual cue showing direction of a series. Pass an
/// `id` that's unique among siblings to keep egui's plot bookkeeping happy.
#[allow(dead_code)]
fn sparkline(ui: &mut egui::Ui, id: &str, points: &[f64], color: egui::Color32, height: f32) {
    if points.len() < 2 {
        ui.add_space(height);
        return;
    }
    let pts: Vec<[f64; 2]> = points
        .iter()
        .enumerate()
        .map(|(i, v)| [i as f64, *v])
        .collect();
    let line = egui_plot::Line::new(id, egui_plot::PlotPoints::new(pts)).color(color);
    egui_plot::Plot::new(format!("sparkline_{id}"))
        .height(height)
        .show_axes(false)
        .show_grid(false)
        .show_x(false)
        .show_y(false)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
        .show_background(false)
        .show(ui, |plot_ui| {
            plot_ui.line(line);
        });
}

/// Draw a circular accuracy/progress gauge — a partial ring with the
/// percentage rendered as text in the center. Used in the Progress tab as
/// the headline accuracy widget. `size` controls overall diameter; the
/// stroke and font sizes scale with it.
fn draw_radial_gauge(ui: &mut egui::Ui, size: f32, percentage: f64, color: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
    let center = rect.center();
    let stroke_w = (size * 0.06).max(6.0);
    let radius = size / 2.0 - stroke_w - 2.0;
    let painter = ui.painter();

    // Background ring — uses hydra_border so it has enough contrast
    // against both the dark and the light page bg. hydra_panel_alt_fill
    // was too close to the light-mode bg and the ring disappeared.
    painter.circle_stroke(
        center,
        radius,
        egui::Stroke::new(stroke_w, hydra_border()),
    );

    // Foreground arc — starts at the top (12 o'clock), sweeps clockwise.
    let pct = (percentage / 100.0).clamp(0.0, 1.0);
    if pct > 0.0 {
        const SEGMENTS: usize = 64;
        let n = ((pct * SEGMENTS as f64).round() as usize).max(1);
        for i in 0..n {
            let t1 = i as f32 / SEGMENTS as f32;
            let t2 = (i + 1) as f32 / SEGMENTS as f32;
            let a1 = -std::f32::consts::FRAC_PI_2 + t1 * 2.0 * std::f32::consts::PI;
            let a2 = -std::f32::consts::FRAC_PI_2 + t2 * 2.0 * std::f32::consts::PI;
            let p1 = egui::pos2(
                center.x + radius * a1.cos(),
                center.y + radius * a1.sin(),
            );
            let p2 = egui::pos2(
                center.x + radius * a2.cos(),
                center.y + radius * a2.sin(),
            );
            painter.line_segment([p1, p2], egui::Stroke::new(stroke_w, color));
        }
    }

    // Center label — big percentage + small "accuracy" subtitle.
    painter.text(
        center + egui::vec2(0.0, -8.0),
        egui::Align2::CENTER_CENTER,
        format!("{percentage:.1}%"),
        egui::FontId::proportional(size * 0.18),
        hydra_text(),
    );
    painter.text(
        center + egui::vec2(0.0, size * 0.12),
        egui::Align2::CENTER_CENTER,
        "accuracy",
        egui::FontId::proportional(size * 0.075),
        hydra_subtle_text(),
    );
}

/// Generous vertical gap with a very faint horizontal hairline at the
/// midpoint — gives the eye a "section ended, new section begins" cue
/// between dashboard rows without the visual weight of a card border.
/// Used as a between-row separator in the Statistics dashboard.
fn subtle_row_separator(ui: &mut egui::Ui) {
    ui.add_space(22.0);
    let (_, rect) = ui.allocate_space(egui::vec2(ui.available_width(), 1.0));
    ui.painter()
        .rect_filled(rect, 0.0, hydra_border().gamma_multiply(0.4));
    ui.add_space(22.0);
}

/// Draw a labeled W/L/D stacked horizontal bar. Used in the Statistics
/// Results card to visualize overall + by-color + by-time-control results
/// as a single comparable widget rather than text rows.
fn draw_result_bar(ui: &mut egui::Ui, label: &str, w: u32, l: u32, d: u32) {
    let total = w + l + d;
    if total == 0 {
        return;
    }
    let w_pct = w as f32 / total as f32;
    let d_pct = d as f32 / total as f32;
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).size(11.0).strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                egui::RichText::new(format!("{:.0}% win", w as f64 / total as f64 * 100.0))
                    .size(10.0)
                    .color(hydra_subtle_text()),
            );
        });
    });
    let bar_w = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(bar_w, 10.0), egui::Sense::hover());
    let p = ui.painter();
    // Background (covers full row, gets overpainted left-to-right).
    p.rect_filled(rect, 5.0, hydra_panel_alt_fill());
    let w_w = bar_w * w_pct;
    let d_w = bar_w * d_pct;
    if w_w > 0.0 {
        let r = egui::Rect::from_min_size(rect.min, egui::vec2(w_w, 10.0));
        p.rect_filled(r, 5.0, hydra_success());
    }
    if d_w > 0.0 {
        let r = egui::Rect::from_min_size(
            egui::pos2(rect.min.x + w_w, rect.min.y),
            egui::vec2(d_w, 10.0),
        );
        p.rect_filled(r, 0.0, hydra_subtle_text());
    }
    let l_w = bar_w - w_w - d_w;
    if l_w > 0.0 {
        let r = egui::Rect::from_min_size(
            egui::pos2(rect.min.x + w_w + d_w, rect.min.y),
            egui::vec2(l_w, 10.0),
        );
        p.rect_filled(r, 5.0, hydra_danger());
    }
    ui.label(
        egui::RichText::new(format!("{w}W  {d}D  {l}L"))
            .size(10.0)
            .color(hydra_subtle_text()),
    );
}

fn hydra_card_frame() -> egui::Frame {
    // Borderless cards: distinguished from the page background only by a
    // soft fill and a very faint shadow. Removing the 1px stroke is what
    // makes a multi-card page feel like one cohesive surface instead of a
    // grid of separate boxes — the original "window in window" complaint
    // came back through the border once we had many cards on a page.
    egui::Frame::new()
        .fill(hydra_panel_fill())
        .corner_radius(8)
        .inner_margin(egui::Margin::same(18))
        .shadow(egui::epaint::Shadow {
            offset: [0, 1],
            blur: 6,
            spread: 0,
            color: egui::Color32::from_black_alpha(10),
        })
}

fn hydra_callout_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(hydra_accent_soft())
        .stroke(egui::Stroke::new(1.0, hydra_accent()))
        .corner_radius(6)
        .inner_margin(egui::Margin::same(10))
}

fn hydra_badge(ui: &mut egui::Ui, label: &str, fill: egui::Color32) {
    let text_color = if fill == hydra_panel_alt_fill() || fill == hydra_panel_raised_fill() {
        hydra_subtle_text()
    } else {
        fill
    };
    ui.label(
        egui::RichText::new(label)
            .size(10.0)
            .strong()
            .color(text_color),
    );
}

fn home_nav_button<'a>(selected: bool, label: &'a str) -> egui::Button<'a> {
    egui::Button::new(
        egui::RichText::new(label)
            .size(12.0)
            .strong()
            .color(if selected { hydra_text() } else { hydra_subtle_text() }),
    )
    .fill(if selected {
        hydra_accent_soft()
    } else {
        egui::Color32::TRANSPARENT
    })
    .stroke(egui::Stroke::new(
        if selected { 1.0 } else { 0.0 },
        if selected { hydra_accent() } else { egui::Color32::TRANSPARENT },
    ))
    .corner_radius(0)
}

fn primary_button(label: &str) -> egui::Button<'_> {
    egui::Button::new(
        egui::RichText::new(label)
            .size(13.0)
            .strong()
            .color(hydra_text_on_accent()),
    )
    .fill(hydra_accent())
    .stroke(egui::Stroke::new(1.0, hydra_accent()))
    .corner_radius(2)
}

fn secondary_button(label: &str) -> egui::Button<'_> {
    egui::Button::new(
        egui::RichText::new(label)
            .size(12.0)
            .strong()
            .color(hydra_text()),
    )
    .fill(egui::Color32::TRANSPARENT)
    .stroke(egui::Stroke::new(1.0, hydra_border()))
    .corner_radius(2)
}

fn danger_button(label: &str) -> egui::Button<'_> {
    egui::Button::new(
        egui::RichText::new(label)
            .size(12.0)
            .strong()
            .color(hydra_danger()),
    )
    .fill(egui::Color32::TRANSPARENT)
    .stroke(egui::Stroke::new(1.0, hydra_danger()))
    .corner_radius(2)
}

fn theme_toggle_button(label: &str) -> egui::Button<'_> {
    egui::Button::new(
        egui::RichText::new(label)
            .size(12.0)
            .strong()
            .color(hydra_text()),
    )
    .fill(egui::Color32::TRANSPARENT)
    .stroke(egui::Stroke::new(1.0, hydra_border()))
    .corner_radius(2)
}

fn draw_engine_settings_controls(
    ui: &mut egui::Ui,
    settings: &mut EngineSettings,
    searching: bool,
) {
    ui.add_enabled_ui(!searching, |ui| {
        ui.horizontal(|ui| {
            ui.radio_value(&mut settings.use_time_limit, true, "Time limit");
            ui.radio_value(&mut settings.use_time_limit, false, "Fixed depth");
        });

        ui.add_space(8.0);
        if settings.use_time_limit {
            ui.label(
                egui::RichText::new("Think time per move")
                    .size(11.0)
                    .color(hydra_subtle_text()),
            );
            let secs = settings.think_time_ms as f32 / 1000.0;
            let mut secs_val = secs;
            ui.add(egui::Slider::new(&mut secs_val, 0.5..=30.0).suffix(" s"));
            settings.think_time_ms = (secs_val * 1000.0) as u64;
        } else {
            ui.label(
                egui::RichText::new("Search depth ceiling")
                    .size(11.0)
                    .color(hydra_subtle_text()),
            );
            let mut depth = settings.max_depth as i32;
            ui.add(egui::Slider::new(&mut depth, 1..=30));
            settings.max_depth = depth as u32;
        }

        ui.add_space(8.0);
        ui.label(
            egui::RichText::new("Transposition table")
                .size(11.0)
                .color(hydra_subtle_text()),
        );
        let mut mb = settings.tt_size_mb as i32;
        ui.add(egui::Slider::new(&mut mb, 1..=256).suffix(" MB"));
        settings.tt_size_mb = mb as usize;

        ui.add_space(8.0);
        ui.label(
            egui::RichText::new("Analysis depth (game review)")
                .size(11.0)
                .color(hydra_subtle_text()),
        );
        let mut analysis_depth = settings.analysis_depth as i32;
        ui.add(egui::Slider::new(&mut analysis_depth, 1..=30));
        settings.analysis_depth = analysis_depth as u32;
    });
}

fn configure_theme(ctx: &egui::Context, theme: UiTheme) {
    ACTIVE_THEME.store(theme.index(), Ordering::Relaxed);

    let mut visuals = match theme {
        UiTheme::Dark => egui::Visuals::dark(),
        UiTheme::Light => egui::Visuals::light(),
    };
    visuals.override_text_color = Some(hydra_text());
    visuals.panel_fill = hydra_bg();
    visuals.window_fill = hydra_panel_fill();
    visuals.faint_bg_color = hydra_panel_fill();
    visuals.extreme_bg_color = hydra_bg();
    visuals.code_bg_color = hydra_panel_alt_fill();
    visuals.hyperlink_color = hydra_accent();
    visuals.selection.bg_fill = hydra_accent();
    visuals.selection.stroke = egui::Stroke::new(1.0, hydra_text_on_accent());
    visuals.window_stroke = egui::Stroke::new(1.0, hydra_border());
    visuals.window_corner_radius = egui::CornerRadius::same(4);
    visuals.menu_corner_radius = egui::CornerRadius::same(2);
    visuals.window_shadow = egui::epaint::Shadow {
        offset: [0, 0],
        blur: 0,
        spread: 0,
        color: egui::Color32::TRANSPARENT,
    };
    visuals.popup_shadow = egui::epaint::Shadow {
        offset: [0, 0],
        blur: 0,
        spread: 0,
        color: egui::Color32::TRANSPARENT,
    };
    visuals.widgets.noninteractive.bg_fill = hydra_panel_fill();
    visuals.widgets.noninteractive.weak_bg_fill = hydra_panel_fill();
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, hydra_border());
    visuals.widgets.noninteractive.corner_radius = egui::CornerRadius::same(6);
    visuals.widgets.inactive.bg_fill = hydra_panel_alt_fill();
    visuals.widgets.inactive.weak_bg_fill = hydra_panel_alt_fill();
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, hydra_border());
    visuals.widgets.inactive.corner_radius = egui::CornerRadius::same(6);
    visuals.widgets.hovered.bg_fill = hydra_panel_raised_fill();
    visuals.widgets.hovered.weak_bg_fill = hydra_panel_raised_fill();
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, hydra_accent());
    visuals.widgets.hovered.corner_radius = egui::CornerRadius::same(6);
    visuals.widgets.active.bg_fill = hydra_panel_raised_fill();
    visuals.widgets.active.weak_bg_fill = hydra_panel_raised_fill();
    visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, hydra_accent());
    visuals.widgets.active.corner_radius = egui::CornerRadius::same(6);
    visuals.widgets.open.bg_fill = hydra_panel_raised_fill();
    visuals.widgets.open.weak_bg_fill = hydra_panel_raised_fill();
    visuals.widgets.open.bg_stroke = egui::Stroke::new(1.0, hydra_accent());
    visuals.widgets.open.corner_radius = egui::CornerRadius::same(6);
    ctx.set_visuals(visuals);

    let mut style = (*ctx.global_style()).clone();
    style.spacing.item_spacing = egui::vec2(12.0, 12.0);
    style.spacing.button_padding = egui::vec2(12.0, 8.0);
    style.spacing.interact_size = egui::vec2(42.0, 30.0);
    style.spacing.menu_margin = egui::Margin::same(12);
    style.spacing.window_margin = egui::Margin::same(16);
    style.visuals.clip_rect_margin = 6.0;
    style.text_styles.insert(
        egui::TextStyle::Heading,
        egui::FontId::proportional(26.0),
    );
    style.text_styles.insert(
        egui::TextStyle::Body,
        egui::FontId::proportional(14.5),
    );
    style.text_styles.insert(
        egui::TextStyle::Button,
        egui::FontId::proportional(13.5),
    );
    style.text_styles.insert(
        egui::TextStyle::Small,
        egui::FontId::proportional(11.5),
    );
    ctx.set_global_style(style);
}

/// Inverse of `board_square_from_pos` — center of a square in screen coords.
fn board_square_center(
    board_rect: egui::Rect,
    sq_size: f32,
    flipped: bool,
    sq_idx: u8,
) -> egui::Pos2 {
    let display_rank = sq_idx / 8;
    let display_file = sq_idx % 8;
    let rank = if flipped { display_rank } else { 7 - display_rank };
    let file = if flipped { 7 - display_file } else { display_file };
    egui::pos2(
        board_rect.min.x + (file as f32 + 0.5) * sq_size,
        board_rect.min.y + (rank as f32 + 0.5) * sq_size,
    )
}

/// Draw a thick colored arrow with a triangular arrowhead, used for annotations.
fn draw_annotation_arrow(
    painter: &egui::Painter,
    from: egui::Pos2,
    to: egui::Pos2,
    color: egui::Color32,
    sq_size: f32,
) {
    let dir = to - from;
    let len = dir.length();
    if len < 1.0 {
        return;
    }
    let unit = dir / len;
    let perp = egui::vec2(-unit.y, unit.x);

    let shaft_width = sq_size * 0.18;
    let head_len = sq_size * 0.36;
    let head_half_width = sq_size * 0.26;

    // Pull both endpoints in so the arrow sits inside the squares, not flush
    // against the edges (looks better visually).
    let from_inset = from + unit * (sq_size * 0.18);
    let to_inset   = to   - unit * (sq_size * 0.10);

    // Shaft stops short of the arrowhead tip so the join looks clean.
    let shaft_end = to_inset - unit * head_len * 0.6;

    painter.line_segment(
        [from_inset, shaft_end],
        egui::Stroke::new(shaft_width, color),
    );

    let tip = to_inset;
    let base_left  = to_inset - unit * head_len + perp * head_half_width;
    let base_right = to_inset - unit * head_len - perp * head_half_width;
    painter.add(egui::Shape::convex_polygon(
        vec![tip, base_left, base_right],
        color,
        egui::Stroke::NONE,
    ));
}

fn board_square_from_pos(
    board_rect: egui::Rect,
    sq_size: f32,
    flipped: bool,
    pos: egui::Pos2,
) -> Option<u8> {
    if !board_rect.contains(pos) {
        return None;
    }

    let file = ((pos.x - board_rect.min.x) / sq_size).floor() as u8;
    let rank = ((pos.y - board_rect.min.y) / sq_size).floor() as u8;
    if file >= 8 || rank >= 8 {
        return None;
    }

    let display_file = if flipped { 7 - file } else { file };
    let display_rank = if flipped { rank } else { 7 - rank };
    Some(display_rank * 8 + display_file)
}

fn draw_piece_image(
    painter: &egui::Painter,
    texture: &egui::TextureHandle,
    rect: egui::Rect,
) {
    painter.image(
        texture.id(),
        rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );
}

/// Compact non-interactive board renderer for History thumbnails. Always
/// shown white-on-bottom. Pieces use Unicode glyphs to avoid loading the
/// big piece textures at thumbnail scale.
fn draw_board_thumbnail(ui: &mut egui::Ui, board: &Board, size: f32) {
    let (response, painter) = ui.allocate_painter(
        egui::vec2(size, size),
        egui::Sense::hover(),
    );
    let rect = response.rect;
    let sq = size / 8.0;
    let light = egui::Color32::from_rgb(227, 218, 201);
    let dark = egui::Color32::from_rgb(120, 99, 81);

    for visual_rank in 0..8u8 {
        // visual_rank 0 = top of board = chess rank 8 = file index 7
        let chess_rank = 7 - visual_rank;
        for file in 0..8u8 {
            let sq_idx = chess_rank * 8 + file;
            let is_light = (file + chess_rank) % 2 == 1;
            let color = if is_light { light } else { dark };
            let x = rect.min.x + file as f32 * sq;
            let y = rect.min.y + visual_rank as f32 * sq;
            painter.rect_filled(
                egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(sq, sq)),
                0.0,
                color,
            );
            if let Some((color_p, piece)) = board.piece_on(Square(sq_idx)) {
                let glyph = piece_unicode(color_p, piece);
                let text_color = if color_p == Color::White {
                    egui::Color32::from_rgb(250, 250, 250)
                } else {
                    egui::Color32::from_rgb(20, 20, 20)
                };
                painter.text(
                    egui::pos2(x + sq / 2.0, y + sq / 2.0),
                    egui::Align2::CENTER_CENTER,
                    glyph,
                    egui::FontId::proportional(sq * 0.85),
                    text_color,
                );
            }
        }
    }
}

fn piece_unicode(color: Color, piece: Piece) -> &'static str {
    match (color, piece) {
        (Color::White, Piece::King) => "\u{2654}",
        (Color::White, Piece::Queen) => "\u{2655}",
        (Color::White, Piece::Rook) => "\u{2656}",
        (Color::White, Piece::Bishop) => "\u{2657}",
        (Color::White, Piece::Knight) => "\u{2658}",
        (Color::White, Piece::Pawn) => "\u{2659}",
        (Color::Black, Piece::King) => "\u{265A}",
        (Color::Black, Piece::Queen) => "\u{265B}",
        (Color::Black, Piece::Rook) => "\u{265C}",
        (Color::Black, Piece::Bishop) => "\u{265D}",
        (Color::Black, Piece::Knight) => "\u{265E}",
        (Color::Black, Piece::Pawn) => "\u{265F}",
    }
}

fn format_nodes(nodes: u64) -> String {
    if nodes >= 1_000_000 {
        format!("{:.1}M", nodes as f64 / 1_000_000.0)
    } else if nodes >= 1_000 {
        format!("{:.1}K", nodes as f64 / 1_000.0)
    } else {
        nodes.to_string()
    }
}

fn elapsed_ms(since: Instant) -> u64 {
    since.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
}

fn format_clock_ms(ms: u64) -> String {
    let minutes = ms / 60_000;
    let seconds = (ms / 1_000) % 60;
    let tenths = (ms % 1_000) / 100;

    if ms < 60_000 {
        format!("{minutes}:{seconds:02}.{tenths}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

fn clock_color(remaining_ms: u64, is_active: bool) -> egui::Color32 {
    if remaining_ms == 0 {
        egui::Color32::RED
    } else if remaining_ms < 10_000 {
        egui::Color32::from_rgb(255, 120, 120)
    } else if is_active {
        hydra_accent()
    } else {
        hydra_text()
    }
}

fn color_name(color: Color) -> &'static str {
    match color {
        Color::White => "White",
        Color::Black => "Black",
    }
}

fn piece_name(piece: Piece) -> &'static str {
    match piece {
        Piece::Queen => "Queen",
        Piece::Rook => "Rook",
        Piece::Bishop => "Bishop",
        Piece::Knight => "Knight",
        Piece::Pawn => "Pawn",
        Piece::King => "King",
    }
}

fn seed_local_history(state: &mut SharedState) {
    state.local_history.clear();
    state.local_history.push(LocalSnapshot {
        board: state.board.clone(),
        white_remaining_ms: state.local_game.time_control.white_remaining_ms,
        black_remaining_ms: state.local_game.time_control.black_remaining_ms,
        move_uci: None,
    });
    state.local_history_cursor = 0;
    sync_local_move_history(state);
}

fn sync_local_move_history(state: &mut SharedState) {
    state.move_history = state
        .local_history
        .iter()
        .take(state.local_history_cursor.saturating_add(1))
        .skip(1)
        .filter_map(|snapshot| snapshot.move_uci.clone())
        .collect();
}

fn record_local_snapshot(state: &mut SharedState, move_uci: String) {
    if state.local_history.is_empty() {
        seed_local_history(state);
    }

    let truncate_len = state.local_history_cursor.saturating_add(1);
    if truncate_len < state.local_history.len() {
        state.local_history.truncate(truncate_len);
    }

    state.local_search_generation = state.local_search_generation.wrapping_add(1);
    state.local_history.push(LocalSnapshot {
        board: state.board.clone(),
        white_remaining_ms: state.local_game.time_control.white_remaining_ms,
        black_remaining_ms: state.local_game.time_control.black_remaining_ms,
        move_uci: Some(move_uci),
    });
    state.local_history_cursor = state.local_history.len().saturating_sub(1);
    sync_local_move_history(state);
}

fn restore_local_snapshot(state: &mut SharedState, snapshot: &LocalSnapshot) {
    state.board = snapshot.board.clone();
    state.local_game.time_control.white_remaining_ms = snapshot.white_remaining_ms;
    state.local_game.time_control.black_remaining_ms = snapshot.black_remaining_ms;
    state.local_game.time_control.clear_active();
}

fn invalidate_local_search(state: &mut SharedState, clear_info: bool) {
    state.local_search_generation = state.local_search_generation.wrapping_add(1);
    if clear_info {
        state.search_info = SearchInfo::default();
    } else {
        state.search_info.searching = false;
    }
}

fn local_history_label(index: usize, move_uci: Option<&str>) -> String {
    match move_uci {
        None => "Start position".to_string(),
        Some(mv) if index % 2 == 1 => format!("{}. {}", (index + 1) / 2, mv),
        Some(mv) => format!("{}... {}", index / 2, mv),
    }
}

fn is_reviewing_history(state: &SharedState) -> bool {
    !state.local_history.is_empty() && state.local_history_cursor + 1 < state.local_history.len()
}

fn local_engine_search_request(
    state: &SharedState,
    strength_config: &crate::strength::StrengthConfig,
) -> Option<LocalSearchRequest> {
    if !state.local_game.active || state.local_game.outcome.is_some() {
        return None;
    }

    let side_to_move = state.board.side_to_move;
    let remaining_ms = state
        .local_game
        .time_control
        .displayed_remaining_ms(side_to_move, Some(side_to_move));
    let safe_max_ms = remaining_ms.saturating_sub(50).max(1);

    if state.local_game.difficulty == LocalDifficulty::Custom {
        let settings = &state.engine_settings;
        let (soft_time_ms, hard_time_ms, depth_cap) = if settings.use_time_limit {
            let hard = settings.think_time_ms.min(safe_max_ms);
            let soft = (hard / 2).max(1);
            (soft, hard, None)
        } else {
            (safe_max_ms, safe_max_ms, Some(settings.max_depth))
        };
        return Some(LocalSearchRequest {
            soft_time_ms,
            hard_time_ms,
            depth_cap,
            generation: state.local_search_generation,
            strength_config: strength_config.clone(),
        });
    }

    let (base_time_ms, _) = uci::allocate_time(
        remaining_ms,
        state.local_game.time_control.increment_ms,
        None,
    );
    let scaled_time_ms = ((base_time_ms as f32) * strength_config.time_scale).round() as u64;
    let min_time_ms = safe_max_ms.min(80);
    let hard_time_ms = scaled_time_ms
        .max(min_time_ms)
        .max(strength_config.min_think_ms)
        .min(safe_max_ms);
    let soft_time_ms = (hard_time_ms * 4 / 10).max(min_time_ms.min(hard_time_ms));

    Some(LocalSearchRequest {
        soft_time_ms,
        hard_time_ms,
        depth_cap: strength_config.max_depth,
        generation: state.local_search_generation,
        strength_config: strength_config.clone(),
    })
}

fn consume_local_turn_time(state: &mut SharedState, mover: Color) -> bool {
    if !state.local_game.active || state.local_game.outcome.is_some() {
        return false;
    }

    let remaining_ms = state.local_game.time_control.consume_running_time(mover);
    if remaining_ms == 0 {
        state.local_game.time_control.flag_side(mover);
        set_local_game_outcome(state, GameOutcome::Timeout(mover.flip()));
        return false;
    }

    true
}

fn handle_active_side_timeout(state: &mut SharedState) -> bool {
    if !state.local_game.active || state.local_game.outcome.is_some() {
        return false;
    }

    let active_side = state.board.side_to_move;
    let expired = state
        .local_game
        .time_control
        .displayed_remaining_ms(active_side, Some(active_side))
        == 0;

    if expired {
        state.local_game.time_control.flag_side(active_side);
        set_local_game_outcome(state, GameOutcome::Timeout(active_side.flip()));
        return true;
    }

    false
}

fn is_draw_by_threefold(state: &SharedState) -> bool {
    let Some(current_snapshot) = state.local_history.get(state.local_history_cursor) else {
        return false;
    };

    let current_hash = current_snapshot.board.hash;
    state
        .local_history
        .iter()
        .take(state.local_history_cursor.saturating_add(1))
        .filter(|snapshot| snapshot.board.hash == current_hash)
        .count()
        >= 3
}

fn detect_game_outcome(state: &SharedState) -> Option<GameOutcome> {
    let board = &state.board;
    let moves = generate_legal_moves(board);
    if moves.is_empty() {
        let king_sq = board.piece_bb(board.side_to_move, Piece::King).lsb();
        let in_check = attacks::is_square_attacked(board, king_sq, board.side_to_move.flip());
        return if in_check {
            Some(GameOutcome::Checkmate(board.side_to_move.flip()))
        } else {
            Some(GameOutcome::Stalemate)
        };
    }

    if board.is_insufficient_material() {
        return Some(GameOutcome::InsufficientMaterial);
    }

    if is_draw_by_threefold(state) {
        return Some(GameOutcome::ThreefoldRepetition);
    }

    if board.halfmove_clock >= 100 {
        return Some(GameOutcome::FiftyMoveRule);
    }

    None
}

fn local_outcome_banner(outcome: GameOutcome, human_color: Color) -> (String, egui::Color32) {
    match outcome {
        GameOutcome::Checkmate(winner) if winner == human_color => (
            "CHECKMATE - You win!".to_string(),
            egui::Color32::LIGHT_GREEN,
        ),
        GameOutcome::Checkmate(_) => (
            "CHECKMATE - Focalors wins".to_string(),
            egui::Color32::RED,
        ),
        GameOutcome::Stalemate => (
            "STALEMATE - Draw".to_string(),
            egui::Color32::YELLOW,
        ),
        GameOutcome::FiftyMoveRule => (
            "DRAW BY 50-MOVE RULE".to_string(),
            egui::Color32::YELLOW,
        ),
        GameOutcome::ThreefoldRepetition => (
            "DRAW BY REPETITION".to_string(),
            egui::Color32::YELLOW,
        ),
        GameOutcome::InsufficientMaterial => (
            "DRAW BY INSUFFICIENT MATERIAL".to_string(),
            egui::Color32::YELLOW,
        ),
        GameOutcome::Timeout(winner) if winner == human_color => (
            "TIMEOUT - You win!".to_string(),
            egui::Color32::LIGHT_GREEN,
        ),
        GameOutcome::Timeout(_) => (
            "TIMEOUT - Focalors wins".to_string(),
            egui::Color32::RED,
        ),
        GameOutcome::Resignation(winner) if winner == human_color => (
            "RESIGNATION - You win!".to_string(),
            egui::Color32::LIGHT_GREEN,
        ),
        GameOutcome::Resignation(_) => (
            "RESIGNATION - Focalors wins".to_string(),
            egui::Color32::RED,
        ),
    }
}

fn local_outcome_status(outcome: GameOutcome, human_color: Color) -> String {
    match outcome {
        GameOutcome::Checkmate(winner) if winner == human_color => {
            "Checkmate. You win!".to_string()
        }
        GameOutcome::Checkmate(_) => "Checkmate. Focalors wins.".to_string(),
        GameOutcome::Stalemate => "Draw by stalemate.".to_string(),
        GameOutcome::FiftyMoveRule => "Draw by 50-move rule.".to_string(),
        GameOutcome::ThreefoldRepetition => "Draw by threefold repetition.".to_string(),
        GameOutcome::InsufficientMaterial => {
            "Draw by insufficient material.".to_string()
        }
        GameOutcome::Timeout(winner) if winner == human_color => {
            "Focalors flagged. You win on time!".to_string()
        }
        GameOutcome::Timeout(_) => "You flagged. Focalors wins on time.".to_string(),
        GameOutcome::Resignation(winner) if winner == human_color => {
            "Focalors resigned. You win!".to_string()
        }
        GameOutcome::Resignation(_) => "You resigned. Focalors wins.".to_string(),
    }
}

fn set_local_game_outcome(state: &mut SharedState, outcome: GameOutcome) {
    invalidate_local_search(state, false);
    state.local_game.outcome = Some(outcome);
    state.local_game.time_control.clear_active();
    state.status_message = local_outcome_status(outcome, state.local_game.human_color);
}

fn update_local_game_outcome(state: &mut SharedState) {
    if !state.local_game.active || state.local_game.outcome.is_some() {
        return;
    }

    if let Some(outcome) = detect_game_outcome(state) {
        set_local_game_outcome(state, outcome);
    }
}

fn classification_color(class: crate::analysis::MoveClass) -> egui::Color32 {
    use crate::analysis::MoveClass;
    match class {
        MoveClass::Best | MoveClass::Brilliant => egui::Color32::from_rgb(100, 200, 100),
        MoveClass::Good | MoveClass::Forced => egui::Color32::from_rgb(180, 180, 180),
        MoveClass::Inaccuracy => egui::Color32::from_rgb(230, 200, 80),
        MoveClass::Mistake => egui::Color32::from_rgb(220, 150, 50),
        MoveClass::Blunder => egui::Color32::from_rgb(220, 80, 80),
    }
}

fn accuracy_color(accuracy: f64) -> egui::Color32 {
    if accuracy >= 90.0 {
        egui::Color32::from_rgb(100, 200, 100)
    } else if accuracy >= 70.0 {
        egui::Color32::from_rgb(200, 200, 80)
    } else if accuracy >= 50.0 {
        egui::Color32::from_rgb(220, 150, 50)
    } else {
        egui::Color32::from_rgb(220, 80, 80)
    }
}

fn chrono_today() -> String {
    // Simple date without pulling in the chrono crate
    use std::time::SystemTime;
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Approximate: days since epoch
    let days = secs / 86400;
    // Zeller-like calculation for year/month/day
    let mut y = 1970i32;
    let mut remaining = days as i64;
    loop {
        let year_days = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
        if remaining < year_days { break; }
        remaining -= year_days;
        y += 1;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let month_days = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut m = 0;
    for md in month_days {
        if remaining < md as i64 { break; }
        remaining -= md as i64;
        m += 1;
    }
    format!("{y:04}.{:02}.{:02}", m + 1, remaining + 1)
}

// ── Launch the GUI ─────────────────────────────────────────────────────────

pub fn run_gui() {
    attacks::init();
    // Initialize NNUE (will use embedded net or fall back to HCE)
    let _ = crate::nnue::init(None);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([950.0, 680.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("Focalors Chess Engine"),
        // Explicit OpenGL backend — avoids wgpu/Vulkan compatibility issues
        // on some Linux setups while being plenty fast for our chess GUI.
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };

    eframe::run_native(
        "Focalors",
        options,
        Box::new(|cc| Ok(Box::new(FocalorsApp::new(cc)))),
    )
    .expect("Failed to launch GUI");
}
