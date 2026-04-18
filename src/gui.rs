use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
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
}

impl Default for EngineSettings {
    fn default() -> Self {
        Self {
            max_depth: 12,
            think_time_ms: 5000,
            tt_size_mb: 64,
            use_time_limit: true,
        }
    }
}

#[derive(Clone)]
pub struct LichessSettings {
    pub token: String,
    pub clock_minutes: u32,
    pub clock_increment: u32,
}

impl Default for LichessSettings {
    fn default() -> Self {
        Self {
            token: String::new(),
            clock_minutes: 10,
            clock_increment: 0,
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

#[derive(Clone, PartialEq)]
enum LichessStatus {
    Disconnected,
    Connected(String), // username
    InGame(String),    // game ID
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
}

impl LocalDifficulty {
    const ALL: [LocalDifficulty; 5] = [
        LocalDifficulty::Beginner,
        LocalDifficulty::Club,
        LocalDifficulty::Tournament,
        LocalDifficulty::Master,
        LocalDifficulty::Adaptive,
    ];

    fn label(self) -> &'static str {
        match self {
            LocalDifficulty::Beginner => "Beginner",
            LocalDifficulty::Club => "Club",
            LocalDifficulty::Tournament => "Tournament",
            LocalDifficulty::Master => "Master",
            LocalDifficulty::Adaptive => "Adaptive",
        }
    }

    fn numeric_level(self, adaptive_level: u32) -> u32 {
        match self {
            LocalDifficulty::Beginner => crate::strength::BEGINNER_LEVEL,
            LocalDifficulty::Club => crate::strength::CLUB_LEVEL,
            LocalDifficulty::Tournament => crate::strength::TOURNAMENT_LEVEL,
            LocalDifficulty::Master => crate::strength::MASTER_LEVEL,
            LocalDifficulty::Adaptive => adaptive_level,
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
    Lichess,
    Import,
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

#[derive(Clone)]
struct LocalSearchRequest {
    time_limit_ms: u64,
    depth_cap: Option<u32>,
    generation: u64,
    strength_config: crate::strength::StrengthConfig,
}

/// State for replaying a saved game.
#[derive(Clone)]
struct ReplayState {
    game: crate::db::SavedGame,
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
    },
}

struct SharedState {
    board: Board,
    move_history: Vec<String>,
    local_history: Vec<LocalSnapshot>,
    local_history_cursor: usize,
    local_search_generation: u64,
    engine_settings: EngineSettings,
    lichess_settings: LichessSettings,
    search_info: SearchInfo,
    lichess_status: LichessStatus,
    lichess_username: String, // always stores the bot's username once connected
    lichess_log: Vec<String>,
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
            lichess_settings: LichessSettings::default(),
            search_info: SearchInfo::default(),
            lichess_status: LichessStatus::Disconnected,
            lichess_username: String::new(),
            lichess_log: Vec::new(),
            local_game: LocalGameState::default(),
            game_saved: false,
            persistent_searcher: {
                let mut s = Searcher::new(64);
                s.use_nnue = crate::nnue::network::get_network().is_some();
                Arc::new(Mutex::new(s))
            },
            status_message: "Idle. Choose local play or connect Lichess.".to_string(),
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
    selected_square: Option<u8>,
    drag_state: Option<DragState>,
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
    stockfish_level: i32,
    piece_textures: HashMap<(Color, Piece), egui::TextureHandle>,
    pgn_import_text: String,
    pgn_import_parsed: Option<crate::pgn::ParsedPgn>,
    pgn_import_error: Option<String>,
    pgn_import_user_color: Color,
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
        configure_theme(&cc.egui_ctx, UiTheme::Light);

        // Load token from environment if available
        let token = std::env::var("LICHESS_TOKEN").unwrap_or_default();
        let state = Arc::new(Mutex::new(SharedState::default()));
        {
            let mut s = state.lock().unwrap();
            s.lichess_settings.token = token;
        }

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
            ui_theme: UiTheme::Light,
            replay_game: None,
            analysis_state: Arc::new(Mutex::new(AnalysisState::Idle)),
            analysis_review_cursor: 0,
            selected_square: None,
            drag_state: None,
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
            stockfish_level: 1,
            piece_textures,
            pgn_import_text: String::new(),
            pgn_import_parsed: None,
            pgn_import_error: None,
            pgn_import_user_color: Color::White,
        }
    }

    fn set_ui_theme(&mut self, ctx: &egui::Context, theme: UiTheme) {
        self.ui_theme = theme;
        configure_theme(ctx, theme);
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
            && matches!(state.lichess_status, LichessStatus::Disconnected)
            && !state.search_info.searching
            && state.board.side_to_move == state.local_game.human_color
            && state
                .local_game
                .time_control
                .displayed_remaining_ms(state.board.side_to_move, Some(state.board.side_to_move))
                > 0
    }

    fn has_running_local_clock(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.local_game.active
            && state.local_game.outcome.is_none()
            && matches!(state.lichess_status, LichessStatus::Disconnected)
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

    fn draw_profile_card(&self, ui: &mut egui::Ui) {
        if let Some(ref profile) = self.profile {
            let (w, l, d) = self.result_counts;
            hydra_card_frame().show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Profile")
                            .size(12.0)
                            .strong()
                            .color(hydra_subtle_text()),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(&profile.name)
                                .size(18.0)
                                .strong()
                                .color(hydra_accent()),
                        );
                    });
                });

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(10.0);
                ui.horizontal_wrapped(|ui| {
                    metric_tile(ui, "Rating", &profile.rating.to_string(), hydra_text());
                    metric_tile(
                        ui,
                        "Games",
                        &profile.games_played.to_string(),
                        hydra_text(),
                    );
                    if profile.games_played > 0 {
                        metric_tile(ui, "Record", &format!("W{w} / L{l} / D{d}"), hydra_subtle_text());
                    }
                });
            });
        }
    }

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

        ui.add_space(8.0);
        hydra_card_frame().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Statistics Dashboard").size(16.0).strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Close").clicked() {
                        self.home_page = HomePage::Overview;
                    }
                });
            });

            egui::ScrollArea::vertical().max_height(500.0).show(ui, |ui| {

                // ── 6.1: Rating Chart ────────────────────────────────
                if rating_history.len() >= 2 {
                    ui.add_space(8.0);
                    ui.label(egui::RichText::new("Rating Over Time").size(13.0).strong());

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
                        .height(120.0)
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
                }

                // ── 6.2: Accuracy Trends ─────────────────────────────
                if accuracy_history.len() >= 2 {
                    ui.add_space(10.0);
                    ui.label(egui::RichText::new("Accuracy Trends").size(13.0).strong());

                    let acc_points: Vec<[f64; 2]> = accuracy_history
                        .iter()
                        .enumerate()
                        .map(|(i, (_, acc))| [i as f64 + 1.0, *acc])
                        .collect();

                    // 5-game rolling average
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
                        .height(120.0)
                        .include_y(0.0)
                        .include_y(100.0)
                        .allow_drag(false)
                        .allow_zoom(false)
                        .allow_scroll(false)
                        .show_axes(true)
                        .y_axis_label("Accuracy %")
                        .legend(egui_plot::Legend::default())
                        .show(ui, |plot_ui| {
                            plot_ui.line(line_acc);
                            plot_ui.line(line_avg);
                        });

                    // Highlight best accuracy
                    if let Some(best) = accuracy_history.iter().map(|(_, a)| *a).max_by(|a, b| a.partial_cmp(b).unwrap()) {
                        ui.label(
                            egui::RichText::new(format!("Personal best: {best:.1}%"))
                                .size(11.0)
                                .color(egui::Color32::from_rgb(80, 200, 80)),
                        );
                    }
                }

                // ── 6.3: Result Breakdown ────────────────────────────
                ui.add_space(10.0);
                ui.label(egui::RichText::new("Results").size(13.0).strong());

                let total_games = total_w + total_l + total_d;
                if total_games > 0 {
                    ui.label(egui::RichText::new(format!(
                        "Overall: {total_games} games — W{total_w} / L{total_l} / D{total_d} ({:.0}% win rate)",
                        total_w as f64 / total_games as f64 * 100.0
                    )).size(11.0));

                    // By color
                    let ((ww, wl, wd), (bw, bl, bd)) = by_color;
                    let white_total = ww + wl + wd;
                    let black_total = bw + bl + bd;
                    if white_total > 0 {
                        ui.label(egui::RichText::new(format!(
                            "  As White: W{ww}/L{wl}/D{wd} ({:.0}%)",
                            ww as f64 / white_total as f64 * 100.0
                        )).size(10.0).color(hydra_subtle_text()));
                    }
                    if black_total > 0 {
                        ui.label(egui::RichText::new(format!(
                            "  As Black: W{bw}/L{bl}/D{bd} ({:.0}%)",
                            bw as f64 / black_total as f64 * 100.0
                        )).size(10.0).color(hydra_subtle_text()));
                    }

                    // By time control
                    if !by_tc.is_empty() {
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new("By Time Control").size(11.0).strong());
                        for (tc, w, l, d) in &by_tc {
                            let total = w + l + d;
                            let rate = if total > 0 { *w as f64 / total as f64 * 100.0 } else { 0.0 };
                            ui.label(egui::RichText::new(format!(
                                "  {tc}: {total}g  W{w}/L{l}/D{d} ({rate:.0}%)"
                            )).size(10.0).color(hydra_subtle_text()));
                        }
                    }
                }

                // ── 6.4: Weakness Analysis ───────────────────────────
                let total_phase_errors = phase_o + phase_m + phase_e;
                if total_phase_errors > 0 || !theme_stats.is_empty() {
                    ui.add_space(10.0);
                    ui.label(egui::RichText::new("Weaknesses").size(13.0).strong());

                    if total_phase_errors > 0 {
                        let phases = [
                            ("Opening (moves 1-15)", phase_o),
                            ("Middlegame (moves 16-35)", phase_m),
                            ("Endgame (moves 36+)", phase_e),
                        ];
                        for (label, count) in &phases {
                            let pct = *count as f64 / total_phase_errors as f64 * 100.0;
                            let bar_width = (pct / 100.0 * 200.0) as f32;
                            let color = if *count == phases.iter().map(|(_, c)| *c).max().unwrap_or(0) {
                                egui::Color32::from_rgb(220, 100, 80)
                            } else {
                                hydra_subtle_text()
                            };
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new(format!("{label}: {count}")).size(10.0).color(color));
                                let (rect, _) = ui.allocate_exact_size(
                                    egui::vec2(bar_width.max(2.0), 10.0),
                                    egui::Sense::hover(),
                                );
                                ui.painter().rect_filled(rect, 2.0, color);
                            });
                        }
                    }

                    // Missed tactical themes
                    let weak: Vec<_> = theme_stats.iter()
                        .filter(|(_, a, s)| *a >= 2 && (*s as f64 / *a as f64) < 0.5)
                        .collect();
                    if !weak.is_empty() {
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new("Weak Puzzle Themes").size(11.0).strong());
                        for (theme, attempts, solved) in &weak {
                            let label = crate::puzzles::PuzzleTheme::from_db_str(theme).label();
                            let rate = *solved as f64 / *attempts as f64 * 100.0;
                            ui.label(egui::RichText::new(format!(
                                "  {label}: {solved}/{attempts} ({rate:.0}%)"
                            )).size(10.0).color(egui::Color32::from_rgb(220, 150, 80)));
                        }
                    }
                }

                // ── 6.5: Session Summary ─────────────────────────────
                ui.add_space(10.0);
                ui.label(egui::RichText::new("This Session").size(13.0).strong());

                let current_games = self.profile.as_ref().map_or(0, |p| p.games_played);
                let current_rating = self.profile.as_ref().map_or(1200, |p| p.rating);
                let session_games = current_games - self.session_start_games;
                let rating_delta = current_rating - self.session_start_rating;
                let sign = if rating_delta >= 0 { "+" } else { "" };

                ui.label(egui::RichText::new(format!("Games played: {session_games}")).size(11.0));
                if session_games > 0 {
                    let rating_color = if rating_delta > 0 {
                        egui::Color32::from_rgb(80, 200, 80)
                    } else if rating_delta < 0 {
                        egui::Color32::from_rgb(220, 100, 80)
                    } else {
                        hydra_text()
                    };
                    ui.label(egui::RichText::new(format!(
                        "Rating: {current_rating} ({sign}{rating_delta})"
                    )).size(11.0).color(rating_color));
                }
                if let Some(best) = self.session_best_accuracy {
                    ui.label(egui::RichText::new(format!(
                        "Best accuracy: {best:.1}%"
                    )).size(11.0).color(accuracy_color(best)));
                }
                if puzzle_total > 0 {
                    ui.label(egui::RichText::new(format!(
                        "Puzzles: {puzzle_solved}/{puzzle_total} solved"
                    )).size(11.0));
                }
            });
        });
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

        ui.add_space(8.0);
        hydra_card_frame().show(ui, |ui| {
            ui.label(egui::RichText::new("Your Openings").size(14.0).strong());
            ui.add_space(4.0);

            for (name, total, wins, losses, draws) in &stats {
                let win_rate = if *total > 0 { *wins as f64 / *total as f64 * 100.0 } else { 0.0 };
                let color = if win_rate >= 60.0 {
                    egui::Color32::from_rgb(80, 200, 80)
                } else if win_rate <= 35.0 {
                    egui::Color32::from_rgb(220, 100, 80)
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
        });
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

        hydra_card_frame().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Progress Report").size(15.0).strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Close").clicked() {
                        self.home_page = HomePage::Overview;
                    }
                });
            });

            // Accuracy trend
            if accuracy_history.len() >= 2 {
                let recent_10: Vec<_> = accuracy_history.iter().rev().take(10).collect();
                let older_10: Vec<_> = accuracy_history.iter().rev().skip(10).take(10).collect();

                let recent_avg: f64 = recent_10.iter().map(|(_, a)| a).sum::<f64>() / recent_10.len() as f64;

                if !older_10.is_empty() {
                    let older_avg: f64 = older_10.iter().map(|(_, a)| a).sum::<f64>() / older_10.len() as f64;
                    let delta = recent_avg - older_avg;
                    let arrow = if delta > 2.0 { "^" } else if delta < -2.0 { "v" } else { "=" };
                    let color = if delta > 2.0 {
                        egui::Color32::from_rgb(80, 200, 80)
                    } else if delta < -2.0 {
                        egui::Color32::from_rgb(220, 100, 80)
                    } else {
                        hydra_text()
                    };
                    ui.label(egui::RichText::new(
                        format!("Accuracy: {recent_avg:.1}% (was {older_avg:.1}%) {arrow}")
                    ).size(13.0).color(color));
                } else {
                    ui.label(egui::RichText::new(
                        format!("Average accuracy: {recent_avg:.1}%")
                    ).size(13.0));
                }
            } else if accuracy_history.len() == 1 {
                ui.label(egui::RichText::new(
                    format!("Accuracy: {:.1}% (analyze more games to see trends)", accuracy_history[0].1)
                ).size(12.0).color(hydra_subtle_text()));
            } else {
                ui.label(egui::RichText::new(
                    "No analyzed games yet. Play a game and click \"Analyze Game\" to get started."
                ).size(12.0).color(hydra_subtle_text()));
                return;
            }

            ui.add_space(4.0);

            // Classification summary
            if !classification_stats.is_empty() {
                let blunders = classification_stats.iter()
                    .find(|(c, _)| c == "blunder").map_or(0, |(_, n)| *n);
                let mistakes = classification_stats.iter()
                    .find(|(c, _)| c == "mistake").map_or(0, |(_, n)| *n);
                let inaccuracies = classification_stats.iter()
                    .find(|(c, _)| c == "inaccuracy").map_or(0, |(_, n)| *n);
                ui.label(egui::RichText::new(
                    format!("Last 10 analyzed games: {blunders} blunders, {mistakes} mistakes, {inaccuracies} inaccuracies")
                ).size(11.0).color(hydra_subtle_text()));
            }

            // Phase weakness
            let (opening, middle, endgame) = phase_weakness;
            let total_errors = opening + middle + endgame;
            if total_errors > 0 {
                let weakest = if opening >= middle && opening >= endgame {
                    "opening"
                } else if middle >= endgame {
                    "middlegame"
                } else {
                    "endgame"
                };
                ui.label(egui::RichText::new(
                    format!("Errors by phase: opening {opening}, middlegame {middle}, endgame {endgame}. Focus on the {weakest}.")
                ).size(11.0));
            }

            // Puzzle theme weaknesses
            let weak_themes: Vec<_> = theme_stats.iter()
                .filter(|(_, attempts, solved)| *attempts >= 3 && (*solved as f64 / *attempts as f64) < 0.5)
                .collect();
            if !weak_themes.is_empty() {
                let names: Vec<_> = weak_themes.iter()
                    .map(|(t, _, _)| crate::puzzles::PuzzleTheme::from_db_str(t).label())
                    .collect();
                ui.label(egui::RichText::new(
                    format!("Weak puzzle themes: {}. Use the puzzle trainer to practice.", names.join(", "))
                ).size(11.0));
            }

            // Suggestions
            ui.add_space(4.0);
            ui.label(egui::RichText::new("Tips").size(12.0).strong());
            let mut tips: Vec<String> = Vec::new();

            if let Some(ref p) = self.profile {
                if p.games_played >= 10 {
                    let (w, l, _) = self.result_counts;
                    if l > w * 2 {
                        tips.push("Try lowering the difficulty or switching to Adaptive mode.".into());
                    }
                }
            }
            if classification_stats.iter().find(|(c, _)| c == "blunder").map_or(0, |(_, n)| *n) > 5 {
                tips.push("You're blundering frequently. Slow down and double-check captures and checks before moving.".into());
            }
            if total_errors > 0 && opening >= middle && opening >= endgame {
                tips.push("Your opening play needs work. Try to develop pieces early and control the center.".into());
            } else if total_errors > 0 && endgame >= opening && endgame >= middle {
                tips.push("Your endgame technique needs improvement. Focus on king activity and passed pawn advancement.".into());
            }
            if tips.is_empty() {
                tips.push("Keep playing and analyzing games to track your improvement!".into());
            }
            for tip in &tips {
                ui.label(egui::RichText::new(format!("  - {tip}")).size(11.0).color(hydra_subtle_text()));
            }
        });
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

            let mut replay_id = None;
            egui::ScrollArea::vertical()
                .max_height(300.0)
                .show(ui, |ui| {
                    for game in &self.recent_games {
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
                            ui.label(
                                egui::RichText::new(result_icon)
                                    .strong()
                                    .color(result_color)
                                    .monospace(),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "{} as {} — {} {}{} ({} moves)",
                                    date,
                                    game.user_color,
                                    reason,
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
                            if ui.small_button("Replay").clicked() {
                                replay_id = Some(game.id);
                            }
                        });
                    }
                });

            if let Some(id) = replay_id {
                self.start_replay(id);
            }
        });
    }

    fn start_replay(&mut self, game_id: i64) {
        let game = match self.db.as_ref().and_then(|db| db.get_game(game_id).ok()) {
            Some(g) => g,
            None => return,
        };
        self.replay_game = Some(ReplayState { game });
    }

    fn draw_replay_panel(&mut self, ui: &mut egui::Ui) {
        let replay = match &self.replay_game {
            Some(r) => r.clone(),
            None => return,
        };

        hydra_card_frame().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Game Replay")
                        .size(14.0)
                        .strong(),
                );
                let result_text = format!(
                    "{} — {} ({})",
                    replay.game.result,
                    replay.game.result_reason.as_deref().unwrap_or(""),
                    replay.game.played_at.get(..10).unwrap_or(""),
                );
                ui.label(
                    egui::RichText::new(result_text)
                        .size(11.0)
                        .color(hydra_subtle_text()),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Close").clicked() {
                        self.replay_game = None;
                        return;
                    }
                });
            });

            ui.add_space(4.0);

            // Show PGN text
            ui.label(egui::RichText::new("PGN:").size(11.0).strong());
            egui::ScrollArea::vertical()
                .max_height(120.0)
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new(&replay.game.pgn)
                            .size(11.0)
                            .monospace(),
                    );
                });
        });
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

        {
            let mut a = analysis_state.lock().unwrap();
            *a = AnalysisState::Running {
                progress: 0,
                total: uci_moves.len(),
            };
        }

        self.analysis_review_cursor = 0;

        let user_rating = self.profile.as_ref().map_or(1200, |p| p.rating);

        thread::spawn(move || {
            crate::attacks::init();
            let use_nnue = crate::nnue::network::get_network().is_some();
            let result = crate::analysis::analyze_game(
                &uci_moves,
                user_color,
                14, // analysis depth
                use_nnue,
                &mut |cur, tot| {
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
            *a = AnalysisState::Complete { analysis: result, puzzles, uci_moves };
        });
    }

    fn draw_analysis_button(&mut self, ui: &mut egui::Ui) {
        let state = self.state.lock().unwrap();
        let analysis = self.analysis_state.lock().unwrap();

        // Show "Analyze Game" button when game is over and not already analyzing
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

            if !uci_moves.is_empty() && ui.button("Analyze Game").clicked() {
                self.start_analysis(uci_moves, user_color);
            }
            return;
        }
        drop(state);
        drop(analysis);
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

    fn draw_analysis_review(&mut self, ui: &mut egui::Ui) {
        // Save analysis data to DB (one-shot on first render of Complete state)
        {
            let analysis = self.analysis_state.lock().unwrap();
            if let AnalysisState::Complete { puzzles, analysis: ga, uci_moves } = &*analysis {
                if let Some(ref db) = self.db {
                    // Save puzzles
                    if !puzzles.is_empty() {
                        for p in puzzles {
                            let _ = db.save_puzzle(p.game_id, &p.fen, &p.solution_uci, p.theme.to_db_str(), p.rating);
                        }
                    }
                    // Persist move analysis and accuracy to DB
                    if !uci_moves.is_empty() {
                        if let Some(recent) = db.get_recent_games(1).ok().and_then(|g| g.into_iter().next()) {
                            let _ = db.save_move_analysis(recent.id, uci_moves, &ga.moves);
                            let _ = db.update_game_accuracy(recent.id, ga.user_accuracy);
                            // Track session best accuracy
                            let acc = ga.user_accuracy;
                            if self.session_best_accuracy.map_or(true, |best| acc > best) {
                                self.session_best_accuracy = Some(acc);
                            }
                        }
                    }
                }
            }
        }
        // Clear puzzles and uci_moves after saving (avoid re-saving on next frame)
        {
            let mut analysis = self.analysis_state.lock().unwrap();
            if let AnalysisState::Complete { puzzles, uci_moves, .. } = &mut *analysis {
                puzzles.clear();
                uci_moves.clear();
            }
        }

        let analysis = self.analysis_state.lock().unwrap().clone();
        let (ga, puzzle_count) = match &analysis {
            AnalysisState::Complete { analysis, puzzles, .. } => (analysis, puzzles.len()),
            _ => return,
        };
        let _ = puzzle_count; // used below

        ui.add_space(8.0);
        hydra_card_frame().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Game Analysis").size(15.0).strong());
                ui.separator();
                ui.label(
                    egui::RichText::new(format!("Accuracy: {:.1}%", ga.user_accuracy))
                        .size(13.0)
                        .color(accuracy_color(ga.user_accuracy)),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Close").clicked() {
                        *self.analysis_state.lock().unwrap() = AnalysisState::Idle;
                    }
                });
            });

            ui.add_space(6.0);

            // ── Eval graph ─────────────────────────────────────────
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

            egui_plot::Plot::new("eval_graph")
                .height(120.0)
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
                    // Cursor line
                    if self.analysis_review_cursor < ga.moves.len() {
                        let vline = egui_plot::VLine::new(
                            "cursor",
                            (self.analysis_review_cursor + 1) as f64,
                        )
                        .color(egui::Color32::from_rgb(255, 200, 50));
                        plot_ui.vline(vline);
                    }
                });

            ui.add_space(6.0);

            // ── Move list (color-coded) ────────────────────────────
            ui.label(egui::RichText::new("Moves").size(12.0).strong());
            egui::ScrollArea::vertical()
                .max_height(180.0)
                .show(ui, |ui| {
                    for (i, ma) in ga.moves.iter().enumerate() {
                        let prefix = if ma.side == Color::White {
                            format!("{}.", ma.move_number)
                        } else {
                            format!("{}...", ma.move_number)
                        };

                        let class_color = classification_color(ma.classification);
                        let symbol = ma.classification.symbol();
                        let label = format!("{prefix} {}{symbol}", ma.move_san);

                        let resp = ui.selectable_label(
                            i == self.analysis_review_cursor,
                            egui::RichText::new(label).size(12.0).color(class_color).monospace(),
                        );

                        if resp.clicked() {
                            self.analysis_review_cursor = i;
                        }

                        // Show explanation for mistakes/blunders when selected
                        if i == self.analysis_review_cursor {
                            if let Some(ref explanation) = ma.explanation {
                                ui.indent("explanation", |ui| {
                                    ui.label(
                                        egui::RichText::new(explanation)
                                            .size(11.0)
                                            .color(hydra_subtle_text()),
                                    );
                                });
                            }
                            // Show CPL for all non-forced moves
                            if !matches!(ma.classification, crate::analysis::MoveClass::Forced) {
                                ui.indent("cpl_info", |ui| {
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "{} — CPL: {} | Eval: {:.2} → {:.2} | Best: {} ({:.2})",
                                            ma.classification.label(),
                                            ma.cpl,
                                            ma.eval_before as f64 / 100.0,
                                            ma.eval_after as f64 / 100.0,
                                            ma.best_move_uci,
                                            ma.best_eval as f64 / 100.0,
                                        ))
                                        .size(10.0)
                                        .color(hydra_subtle_text()),
                                    );
                                });
                            }
                        }
                    }
                });

            // ── Navigation ─────────────────────────────────────────
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui.button("◀ Prev").clicked() && self.analysis_review_cursor > 0 {
                    self.analysis_review_cursor -= 1;
                }
                if ui.button("Next ▶").clicked()
                    && self.analysis_review_cursor + 1 < ga.moves.len()
                {
                    self.analysis_review_cursor += 1;
                }
                ui.separator();

                // Summary counts
                let mut counts = [0u32; 5]; // best, good, inaccuracy, mistake, blunder
                for m in &ga.moves {
                    if m.side != ga.user_color { continue; }
                    match m.classification {
                        crate::analysis::MoveClass::Best => counts[0] += 1,
                        crate::analysis::MoveClass::Good => counts[1] += 1,
                        crate::analysis::MoveClass::Inaccuracy => counts[2] += 1,
                        crate::analysis::MoveClass::Mistake => counts[3] += 1,
                        crate::analysis::MoveClass::Blunder => counts[4] += 1,
                        _ => {}
                    }
                }
                ui.label(egui::RichText::new(format!(
                    "Best:{} Good:{} Inaccuracy:{} Mistake:{} Blunder:{}",
                    counts[0], counts[1], counts[2], counts[3], counts[4],
                )).size(10.0).color(hydra_subtle_text()));
            });
        });
    }

    fn draw_idle_home(&mut self, ui: &mut egui::Ui) {
        let (searching, token_set, lichess_status, status_message, lichess_log) = {
            let state = self.state.lock().unwrap();
            (
                state.search_info.searching,
                !state.lichess_settings.token.is_empty(),
                state.lichess_status.clone(),
                state.status_message.clone(),
                state.lichess_log.clone(),
            )
        };

        let ctx = ui.ctx().clone();
        let status_label = match &lichess_status {
            LichessStatus::Disconnected => "Offline local play ready".to_string(),
            LichessStatus::Connected(user) => format!("Lichess connected as {user}"),
            LichessStatus::InGame(id) => format!("Lichess game active: {id}"),
        };
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

        ui.add_space(12.0);
        ui.horizontal_wrapped(|ui| {
            ui.vertical(|ui| {
                ui.label(
                    egui::RichText::new("Focalors")
                        .size(28.0)
                        .strong()
                        .color(hydra_accent()),
                );
                ui.label(
                    egui::RichText::new("Train, play, and review chess in one focused workspace.")
                        .size(13.0)
                        .color(hydra_subtle_text()),
                );
            });

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                hydra_badge(ui, &status_label, hydra_panel_alt_fill());
            });
        });

        ui.add_space(12.0);
        self.draw_profile_card(ui);
        ui.add_space(14.0);
        ui.horizontal_wrapped(|ui| {
            if ui.add(home_nav_button(self.home_page == HomePage::Overview, "Overview")).clicked() {
                self.home_page = HomePage::Overview;
            }

            if ui.add(home_nav_button(self.home_page == HomePage::Lichess, "Lichess")).clicked() {
                self.home_page = HomePage::Lichess;
            }

            if ui.add(home_nav_button(self.home_page == HomePage::Import, "Import")).clicked() {
                self.home_page = HomePage::Import;
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
                self.draw_local_setup_card(ui, searching, &lichess_status);

                ui.add_space(12.0);
                ui.label(
                    egui::RichText::new(status_message)
                        .size(12.0)
                        .color(hydra_subtle_text()),
                );

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

                if has_games {
                    ui.add_space(14.0);
                    self.draw_opening_stats(ui);
                }
            }
            HomePage::Lichess => {
                self.draw_lichess_setup_card(ui, &lichess_status, searching, token_set);

                if !lichess_log.is_empty() {
                    ui.add_space(12.0);
                    hydra_card_frame().show(ui, |ui| {
                        ui.label(
                            egui::RichText::new("Recent Activity")
                                .size(13.0)
                                .strong()
                                .color(hydra_accent()),
                        );
                        ui.add_space(8.0);
                        egui::ScrollArea::vertical()
                            .max_height(220.0)
                            .stick_to_bottom(true)
                            .show(ui, |ui| {
                                for log in lichess_log.iter().rev().take(40).rev() {
                                    ui.label(egui::RichText::new(log).size(10.5).monospace().color(hydra_subtle_text()));
                                }
                            });
                    });
                }
            }
            HomePage::Import => self.draw_pgn_import(ui),
            HomePage::Progress => self.draw_coaching_report(ui),
            HomePage::Statistics => self.draw_statistics_panel(ui),
            HomePage::History => {
                self.draw_game_history(ui);
                if self.replay_game.is_some() {
                    ui.add_space(10.0);
                    self.draw_replay_panel(ui);
                }
            }
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

    fn draw_local_setup_card(
        &mut self,
        ui: &mut egui::Ui,
        searching: bool,
        lichess_status: &LichessStatus,
    ) {
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
                        if ui.add(secondary_button("Advanced")).clicked() {
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
            let can_start = !searching && matches!(lichess_status, LichessStatus::Disconnected);
            let start_clicked = ui
                .add_enabled_ui(can_start, |ui| {
                    ui.add_sized([ui.available_width(), 38.0], primary_button("Start Local Game"))
                        .clicked()
                })
                .inner;
            if start_clicked {
                self.begin_local_game(self.local_side_choice.resolve());
            }

            if !matches!(lichess_status, LichessStatus::Disconnected) {
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new("Disconnect Lichess before starting an offline local game.")
                        .size(11.0)
                        .color(hydra_warning()),
                );
            }
        });
    }

    fn draw_pgn_import(&mut self, ui: &mut egui::Ui) {
        hydra_card_frame().show(ui, |ui| {
            ui.label(
                egui::RichText::new("Import PGN")
                    .strong()
                    .size(16.0)
                    .color(hydra_accent()),
            );
            ui.label(
                egui::RichText::new(
                    "Paste a game exported from Lichess or Chess.com to run a full engine review.",
                )
                .size(12.0)
                .color(hydra_subtle_text()),
            );

            ui.add_space(10.0);
            ui.add(
                egui::TextEdit::multiline(&mut self.pgn_import_text)
                    .font(egui::TextStyle::Monospace)
                    .desired_rows(12)
                    .desired_width(f32::INFINITY)
                    .hint_text("[Event \"…\"]\n[White \"…\"]\n…\n\n1. e4 e5 2. Nf3 Nc6 …"),
            );

            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if ui.add(primary_button("Parse PGN")).clicked() {
                    self.pgn_import_error = None;
                    match crate::pgn::parse_pgn(&self.pgn_import_text) {
                        Ok(parsed) => {
                            let profile_name = self.profile.as_ref().map(|p| p.name.clone());
                            self.pgn_import_user_color = crate::pgn::user_color_from_headers(
                                &parsed,
                                profile_name.as_deref(),
                            );
                            self.pgn_import_parsed = Some(parsed);
                        }
                        Err(e) => {
                            self.pgn_import_parsed = None;
                            self.pgn_import_error = Some(e);
                        }
                    }
                }
                if !self.pgn_import_text.is_empty()
                    && ui.add(secondary_button("Clear")).clicked()
                {
                    self.pgn_import_text.clear();
                    self.pgn_import_parsed = None;
                    self.pgn_import_error = None;
                }
            });
        });

        if let Some(err) = self.pgn_import_error.clone() {
            ui.add_space(10.0);
            hydra_callout_frame().show(ui, |ui| {
                ui.label(
                    egui::RichText::new(err)
                        .size(12.0)
                        .strong()
                        .color(hydra_danger()),
                );
            });
        }

        if let Some(parsed) = self.pgn_import_parsed.clone() {
            ui.add_space(12.0);
            hydra_card_frame().show(ui, |ui| {
                ui.label(
                    egui::RichText::new("Parsed game")
                        .strong()
                        .size(13.0)
                        .color(hydra_accent()),
                );
                ui.add_space(6.0);
                let row = |ui: &mut egui::Ui, label: &str, value: &str| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(label)
                                .size(11.0)
                                .strong()
                                .color(hydra_subtle_text()),
                        );
                        ui.label(egui::RichText::new(value).size(12.0).color(hydra_text()));
                    });
                };
                row(ui, "White", parsed.white.as_deref().unwrap_or("—"));
                row(ui, "Black", parsed.black.as_deref().unwrap_or("—"));
                row(ui, "Result", parsed.result.as_deref().unwrap_or("—"));
                row(ui, "Moves", &parsed.uci_moves.len().to_string());

                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Analyze as")
                            .size(11.0)
                            .strong()
                            .color(hydra_subtle_text()),
                    );
                    if ui
                        .selectable_label(self.pgn_import_user_color == Color::White, "White")
                        .clicked()
                    {
                        self.pgn_import_user_color = Color::White;
                    }
                    if ui
                        .selectable_label(self.pgn_import_user_color == Color::Black, "Black")
                        .clicked()
                    {
                        self.pgn_import_user_color = Color::Black;
                    }
                });

                ui.add_space(10.0);
                let analysis_idle = matches!(
                    *self.analysis_state.lock().unwrap(),
                    AnalysisState::Idle
                );
                let analyze = ui.add_enabled(
                    analysis_idle,
                    primary_button(if analysis_idle {
                        "Analyze with Engine"
                    } else {
                        "Analysis already running"
                    }),
                );
                if analyze.clicked() {
                    let uci_moves = parsed.uci_moves.clone();
                    let user_color = self.pgn_import_user_color;
                    self.start_analysis(uci_moves, user_color);
                    self.home_page = HomePage::Progress;
                }
            });
        }
    }

    fn draw_lichess_setup_card(
        &mut self,
        ui: &mut egui::Ui,
        lichess_status: &LichessStatus,
        searching: bool,
        token_set: bool,
    ) {
        hydra_card_frame().show(ui, |ui| {
            ui.label(
                egui::RichText::new("Lichess")
                    .strong()
                    .size(16.0)
                    .color(hydra_accent()),
            );
            ui.label(
                egui::RichText::new(
                    "Use Focalors online only when you want bot games or API-driven testing against Lichess AI.",
                )
                .size(12.0)
                .color(hydra_subtle_text()),
            );

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                let (label, tone) = match lichess_status {
                    LichessStatus::Disconnected => ("Disconnected".to_string(), hydra_subtle_text()),
                    LichessStatus::Connected(user) => (format!("Connected as {user}"), hydra_success()),
                    LichessStatus::InGame(id) => (format!("In game {id}"), hydra_success()),
                };
                ui.label(
                    egui::RichText::new("Status")
                        .size(11.0)
                        .strong()
                        .color(hydra_subtle_text()),
                );
                ui.label(egui::RichText::new(label).size(12.0).strong().color(tone));
            });

            ui.add_space(12.0);
            let mut token = self.state.lock().unwrap().lichess_settings.token.clone();
            ui.label(
                egui::RichText::new("API Token")
                    .size(11.0)
                    .strong()
                    .color(hydra_subtle_text()),
            );
            ui.add(
                egui::TextEdit::singleline(&mut token)
                    .password(true)
                    .desired_width(f32::INFINITY)
                    .hint_text("Paste your Lichess bot token"),
            );
            self.state.lock().unwrap().lichess_settings.token = token;

            ui.add_space(10.0);
            let connect_clicked = ui
                .add_enabled_ui(!searching, |ui| {
                    let button = match lichess_status {
                        LichessStatus::Disconnected => primary_button("Connect to Lichess"),
                        _ => danger_button("Disconnect Lichess"),
                    };
                    ui.add_sized([ui.available_width(), 38.0], button).clicked()
                })
                .inner;
            if connect_clicked {
                match lichess_status {
                    LichessStatus::Disconnected => self.start_lichess(),
                    _ => {
                        let mut s = self.state.lock().unwrap();
                        s.lichess_status = LichessStatus::Disconnected;
                        s.lichess_log.push("Disconnected".to_string());
                        s.status_message = "Disconnected from Lichess.".to_string();
                    }
                }
            }

            if matches!(lichess_status, LichessStatus::Connected(_)) {
                ui.add_space(10.0);
                ui.separator();
                ui.add_space(8.0);
                ui.label(egui::RichText::new("Challenge Setup").strong().color(hydra_accent()));
                ui.label(
                    egui::RichText::new("Tune the AI level and clock before sending a challenge.")
                        .size(11.0)
                        .color(hydra_subtle_text()),
                );

                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new("Stockfish level")
                        .size(11.0)
                        .strong()
                        .color(hydra_subtle_text()),
                );
                ui.add(egui::Slider::new(&mut self.stockfish_level, 1..=8));

                let mut ls = self.state.lock().unwrap().lichess_settings.clone();
                ui.add_space(6.0);
                egui::Grid::new("lichess_challenge_form")
                    .num_columns(2)
                    .spacing([12.0, 8.0])
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new("Minutes")
                                .size(11.0)
                                .strong()
                                .color(hydra_subtle_text()),
                        );
                        let mut mins = ls.clock_minutes as i32;
                        ui.add(egui::DragValue::new(&mut mins).range(1..=180).suffix(" min"));
                        ls.clock_minutes = mins as u32;
                        ui.end_row();

                        ui.label(
                            egui::RichText::new("Increment")
                                .size(11.0)
                                .strong()
                                .color(hydra_subtle_text()),
                        );
                        let mut inc = ls.clock_increment as i32;
                        ui.add(egui::DragValue::new(&mut inc).range(0..=60).suffix(" s"));
                        ls.clock_increment = inc as u32;
                        ui.end_row();
                    });
                self.state.lock().unwrap().lichess_settings = ls;

                ui.add_space(8.0);
                if ui
                    .add_sized([ui.available_width(), 34.0], secondary_button("Challenge Stockfish"))
                    .clicked()
                {
                    self.challenge_stockfish();
                }
            }

            ui.add_space(8.0);
            ui.label(
                egui::RichText::new(if token_set {
                    "Token detected and ready for online play."
                } else {
                    "No token loaded yet."
                })
                .size(11.0)
                .color(hydra_subtle_text()),
            );
        });
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
                        "Manual tuning for analysis runs and Lichess fallback. Timed local games still use the selected difficulty profile.",
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
            let show_home_screen = !state.local_game.active
                && !matches!(state.lichess_status, LichessStatus::InGame(_));
            let mode_label = if show_home_screen {
                "Home"
            } else if state.local_game.active {
                "Local Play"
            } else {
                "Lichess"
            };
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
                        self.draw_idle_home(ui);
                    });
            } else {
                ui.add_space(6.0);
                self.draw_board(ui);
            }
        });
    }
}

impl FocalorsApp {
    // ── Chess board ────────────────────────────────────────────────────

    fn draw_board(&mut self, ui: &mut egui::Ui) {
        let board = self.state.lock().unwrap().board.clone();
        let available = ui.available_size();
        let max_height = (available.y - 18.0).max(180.0);
        let board_size = available.x.min(max_height);
        let sq_size = board_size / 8.0;
        let board_total_height = board_size + 28.0;

        let light = egui::Color32::from_rgb(227, 218, 201);
        let dark = egui::Color32::from_rgb(120, 99, 81);
        let selected_color = egui::Color32::from_rgba_premultiplied(216, 179, 92, 180);
        let legal_color = egui::Color32::from_rgba_premultiplied(148, 126, 98, 110);

        // Compute legal moves from selected square
        let legal_targets: Vec<u8> = if let Some(from_sq) = self.selected_square {
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
        let allow_input = self.local_input_enabled();
        let dragged_from_sq = self.drag_state.map(|drag| drag.from_sq);
        let drag_hover_sq = self.drag_state.and_then(|drag| {
            board_square_from_pos(board_rect, sq_size, self.flipped, drag.pointer_pos)
        });

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

                // Square color
                let is_light = (display_rank + display_file) % 2 == 0;
                let base_color = if is_light { light } else { dark };
                let color = if self.selected_square == Some(sq_idx) {
                    selected_color
                } else if drag_hover_sq == Some(sq_idx) {
                    egui::Color32::from_rgba_premultiplied(201, 150, 74, 135)
                } else if legal_targets.contains(&sq_idx) {
                    legal_color
                } else {
                    base_color
                };

                painter.rect_filled(rect, 0.0, color);

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

        if let Some(drag_state) = &mut self.drag_state {
            if let Some(pos) = response.interact_pointer_pos() {
                drag_state.pointer_pos = pos;
            }
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
        if let Some(drag_state) = self.drag_state {
            if let Some(texture) = self
                .piece_textures
                .get(&(drag_state.piece_color, drag_state.piece))
            {
                let piece_size = sq_size * 0.94;
                let shadow_rect = egui::Rect::from_center_size(
                    egui::pos2(drag_state.pointer_pos.x + 3.0, drag_state.pointer_pos.y + 5.0),
                    egui::vec2(piece_size, piece_size),
                );
                painter.rect_filled(
                    shadow_rect,
                    6.0,
                    egui::Color32::from_rgba_premultiplied(0, 0, 0, 70),
                );
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

        self.draw_promotion_window(ui.ctx());
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
                    || !matches!(s.lichess_status, LichessStatus::Disconnected)
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
            searcher.set_position_history(position_hashes);

            let result = if let Some(ref request) = local_request {
                if let Some(depth_cap) = request.depth_cap {
                    searcher.search_timed_with_depth_cap(&board, request.time_limit_ms, depth_cap)
                } else {
                    searcher.search_timed(&board, request.time_limit_ms)
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
                    let score = state.search_info.score;
                    let score_text = if score.abs() > 20000 {
                        format!("M{}", (29000 - score.abs() + 1) / 2)
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
        let (local_active, lichess_status) = {
            let state = self.state.lock().unwrap();
            (state.local_game.active, state.lichess_status.clone())
        };

        if local_active {
            self.draw_local_game_controls(ui);
        } else if matches!(lichess_status, LichessStatus::InGame(_)) {
            self.draw_lichess_game_controls(ui);
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
            self.draw_analysis_progress(ui);
            self.draw_analysis_review(ui);

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

    fn draw_lichess_game_controls(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            let (status, logs) = {
                let state = self.state.lock().unwrap();
                (state.lichess_status.clone(), state.lichess_log.clone())
            };

            hydra_card_frame().show(ui, |ui| {
                ui.label(
                    egui::RichText::new("Lichess Session")
                        .strong()
                        .size(16.0)
                        .color(hydra_accent()),
                );
                ui.add_space(10.0);

                match &status {
                    LichessStatus::InGame(id) => {
                        hydra_badge(ui, &format!("Current game: {id}"), hydra_success());
                    }
                    LichessStatus::Connected(user) => {
                        hydra_badge(ui, &format!("Connected as {user}"), hydra_success());
                    }
                    LichessStatus::Disconnected => {
                        hydra_badge(ui, "Disconnected", hydra_danger());
                    }
                }

                ui.add_space(10.0);
                ui.horizontal_wrapped(|ui| {
                    if ui.add_sized([138.0, 34.0], secondary_button("Flip Board")).clicked() {
                        self.flipped = !self.flipped;
                    }
                    if ui.add_sized([138.0, 34.0], danger_button("Disconnect")).clicked() {
                        let mut s = self.state.lock().unwrap();
                        s.lichess_status = LichessStatus::Disconnected;
                        s.lichess_log.push("Disconnected".to_string());
                        s.status_message = "Disconnected from Lichess.".to_string();
                    }
                });

                if !logs.is_empty() {
                    ui.add_space(10.0);
                    ui.separator();
                    ui.add_space(8.0);
                    ui.label(egui::RichText::new("Log").strong().color(hydra_accent()));
                    egui::ScrollArea::vertical()
                        .max_height(180.0)
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            for log in logs.iter().rev().take(20).rev() {
                                ui.label(egui::RichText::new(log).size(10.0).monospace());
                            }
                        });
                }
            });
        });
    }

    // ── Lichess integration ────────────────────────────────────────────

    fn start_lichess(&mut self) {
        let (state, token, local_before_connect) = {
            let state = self.state.clone();
            let s = self.state.lock().unwrap();
            let mut paused_local_game = s.local_game.clone();
            paused_local_game.time_control.clear_active();
            (state, s.lichess_settings.token.clone(), paused_local_game)
        };

        if token.is_empty() {
            let mut s = state.lock().unwrap();
            s.lichess_log.push("Error: No token set".to_string());
            return;
        }

        {
            let mut s = self.state.lock().unwrap();
            invalidate_local_search(&mut s, false);
            s.local_game.time_control.clear_active();
            s.local_game.active = false;
            s.local_game.outcome = None;
            s.status_message = "Connecting to Lichess... Local play paused.".to_string();
        }
        self.selected_square = None;
        self.drag_state = None;
        self.pending_promotion = None;

        thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let client = reqwest::Client::new();

                let resp = client
                    .get("https://lichess.org/api/account")
                    .bearer_auth(&token)
                    .send()
                    .await;

                match resp {
                    Ok(r) if r.status().is_success() => {
                        let account: serde_json::Value = r.json().await.unwrap_or_default();
                        let username = account["username"]
                            .as_str()
                            .unwrap_or("unknown")
                            .to_string();
                        let mut s = state.lock().unwrap();
                        s.lichess_username = username.to_lowercase();
                        s.lichess_status = LichessStatus::Connected(username.clone());
                        s.lichess_log.push(format!("Connected as {username}"));
                    }
                    Ok(r) => {
                        let body = r.text().await.unwrap_or_default();
                        let mut s = state.lock().unwrap();
                        let mut resumed_game = local_before_connect.clone();
                        if resumed_game.active && resumed_game.outcome.is_none() {
                            resumed_game.time_control.start_turn_now();
                        }
                        s.local_game = resumed_game;
                        s.status_message = "Lichess auth failed. Local play resumed.".to_string();
                        s.lichess_log.push(format!("Auth failed: {body}"));
                    }
                    Err(e) => {
                        let mut s = state.lock().unwrap();
                        let mut resumed_game = local_before_connect.clone();
                        if resumed_game.active && resumed_game.outcome.is_none() {
                            resumed_game.time_control.start_turn_now();
                        }
                        s.local_game = resumed_game;
                        s.status_message = "Lichess connection failed. Local play resumed.".to_string();
                        s.lichess_log.push(format!("Connection failed: {e}"));
                    }
                }

                lichess_event_loop(state.clone(), token.clone()).await;
            });
        });
    }

    fn challenge_stockfish(&self) {
        let state = self.state.clone();
        let token = self.state.lock().unwrap().lichess_settings.token.clone();
        let level = self.stockfish_level;
        let ls = self.state.lock().unwrap().lichess_settings.clone();

        thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let client = reqwest::Client::new();
                let resp = client
                    .post(format!("https://lichess.org/api/challenge/ai"))
                    .bearer_auth(&token)
                    .form(&[
                        ("level", level.to_string()),
                        ("clock.limit", (ls.clock_minutes * 60).to_string()),
                        ("clock.increment", ls.clock_increment.to_string()),
                    ])
                    .send()
                    .await;

                match resp {
                    Ok(r) if r.status().is_success() => {
                        let body: serde_json::Value = r.json().await.unwrap_or_default();
                        let game_id = body["id"].as_str().unwrap_or("unknown");
                        let mut s = state.lock().unwrap();
                        s.lichess_log.push(format!("Challenge sent! Game: {game_id}"));
                        s.lichess_status = LichessStatus::InGame(game_id.to_string());
                    }
                    Ok(r) => {
                        let body = r.text().await.unwrap_or_default();
                        let mut s = state.lock().unwrap();
                        s.lichess_log.push(format!("Challenge failed: {body}"));
                    }
                    Err(e) => {
                        let mut s = state.lock().unwrap();
                        s.lichess_log.push(format!("Challenge error: {e}"));
                    }
                }
            });
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
    theme_rgb([30, 35, 43], [249, 250, 252])
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

fn hydra_card_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(hydra_panel_fill())
        .stroke(egui::Stroke::new(1.0, hydra_border()))
        .corner_radius(3)
        .inner_margin(egui::Margin::same(18))
        .shadow(egui::epaint::Shadow {
            offset: [0, 0],
            blur: 0,
            spread: 0,
            color: egui::Color32::TRANSPARENT,
        })
}

fn hydra_callout_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(hydra_accent_soft())
        .stroke(egui::Stroke::new(1.0, hydra_accent()))
        .corner_radius(2)
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
    });
}

fn metric_tile(ui: &mut egui::Ui, label: &str, value: &str, tone: egui::Color32) {
    ui.vertical(|ui| {
        ui.set_min_width(96.0);
        ui.label(
            egui::RichText::new(label)
                .size(10.0)
                .strong()
                .color(hydra_subtle_text()),
        );
        ui.add_space(2.0);
        ui.label(
            egui::RichText::new(value)
                .size(16.0)
                .strong()
                .color(tone),
        );
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
    visuals.widgets.noninteractive.corner_radius = egui::CornerRadius::same(2);
    visuals.widgets.inactive.bg_fill = hydra_panel_alt_fill();
    visuals.widgets.inactive.weak_bg_fill = hydra_panel_alt_fill();
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, hydra_border());
    visuals.widgets.inactive.corner_radius = egui::CornerRadius::same(2);
    visuals.widgets.hovered.bg_fill = hydra_panel_raised_fill();
    visuals.widgets.hovered.weak_bg_fill = hydra_panel_raised_fill();
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, hydra_accent());
    visuals.widgets.hovered.corner_radius = egui::CornerRadius::same(2);
    visuals.widgets.active.bg_fill = hydra_panel_raised_fill();
    visuals.widgets.active.weak_bg_fill = hydra_panel_raised_fill();
    visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, hydra_accent());
    visuals.widgets.active.corner_radius = egui::CornerRadius::same(2);
    visuals.widgets.open.bg_fill = hydra_panel_raised_fill();
    visuals.widgets.open.weak_bg_fill = hydra_panel_raised_fill();
    visuals.widgets.open.bg_stroke = egui::Stroke::new(1.0, hydra_accent());
    visuals.widgets.open.corner_radius = egui::CornerRadius::same(2);
    ctx.set_visuals(visuals);

    let mut style = (*ctx.global_style()).clone();
    style.spacing.item_spacing = egui::vec2(10.0, 8.0);
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

    let (base_time_ms, _) = uci::allocate_time(
        remaining_ms,
        state.local_game.time_control.increment_ms,
        None,
    );
    let scaled_time_ms = ((base_time_ms as f32) * strength_config.time_scale).round() as u64;
    let safe_max_ms = remaining_ms.saturating_sub(50).max(1);
    let min_time_ms = safe_max_ms.min(80);
    let time_limit_ms = scaled_time_ms
        .max(min_time_ms)
        .max(strength_config.min_think_ms)
        .min(safe_max_ms);

    Some(LocalSearchRequest {
        time_limit_ms,
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

// ── Lichess event loop (runs in background) ────────────────────────────────

async fn lichess_event_loop(state: Arc<Mutex<SharedState>>, token: String) {
    use futures_util::StreamExt;

    let client = reqwest::Client::new();
    let resp = match client
        .get("https://lichess.org/api/stream/event")
        .bearer_auth(&token)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            let mut s = state.lock().unwrap();
            s.lichess_log.push(format!("Event stream error: {e}"));
            s.lichess_status = LichessStatus::Disconnected;
            return;
        }
    };

    let mut stream = resp.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        // Check if we should disconnect
        {
            let s = state.lock().unwrap();
            if matches!(s.lichess_status, LichessStatus::Disconnected) {
                return;
            }
        }

        let chunk = match chunk {
            Ok(c) => c,
            Err(_) => continue,
        };

        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(pos) = buffer.find('\n') {
            let line: String = buffer.drain(..=pos).collect();
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Ok(event) = serde_json::from_str::<serde_json::Value>(line) {
                let event_type = event["type"].as_str().unwrap_or("");

                match event_type {
                    "gameStart" => {
                        let game_id = event["game"]["gameId"]
                            .as_str()
                            .unwrap_or("")
                            .to_string();
                        if !game_id.is_empty() {
                            {
                                let mut s = state.lock().unwrap();
                                s.lichess_log
                                    .push(format!("Game started: {game_id}"));
                                s.lichess_status = LichessStatus::InGame(game_id.clone());
                            }

                            // Play the game in a separate task
                            let state2 = state.clone();
                            let token2 = token.clone();
                            tokio::spawn(async move {
                                play_lichess_game(state2, token2, game_id).await;
                            });
                        }
                    }
                    "gameFinish" => {
                        let mut s = state.lock().unwrap();
                        s.lichess_log.push("Game finished".to_string());
                        let username = s.lichess_username.clone();
                        s.lichess_status = LichessStatus::Connected(username);
                    }
                    "challenge" => {
                        // Auto-accept challenges
                        let challenge_id =
                            event["challenge"]["id"].as_str().unwrap_or("").to_string();
                        if !challenge_id.is_empty() {
                            let mut s = state.lock().unwrap();
                            s.lichess_log
                                .push(format!("Accepting challenge: {challenge_id}"));
                            drop(s);
                            let _ = client
                                .post(format!(
                                    "https://lichess.org/api/challenge/{challenge_id}/accept"
                                ))
                                .bearer_auth(&token)
                                .send()
                                .await;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

async fn play_lichess_game(
    state: Arc<Mutex<SharedState>>,
    token: String,
    game_id: String,
) {
    use futures_util::StreamExt;

    let client = reqwest::Client::new();
    let resp = match client
        .get(format!(
            "https://lichess.org/api/bot/game/stream/{game_id}"
        ))
        .bearer_auth(&token)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            let mut s = state.lock().unwrap();
            s.lichess_log
                .push(format!("Game stream error: {e}"));
            return;
        }
    };

    let mut stream = resp.bytes_stream();
    let mut buffer = String::new();
    let mut our_color: Option<Color> = None;
    let bot_id = state.lock().unwrap().lichess_username.clone();

    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(_) => continue,
        };

        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(pos) = buffer.find('\n') {
            let line: String = buffer.drain(..=pos).collect();
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let event: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let event_type = event["type"].as_str().unwrap_or("");

            match event_type {
                "gameFull" => {
                    // Determine color
                    let white_id = event["white"]["id"].as_str().unwrap_or("").to_lowercase();
                    let black_id = event["black"]["id"].as_str().unwrap_or("").to_lowercase();
                    if white_id == bot_id {
                        our_color = Some(Color::White);
                    } else if black_id == bot_id {
                        our_color = Some(Color::Black);
                    }

                    let moves_str = event["state"]["moves"].as_str().unwrap_or("");
                    let wtime = event["state"]["wtime"].as_u64();
                    let btime = event["state"]["btime"].as_u64();

                    {
                        let mut s = state.lock().unwrap();
                        s.lichess_log.push(format!(
                            "Playing as {}",
                            if our_color == Some(Color::White) { "White" } else { "Black" }
                        ));
                    }

                    if should_move(moves_str, our_color) {
                        let our_time = match our_color {
                            Some(Color::White) => wtime,
                            Some(Color::Black) => btime,
                            None => None,
                        };
                        let best = calculate_and_play(
                            &state, &client, &token, &game_id, moves_str, our_time,
                        )
                        .await;
                        if let Some(mv) = best {
                            let mut s = state.lock().unwrap();
                            s.lichess_log.push(format!("Played: {mv}"));
                        }
                    }
                }
                "gameState" => {
                    let status = event["status"].as_str().unwrap_or("started");
                    if status != "started" {
                        let mut s = state.lock().unwrap();
                        s.lichess_log.push(format!("Game ended: {status}"));
                        return;
                    }

                    let moves_str = event["moves"].as_str().unwrap_or("");
                    let wtime = event["wtime"].as_u64();
                    let btime = event["btime"].as_u64();

                    if should_move(moves_str, our_color) {
                        let our_time = match our_color {
                            Some(Color::White) => wtime,
                            Some(Color::Black) => btime,
                            None => None,
                        };
                        let best = calculate_and_play(
                            &state, &client, &token, &game_id, moves_str, our_time,
                        )
                        .await;
                        if let Some(mv) = best {
                            let mut s = state.lock().unwrap();
                            s.lichess_log.push(format!("Played: {mv}"));
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

fn should_move(moves_str: &str, our_color: Option<Color>) -> bool {
    let count = if moves_str.is_empty() {
        0
    } else {
        moves_str.split_whitespace().count()
    };
    match our_color {
        Some(Color::White) => count % 2 == 0,
        Some(Color::Black) => count % 2 == 1,
        None => false,
    }
}

async fn calculate_and_play(
    state: &Arc<Mutex<SharedState>>,
    client: &reqwest::Client,
    token: &str,
    game_id: &str,
    moves_str: &str,
    our_time: Option<u64>,
) -> Option<String> {
    let settings = state.lock().unwrap().engine_settings.clone();

    // Build the board
    let mut board = Board::startpos();
    if !moves_str.is_empty() {
        for mv_str in moves_str.split_whitespace() {
            if let Some(mv) = uci::parse_move(&board, mv_str) {
                make_move(&mut board, mv);
            }
        }
    }

    // Update the GUI board
    {
        let mut s = state.lock().unwrap();
        s.board = board.clone();
        s.move_history = moves_str.split_whitespace().map(|s| s.to_string()).collect();
        s.search_info.searching = true;
    }

    // Calculate time
    let time_for_move = match our_time {
        Some(ms) if ms > 0 && ms < 2_000_000_000 => {
            let base = ms / 30;
            let safe = ms.saturating_sub(100);
            base.min(safe).max(200).min(10000)
        }
        _ => settings.think_time_ms,
    };

    // Build position history for repetition detection
    let mut position_hashes = vec![Board::startpos().hash];
    {
        let mut replay_board = Board::startpos();
        if !moves_str.is_empty() {
            for mv_str in moves_str.split_whitespace() {
                if let Some(mv) = crate::uci::parse_move(&replay_board, mv_str) {
                    make_move(&mut replay_board, mv);
                    position_hashes.push(replay_board.hash);
                }
            }
        }
    }

    // Search (run synchronously, then release lock before async GUI update)
    let result = {
        let searcher_arc = state.lock().unwrap().persistent_searcher.clone();
        let mut searcher = searcher_arc.lock().unwrap();
        searcher.set_position_history(position_hashes);
        searcher.search_timed(&board, time_for_move)
    };

    // Update GUI
    {
        let mut s = state.lock().unwrap();
        s.search_info.depth = result.depth;
        s.search_info.score = result.score;
        s.search_info.nodes = result.nodes;
        s.search_info.best_move = result.best_move.to_uci();
        s.search_info.searching = false;
        make_move(&mut s.board, result.best_move);
        s.move_history.push(result.best_move.to_uci());
    }

    // Send move to Lichess
    let mv_str = result.best_move.to_uci();
    let resp = client
        .post(format!(
            "https://lichess.org/api/bot/game/{game_id}/move/{mv_str}"
        ))
        .bearer_auth(token)
        .send()
        .await;

    match resp {
        Ok(r) if r.status().is_success() => Some(mv_str),
        Ok(r) => {
            let body = r.text().await.unwrap_or_default();
            let mut s = state.lock().unwrap();
            s.lichess_log
                .push(format!("Move failed: {body}"));
            None
        }
        Err(e) => {
            let mut s = state.lock().unwrap();
            s.lichess_log
                .push(format!("Move error: {e}"));
            None
        }
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
