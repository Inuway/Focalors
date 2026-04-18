use std::env;

use futures_util::StreamExt;
use reqwest::Client;

extern crate serde;
extern crate serde_json;

use crate::board::Board;
use crate::movegen::make_move;
use crate::uci;

const LICHESS_API: &str = "https://lichess.org/api";

// ── Event stream types ─────────────────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
struct StreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    game: Option<GameStartInfo>,
    challenge: Option<ChallengeInfo>,
}

#[derive(Debug, serde::Deserialize)]
struct GameStartInfo {
    #[serde(rename = "gameId")]
    game_id: String,
}

#[derive(Debug, serde::Deserialize)]
struct ChallengeInfo {
    id: String,
}

// ── Game stream types ──────────────────────────────────────────────────────
// gameFull has players at top level + a nested `state` object.
// gameState has moves/times at top level.

#[derive(Debug, serde::Deserialize)]
struct GameStateData {
    moves: Option<String>,
    wtime: Option<u64>,
    btime: Option<u64>,
}

#[derive(Debug, serde::Deserialize)]
struct PlayerInfo {
    id: Option<String>,
    name: Option<String>,
}

// Wrapper to handle both event types from the game stream
#[derive(Debug, serde::Deserialize)]
struct GameStreamEvent {
    #[serde(rename = "type")]
    event_type: String,

    // gameFull fields
    white: Option<PlayerInfo>,
    black: Option<PlayerInfo>,
    state: Option<GameStateData>,

    // gameState fields (top-level on gameState events)
    moves: Option<String>,
    wtime: Option<u64>,
    btime: Option<u64>,
    status: Option<String>,
}

fn get_token() -> String {
    env::var("LICHESS_TOKEN").expect(
        "LICHESS_TOKEN environment variable not set.\n\
         Set it with: export LICHESS_TOKEN=lip_yourtoken",
    )
}

pub async fn run() {
    let token = get_token();
    let client = Client::new();

    eprintln!("Connecting to Lichess...");
    let resp = client
        .get(format!("{LICHESS_API}/account"))
        .bearer_auth(&token)
        .send()
        .await
        .expect("Failed to connect to Lichess");

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        panic!("Lichess auth failed ({status}): {body}");
    }

    let account: serde_json::Value = resp.json().await.unwrap();
    let username = account["username"].as_str().unwrap_or("unknown");
    eprintln!("Logged in as: {username}");

    let bot_check = account["title"].as_str().unwrap_or("");
    if bot_check != "BOT" {
        eprintln!("Account is not a BOT account. Attempting to upgrade...");
        let resp = client
            .post(format!("{LICHESS_API}/bot/account/upgrade"))
            .bearer_auth(&token)
            .send()
            .await;
        match resp {
            Ok(r) if r.status().is_success() => eprintln!("Successfully upgraded to BOT account!"),
            Ok(r) => {
                let body = r.text().await.unwrap_or_default();
                eprintln!("Could not upgrade to BOT: {body}");
            }
            Err(e) => eprintln!("Upgrade request failed: {e}"),
        }
    }

    let bot_id = username.to_lowercase();

    eprintln!("Listening for challenges and games...");
    let resp = client
        .get(format!("{LICHESS_API}/stream/event"))
        .bearer_auth(&token)
        .send()
        .await
        .expect("Failed to start event stream");

    let mut stream = resp.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => { eprintln!("Stream error: {e}"); continue; }
        };

        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(pos) = buffer.find('\n') {
            let line: String = buffer.drain(..=pos).collect();
            let line = line.trim();
            if line.is_empty() { continue; }

            match serde_json::from_str::<StreamEvent>(line) {
                Ok(event) => handle_event(&client, &token, &bot_id, &event).await,
                Err(e) => eprintln!("Failed to parse event: {e}"),
            }
        }
    }
}

async fn handle_event(client: &Client, token: &str, bot_id: &str, event: &StreamEvent) {
    match event.event_type.as_str() {
        "challenge" => {
            if let Some(challenge) = &event.challenge {
                eprintln!("Accepting challenge: {}", challenge.id);
                let resp = client
                    .post(format!("{LICHESS_API}/challenge/{}/accept", challenge.id))
                    .bearer_auth(token)
                    .send()
                    .await;
                match resp {
                    Ok(r) if r.status().is_success() => eprintln!("Challenge accepted!"),
                    Ok(r) => eprintln!("Failed to accept: {}", r.text().await.unwrap_or_default()),
                    Err(e) => eprintln!("Challenge error: {e}"),
                }
            }
        }
        "gameStart" => {
            if let Some(game) = &event.game {
                let game_id = game.game_id.clone();
                let token = token.to_string();
                let bot_id = bot_id.to_string();
                let client = client.clone();

                tokio::spawn(async move {
                    if let Err(e) = play_game(&client, &token, &bot_id, &game_id).await {
                        eprintln!("Game {game_id} error: {e}");
                    }
                });
            }
        }
        "gameFinish" => eprintln!("Game finished."),
        other => eprintln!("Unhandled event: {other}"),
    }
}

async fn play_game(
    client: &Client,
    token: &str,
    bot_id: &str,
    game_id: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    eprintln!("Playing game: {game_id}");

    let resp = client
        .get(format!("{LICHESS_API}/bot/game/stream/{game_id}"))
        .bearer_auth(token)
        .send()
        .await?;

    let mut stream = resp.bytes_stream();
    let mut buffer = String::new();
    let mut our_color: Option<crate::types::Color> = None;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(pos) = buffer.find('\n') {
            let line: String = buffer.drain(..=pos).collect();
            let line = line.trim();
            if line.is_empty() { continue; }

            let event: GameStreamEvent = match serde_json::from_str(line) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("Game parse error: {e} — {line}");
                    continue;
                }
            };

            match event.event_type.as_str() {
                "gameFull" => {
                    // Determine our color from top-level white/black
                    our_color = determine_color(&event, bot_id);
                    let color_str = match our_color {
                        Some(crate::types::Color::White) => "White",
                        Some(crate::types::Color::Black) => "Black",
                        None => "Unknown",
                    };
                    eprintln!("Game {game_id}: playing as {color_str}");

                    // Moves and times are in the nested `state` object
                    if let Some(state) = &event.state {
                        let moves_str = state.moves.as_deref().unwrap_or("");
                        eprintln!("Game {game_id}: initial moves: '{moves_str}'");
                        if should_move(moves_str, our_color) {
                            let our_time = get_our_time(state, our_color);
                            let best = calculate_move(moves_str, our_time);
                            send_move(client, token, game_id, &best).await;
                        }
                    }
                }
                "gameState" => {
                    // Check game status
                    if let Some(status) = &event.status {
                        if status != "started" {
                            eprintln!("Game {game_id} ended: {status}");
                            return Ok(());
                        }
                    }

                    // Moves and times are at top level on gameState events
                    let moves_str = event.moves.as_deref().unwrap_or("");
                    if should_move(moves_str, our_color) {
                        let our_time = match our_color {
                            Some(crate::types::Color::White) => event.wtime,
                            Some(crate::types::Color::Black) => event.btime,
                            None => None,
                        };
                        let best = calculate_move(moves_str, our_time);
                        send_move(client, token, game_id, &best).await;
                    }
                }
                "chatLine" => {} // ignore chat
                other => eprintln!("Game {game_id}: unhandled event type: {other}"),
            }
        }
    }

    Ok(())
}

fn determine_color(event: &GameStreamEvent, bot_id: &str) -> Option<crate::types::Color> {
    if let Some(white) = &event.white {
        let id = white.id.as_deref().unwrap_or("").to_lowercase();
        let name = white.name.as_deref().unwrap_or("").to_lowercase();
        if id == bot_id || name == bot_id {
            return Some(crate::types::Color::White);
        }
    }
    if let Some(black) = &event.black {
        let id = black.id.as_deref().unwrap_or("").to_lowercase();
        let name = black.name.as_deref().unwrap_or("").to_lowercase();
        if id == bot_id || name == bot_id {
            return Some(crate::types::Color::Black);
        }
    }
    Some(crate::types::Color::White) // fallback
}

fn should_move(moves_str: &str, our_color: Option<crate::types::Color>) -> bool {
    let count = if moves_str.is_empty() { 0 } else { moves_str.split_whitespace().count() };
    match our_color {
        Some(crate::types::Color::White) => count % 2 == 0,
        Some(crate::types::Color::Black) => count % 2 == 1,
        None => false,
    }
}

fn get_our_time(state: &GameStateData, color: Option<crate::types::Color>) -> Option<u64> {
    match color {
        Some(crate::types::Color::White) => state.wtime,
        Some(crate::types::Color::Black) => state.btime,
        None => None,
    }
}

fn calculate_move(moves_str: &str, our_time_ms: Option<u64>) -> String {
    let mut board = Board::startpos();

    if !moves_str.is_empty() {
        for mv_str in moves_str.split_whitespace() {
            if let Some(mv) = uci::parse_move(&board, mv_str) {
                make_move(&mut board, mv);
            }
        }
    }

    let time_for_move = match our_time_ms {
        Some(ms) if ms > 0 => {
            let base = ms / 30;
            let safe = ms.saturating_sub(100);
            base.min(safe).max(200).min(10000)
        }
        _ => 3000,
    };

    // Build position history for repetition detection
    let mut position_hashes = vec![Board::startpos().hash];
    {
        let mut replay = Board::startpos();
        if !moves_str.is_empty() {
            for mv_str in moves_str.split_whitespace() {
                if let Some(mv) = uci::parse_move(&replay, mv_str) {
                    make_move(&mut replay, mv);
                    position_hashes.push(replay.hash);
                }
            }
        }
    }

    eprintln!("Thinking for {time_for_move}ms (clock: {our_time_ms:?})");
    let mut searcher = crate::search::Searcher::new(32);
    searcher.use_nnue = crate::nnue::network::get_network().is_some();
    searcher.set_position_history(position_hashes);
    let result = searcher.search_timed(&board, time_for_move);
    eprintln!("Chose: {}", result.best_move.to_uci());
    result.best_move.to_uci()
}

async fn send_move(client: &Client, token: &str, game_id: &str, mv: &str) {
    eprintln!("Playing: {mv}");
    let resp = client
        .post(format!("{LICHESS_API}/bot/game/{game_id}/move/{mv}"))
        .bearer_auth(token)
        .send()
        .await;

    match resp {
        Ok(r) if r.status().is_success() => {}
        Ok(r) => eprintln!("Failed to send move: {}", r.text().await.unwrap_or_default()),
        Err(e) => eprintln!("Move send error: {e}"),
    }
}
