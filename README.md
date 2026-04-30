# Focalors

Focalors is a chess engine and learning app I built in Rust. It runs entirely offline - local play against the engine, post-game review with explanations, puzzles drawn from your own mistakes, and a stats view that lives in a small SQLite file on your machine. No accounts, no cloud, no telemetry.

![Focalors desktop GUI showing a live local game](assets/screenshots/hero.png)

The thing I wanted to figure out is whether you can have both a strong engine *and* readable explanations of what it's doing. Most engines optimize one or the other. Focalors keeps two evaluations side by side: an NNUE network drives playing strength during search, and a hand-crafted evaluation runs alongside it whenever the app needs to explain *why* a move was bad - a hanging piece, a worsening pawn structure, lost king safety. The same core powers the desktop GUI and a standard UCI engine for other front-ends.

![Focalors showcase with the redesigned local play and statistics views](assets/screenshots/showcase.png)

## Elo 2200-2400~? (And why it's not (very) important)
My goal with this project is to make the learning experience for chess possible without the need of having a permanent internet connection or paying a subscription. As earlier mentioned, I want users to be capable of having the full control over their data, games and progress, all on their own machine, always accessible. While of course the strength of the engine is to be further improved, this project does not aim to be "the strongest" or compete with state-of-the-art engines such as Stockfish. The engine just needs to be strong enough to teach human understandable positions and help the average, advanced, or possibly even masters improve their understanding. (With the sole exception of Satoru Gojo aka Magnus Carlsen) I love chess and wanna do a small contribution to people trying to get further into the game. (I also lack the hardware for training huge NNUE nets so if anybody wanna help feel free lol. More to that in TECHNICAL.md further mentioned below)


## Running it

The simplest path is to grab a prebuilt binary from the [Releases page](https://github.com/Inuway/Focalors/releases) - Linux, macOS, and Windows builds are attached to each tagged release. Run the binary and the GUI opens.

If you'd rather build from source you'll need a Rust toolchain, then:

```bash
cargo build --release
./target/release/focalors gui
```

If you want to plug Focalors into another chess GUI, `focalors uci` runs the standard UCI protocol on stdio.

## Where to dig in

[docs/TECHNICAL.md](docs/TECHNICAL.md) has the engine internals, evaluation design, and the NNUE training workflow. [CONTRIBUTING.md](CONTRIBUTING.md) has the practical workflow if you want to send a patch.

## License

GPL-3.0-or-later.
