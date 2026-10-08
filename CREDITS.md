# Credits

Attribution for third-party assets included in this repository.

## Chess piece graphics

Each piece set lives in its own directory under
[`assets/pieces/`](assets/pieces/), with the SVG sources and the 128px PNG
rasterizations the app embeds. The sets keep their own licenses; the rest of
Focalors is licensed under GPL-3.0-or-later (see [`LICENSE`](LICENSE)).

### Cburnett (default set)

The pieces in [`assets/pieces/cburnett/`](assets/pieces/cburnett/) are by
**Colin M.L. Burnett** (Wikipedia user
[Cburnett](https://en.wikipedia.org/wiki/User:Cburnett)). They are the same
piece set that originated on Wikipedia in 2006 and is now used by lichess and
many other open-source chess projects.

- **Source:** <https://commons.wikimedia.org/wiki/Category:SVG_chess_pieces>
- **License:** [Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/)

The original SVGs are included unmodified. The PNGs alongside them are
rasterizations of those SVGs and are distributed under the same CC BY-SA 3.0
license.

### RhosGFX

The pieces in [`assets/pieces/rhosgfx/`](assets/pieces/rhosgfx/) are the
"Outline" variant of the **Vector Chess Pieces Pack** by **RhosGFX**.

- **Source:** <https://rhosgfx.itch.io/vector-chess-pieces>
- **License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) (public domain dedication, as stated in the pack's License.txt)

The SVGs are the pack's files renamed to the `wK` to `bP` scheme shared by
every set here, otherwise unmodified. The PNGs are rasterizations of those
SVGs. CC0 asks for no attribution; it is given here with thanks.

## Typeface

The UI font is **Inter** by Rasmus Andersson and the Inter Project Authors
(<https://github.com/rsms/inter>), embedded from
[`assets/fonts/`](assets/fonts/) (Regular and SemiBold) under the
[SIL Open Font License 1.1](assets/fonts/LICENSE-Inter.txt).

## NNUE network

The shipping NNUE network ([`nets/current.nnue`](nets/current.nnue)) is
trained in-house through generations of self-play fine-tuning, starting
from an initial net trained from scratch for this project.
**Luc Vedrenne**
([@ListIndexOutOfRange](https://github.com/ListIndexOutOfRange))
contributed ten generations of that fine-tuning (a chained estimate of
roughly +270 elo across the accepted promotions), together with the
generation recipe still in use, via
[PR #2](https://github.com/Inuway/Focalors/pull/2).

## GPU training pipeline (optional `gpu-training` feature)

The optional GPU NNUE training path uses the [Burn](https://burn.dev/)
machine learning framework by [Tracel AI](https://github.com/tracel-ai),
licensed under Apache-2.0 OR MIT. Burn is pulled in **only** when the
`gpu-training` Cargo feature is enabled; the shipping `focalors` binary
does not depend on Burn. See
[`docs/GPU_TRAINING.md`](docs/GPU_TRAINING.md).

The pure-Rust CPU trainer ([`src/trainer.rs`](src/trainer.rs)) does not
depend on Burn and produces byte-identical nets; either trainer can be
used interchangeably.
