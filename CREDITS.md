# Credits

Attribution for third-party assets included in this repository.

## Chess piece graphics

The chess piece images in [`assets/pieces/`](assets/pieces/) (both SVG and PNG)
are by **Colin M.L. Burnett** (Wikipedia user
[Cburnett](https://en.wikipedia.org/wiki/User:Cburnett)). They are the same
piece set that originated on Wikipedia in 2006 and is now used by lichess and
many other open-source chess projects.

- **Source:** <https://commons.wikimedia.org/wiki/Category:SVG_chess_pieces>
- **License:** [Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/)

The original SVGs are included unmodified. The PNGs alongside them are
rasterizations of those SVGs and are distributed under the same CC BY-SA 3.0
license.

These piece images retain their original CC BY-SA 3.0 license; the rest of
Focalors is licensed under GPL-3.0-or-later (see [`LICENSE`](LICENSE)).

## NNUE network

The current shipping NNUE network ([`nets/current.nnue`](nets/current.nnue), `gen10`)
was trained by **Luc Vedrenne** ([@ListIndexOutOfRange](https://github.com/ListIndexOutOfRange))
through 10 generations of self-play fine-tuning, contributed via
[PR #2](https://github.com/Inuway/Focalors/pull/2).

## GPU training pipeline (experimental, `gpu-training` branch only)

The optional GPU NNUE training path uses the [Burn](https://burn.dev/)
machine learning framework by [Tracel AI](https://github.com/tracel-ai),
licensed under Apache-2.0 OR MIT. Burn is pulled in **only** when the
`gpu-training` Cargo feature is enabled; the shipping `focalors` binary
on `main` does not depend on Burn.

CPU training (used to produce all shipping nets to date) does not depend
on Burn — see [`src/trainer.rs`](src/trainer.rs).
