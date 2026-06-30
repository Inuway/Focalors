# Mascot art

`placeholder.png` is a temporary stand-in for the coaching mascot shown on the
Game Review page. Replace it with real art.

When you add real sprites, the intended direction (see `draw_review_mascot` and
`FocalorsApp::new` in `src/gui.rs`) is a small per-mood set keyed to the move
classification:

- `neutral.png`  - default / Good / Forced / Inaccuracy
- `happy.png`    - Best / Brilliant
- `worried.png`  - Mistake / Blunder

Guidelines:
- Square canvas, transparent background, roughly 256x256 (drawn at ~56px today,
  so leave it crisp at small sizes).
- Embedded into the binary via `include_bytes!`, so the shipped build always
  matches what users download (no load-from-disk).
