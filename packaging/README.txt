Focalors - a chess coach that runs on your computer

Thank you for downloading Focalors. It runs entirely on your own
machine: play against the engine, review your games with plain-language
explanations of what went wrong, and train on puzzles built from your
own mistakes. No account, no internet required, nothing leaves your
computer.


HOW TO RUN IT

Windows
  Double-click "focalors".
  If Windows shows a blue "Windows protected your PC" box, click
  "More info", then "Run anyway". This only happens the first time, and
  only because the app is made by an independent developer rather than a
  big company. It is not a sign that anything is wrong.

macOS (Apple Silicon, M1 to M4)
  macOS blocks unsigned apps by default. Open the Terminal app in this
  folder and run these three lines:
    xattr -d com.apple.quarantine focalors
    chmod +x focalors
    ./focalors

Linux
  Make it executable, then run it:
    chmod +x focalors
    ./focalors


YOUR DATA

Your games, stats, and puzzles are saved in a small database file in
your user data folder. Nothing is uploaded anywhere. Deleting that file
resets the app.


MORE

  Website:                     https://focalors-chess.com
  Source code and updates:     https://github.com/Inuway/Focalors
  Want to contribute? See the repository above.

Focalors is free and open source under the GNU GPL-3.0 license. The full
license text is in the LICENSE file included here.
