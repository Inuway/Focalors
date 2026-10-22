#!/bin/sh
# Adds Focalors to your app menu, or removes it again with --remove.
# Writes a single launcher file into your home folder; nothing needs root.
# Keep the Focalors folder where it is, or run this again after moving it.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
entry="$apps/focalors.desktop"

if [ "${1:-}" = "--remove" ]; then
    rm -f "$entry"
    echo "Removed the Focalors launcher ($entry)."
    exit 0
fi

if [ ! -x "$here/focalors" ]; then
    echo "The focalors program was not found next to this script ($here)." >&2
    exit 1
fi

mkdir -p "$apps"
cat > "$entry" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Focalors
Comment=Chess coach that runs on your computer
Exec="$here/focalors"
Icon=$here/focalors.png
Terminal=false
Categories=Game;BoardGame;
Keywords=chess;engine;puzzles;training;
StartupWMClass=focalors
DESKTOP
chmod 644 "$entry"
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$apps" 2>/dev/null || true
fi
echo "Added Focalors to your app menu ($entry)."
