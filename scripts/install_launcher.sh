#!/usr/bin/env bash
# Install the Reachy Teacher desktop launcher.
#
# Copies the .desktop file into ~/.local/share/applications (so it shows up in
# the activities/app grid) and onto ~/Desktop (so the elderly user has a
# clickable icon), marks both as trusted/executable, and refreshes the desktop
# database.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

LAUNCHER_SH="$SCRIPT_DIR/launch_teacher_app.sh"
STOP_SH="$SCRIPT_DIR/stop_teacher_app.sh"
DESKTOP_SRC="$SCRIPT_DIR/reachy-mini-teacher-app.desktop"
DESKTOP_STOP_SRC="$SCRIPT_DIR/reachy-mini-teacher-app-stop.desktop"

APPS_DIR="$HOME/.local/share/applications"
DESKTOP_DIR="${XDG_DESKTOP_DIR:-$HOME/Desktop}"

mkdir -p "$APPS_DIR" "$DESKTOP_DIR"

# install_entry SRC EXEC BASENAME
#   SRC      — source .desktop file in scripts/
#   EXEC     — absolute path to substitute into Exec=
#   BASENAME — output filename (without directory)
install_entry() {
    local src="$1" exec_path="$2" basename="$3"
    local apps_path="$APPS_DIR/$basename"
    local user_path="$DESKTOP_DIR/$basename"

    [ -f "$src" ]       || { echo "Missing: $src" >&2; exit 1; }
    [ -x "$exec_path" ] || { echo "Missing or not executable: $exec_path" >&2; exit 1; }

    sed -e "s|^Exec=.*|Exec=$exec_path|" "$src" > "$apps_path"
    chmod +x "$apps_path"
    cp "$apps_path" "$user_path"
    chmod +x "$user_path"

    # GNOME ≥ 42 requires desktop files on the user's Desktop to be marked
    # "trusted" via the gio metadata attribute, otherwise they show up as a
    # plain text file.
    if command -v gio >/dev/null 2>&1; then
        gio set "$user_path" metadata::trusted true 2>/dev/null || true
    fi

    echo "  $apps_path"
    echo "  $user_path"
}

chmod +x "$LAUNCHER_SH" "$STOP_SH"

echo "Installed:"
install_entry "$DESKTOP_SRC"      "$LAUNCHER_SH" "reachy-mini-teacher-app.desktop"
install_entry "$DESKTOP_STOP_SRC" "$STOP_SH"     "reachy-mini-teacher-app-stop.desktop"

# Refresh the menu cache so the apps appear immediately.
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APPS_DIR" 2>/dev/null || true
fi

echo
echo "Scripts:"
echo "  $LAUNCHER_SH"
echo "  $STOP_SH"
echo
if ! command -v zenity >/dev/null 2>&1; then
    echo "WARNING: 'zenity' is not installed — error dialogs will fall back to"
    echo "         desktop notifications only. Install it with:"
    echo "             sudo apt-get install zenity"
fi
echo "Done. Double-click 'Reachy Teacher' on the desktop to launch,"
echo "and 'Stop Reachy Teacher' to stop it."
