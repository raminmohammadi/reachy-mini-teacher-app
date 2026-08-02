#!/usr/bin/env bash
# Stop the running Reachy Mini Teacher App.
#
# Sends SIGTERM (which main.py handles by calling _do_shutdown), waits a few
# seconds, then SIGKILL if anything is still alive. Reports the result with a
# desktop notification / zenity dialog.

set -u

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
APP_BIN="$REPO_DIR/venv/bin/reachy-mini-teacher-app"
APP_LOG="$HOME/.local/share/reachy-mini-teacher-app/app.log"
# Generous grace so the Gemini Live handler can finish handler.shutdown(),
# which performs a network call to generate the session summary before the
# DB rows are committed.  Anything less and the user loses their session info.
GRACE_SECONDS=30

DAEMON_BASE="http://reachy-mini.local:8000/api"

# Close the lingering "Reachy Teacher" gnome-terminal log window that the
# launcher leaves open with `exec tail -F $APP_LOG`.  We do this AFTER the
# app has fully shut down (so the user can see the final log lines briefly).
close_log_window() {
    # 1. Kill any `tail -F` processes that the launcher is using to stream
    #    $APP_LOG.  Matching on the absolute log path avoids killing unrelated
    #    tails the user may have running.
    local tail_pids
    tail_pids=$(pgrep -u "$USER" -f "tail .* $APP_LOG" 2>/dev/null || true)
    if [ -n "$tail_pids" ]; then
        echo "Closing log window (killing tail pids: $tail_pids)"
        # SIGHUP makes tail exit cleanly; gnome-terminal closes when its
        # only foreground child exits.
        kill -HUP $tail_pids 2>/dev/null || true
    fi
}

# Print the daemon state ("running"/"stopping"/"stopped"/"starting"/"?") or
# "?" on any failure.  Uses single-quoted python -c to avoid bash escaping
# headaches.
read_daemon_state() {
    curl -fsS --max-time 2 "$DAEMON_BASE/daemon/status" 2>/dev/null | python3 -c '
import json, sys
try:
    print(json.load(sys.stdin).get("state", "?"))
except Exception:
    print("?")
' 2>/dev/null || echo "?"
}

# After the app is gone, stop the daemon with goto_sleep=true.  This mirrors
# the GUI's Stop button: the daemon plays the goto_sleep move (head tucks
# into the shell), disables motors, and transitions to the STOPPED state.
# A subsequent launcher run will trigger /daemon/start?wake_up=true which
# is the only reliable way to make wake_up play visibly.
goto_sleep() {
    if ! curl -fsS --max-time 3 "$DAEMON_BASE/daemon/status" -o /dev/null 2>&1; then
        echo "Daemon not reachable — skipping goto_sleep."
        return 0
    fi
    local state
    state=$(read_daemon_state)
    if [ "$state" = "stopped" ]; then
        echo "Daemon already stopped — robot is already asleep."
        return 0
    fi
    echo "POST /api/daemon/stop?goto_sleep=true"
    curl -fsS --max-time 5 -X POST "$DAEMON_BASE/daemon/stop?goto_sleep=true" \
         -o /dev/null 2>&1 || { echo "  (/daemon/stop call failed — ignored)"; return 0; }
    # Poll for STOPPED state; goto_sleep + motor disable takes a few seconds.
    for i in $(seq 1 30); do
        state=$(read_daemon_state)
        if [ "$state" = "stopped" ]; then
            echo "Reachy is asleep (daemon stopped after ${i}s)."
            return 0
        fi
        sleep 1
    done
    echo "  (daemon did not reach stopped state in 30s — last=$state)"
}

LOG_DIR="$HOME/.local/share/reachy-mini-teacher-app"
STOP_LOG="$LOG_DIR/stop.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$STOP_LOG") 2>&1
echo
echo "===== Stop at $(date -Iseconds) ====="

notify_ok() {
    local body="$1"
    notify-send "Reachy Teacher App" "$body" 2>/dev/null || true
}

notify_err() {
    local body="$1"
    if command -v zenity >/dev/null 2>&1; then
        zenity --error --title="Reachy Teacher App — Stop" --width=420 \
               --text="$body" 2>/dev/null || true
    else
        notify-send "Reachy Teacher App" "$body" 2>/dev/null || echo "$body" >&2
    fi
}

# Find PIDs whose command line contains the teacher-app entry point AND whose
# /proc/<pid>/exe is a Python interpreter — this excludes unrelated shells that
# may happen to mention the path string (e.g. this script's own parent).
mapfile -t CANDIDATES < <(pgrep -u "$USER" -f "$APP_BIN" || true)
PIDS=()
for pid in "${CANDIDATES[@]}"; do
    # Skip ourselves and our direct ancestors.
    if [ "$pid" = "$$" ] || [ "$pid" = "$PPID" ]; then
        continue
    fi
    exe=$(readlink "/proc/$pid/exe" 2>/dev/null || true)
    case "$exe" in
        */python|*/python3|*/python3.*) PIDS+=("$pid") ;;
    esac
done

if [ "${#PIDS[@]}" -eq 0 ]; then
    # The app may have already exited on its own (crash, manual Ctrl-C, etc.)
    # but the user clicked Stop because they want the robot asleep, so still
    # send it to sleep.
    echo "No running reachy-mini-teacher-app process found."
    goto_sleep
    close_log_window
    notify_ok "App was not running — robot sent to sleep."
    exit 0
fi

# Returns 0 if pid is alive AND not a zombie, non-zero otherwise.
# Zombies (process state 'Z' in /proc/<pid>/status) are functionally dead:
# the program has exited but the parent hasn't reaped them yet.  This
# happens here because the launcher does `exec tail -F`, so the teacher
# app's parent is `tail`, which never calls wait().  SIGKILL has no
# effect on a zombie, so we must treat them as dead.
is_alive() {
    local pid="$1"
    kill -0 "$pid" 2>/dev/null || return 1
    local state
    state=$(awk '/^State:/ {print $2; exit}' "/proc/$pid/status" 2>/dev/null)
    [ "$state" != "Z" ]
}

echo "Found pids: ${PIDS[*]}"
kill -TERM "${PIDS[@]}" 2>/dev/null || true

# Wait up to GRACE_SECONDS for graceful exit.
for i in $(seq 1 "$GRACE_SECONDS"); do
    still_alive=()
    for pid in "${PIDS[@]}"; do
        if is_alive "$pid"; then
            still_alive+=("$pid")
        fi
    done
    if [ "${#still_alive[@]}" -eq 0 ]; then
        echo "All processes exited cleanly after ${i}s."
        goto_sleep
        close_log_window
        notify_ok "App stopped — session saved."
        exit 0
    fi
    sleep 1
done

# Anything left → SIGKILL.
echo "Still alive after ${GRACE_SECONDS}s, sending SIGKILL: ${still_alive[*]}"
kill -KILL "${still_alive[@]}" 2>/dev/null || true
sleep 1

# Final check.
final=()
for pid in "${still_alive[@]}"; do
    if is_alive "$pid"; then
        final+=("$pid")
    fi
done

if [ "${#final[@]}" -eq 0 ]; then
    goto_sleep
    close_log_window
    notify_ok "App stopped (forced — session may be incomplete)."
    exit 0
fi

notify_err "Could not stop the app.
Remaining pids: ${final[*]}
See: $STOP_LOG"
exit 1
