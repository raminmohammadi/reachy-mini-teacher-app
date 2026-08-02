#!/usr/bin/env bash
# Desktop launcher for the Reachy Mini Teacher App.
#
# Flow:
#   1. Check the wireless Reachy daemon is reachable on the network.
#   2. Start reachy-mini-teacher-app from the project venv (background).
#   3. Wait for the FastAPI dashboard at http://localhost:7860 to come up.
#   4. Open the dashboard in the default browser.
#   5. On any failure, show a zenity dialog with the tail of the log.

set -u

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_DIR/venv"
APP_BIN="$VENV_DIR/bin/reachy-mini-teacher-app"
DAEMON_HOST="reachy-mini.local"
DAEMON_PORT="8000"
DAEMON_BASE="http://$DAEMON_HOST:$DAEMON_PORT/api"
DAEMON_URL="$DAEMON_BASE/daemon/status"
WEB_UI_URL="http://localhost:7860"
DAEMON_TIMEOUT=45       # seconds to wait for the daemon HTTP API
WAKE_TIMEOUT=30         # seconds to wait for daemon to transition to RUNNING
WEB_UI_TIMEOUT=60       # seconds to wait for the dashboard FastAPI
                        # (torch + YOLO model load is ~11 s on CPU)

LOG_DIR="$HOME/.local/share/reachy-mini-teacher-app"
LAUNCHER_LOG="$LOG_DIR/launcher.log"
APP_LOG="$LOG_DIR/app.log"

mkdir -p "$LOG_DIR"

# When started from the desktop icon there is no visible window, so the
# elderly user has no idea whether anything is happening.  Re-launch
# ourselves inside a gnome-terminal so progress + live logs are visible.
# REACHY_LAUNCHER_IN_TERMINAL prevents an infinite re-exec loop.
if [ -z "${REACHY_LAUNCHER_IN_TERMINAL:-}" ] \
   && [ -z "${REACHY_LAUNCHER_NO_TERMINAL:-}" ] \
   && command -v gnome-terminal >/dev/null 2>&1; then
    export REACHY_LAUNCHER_IN_TERMINAL=1
    exec gnome-terminal --title="Reachy Teacher" -- bash "$0" "$@"
fi

exec > >(tee -a "$LAUNCHER_LOG") 2>&1
echo
echo "===== Launch at $(date -Iseconds) ====="

show_error() {
    local title="$1" body="$2"
    if command -v zenity >/dev/null 2>&1; then
        zenity --error --title="$title" --width=520 --text="$body" 2>/dev/null || true
    else
        notify-send "$title" "$body" 2>/dev/null || echo -e "$title\n$body" >&2
    fi
}

show_log() {
    local title="$1" file="$2"
    [ -f "$file" ] || return 0
    if command -v zenity >/dev/null 2>&1; then
        zenity --text-info --title="$title" --width=900 --height=600 \
               --filename="$file" 2>/dev/null || true
    fi
}

fail() {
    local msg="$1"
    echo "FAIL: $msg"
    show_error "Reachy Teacher App — Error" "$msg

Click OK to see the log."
    if [ -s "$APP_LOG" ]; then
        show_log "Reachy Teacher App — App log" "$APP_LOG"
    else
        show_log "Reachy Teacher App — Launcher log" "$LAUNCHER_LOG"
    fi
    # Keep the terminal window open so the user can read the messages above.
    if [ -t 0 ]; then
        echo
        echo "Press Enter to close this window..."
        read -r _ || true
    fi
    exit 1
}

# ---------------------------------------------------------------------------
# 0. Sanity checks
# ---------------------------------------------------------------------------
[ -d "$VENV_DIR" ] || fail "Project virtualenv missing at:
$VENV_DIR"
[ -x "$APP_BIN" ]  || fail "reachy-mini-teacher-app not installed in venv:
$APP_BIN

Run 'pip install -e .' inside the project venv first."

# Already running? Just open the browser.
if curl -fsS --max-time 2 "$WEB_UI_URL" -o /dev/null; then
    echo "Dashboard already up — opening browser."
    xdg-open "$WEB_UI_URL" >/dev/null 2>&1 &
    exit 0
fi

# ---------------------------------------------------------------------------
# 1. Wait for the wireless Reachy daemon
# ---------------------------------------------------------------------------
echo "Checking daemon at $DAEMON_URL ..."
if ! curl -fsS --max-time 3 "$DAEMON_URL" -o /dev/null; then
    notify-send "Reachy Teacher App" "Looking for Reachy on the network…" 2>/dev/null || true
    ok=0
    for i in $(seq 1 "$DAEMON_TIMEOUT"); do
        if curl -fsS --max-time 2 "$DAEMON_URL" -o /dev/null; then
            ok=1; break
        fi
        sleep 1
    done
    if [ "$ok" != "1" ]; then
        fail "Could not reach Reachy on the network.

Tried: $DAEMON_URL
Waited: ${DAEMON_TIMEOUT}s

Please make sure:
  • Reachy is turned on
  • Reachy is connected to the same WiFi as this computer"
    fi
fi
echo "Daemon reachable."

# ---------------------------------------------------------------------------
# 1b. Make sure the daemon is RUNNING with motors ENABLED, and the robot has
#     been physically woken up (head out of the shell).
# ---------------------------------------------------------------------------
# This mirrors what the "Start" button in Reachy Mini Control does:
# POST /api/daemon/start?wake_up=true.  That endpoint is a no-op if the
# daemon is already RUNNING, so when we get here in RUNNING state and the
# robot might be in sleep pose (from a previous Stop click), we first call
# /daemon/stop?goto_sleep=false to transition to STOPPED, then /daemon/start
# to come back up cleanly with wake_up=true.

# Print "<state> <motor_mode>" parsed from /api/daemon/status, else "unknown unknown".
read_robot_state() {
    curl -fsS --max-time 3 "$DAEMON_URL" 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("unknown unknown"); sys.exit(0)
bs = d.get("backend_status") or {}
print(d.get("state", "unknown"), bs.get("motor_control_mode", "unknown"))
' 2>/dev/null || echo "unknown unknown"
}

wait_for_state() {
    local target="$1" timeout="$2" label="$3"
    for i in $(seq 1 "$timeout"); do
        read STATE MOTOR_MODE < <(read_robot_state)
        if [ "$STATE" = "$target" ]; then
            return 0
        fi
        sleep 1
    done
    fail "Daemon did not reach '$target' state within ${timeout}s ($label).
Last state: $STATE motor_mode=$MOTOR_MODE"
}

read STATE MOTOR_MODE < <(read_robot_state)
echo "Daemon state=$STATE motor_mode=$MOTOR_MODE"

# If daemon is running but we just relaunched (i.e., the user clicked
# Stop then Start), we still need to fire a fresh wake_up.  /daemon/start
# is a no-op while RUNNING — so first stop without going to sleep, then
# start with wake_up.
if [ "$STATE" = "running" ]; then
    echo "Daemon already running — cycling so wake_up actually fires."
    echo "POST /api/daemon/stop?goto_sleep=false"
    curl -fsS --max-time 5 -X POST "$DAEMON_BASE/daemon/stop?goto_sleep=false" -o /dev/null || \
        fail "Could not stop the running Reachy daemon."
    wait_for_state "stopped" 20 "after /daemon/stop"
fi

echo "POST /api/daemon/start?wake_up=true"
curl -fsS --max-time 5 -X POST "$DAEMON_BASE/daemon/start?wake_up=true" -o /dev/null || \
    fail "Could not start the Reachy daemon at $DAEMON_HOST."
wait_for_state "running" "$WAKE_TIMEOUT" "after /daemon/start"

# /daemon/start with wake_up=true enables motors and plays wake_up internally
# before returning to RUNNING state.  Double-check motors are enabled.
read STATE MOTOR_MODE < <(read_robot_state)
if [ "$MOTOR_MODE" != "enabled" ]; then
    echo "Motors $MOTOR_MODE after start — forcing enabled."
    curl -fsS --max-time 5 -X POST "$DAEMON_BASE/motors/set_mode/enabled" -o /dev/null || \
        fail "Failed to enable Reachy's motors."
    for i in $(seq 1 10); do
        read STATE MOTOR_MODE < <(read_robot_state)
        [ "$MOTOR_MODE" = "enabled" ] && break
        sleep 1
    done
fi
echo "Daemon ready: state=$STATE motors=$MOTOR_MODE"

# ---------------------------------------------------------------------------
# 2. Launch the teacher app in the background
# ---------------------------------------------------------------------------
cd "$REPO_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# The NVIDIA driver on this machine is non-functional (nvidia-smi fails),
# yet a discrete GPU is physically present.  Without this guard, torch /
# ultralytics probe the broken kernel module during YOLO init and the
# process aborts (C-level crash, no Python traceback).  Forcing the CUDA
# enumeration to be empty keeps everything on CPU, which is plenty fast
# enough for 25 Hz face tracking.
export CUDA_VISIBLE_DEVICES=""

: > "$APP_LOG"
echo "Starting reachy-mini-teacher-app ..."
nohup "$APP_BIN" >> "$APP_LOG" 2>&1 &
APP_PID=$!
disown "$APP_PID" 2>/dev/null || true
echo "App pid=$APP_PID"

# ---------------------------------------------------------------------------
# 3. Wait for the FastAPI dashboard
# ---------------------------------------------------------------------------
ok=0
for i in $(seq 1 "$WEB_UI_TIMEOUT"); do
    if curl -fsS --max-time 2 "$WEB_UI_URL" -o /dev/null; then
        ok=1; break
    fi
    if ! kill -0 "$APP_PID" 2>/dev/null; then
        ERR_TAIL=$(tail -n 30 "$APP_LOG" 2>/dev/null || true)
        fail "The teacher app crashed before the dashboard started.

Last log lines:
$ERR_TAIL"
    fi
    sleep 1
done

if [ "$ok" != "1" ]; then
    fail "The dashboard did not start within ${WEB_UI_TIMEOUT}s.

Tried: $WEB_UI_URL
Log: $APP_LOG"
fi

# ---------------------------------------------------------------------------
# 4. Open the browser
# ---------------------------------------------------------------------------
echo "Opening browser at $WEB_UI_URL"
xdg-open "$WEB_UI_URL" >/dev/null 2>&1 &
echo "===== Launcher finished OK ====="

# ---------------------------------------------------------------------------
# 5. Keep the terminal open and stream live app logs so the user can see
#    what the robot is doing.  The app itself is detached (nohup) so closing
#    this window does NOT stop Reachy — use the 'Stop Reachy Teacher' icon.
# ---------------------------------------------------------------------------
if [ -t 0 ]; then
    echo
    echo "--------------------------------------------------------------------"
    echo " Reachy Teacher is running."
    echo " Live logs below.  Close this window or press Ctrl-C to hide them."
    echo " (The robot keeps running — use 'Stop Reachy Teacher' to stop it.)"
    echo "--------------------------------------------------------------------"
    echo
    exec tail -n 0 -F "$APP_LOG"
fi
exit 0
