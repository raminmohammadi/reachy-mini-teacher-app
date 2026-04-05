"use strict";

const API = "";          // same origin
const POLL_MS = 2000;    // refresh interval

let knownMsgIds = new Set();
let clearViewFlag = false;

// ── Utility ──────────────────────────────────────────────────────────────────
function fmtTime(ts) {
  if (!ts) return "";
  try {
    const d = new Date(ts.includes("T") ? ts : ts + "Z");
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch { return ts; }
}

function setStatus(text, cls) {
  const el = document.getElementById("status-badge");
  el.textContent = text;
  el.className = "status-badge" + (cls ? " " + cls : "");
}

// ── Status poll ───────────────────────────────────────────────────────────────
async function fetchStatus() {
  const r = await fetch(`${API}/api/status`);
  if (!r.ok) throw new Error(r.status);
  return r.json();
}

function applyStatus(data) {
  document.getElementById("val-mode").textContent    = data.mode    || "—";
  document.getElementById("val-profile").textContent = data.profile || "default";
  document.getElementById("val-session").textContent = data.session_id ?? "—";
  document.getElementById("val-msgs").textContent    = data.message_count ?? "—";
  document.getElementById("val-start").textContent   = fmtTime(data.start_time) || "—";
  setStatus("🟢 Online", "online");

  // Profile buttons
  const profiles = data.profiles || ["default"];
  const active   = data.profile  || "default";
  const container = document.getElementById("profile-buttons");
  container.innerHTML = "";
  profiles.forEach(name => {
    const btn = document.createElement("button");
    btn.className = "profile-btn" + (name === active ? " active" : "");
    btn.textContent = name === "default" ? "🌐 Default" : "📚 " + name.replace(/_/g, " ");
    btn.onclick = () => switchProfile(name);
    container.appendChild(btn);
  });
}

// ── Messages poll ─────────────────────────────────────────────────────────────
async function fetchMessages(sessionId) {
  const r = await fetch(`${API}/api/messages?session_id=${sessionId}&limit=60`);
  if (!r.ok) throw new Error(r.status);
  return r.json();
}

function renderMessages(msgs) {
  const box = document.getElementById("transcript");
  if (!msgs || !msgs.length) return;

  const empty = box.querySelector(".empty-state");
  if (empty) empty.remove();

  let added = false;
  msgs.forEach((m, idx) => {
    const key = `${m.session_id || 0}-${idx}-${m.role}-${(m.content || "").slice(0, 20)}`;
    if (clearViewFlag || !knownMsgIds.has(key)) {
      knownMsgIds.add(key);
      const div = document.createElement("div");
      div.className = "msg " + (m.role === "user" ? "user" : "assistant");
      div.innerHTML = `
        <span class="role">${m.role === "user" ? "🗣 User" : "🤖 Assistant"}</span>
        <span class="content">${escHtml(m.content || "")}</span>
        <span class="time">${fmtTime(m.timestamp)}</span>`;
      box.appendChild(div);
      added = true;
    }
  });
  clearViewFlag = false;
  if (added) box.scrollTop = box.scrollHeight;
}

function escHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// ── Profile switch ────────────────────────────────────────────────────────────
async function switchProfile(name) {
  const msg = document.getElementById("switch-msg");
  msg.textContent = "Switching…";
  try {
    const r = await fetch(`${API}/api/profile/${encodeURIComponent(name)}`, { method: "POST" });
    const data = await r.json();
    if (data.ok) {
      msg.textContent = `✓ Switched to ${name}`;
      knownMsgIds.clear();
    } else {
      msg.textContent = data.error || "Failed";
    }
  } catch (e) {
    msg.textContent = "Error: " + e.message;
  }
  setTimeout(() => { msg.textContent = ""; }, 3000);
}

// ── Clear view ────────────────────────────────────────────────────────────────
function clearView() {
  document.getElementById("transcript").innerHTML =
    '<div class="empty-state">Waiting for conversation…</div>';
  knownMsgIds.clear();
  clearViewFlag = true;
}

// ── Main poll loop ────────────────────────────────────────────────────────────
let lastSessionId = null;

async function poll() {
  try {
    const status = await fetchStatus();
    applyStatus(status);
    const sid = status.session_id;
    if (sid !== null && sid !== undefined) {
      if (sid !== lastSessionId) {
        knownMsgIds.clear();
        lastSessionId = sid;
      }
      const { messages } = await fetchMessages(sid);
      renderMessages(messages);
    }
  } catch (e) {
    setStatus("🔴 Offline", "error");
  }
}

poll();
setInterval(poll, POLL_MS);
