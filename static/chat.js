// DEFAULTS is injected by the template as window.DEFAULTS before this script loads.
let modelDefaults = {}; // populated from API
const state = {
  sessions: [],
  current: null,    // full session object
  pendingFiles: [],
  busy: false,
  generating: null, // { sid, step, total } while a generation is in flight
  editingMsgId: null, // message id being edited (rewind target)
};

const $ = (id) => document.getElementById(id);

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok && res.status !== 204) {
    let msg = res.statusText;
    try { msg = (await res.json()).error || msg; } catch {}
    throw new Error(msg);
  }
  if (res.status === 204) return null;
  return res.json();
}

async function loadSessions() {
  state.sessions = await api("/api/sessions");
  renderSessionList();
}

function renderSessionList() {
  const list = $("session-list");
  list.innerHTML = "";
  for (const s of state.sessions) {
    const el = document.createElement("div");
    el.className = "session-item" + (state.current && state.current.id === s.id ? " active" : "");
    el.innerHTML = `
      <div class="session-name">
        <span>${escapeHtml(s.name)}</span>
        <button class="session-del" title="Delete">✕</button>
      </div>
      <div class="session-meta">${s.config.model} · ${s.message_count} msg</div>
    `;
    el.addEventListener("click", (ev) => {
      if (ev.target.closest(".session-del")) return;
      selectSession(s.id);
    });
    el.querySelector(".session-del").addEventListener("click", async (ev) => {
      ev.stopPropagation();
      if (!confirm(`Delete "${s.name}"?`)) return;
      await api(`/api/sessions/${s.id}`, { method: "DELETE" });
      if (state.current && state.current.id === s.id) state.current = null;
      await loadSessions();
      renderChat();
    });
    list.appendChild(el);
  }
}

async function getModelDefaults(modelKey) {
  if (modelDefaults[modelKey]) {
    return modelDefaults[modelKey];
  }
  try {
    const data = await api("/api/models");
    modelDefaults = data.model_defaults || {};
    return modelDefaults[modelKey] || DEFAULTS;
  } catch {
    return DEFAULTS;
  }
}

async function createSession() {
  const modelKey = $("cfg-model").value || "flux2-klein-4b";
  const defaults = await getModelDefaults(modelKey);
  const data = await api("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: `Session ${new Date().toLocaleString()}`,
      config: {
        model: modelKey,
        steps: defaults.steps, guidance: defaults.guidance,
        seed: defaults.seed, width: defaults.width, height: defaults.height,
        use_small_decoder: isKleinModel(modelKey) && $("cfg-small-decoder").checked,
      },
    }),
  });
  await loadSessions();
  await selectSession(data.id);
  return data;
}

async function selectSession(id) {
  state.current = await api(`/api/sessions/${id}`);
  $("session-name").disabled = false;
  $("session-name").value = state.current.name;
  loadConfigUI(state.current.config);
  renderSessionList();
  renderChat();
}

function loadConfigUI(cfg) {
  $('cfg-model').value = cfg.model;
  $('cfg-steps').value = cfg.steps;
  $('cfg-guidance').value = cfg.guidance;
  $('cfg-seed').value = cfg.seed;
  $('cfg-width').value = cfg.width;
  $('cfg-height').value = cfg.height;
  const supportsSmallDecoder = isKleinModel(cfg.model);
  $('small-decoder-label').style.display = supportsSmallDecoder ? '' : 'none';
  $('cfg-small-decoder').checked = supportsSmallDecoder && !!cfg.use_small_decoder;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function makePendingEl() {
  const el = document.createElement("div");
  el.className = "pending";
  el.innerHTML = `
    <div class="pending-row"><span class="spinner"></span> <span class="progress-label" id="prog-label">generating…</span></div>
    <div class="progress-bar-wrap"><div class="progress-bar-fill" id="prog-fill"></div></div>`;
  return el;
}

function renderChat() {
  const chat = $("chat");
  chat.innerHTML = "";
  if (!state.current) {
    chat.innerHTML = `<div class="empty">Create a new session to get started.</div>`;
    return;
  }
  if (state.current.messages.length === 0 && !state.generating) {
    chat.innerHTML = `<div class="empty">Send a message to generate your first image.</div>`;
    return;
  }
  for (const m of state.current.messages) {
    chat.appendChild(renderMessage(m));
  }
  // Re-attach progress bar if this session has an active generation.
  if (state.generating && state.generating.sid === state.current.id) {
    const pending = makePendingEl();
    chat.appendChild(pending);
    // Restore last-known progress immediately so there's no blank bar flicker.
    const { step, total } = state.generating;
    if (total > 0) {
      pending.querySelector("#prog-fill").style.width = Math.round((step / total) * 100) + "%";
      pending.querySelector("#prog-label").textContent = `step ${step} / ${total}`;
    }
  }
  chat.scrollTop = chat.scrollHeight;
}

function renderMessage(m) {
  const el = document.createElement("div");
  el.className = `msg ${m.role}`;
  const sid = state.current.id;
  if (m.role === "user") {
    let thumbs = "";
    if (m.images && m.images.length) {
      thumbs = `<div class="thumbs">${m.images.map(n =>
        `<img src="/api/sessions/${sid}/files/${encodeURIComponent(n)}" />`).join("")}</div>`;
    }
    el.innerHTML = `<div class="bubble">${escapeHtml(m.text || "")}${thumbs}</div>`;
    if (m.id) {
      const editBtn = document.createElement("button");
      editBtn.className = "edit-btn";
      editBtn.title = "Edit and resend";
      editBtn.textContent = "✎";
      editBtn.addEventListener("click", () => {
        state.editingMsgId = m.id;
        $("prompt-input").value = m.text || "";
        $("prompt-input").style.height = "auto";
        $("prompt-input").style.height = Math.min($("prompt-input").scrollHeight, 200) + "px";
        $("prompt-input").focus();
        $("send-btn").textContent = "Resend";
      });
      el.appendChild(editBtn);
    }
  } else {
    if (m.error) {
      el.innerHTML = `<div class="bubble err">⚠ ${escapeHtml(m.error)}</div>`;
    } else {
      const cfg = m.config_snapshot || {};
      const smallDecInfo = cfg.use_small_decoder ? " · small decoder" : "";
      const meta = `${cfg.model || ""} · steps ${cfg.steps} · g ${cfg.guidance} · seed ${cfg.seed} · ${cfg.width}×${cfg.height}${smallDecInfo}${m.elapsed_s ? ` · ${m.elapsed_s}s` : ""}` ;
      el.innerHTML = `
        <div class="bubble">
          <img src="/api/sessions/${sid}/files/${encodeURIComponent(m.image)}" />
          <div class="meta">${escapeHtml(meta)}</div>
        </div>`;
    }
  }
  return el;
}

function addPendingFiles(files) {
  for (const f of files) state.pendingFiles.push(f);
  renderPendingUploads();
}

function renderPendingUploads() {
  const c = $("pending-uploads");
  c.innerHTML = "";
  state.pendingFiles.forEach((f, i) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `📎 ${escapeHtml(f.name)} <button title="Remove">✕</button>`;
    chip.querySelector("button").addEventListener("click", () => {
      state.pendingFiles.splice(i, 1);
      renderPendingUploads();
    });
    c.appendChild(chip);
  });
}

async function sendMessage() {
  if (state.busy) return;
  const prompt = $("prompt-input").value.trim();
  if (!prompt) return;

  if (!state.current) await createSession();

  await saveConfig();

  const fd = new FormData();
  fd.append("prompt", prompt);
  for (const f of state.pendingFiles) fd.append("images", f);
  if ($("chain-output").checked) fd.append("use_last_output", "1");
  if (state.editingMsgId) fd.append("rewind_to", state.editingMsgId);

  state.busy = true;
  $("send-btn").disabled = true;
  $("send-btn").textContent = "Send";
  $("prompt-input").value = "";
  state.pendingFiles = [];
  renderPendingUploads();

  // Truncate local messages optimistically if rewinding.
  if (state.editingMsgId) {
    const idx = state.current.messages.findIndex(m => m.id === state.editingMsgId);
    if (idx !== -1) state.current.messages = state.current.messages.slice(0, idx);
  }
  state.editingMsgId = null;

  const optimistic = { role: "user", text: prompt, images: [] };
  state.current.messages.push(optimistic);
  const sid = state.current.id;
  state.generating = { sid, step: 0, total: 0 };
  renderChat();
  $("chat").scrollTop = $("chat").scrollHeight;

  const pollTimer = setInterval(async () => {
    try {
      const p = await api(`/api/sessions/${sid}/progress`);
      if (p && p.total > 0) {
        // Always update state so a session switch-back can restore the value.
        state.generating = { sid, step: p.step, total: p.total };
        const pct = Math.round((p.step / p.total) * 100);
        const fill = document.getElementById("prog-fill");
        const label = document.getElementById("prog-label");
        if (fill) fill.style.width = pct + "%";
        if (label) label.textContent = `step ${p.step} / ${p.total}`;
      }
    } catch {}
  }, 300);

  try {
    const res = await fetch(`/api/sessions/${sid}/messages`, {
      method: "POST", body: fd,
    });
    const data = await res.json();
    state.current.messages.pop();
    if (data.user) state.current.messages.push(data.user);
    if (data.assistant) state.current.messages.push(data.assistant);
    state.generating = null;
    renderChat();
    await loadSessions();
  } catch (e) {
    state.generating = null;
    alert("Generation failed: " + e.message);
  } finally {
    clearInterval(pollTimer);
    state.generating = null;
    state.busy = false;
    $("send-btn").disabled = false;
  }
}

async function saveConfig() {
  if (!state.current) return;
  const cfg = {
    model: $("cfg-model").value,
    steps: parseInt($("cfg-steps").value, 10),
    guidance: parseFloat($("cfg-guidance").value),
    seed: parseInt($("cfg-seed").value, 10),
    width: parseInt($("cfg-width").value, 10),
    height: parseInt($("cfg-height").value, 10),
    use_small_decoder: isKleinModel($("cfg-model").value) && $("cfg-small-decoder").checked,
  };
  const name = $("session-name").value.trim() || state.current.name;
  state.current = await api(`/api/sessions/${state.current.id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, config: cfg }),
  });
  renderSessionList();
}

// --- wire up UI ---
function isKleinModel(modelKey) {
  return modelKey && modelKey.startsWith('flux2-klein');
}

$('new-session-btn').addEventListener('click', createSession);
$('config-toggle').addEventListener('click', () => $('config-panel').classList.toggle('open'));
$('attach-btn').addEventListener('click', () => $('file-input').click());
$('file-input').addEventListener('change', (e) => {
  addPendingFiles(e.target.files);
  e.target.value = '';
});
$('send-btn').addEventListener('click', sendMessage);
$('prompt-input').addEventListener('keydown', (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
  if (e.key === "Escape" && state.editingMsgId) {
    state.editingMsgId = null;
    $("prompt-input").value = "";
    $("send-btn").textContent = "Send";
  }
});
$("prompt-input").addEventListener("input", (e) => {
  e.target.style.height = "auto";
  e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
});
$("session-name").addEventListener("change", saveConfig);
$("cfg-model").addEventListener("change", async (e) => {
  const defaults = await getModelDefaults(e.target.value);
  $("cfg-steps").value = defaults.steps;
  $("cfg-guidance").value = defaults.guidance;
  // Show/hide the small decoder option based on whether the new model supports it.
  const klein = isKleinModel(e.target.value);
  $("small-decoder-label").style.display = klein ? '' : 'none';
  if (!klein) $("cfg-small-decoder").checked = false;
  await saveConfig();
});
for (const id of ["cfg-steps", "cfg-guidance", "cfg-seed", "cfg-width", "cfg-height"]) {
  $(id).addEventListener("change", saveConfig);
}
$("cfg-small-decoder").addEventListener("change", saveConfig);

// --- theme ---
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  $("theme-toggle").textContent = theme === "light" ? "\u263D" : "\u2600\uFE0E";
}
(function initTheme() {
  const saved = localStorage.getItem("theme") ||
    (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  applyTheme(saved);
})();
$("theme-toggle").addEventListener("click", () => {
  const next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
  applyTheme(next);
  localStorage.setItem("theme", next);
});

(async () => {
  // Fetch model defaults from API for later use
  try {
    const data = await api("/api/models");
    modelDefaults = data.model_defaults || {};
  } catch (e) {
    console.warn("Failed to fetch model defaults:", e);
  }
  
  await loadSessions();
  if (state.sessions.length) {
    await selectSession(state.sessions[0].id);
  } else {
    const defaultModel = "flux2-klein-4b";
    const defaults = modelDefaults[defaultModel] || DEFAULTS;
    loadConfigUI({
      model: defaultModel,
      steps: defaults.steps, guidance: defaults.guidance,
      seed: defaults.seed, width: defaults.width, height: defaults.height,
      use_small_decoder: false,
    });
  }
})();
